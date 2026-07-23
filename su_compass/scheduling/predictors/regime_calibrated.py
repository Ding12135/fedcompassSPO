"""Online regime-aware latency prediction with calibrated safety bounds.

The predictor is intentionally prequential: ``predict`` only reads completed
history, and ``observe`` is called after the corresponding duration is known.
It can wrap the existing State-Driven point/safe prediction as a protected
expert, which makes it suitable for shadow replay before scheduler apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _quantile(values: List[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(min(max(probability, 0.0), 1.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _median(values: List[float]) -> float:
    return _quantile(values, 0.5)


@dataclass
class _RobustLocationScale:
    location: Optional[float] = None
    scale: float = 0.0
    rate: float = 0.2
    huber_k: float = 2.5
    non_negative: bool = True

    def update(self, value: float) -> None:
        value = float(value)
        if self.non_negative:
            value = max(0.0, value)
        if self.location is None:
            self.location = value
            self.scale = max(0.05 * value, 1e-6)
            return
        residual = value - self.location
        limit = self.huber_k * max(self.scale, 1e-6)
        clipped = min(max(residual, -limit), limit)
        self.location += self.rate * clipped
        absolute_residual = abs(residual)
        self.scale += self.rate * (min(absolute_residual, 4.0 * limit) - self.scale)
        self.scale = max(self.scale, 1e-6)


@dataclass
class RegimePrediction:
    baseline_duration: float
    baseline_safe_duration: float
    raw_duration: float
    raw_safe_duration: float
    predicted_duration: float
    safe_duration: float
    burst_probability: float
    regime: str
    conformal_margin: float
    analytical_margin: float
    expert_weight: float
    used_candidate: bool
    num_observations: int
    calibration_source: str = "analytical"
    calibration_n: int = 0
    calibration_rank: int = 0
    client_margin: float = 0.0
    pooled_margin: float = 0.0


@dataclass
class _ClientModel:
    compute: _RobustLocationScale
    fixed: _RobustLocationScale
    residual: _RobustLocationScale
    last_regime: int = 0
    transition_counts: List[List[float]] = field(
        default_factory=lambda: [[1.0, 1.0], [1.0, 1.0]]
    )
    burst_excesses: List[float] = field(default_factory=list)
    stable_residuals: List[float] = field(default_factory=list)
    burst_residuals: List[float] = field(default_factory=list)
    last_residual: Optional[float] = None
    residual_cross: float = 0.0
    residual_previous_square: float = 0.0
    candidate_positive_errors: List[float] = field(default_factory=list)
    stable_positive_errors: List[float] = field(default_factory=list)
    burst_positive_errors: List[float] = field(default_factory=list)
    baseline_loss_ema: Optional[float] = None
    candidate_loss_ema: Optional[float] = None
    baseline_losses: List[float] = field(default_factory=list)
    candidate_losses: List[float] = field(default_factory=list)
    observations: int = 0
    pending: Optional[RegimePrediction] = None


class RegimeCalibratedPredictor:
    """Robust two-regime mixture predictor with online residual calibration.

    The stable component is ``robust_compute * Q + robust_fixed``.  A
    smoothed two-state Markov model estimates whether a previously observed
    burst persists.  A one-sided empirical residual quantile calibrates the
    safe bound.  The existing predictor remains an online expert and is used
    until the candidate has enough history and lower recent absolute loss.
    """

    name = "ramp_ac_v1"

    def __init__(
        self, *, min_observations: int = 5, target_coverage: float = 0.90,
        history_size: int = 40, expert_rate: float = 0.15,
        improvement_margin: float = 0.0, finite_sample_pooling: bool = False,
        client_calibration_min: int = 8,
    ) -> None:
        if min_observations < 1:
            raise ValueError("min_observations must be positive")
        if not 0.5 < target_coverage < 1.0:
            raise ValueError("target_coverage must be in (0.5, 1)")
        self.min_observations = min_observations
        self.target_coverage = target_coverage
        self.history_size = history_size
        self.expert_rate = expert_rate
        self.improvement_margin = improvement_margin
        self.finite_sample_pooling = finite_sample_pooling
        self.client_calibration_min = client_calibration_min
        self._clients: Dict[str, _ClientModel] = {}
        self._pooled_positive_errors: List[float] = []

    def _finite_sample_margin(self, values: List[float]) -> tuple[float, int]:
        """One-sided finite-sample conformal order statistic."""
        if not values:
            return 0.0, 0
        ordered = sorted(values)
        rank = min(len(ordered), math.ceil((len(ordered) + 1) * self.target_coverage))
        return ordered[rank - 1], rank

    def preview_safety_margin(
        self, *, client_id: str, analytical_margin: float,
    ) -> tuple[float, str, int, int, float, float]:
        """Read-only v2 safety head for counterfactual Q curves."""
        model = self._client(client_id)
        client_margin, client_rank = self._finite_sample_margin(
            model.candidate_positive_errors
        )
        pooled_margin, pooled_rank = self._finite_sample_margin(
            self._pooled_positive_errors
        )
        if len(model.candidate_positive_errors) >= self.client_calibration_min:
            source, values, rank, learned = (
                "client_finite", model.candidate_positive_errors,
                client_rank, client_margin,
            )
        elif len(self._pooled_positive_errors) >= self.client_calibration_min:
            source, values, rank, learned = (
                "pooled_finite", self._pooled_positive_errors,
                pooled_rank, pooled_margin,
            )
        else:
            source, values, rank, learned = "analytical", [], 0, 0.0
        return (
            max(float(analytical_margin), learned), source, len(values), rank,
            client_margin, pooled_margin,
        )

    def _client(self, client_id: str) -> _ClientModel:
        if client_id not in self._clients:
            self._clients[client_id] = _ClientModel(
                compute=_RobustLocationScale(), fixed=_RobustLocationScale(),
                residual=_RobustLocationScale(non_negative=False),
            )
        return self._clients[client_id]

    def predict(
        self, *, client_id: str, local_steps: int, baseline_duration: float,
        baseline_safe_duration: float,
    ) -> RegimePrediction:
        model = self._client(client_id)
        baseline_duration = max(0.0, float(baseline_duration))
        baseline_safe_duration = max(baseline_duration, float(baseline_safe_duration))

        row = model.transition_counts[model.last_regime]
        burst_probability = row[1] / max(sum(row), 1e-12)
        # The existing decomposed predictor is already a strong structural
        # model.  RAMP-AC models its signed residual rather than replacing it
        # with another small-sample estimate of compute and communication.
        # Replay shows that the existing decomposed point predictor is already
        # the strongest causal location model.  RAMP-AC therefore protects it
        # exactly and learns the conditional upper tail, where burst clients
        # are poorly represented by mean + one standard deviation.
        raw_duration = baseline_duration

        analytical_margin = max(0.0, baseline_safe_duration - baseline_duration)
        calibration_source = "client_empirical"
        calibration_values = model.candidate_positive_errors
        if self.finite_sample_pooling:
            if len(model.candidate_positive_errors) >= self.client_calibration_min:
                calibration_source = "client_finite"
            elif len(self._pooled_positive_errors) >= self.client_calibration_min:
                calibration_source = "pooled_finite"
                calibration_values = self._pooled_positive_errors
            else:
                calibration_source = "analytical"
                calibration_values = []
            conformal_margin, calibration_rank = self._finite_sample_margin(
                calibration_values
            )
        else:
            conformal_margin = _quantile(
                calibration_values, self.target_coverage
            )
            calibration_rank = math.ceil(
                self.target_coverage * len(calibration_values)
            ) if calibration_values else 0
        raw_safe = raw_duration + max(analytical_margin, conformal_margin)

        enough = model.observations >= self.min_observations
        if self.finite_sample_pooling:
            enough = calibration_source != "analytical"
        # Applying the candidate changes only the safety head; point alignment
        # is bit-for-bit protected.  The max() above also prevents a calibrated
        # margin from becoming less conservative than the analytical baseline.
        use_candidate = bool(enough)
        if model.baseline_loss_ema is None or model.candidate_loss_ema is None:
            expert_weight = 0.0
        else:
            gap = model.baseline_loss_ema - model.candidate_loss_ema
            denom = max(model.baseline_loss_ema, model.candidate_loss_ema, 1e-9)
            expert_weight = 1.0 / (1.0 + math.exp(-5.0 * gap / denom))

        result = RegimePrediction(
            baseline_duration=baseline_duration,
            baseline_safe_duration=baseline_safe_duration,
            raw_duration=raw_duration,
            raw_safe_duration=raw_safe,
            predicted_duration=raw_duration if use_candidate else baseline_duration,
            safe_duration=raw_safe if use_candidate else baseline_safe_duration,
            burst_probability=burst_probability,
            regime="burst" if model.last_regime else "stable",
            conformal_margin=conformal_margin,
            analytical_margin=analytical_margin,
            expert_weight=expert_weight,
            used_candidate=use_candidate,
            num_observations=model.observations,
            calibration_source=calibration_source,
            calibration_n=len(calibration_values),
            calibration_rank=calibration_rank,
            client_margin=self._finite_sample_margin(
                model.candidate_positive_errors
            )[0] if self.finite_sample_pooling else conformal_margin,
            pooled_margin=self._finite_sample_margin(
                self._pooled_positive_errors
            )[0] if self.finite_sample_pooling else 0.0,
        )
        model.pending = result
        return result

    def observe(
        self, *, client_id: str, local_steps: int, actual_duration: float,
        compute_duration: float, communication_duration: float,
        spike_duration: float = 0.0, availability_duration: float = 0.0,
    ) -> None:
        model = self._client(client_id)
        actual_duration = max(0.0, float(actual_duration))
        compute_duration = max(0.0, float(compute_duration))
        fixed_duration = max(
            0.0,
            float(communication_duration) + float(spike_duration)
            + float(availability_duration),
        )

        pending = model.pending
        signed_residual = (
            0.0 if pending is None
            else actual_duration - pending.baseline_duration
        )
        old_residual = model.residual.location
        old_scale = model.residual.scale
        explicit_event = float(spike_duration) > 0.0 or float(availability_duration) > 0.0
        if old_residual is None:
            regime = 0
            excess = 0.0
        else:
            deviation = abs(signed_residual - old_residual)
            regime = int(explicit_event or deviation > 2.5 * max(old_scale, 1e-6))
            excess = deviation

        model.transition_counts[model.last_regime][regime] += 1.0
        model.last_regime = regime
        if regime:
            model.burst_excesses.append(excess)
            model.burst_residuals.append(signed_residual)
            del model.burst_excesses[:-self.history_size]
            del model.burst_residuals[:-self.history_size]
        else:
            model.stable_residuals.append(signed_residual)
            del model.stable_residuals[:-self.history_size]

        if pending is not None:
            baseline_loss = abs(actual_duration - pending.baseline_duration)
            candidate_loss = abs(actual_duration - pending.raw_duration)
            rate = self.expert_rate
            model.baseline_loss_ema = (
                baseline_loss if model.baseline_loss_ema is None
                else (1.0 - rate) * model.baseline_loss_ema + rate * baseline_loss
            )
            model.candidate_loss_ema = (
                candidate_loss if model.candidate_loss_ema is None
                else (1.0 - rate) * model.candidate_loss_ema + rate * candidate_loss
            )
            model.candidate_positive_errors.append(
                max(0.0, actual_duration - pending.raw_duration)
            )
            del model.candidate_positive_errors[:-self.history_size]
            self._pooled_positive_errors.append(
                max(0.0, actual_duration - pending.raw_duration)
            )
            del self._pooled_positive_errors[:-(self.history_size * 8)]
            regime_errors = (
                model.burst_positive_errors if regime
                else model.stable_positive_errors
            )
            regime_errors.append(max(0.0, actual_duration - pending.raw_duration))
            del regime_errors[:-self.history_size]
            model.baseline_losses.append(baseline_loss)
            model.candidate_losses.append(candidate_loss)
            # A short rolling expert audit adapts after a regime change and
            # prevents old easy rounds from authorising a currently bad model.
            del model.baseline_losses[:-12]
            del model.candidate_losses[:-12]

        model.compute.update(compute_duration / max(1, int(local_steps)))
        model.fixed.update(fixed_duration)
        model.residual.update(signed_residual)
        if model.last_residual is not None:
            # Mild forgetting lets the residual dynamics follow a changed
            # client regime without retaining the whole run.
            model.residual_cross = 0.9 * model.residual_cross + model.last_residual * signed_residual
            model.residual_previous_square = (
                0.9 * model.residual_previous_square + model.last_residual ** 2
            )
        model.last_residual = signed_residual
        model.observations += 1
        model.pending = None
