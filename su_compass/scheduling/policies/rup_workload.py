"""Composable RUP-Compass workload policy.

Every layer has an independent switch.  The policy is pure with respect to the
FedCompass controller: it consumes one existing-group decision and returns a
decision plus a complete audit row.  ``mode=shadow`` computes the full policy
while dispatching the FedCompass Q; ``mode=off`` is a strict no-op.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Iterable, Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


@dataclass(frozen=True)
class RUPConfig:
    mode: str = "apply"                 # off | shadow | apply
    group_admission_mode: str = "conservative" # off | shadow | apply | conservative
    group_admission_min_group_size: int = 3
    group_admission_late_slack_ratio: float = 0.05
    state_enabled: bool = True
    residual_risk_enabled: bool = True
    trust_region_enabled: bool = True
    soft_boundary_enabled: bool = True
    utility_enabled: bool = True
    budget_enabled: bool = True
    state_warmup_reports: int = 3
    residual_min_reports: int = 5
    residual_window: int = 20
    residual_quantile: float = 0.80
    trust_fc_lower: float = 0.80
    trust_fc_upper: float = 1.20
    trust_prev_lower: float = 0.75
    trust_prev_upper: float = 1.25
    soft_qmin: int = 50
    soft_qmax: int = 180
    utility_beta: float = 0.20
    utility_min_reports: int = 5
    utility_lower: float = 0.90
    utility_upper: float = 1.10
    budget_window: int = 50
    budget_lower_ratio: float = 0.97
    budget_upper_ratio: float = 1.03
    budget_max_adjust_ratio: float = 0.10
    accuracy_priority_enabled: bool = True
    accuracy_q_floor_ratio: float = 1.0
    accuracy_q_boost_ratio: float = 0.05
    accuracy_q_boost_min_confidence: float = 0.5
    accuracy_q_boost_start_accuracy: float = 50.0
    risk_gated_floor_enabled: bool = False
    risk_gated_floor_min_safe_candidates: int = 10
    risk_gated_floor_slack_ratio: float = 0.05
    q_smoothness_enabled: bool = False
    q_smooth_max_increase_ratio: float = 0.10
    q_smooth_max_decrease_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "apply"}:
            raise ValueError("RUP mode must be off, shadow or apply")
        if self.group_admission_mode not in {"off", "shadow", "apply", "conservative"}:
            raise ValueError("RUP group admission mode must be off, shadow, apply or conservative")
        if self.group_admission_min_group_size < 1:
            raise ValueError("group_admission_min_group_size must be positive")
        if self.group_admission_late_slack_ratio < 0:
            raise ValueError("group_admission_late_slack_ratio must be non-negative")
        if not 0.5 <= self.residual_quantile <= 1.0:
            raise ValueError("residual_quantile must be in [0.5, 1]")
        if not 0 < self.utility_lower <= 1 <= self.utility_upper:
            raise ValueError("utility bounds must contain 1")
        if not 0 < self.budget_lower_ratio <= 1 <= self.budget_upper_ratio:
            raise ValueError("budget bounds must contain 1")
        if self.residual_window < 1 or self.budget_window < 1:
            raise ValueError("residual and budget windows must be positive")
        if self.state_warmup_reports < 0 or self.utility_min_reports < 1:
            raise ValueError("invalid warmup/report threshold")
        if not 0 < self.trust_fc_lower <= 1 <= self.trust_fc_upper:
            raise ValueError("FedCompass trust bounds must contain 1")
        if not 0 < self.trust_prev_lower <= 1 <= self.trust_prev_upper:
            raise ValueError("previous-Q trust bounds must contain 1")
        if self.soft_qmin > self.soft_qmax:
            raise ValueError("soft_qmin must not exceed soft_qmax")
        if self.accuracy_q_floor_ratio < 0:
            raise ValueError("accuracy_q_floor_ratio must be non-negative")
        if self.accuracy_q_boost_ratio < 0:
            raise ValueError("accuracy_q_boost_ratio must be non-negative")
        if not 0 <= self.accuracy_q_boost_min_confidence <= 1:
            raise ValueError("accuracy_q_boost_min_confidence must be in [0, 1]")
        if self.accuracy_q_boost_start_accuracy < 0:
            raise ValueError("accuracy_q_boost_start_accuracy must be non-negative")
        if self.risk_gated_floor_min_safe_candidates < 0:
            raise ValueError("risk_gated_floor_min_safe_candidates must be non-negative")
        if self.risk_gated_floor_slack_ratio < 0:
            raise ValueError("risk_gated_floor_slack_ratio must be non-negative")
        if self.q_smooth_max_increase_ratio < 0 or self.q_smooth_max_decrease_ratio < 0:
            raise ValueError("Q smoothness ratios must be non-negative")


@dataclass
class _UtilityState:
    loss_ema: Optional[float] = None
    progress_ema: Optional[float] = None
    num_samples: int = 0
    reports: int = 0


@dataclass(frozen=True)
class RUPDecision:
    client_id: str
    mode: str
    baseline_q: int
    raw_state_q: int
    trust_q: int
    soft_q: int
    utility_q: int
    budget_q: int
    recommended_q: int
    applied_q: int
    state_safe_feasible: bool
    num_safe_candidates: int
    predicted_finish_time: float
    safe_finish_time: float
    expected_arrival_time: float
    latest_arrival_time: float
    residual_margin: float
    residual_count: int
    trust_lower: int
    trust_upper: int
    utility_raw: float
    utility_normalized: float
    utility_confidence: float
    utility_reports: int
    loss_ema: Optional[float]
    progress_ema: Optional[float]
    budget_ratio_before: float
    budget_debt_before: int
    budget_adjustment: int
    pre_accuracy_q: int
    accuracy_priority_q: int
    accuracy_floor_applied: bool
    accuracy_boost_applied: bool
    current_global_accuracy: Optional[float]
    accuracy_boost_stage_active: bool
    risk_gated_floor_allowed: bool
    pre_smooth_q: int
    smooth_q: int
    q_smooth_applied: bool
    previous_q: int
    hit_hard_qmin: bool
    hit_hard_qmax: bool
    hit_soft_qmin: bool
    hit_soft_qmax: bool
    fallback_reason: str
    enabled_layers: str

    def to_trace(self, virtual_time: float, assigned_group: int) -> Dict[str, Any]:
        row = asdict(self)
        row.update({"virtual_time": virtual_time, "assigned_group": assigned_group})
        for key in (
            "state_safe_feasible", "accuracy_floor_applied",
            "accuracy_boost_applied", "accuracy_boost_stage_active",
            "risk_gated_floor_allowed", "q_smooth_applied",
            "hit_hard_qmin", "hit_hard_qmax",
            "hit_soft_qmin", "hit_soft_qmax",
        ):
            row[key] = int(bool(row[key]))
        return row


class RUPWorkloadPolicy:
    """Risk-safe, utility-aware and budget-neutral local-work controller."""

    def __init__(self, predictor: LatencyPredictor, config: RUPConfig) -> None:
        self.predictor = predictor
        self.config = config
        self._utility: Dict[str, _UtilityState] = defaultdict(_UtilityState)
        self._residuals: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=config.residual_window)
        )
        self._pending_prediction: Dict[str, float] = {}
        self._previous_q: Dict[str, int] = {}
        self._budget: Deque[tuple[int, int]] = deque(maxlen=config.budget_window)
        self._current_global_accuracy: Optional[float] = None

    def update_global_accuracy(self, accuracy: Optional[float]) -> None:
        if accuracy is None or not math.isfinite(float(accuracy)):
            return
        self._current_global_accuracy = float(accuracy)

    def observe_upload(self, client_id: str, actual_duration: float, observation: Any = None) -> None:
        predicted = self._pending_prediction.pop(client_id, None)
        if predicted is not None and math.isfinite(actual_duration):
            self._residuals[client_id].append(float(actual_duration) - predicted)
        if observation is None or not getattr(observation, "finite", False):
            return
        loss = getattr(observation, "loss_before", None)
        progress = getattr(observation, "loss_delta_per_step", None)
        if loss is None or not math.isfinite(float(loss)):
            return
        state = self._utility[client_id]
        beta = self.config.utility_beta
        state.loss_ema = float(loss) if state.loss_ema is None else (1 - beta) * state.loss_ema + beta * float(loss)
        if progress is not None and math.isfinite(float(progress)):
            state.progress_ema = (
                float(progress) if state.progress_ema is None
                else (1 - beta) * state.progress_ema + beta * float(progress)
            )
        state.num_samples = max(1, int(getattr(observation, "num_train_samples", 1)))
        state.reports += 1

    def passthrough(
        self, *, client_id: str, q: int, dispatch_time: float,
        expected_arrival_time: float, latest_arrival_time: float, reason: str,
    ) -> RUPDecision:
        """Account for first/new-group dispatches without changing their semantics."""
        previous = self._previous_q.get(client_id, q)
        enabled = [name for name, value in (
            ("state", self.config.state_enabled),
            ("residual_risk", self.config.residual_risk_enabled),
            ("trust", self.config.trust_region_enabled),
            ("soft_boundary", self.config.soft_boundary_enabled),
            ("utility", self.config.utility_enabled),
            ("budget", self.config.budget_enabled),
            ("risk_gated_floor", self.config.risk_gated_floor_enabled),
            ("q_smoothness", self.config.q_smoothness_enabled),
        ) if value]
        return self._finish_noop(
            client_id, q, previous, dispatch_time,
            expected_arrival_time, latest_arrival_time, reason, enabled,
        )

    def decide(
        self, *, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], expected_arrival_time: float,
        latest_arrival_time: float, qmin: int, qmax: int, baseline_q: int,
    ) -> RUPDecision:
        cfg = self.config
        previous_q = self._previous_q.get(client_id, baseline_q)
        enabled = [name for name, value in (
            ("state", cfg.state_enabled), ("residual_risk", cfg.residual_risk_enabled),
            ("trust", cfg.trust_region_enabled), ("soft_boundary", cfg.soft_boundary_enabled),
            ("utility", cfg.utility_enabled), ("budget", cfg.budget_enabled),
            ("risk_gated_floor", cfg.risk_gated_floor_enabled),
            ("q_smoothness", cfg.q_smoothness_enabled),
        ) if value]
        if cfg.mode == "off":
            return self._finish_noop(
                client_id, baseline_q, previous_q, dispatch_time,
                expected_arrival_time, latest_arrival_time, "mode_off", enabled,
            )

        predictions = {}
        safe_qs = []
        residual_margin = self._residual_margin(client_id)
        reports = int(getattr(runtime_state, "num_reports", 0)) if runtime_state else 0
        state_ready = cfg.state_enabled and reports >= cfg.state_warmup_reports
        if state_ready:
            for q in range(qmin, qmax + 1):
                pred = self.predictor.predict(PredictionContext(
                    client_id=client_id, dispatch_time=dispatch_time, local_steps=q,
                    speed_smoothed=speed_smoothed, runtime_state=runtime_state,
                ))
                intrinsic_margin = max(0.0, pred.safe_duration - pred.predicted_duration)
                margin = max(intrinsic_margin, residual_margin) if cfg.residual_risk_enabled else intrinsic_margin
                safe_finish = pred.predicted_finish_time + margin
                predictions[q] = (pred, safe_finish, margin)
                if safe_finish <= latest_arrival_time:
                    safe_qs.append(q)

        fallback = ""
        if not state_ready:
            raw_state_q = baseline_q
            safe_qs = list(range(qmin, qmax + 1))
            fallback = "state_warmup_keep_fedcompass"
        elif not safe_qs:
            raw_state_q = baseline_q
            fallback = "no_safe_q_keep_fedcompass"
        else:
            raw_state_q = min(
                safe_qs,
                key=lambda q: (abs(predictions[q][0].predicted_finish_time - expected_arrival_time), -q),
            )

        trust_lower, trust_upper = qmin, qmax
        trust_q = raw_state_q
        if cfg.trust_region_enabled:
            trust_lower = max(qmin, math.floor(cfg.trust_fc_lower * baseline_q), math.floor(cfg.trust_prev_lower * previous_q))
            trust_upper = min(qmax, math.ceil(cfg.trust_fc_upper * baseline_q), math.ceil(cfg.trust_prev_upper * previous_q))
            stable = [q for q in safe_qs if trust_lower <= q <= trust_upper]
            if stable:
                trust_q = min(stable, key=lambda q: (abs(q - raw_state_q), -q))
            elif safe_qs:
                # Safety outranks the lower trust bound.
                trust_q = min(safe_qs, key=lambda q: (abs(q - raw_state_q), -q))
                fallback = fallback or "trust_empty_safety_priority"

        soft_q = trust_q
        if cfg.soft_boundary_enabled and safe_qs:
            if soft_q <= qmin and cfg.soft_qmin in safe_qs and cfg.soft_qmin <= trust_upper:
                soft_q = cfg.soft_qmin
            if soft_q >= qmax and cfg.soft_qmax in safe_qs:
                pred = predictions.get(cfg.soft_qmax)
                span = max(latest_arrival_time - expected_arrival_time, 1e-8)
                sufficiently_used = pred is not None and pred[0].predicted_finish_time >= expected_arrival_time - 0.25 * span
                if sufficiently_used:
                    soft_q = cfg.soft_qmax

        utility_raw, utility_norm, utility_conf, utility_state = self._utility_score(client_id)
        utility_q = soft_q
        if cfg.utility_enabled and utility_conf > 0 and safe_qs:
            target = int(round(soft_q * utility_norm))
            eligible = [q for q in safe_qs if trust_lower <= q <= trust_upper]
            if eligible:
                utility_q = min(eligible, key=lambda q: (abs(q - target), -q))

        ratio_before, debt_before = self._budget_status()
        budget_q = utility_q
        budget_adjustment = 0
        if cfg.budget_enabled and safe_qs:
            limit = max(1, int(round(baseline_q * cfg.budget_max_adjust_ratio)))
            eligible = [q for q in safe_qs if trust_lower <= q <= trust_upper]
            if ratio_before < cfg.budget_lower_ratio and debt_before > 0 and utility_norm >= 1.0 and eligible:
                cap = min(max(eligible), utility_q + limit, utility_q + debt_before)
                budget_q = max(q for q in eligible if q <= cap)
            elif ratio_before > cfg.budget_upper_ratio and debt_before < 0 and utility_norm <= 1.0 and eligible:
                floor = max(min(eligible), utility_q - limit, utility_q + debt_before)
                budget_q = min(q for q in eligible if q >= floor)
            budget_adjustment = budget_q - utility_q

        pre_accuracy_q = int(budget_q)
        accuracy_floor_applied = False
        accuracy_boost_applied = False
        current_accuracy = self._current_global_accuracy
        boost_stage_active = (
            current_accuracy is not None
            and current_accuracy >= cfg.accuracy_q_boost_start_accuracy
        )
        risk_gated_floor_allowed = True
        if cfg.accuracy_priority_enabled and safe_qs:
            eligible = [q for q in safe_qs if trust_lower <= q <= trust_upper]
            if eligible:
                floor_target = int(math.ceil(baseline_q * cfg.accuracy_q_floor_ratio))
                floor_candidates = [q for q in eligible if q >= floor_target]
                if floor_candidates:
                    floor_q = min(floor_candidates)
                    risk_gated_floor_allowed = self._floor_risk_allowed(
                        q=floor_q,
                        predictions=predictions,
                        safe_qs=safe_qs,
                        expected_arrival_time=expected_arrival_time,
                        latest_arrival_time=latest_arrival_time,
                    )
                    if budget_q < floor_q:
                        if not cfg.risk_gated_floor_enabled or risk_gated_floor_allowed:
                            budget_q = floor_q
                            accuracy_floor_applied = True
                if (
                    boost_stage_active
                    and
                    utility_norm >= 1.0
                    and utility_conf >= cfg.accuracy_q_boost_min_confidence
                    and cfg.accuracy_q_boost_ratio > 0
                ):
                    boost_target = int(math.ceil(baseline_q * (1.0 + cfg.accuracy_q_boost_ratio)))
                    boost_cap = min(max(eligible), boost_target)
                    boost_candidates = [q for q in eligible if q <= boost_cap]
                    if boost_candidates:
                        boost_q = max(boost_candidates)
                        if budget_q < boost_q:
                            budget_q = boost_q
                            accuracy_boost_applied = True

        pre_smooth_q = int(budget_q)
        smooth_q = pre_smooth_q
        q_smooth_applied = False
        if cfg.q_smoothness_enabled:
            lower = max(qmin, math.floor(previous_q * (1.0 - cfg.q_smooth_max_decrease_ratio)))
            upper = min(qmax, math.ceil(previous_q * (1.0 + cfg.q_smooth_max_increase_ratio)))
            smooth_target = min(max(pre_smooth_q, lower), upper)
            if smooth_target != pre_smooth_q:
                q_smooth_applied = True
            if safe_qs:
                eligible = [q for q in safe_qs if lower <= q <= upper]
                if cfg.trust_region_enabled:
                    eligible = [
                        q for q in eligible
                        if trust_lower <= q <= trust_upper
                    ] or eligible
                if eligible:
                    smooth_q = min(eligible, key=lambda q: (abs(q - smooth_target), -q))
                else:
                    smooth_q = smooth_target
            else:
                smooth_q = smooth_target

        recommended_q = min(max(int(smooth_q), qmin), qmax)
        applied_q = baseline_q if cfg.mode == "shadow" else recommended_q
        chosen = predictions.get(recommended_q)
        if chosen is None:
            predicted_finish = dispatch_time + baseline_q * speed_smoothed
            safe_finish = predicted_finish
            chosen_margin = 0.0
        else:
            predicted_finish = chosen[0].predicted_finish_time
            safe_finish = chosen[1]
            chosen_margin = chosen[2]
        actual_prediction = predictions.get(applied_q)
        if actual_prediction is not None:
            # Residual calibration must follow the Q that was really dispatched;
            # in shadow mode this is the FedCompass Q, not the recommendation.
            self._pending_prediction[client_id] = actual_prediction[0].predicted_duration

        self._previous_q[client_id] = applied_q
        self._budget.append((baseline_q, applied_q))
        return RUPDecision(
            client_id=client_id, mode=cfg.mode, baseline_q=baseline_q,
            raw_state_q=raw_state_q, trust_q=trust_q, soft_q=soft_q,
            utility_q=utility_q, budget_q=budget_q, recommended_q=recommended_q,
            applied_q=applied_q, state_safe_feasible=bool(safe_qs),
            num_safe_candidates=len(safe_qs), predicted_finish_time=predicted_finish,
            safe_finish_time=safe_finish, expected_arrival_time=expected_arrival_time,
            latest_arrival_time=latest_arrival_time, residual_margin=chosen_margin,
            residual_count=len(self._residuals[client_id]), trust_lower=trust_lower,
            trust_upper=trust_upper, utility_raw=utility_raw,
            utility_normalized=utility_norm, utility_confidence=utility_conf,
            utility_reports=utility_state.reports, loss_ema=utility_state.loss_ema,
            progress_ema=utility_state.progress_ema, budget_ratio_before=ratio_before,
            budget_debt_before=debt_before, budget_adjustment=budget_adjustment,
            pre_accuracy_q=pre_accuracy_q, accuracy_priority_q=recommended_q,
            accuracy_floor_applied=accuracy_floor_applied,
            accuracy_boost_applied=accuracy_boost_applied,
            current_global_accuracy=current_accuracy,
            accuracy_boost_stage_active=boost_stage_active,
            risk_gated_floor_allowed=risk_gated_floor_allowed,
            pre_smooth_q=pre_smooth_q,
            smooth_q=smooth_q,
            q_smooth_applied=q_smooth_applied,
            previous_q=previous_q, hit_hard_qmin=recommended_q == qmin,
            hit_hard_qmax=recommended_q == qmax, hit_soft_qmin=recommended_q <= cfg.soft_qmin,
            hit_soft_qmax=recommended_q >= cfg.soft_qmax, fallback_reason=fallback,
            enabled_layers=",".join(enabled),
        )

    def _floor_risk_allowed(
        self, *, q: int, predictions: Dict[int, tuple],
        safe_qs: list[int], expected_arrival_time: float,
        latest_arrival_time: float,
    ) -> bool:
        cfg = self.config
        if len(safe_qs) < cfg.risk_gated_floor_min_safe_candidates:
            return False
        chosen = predictions.get(q)
        if chosen is None:
            return True
        window = max(latest_arrival_time - expected_arrival_time, 0.0)
        required_slack = window * cfg.risk_gated_floor_slack_ratio
        return chosen[1] <= latest_arrival_time - required_slack

    def _utility_score(self, client_id: str):
        cfg = self.config
        current = self._utility[client_id]
        valid = [s for s in self._utility.values() if s.loss_ema is not None and s.num_samples > 0]
        if current.loss_ema is None or current.reports < cfg.utility_min_reports or not valid:
            return 1.0, 1.0, 0.0, current
        raws = sorted(math.sqrt(s.num_samples) * float(s.loss_ema) for s in valid)
        median = _percentile(raws, 0.5)
        raw = math.sqrt(current.num_samples) * float(current.loss_ema)
        confidence = min(1.0, current.reports / max(cfg.utility_min_reports, 1))
        if current.progress_ema is None or current.progress_ema <= 0:
            confidence = 0.0
        normalized = raw / max(median, 1e-8)
        shrunk = 1.0 + confidence * (normalized - 1.0)
        return raw, min(max(shrunk, cfg.utility_lower), cfg.utility_upper), confidence, current

    def _residual_margin(self, client_id: str) -> float:
        values = list(self._residuals[client_id])
        if len(values) < self.config.residual_min_reports:
            return 0.0
        return max(0.0, _percentile(sorted(values), self.config.residual_quantile))

    def _budget_status(self) -> tuple[float, int]:
        baseline = sum(x for x, _ in self._budget)
        applied = sum(y for _, y in self._budget)
        ratio = applied / baseline if baseline else 1.0
        return ratio, baseline - applied

    def _finish_noop(self, client_id, q, previous_q, dispatch_time, expected, latest, reason, enabled):
        ratio_before, debt_before = self._budget_status()
        utility_raw, utility_norm, utility_conf, utility_state = self._utility_score(client_id)
        self._previous_q[client_id] = q
        self._budget.append((q, q))
        return RUPDecision(
            client_id=client_id, mode=self.config.mode, baseline_q=q, raw_state_q=q,
            trust_q=q, soft_q=q, utility_q=q, budget_q=q, recommended_q=q,
            applied_q=q, state_safe_feasible=True, num_safe_candidates=0,
            predicted_finish_time=dispatch_time, safe_finish_time=dispatch_time,
            expected_arrival_time=expected, latest_arrival_time=latest,
            residual_margin=0.0, residual_count=len(self._residuals[client_id]),
            trust_lower=q, trust_upper=q, utility_raw=utility_raw,
            utility_normalized=utility_norm, utility_confidence=utility_conf,
            utility_reports=utility_state.reports, loss_ema=utility_state.loss_ema,
            progress_ema=utility_state.progress_ema, budget_ratio_before=ratio_before,
            budget_debt_before=debt_before,
            budget_adjustment=0, pre_accuracy_q=q, accuracy_priority_q=q,
            accuracy_floor_applied=False, accuracy_boost_applied=False,
            current_global_accuracy=self._current_global_accuracy,
            accuracy_boost_stage_active=(
                self._current_global_accuracy is not None
                and self._current_global_accuracy >= self.config.accuracy_q_boost_start_accuracy
            ),
            risk_gated_floor_allowed=True,
            pre_smooth_q=q,
            smooth_q=q,
            q_smooth_applied=False,
            previous_q=previous_q, hit_hard_qmin=False,
            hit_hard_qmax=False, hit_soft_qmin=False, hit_soft_qmax=False,
            fallback_reason=reason, enabled_layers=",".join(enabled),
        )


def _percentile(sorted_values: Iterable[float], quantile: float) -> float:
    values = list(sorted_values)
    if not values:
        return 0.0
    position = quantile * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1 - fraction) + values[upper] * fraction)
