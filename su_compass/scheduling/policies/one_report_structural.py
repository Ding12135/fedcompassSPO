"""Conservative one-report latency prior for communication-dominated clients.

FedCompass scales the complete previous round duration with Q.  That is a poor
cold-start prior when communication is a round-level fixed cost.  After one
completed report we can already preserve the observed decomposition:

    T(Q) = communication + spike + availability + Q * compute_per_step

The helper is deliberately pure and is intended for Shadow/replay first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OneReportStructuralPrediction:
    eligible: bool
    reason: str
    q: int
    predicted_duration: float
    safe_duration: float
    compute_duration: float
    fixed_duration: float
    communication_ratio: float


def predict_one_report_structural(
    *,
    q: int,
    observed_q: int,
    observed_round_duration: float,
    observed_compute_duration: float,
    observed_communication_duration: float,
    observed_spike_duration: float = 0.0,
    observed_availability_duration: float = 0.0,
    num_reports: int = 1,
    communication_ratio_gate: float = 0.95,
    safety_fraction: float = 0.10,
) -> OneReportStructuralPrediction:
    """Return a conservative structural prior without changing Q.

    The safe duration uses the larger of the structural point estimate and the
    observed round duration, then adds a fractional one-sample margin.  It is a
    risk boundary, not a statistical coverage guarantee.
    """
    if q <= 0 or observed_q <= 0:
        raise ValueError("q and observed_q must be positive")
    if safety_fraction < 0:
        raise ValueError("safety_fraction must be non-negative")

    round_duration = max(0.0, float(observed_round_duration))
    compute = max(0.0, float(observed_compute_duration))
    communication = max(0.0, float(observed_communication_duration))
    spike = max(0.0, float(observed_spike_duration))
    availability = max(0.0, float(observed_availability_duration))
    fixed = communication + spike + availability
    communication_ratio = communication / max(round_duration, 1e-12)
    eligible = bool(
        num_reports == 1
        and round_duration > 0.0
        and communication_ratio >= communication_ratio_gate
    )
    point_compute = compute / observed_q * q
    point = fixed + point_compute
    safe = max(point, round_duration) * (1.0 + safety_fraction)
    reason = (
        "one_report_communication_structural_prior"
        if eligible
        else "gate_closed"
    )
    return OneReportStructuralPrediction(
        eligible=eligible,
        reason=reason,
        q=int(q),
        predicted_duration=point,
        safe_duration=safe,
        compute_duration=point_compute,
        fixed_duration=fixed,
        communication_ratio=communication_ratio,
    )
