"""Pure reason-aware routing helpers for Effective-Service Shadow.

The module classifies *why* a client is slow and recommends a service lane.
It deliberately does not choose or modify Q.  Online application remains
owned by the existing Effective-Service scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from su_compass.scheduling.policies.lyapunov_group_q import LyapunovAction
from su_compass.scheduling.state_time_model import QTimeCandidate


@dataclass(frozen=True)
class SlowCause:
    label: str
    confidence: float
    compute_ratio: float
    communication_ratio: float
    availability_ratio: float
    spike_ratio: float
    mature: bool


@dataclass(frozen=True)
class ReasonAwareRoute:
    lane: str
    mode: str
    group_id: int
    q: int
    reason: str
    compatible_background_group: int
    anchor_eligible: bool
    changed: bool


def classify_slow_cause(point: QTimeCandidate | None) -> SlowCause:
    """Classify the dominant predicted latency cause without client IDs."""
    if point is None:
        return SlowCause("cold_start", 0.0, 0.0, 0.0, 0.0, 0.0, False)
    total = max(float(point.predicted_duration), 1e-12)
    compute = max(0.0, float(point.compute_duration)) / total
    communication = max(0.0, float(point.communication_duration)) / total
    availability = max(
        0.0,
        float(point.availability_duration)
        + float(point.availability_risk_duration),
    ) / total
    spike = max(0.0, float(point.spike_duration)) / total
    mature = bool(
        point.num_reports >= 2
        and not point.used_fallback
        and point.predictor_source not in {"", "fedcompass_fallback"}
    )
    if not mature:
        label = "cold_start"
    elif availability >= 0.25:
        label = "availability_bound"
    elif spike >= 0.15:
        label = "volatile"
    elif communication >= 0.98 and compute <= 0.02:
        label = "extreme_communication_bound"
    elif communication >= 0.90:
        label = "communication_bound"
    else:
        label = "balanced"
    components = sorted((compute, communication, availability, spike), reverse=True)
    dominance = components[0] - components[1] if len(components) > 1 else components[0]
    sample_confidence = min(1.0, point.num_reports / 4.0)
    confidence = sample_confidence * min(1.0, max(0.0, dominance))
    return SlowCause(
        label, confidence, compute, communication, availability, spike, mature,
    )


def recommend_reason_aware_route(
    *,
    cause: SlowCause,
    v23_action: LyapunovAction | None,
    scored_actions: Iterable[LyapunovAction],
    service_age_periods: float,
    minimum_anchor_age_periods: float,
    system_healthy: bool,
    background_sojourn_periods: float,
    rhythm_target: float,
) -> ReasonAwareRoute:
    """Recommend a lane while preserving the V2.3-selected Q exactly."""
    if v23_action is None:
        return ReasonAwareRoute(
            "fallback", "", -1, -1, "no_v23_action", -1, False, False,
        )
    base = ReasonAwareRoute(
        "fast", v23_action.mode, v23_action.group_id, v23_action.q,
        "v23_route_preserved", -1, False, False,
    )
    if cause.label == "cold_start":
        return ReasonAwareRoute(
            "fallback", base.mode, base.group_id, base.q,
            "cold_start_preserve_v23", -1, False, False,
        )
    if cause.label in {"availability_bound", "volatile"}:
        return ReasonAwareRoute(
            "guarded", base.mode, base.group_id, base.q,
            "unstable_state_preserve_v23", -1, False, False,
        )
    if cause.label != "extreme_communication_bound":
        return base

    background_joins = [
        action for action in scored_actions
        if (
            action.legal
            and action.mode == "join"
            and action.q == v23_action.q
            and action.predicted_sojourn
            >= background_sojourn_periods * rhythm_target
        )
    ]
    compatible = min(
        background_joins,
        key=lambda action: (action.score, action.predicted_sojourn, action.group_id),
        default=None,
    )
    aged = service_age_periods >= minimum_anchor_age_periods
    anchor_eligible = bool(aged and system_healthy and compatible is None)
    selected = compatible
    reason = "compatible_background_join"
    if selected is None and anchor_eligible:
        creates = [
            action for action in scored_actions
            if (
                action.legal
                and action.mode == "create"
                and action.q == v23_action.q
            )
        ]
        selected = min(
            creates,
            key=lambda action: (action.score, action.predicted_sojourn, action.q),
            default=None,
        )
        reason = "aged_background_anchor" if selected else "no_legal_background_action"
    if selected is None:
        return ReasonAwareRoute(
            "background", base.mode, base.group_id, base.q,
            "background_gate_closed_preserve_v23", -1, anchor_eligible, False,
        )
    # Q is intentionally inherited from V2.3 even when Shadow changes mode/group.
    changed = selected.mode != base.mode or selected.group_id != base.group_id
    return ReasonAwareRoute(
        "background", selected.mode, selected.group_id, base.q, reason,
        compatible.group_id if compatible else -1, anchor_eligible, changed,
    )
