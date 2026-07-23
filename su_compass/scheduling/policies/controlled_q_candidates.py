"""Controlled dual-anchor Q candidate compression.

The state predictor may inspect the complete Q curve, but the scheduler only
receives a small, auditable candidate set.  The upper trust anchor is a frozen
per-client FedCompass calibration statistic, deliberately independent of the
currently selected group's (possibly very long) deadline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from su_compass.scheduling.state_time_model import QTimeCandidate


@dataclass(frozen=True)
class ControlledQSet:
    reliable: bool
    reference_q: int
    raw_align_q: int
    controlled_align_q: int
    trust_upper_q: int
    candidate_qs: tuple[int, ...]
    reason: str


def controlled_join_qs(
    *, curve: Iterable[QTimeCandidate], target_time: float, deadline: float,
    group_safe_frontier: float, reference_q: int, qmin: int, qmax: int,
    trust_eta: float,
) -> ControlledQSet:
    """Compress a full state-predicted curve into safe representative Qs.

    Downward correction is unrestricted down to Qmin.  Upward correction is
    capped relative to a group-independent robust reference so that a slow
    group frontier cannot turn alignment into a hidden Qmax policy.
    """
    points = sorted(curve, key=lambda point: point.q)
    ref = max(qmin, min(qmax, int(reference_q)))
    trust_upper = max(qmin, min(qmax, int(math.ceil(trust_eta * ref))))
    reliable = bool(points) and all(
        not point.used_fallback and point.predictor_source == "mature_state"
        for point in points
    ) and all(
        points[idx].predicted_duration <= points[idx + 1].predicted_duration
        and points[idx].safe_duration <= points[idx + 1].safe_duration
        for idx in range(len(points) - 1)
    )
    if group_safe_frontier > deadline:
        return ControlledQSet(
            reliable, ref, -1, -1, trust_upper, (), "group_already_at_risk",
        )

    safe = [point for point in points if point.safe_finish_time <= deadline]
    if not safe:
        return ControlledQSet(
            reliable, ref, -1, -1, trust_upper, (), "no_deadline_safe_q",
        )

    raw = min(safe, key=lambda point: (
        abs(point.predicted_finish_time - target_time), point.q,
    ))
    bounded = [point for point in safe if point.q <= trust_upper]
    controlled = min(bounded, key=lambda point: (
        abs(point.predicted_finish_time - target_time),
        abs(point.q - ref), point.q,
    )) if reliable and bounded else None

    requested = {qmin, round(0.8 * ref), ref, round(1.1 * ref)}
    if reliable and controlled is not None:
        requested.add(controlled.q)
    available = {point.q for point in safe}
    candidates = tuple(sorted(
        max(qmin, min(qmax, int(q))) for q in requested
        if max(qmin, min(qmax, int(q))) in available
    ))
    reason = "controlled_dual_anchor" if reliable else "predictor_unreliable_reference_only"
    return ControlledQSet(
        reliable=reliable, reference_q=ref, raw_align_q=raw.q,
        controlled_align_q=controlled.q if controlled is not None else -1,
        trust_upper_q=trust_upper, candidate_qs=candidates, reason=reason,
    )


def controlled_create_qs(
    *, curve: Iterable[QTimeCandidate], reference_q: int,
    qmin: int, qmax: int, trust_eta: float,
) -> tuple[int, ...]:
    """Group-independent create candidates; never align Q to a future group."""
    points = list(curve)
    if not points:
        return ()
    ref = max(qmin, min(qmax, int(reference_q)))
    upper = max(qmin, min(qmax, int(math.ceil(trust_eta * ref))))
    available = {point.q for point in points}
    requested = {qmin, round(0.8 * ref), ref, round(1.1 * ref)}
    return tuple(sorted(
        q for q in (max(qmin, min(upper, int(value))) for value in requested)
        if q in available
    ))
