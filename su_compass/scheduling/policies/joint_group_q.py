"""Pure joint existing-group and local-work selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from su_compass.scheduling.state_time_model import QTimeCandidate


@dataclass(frozen=True)
class GroupQCandidate:
    group_id: int
    group_size: int
    q: int
    expected_arrival_time: float
    latest_arrival_time: float
    predicted_finish_time: float
    safe_finish_time: float
    uncertainty: float
    alignment_error: float
    safe_slack: float
    target_tolerance: float
    deadline_safe: bool
    target_aligned: bool
    expected_already_passed: bool
    predictor_source: str
    num_reports: int

    def to_trace(self) -> dict:
        row = self.__dict__.copy()
        for field in ("deadline_safe", "target_aligned", "expected_already_passed"):
            row[field] = int(row[field])
        return row


@dataclass(frozen=True)
class JointGroupQDecision:
    feasible: bool
    group_id: int = -1
    q: int = -1
    predicted_finish_time: float = 0.0
    safe_finish_time: float = 0.0
    alignment_error: float = 0.0
    safe_slack: float = 0.0
    predictor_source: str = ""
    num_reports: int = 0
    group_size: int = 0
    reason: str = "no_deadline_safe_existing_group"


class JointGroupQPolicy:
    def __init__(
        self, target_band_ratio: float = 0.05,
        alignment_equivalence_band: float = 0.5,
    ) -> None:
        if target_band_ratio < 0:
            raise ValueError("target_band_ratio must be non-negative")
        if alignment_equivalence_band < 0:
            raise ValueError("alignment_equivalence_band must be non-negative")
        self.target_band_ratio = target_band_ratio
        self.alignment_equivalence_band = alignment_equivalence_band

    def enumerate_candidates(
        self, *, now: float, groups: Mapping[int, dict],
        curve: Iterable[QTimeCandidate],
    ) -> list[GroupQCandidate]:
        rows = []
        for group_id, group in groups.items():
            expected = float(group["expected_arrival_time"])
            latest = float(group["latest_arrival_time"])
            group_size = len(group.get("clients", [])) + len(
                group.get("arrived_clients", [])
            )
            tolerance = self.target_band_ratio * max(0.0, latest - expected)
            for point in curve:
                error = abs(point.predicted_finish_time - expected)
                rows.append(GroupQCandidate(
                    group_id=int(group_id), group_size=group_size, q=point.q,
                    expected_arrival_time=expected,
                    latest_arrival_time=latest,
                    predicted_finish_time=point.predicted_finish_time,
                    safe_finish_time=point.safe_finish_time,
                    uncertainty=point.uncertainty,
                    alignment_error=error,
                    safe_slack=latest - point.safe_finish_time,
                    target_tolerance=tolerance,
                    deadline_safe=point.safe_finish_time <= latest,
                    target_aligned=error <= tolerance,
                    expected_already_passed=expected <= now,
                    predictor_source=point.predictor_source,
                    num_reports=point.num_reports,
                ))
        return rows

    def choose(self, candidates: Iterable[GroupQCandidate]) -> JointGroupQDecision:
        # Deadline safety is the only admission hard gate.  The expected time
        # is a synchronization target, not an expiry time; an active group can
        # still accept a client after expected if a legal Q remains safe before
        # latest.  Alignment is retained below as a soft group-ranking signal.
        safe = [c for c in candidates if c.deadline_safe]
        if not safe:
            return JointGroupQDecision(feasible=False)

        # Preserve local work and avoid unnecessary early waiting: for every
        # group first retain the largest deadline-safe legal Q.
        per_group: dict[int, GroupQCandidate] = {}
        for candidate in safe:
            current = per_group.get(candidate.group_id)
            if current is None or candidate.q > current.q:
                per_group[candidate.group_id] = candidate

        group_rows = list(per_group.values())
        min_error = min(c.alignment_error for c in group_rows)
        near_best = [
            c for c in group_rows
            if c.alignment_error <= min_error + self.alignment_equivalence_band
        ]
        # Within a small alignment-equivalent band, consolidate into the
        # largest group; then close earlier groups first and preserve Q.
        best = min(near_best, key=lambda c: (
            -c.group_size, c.latest_arrival_time, -c.q, c.group_id,
        ))
        return JointGroupQDecision(
            feasible=True, group_id=best.group_id, q=best.q,
            predicted_finish_time=best.predicted_finish_time,
            safe_finish_time=best.safe_finish_time,
            alignment_error=best.alignment_error,
            safe_slack=best.safe_slack,
            predictor_source=best.predictor_source,
            num_reports=best.num_reports,
            group_size=best.group_size,
            reason="deadline_safe_reuse_first",
        )
