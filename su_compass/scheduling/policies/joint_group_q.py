"""Pure joint existing-group and local-work selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from su_compass.scheduling.state_time_model import QTimeCandidate


@dataclass(frozen=True)
class GroupQCandidate:
    group_id: int
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
    reason: str = "no_target_aligned_safe_group"


class JointGroupQPolicy:
    def __init__(self, target_band_ratio: float = 0.05) -> None:
        if target_band_ratio < 0:
            raise ValueError("target_band_ratio must be non-negative")
        self.target_band_ratio = target_band_ratio

    def enumerate_candidates(
        self, *, now: float, groups: Mapping[int, dict],
        curve: Iterable[QTimeCandidate],
    ) -> list[GroupQCandidate]:
        rows = []
        for group_id, group in groups.items():
            expected = float(group["expected_arrival_time"])
            latest = float(group["latest_arrival_time"])
            tolerance = self.target_band_ratio * max(0.0, latest - expected)
            for point in curve:
                error = abs(point.predicted_finish_time - expected)
                rows.append(GroupQCandidate(
                    group_id=int(group_id), q=point.q,
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
        valid = [c for c in candidates if (
            not c.expected_already_passed and c.deadline_safe and c.target_aligned
        )]
        if not valid:
            return JointGroupQDecision(feasible=False)
        earliest = min(c.expected_arrival_time for c in valid)
        earliest_rows = [c for c in valid if c.expected_arrival_time == earliest]
        best = min(earliest_rows, key=lambda c: (c.alignment_error, -c.q))
        return JointGroupQDecision(
            feasible=True, group_id=best.group_id, q=best.q,
            predicted_finish_time=best.predicted_finish_time,
            safe_finish_time=best.safe_finish_time,
            alignment_error=best.alignment_error,
            safe_slack=best.safe_slack,
            predictor_source=best.predictor_source,
            num_reports=best.num_reports,
            reason="earliest_target_aligned_safe_group",
        )
