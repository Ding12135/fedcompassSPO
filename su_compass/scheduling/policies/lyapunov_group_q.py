"""Pure Lyapunov-guided joint group/Q action policy.

The policy owns no scheduler state.  It scores an already predicted action
set, which keeps prediction, queue accounting, and mutation independently
testable and makes ``off``/``shadow``/``apply`` modes mechanically reversible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LyapunovAction:
    mode: str
    group_id: int
    q: int
    predicted_finish_time: float
    safe_finish_time: float
    group_frontier_time: float
    latest_arrival_time: float
    deadline_safe: bool
    holding_wait: float
    external_wait: float
    predicted_sojourn: float
    effective_work: float
    utility: float
    score: float = math.inf
    legal: bool = False
    rejection_reason: str = ""

    def to_trace(self) -> dict:
        row = self.__dict__.copy()
        row["deadline_safe"] = int(self.deadline_safe)
        row["legal"] = int(self.legal)
        return row


@dataclass(frozen=True)
class LyapunovDecision:
    feasible: bool
    action: LyapunovAction | None = None
    reason: str = "no_legal_action"


class LyapunovGroupQPolicy:
    """Score legal actions with normalized drift-plus-penalty terms."""

    def __init__(
        self, *, rhythm_target: float, tradeoff_v: float,
        max_holding_wait: float, q_trust_eta: float,
        create_penalty: float, enable_rhythm_queue: bool = True,
        enable_workload_queue: bool = True,
    ) -> None:
        if rhythm_target <= 0 or tradeoff_v < 0:
            raise ValueError("rhythm_target must be positive and V non-negative")
        if max_holding_wait < 0 or q_trust_eta < 1.0 or create_penalty < 0:
            raise ValueError("invalid Lyapunov safety configuration")
        self.rhythm_target = rhythm_target
        self.tradeoff_v = tradeoff_v
        self.max_holding_wait = max_holding_wait
        self.q_trust_eta = q_trust_eta
        self.create_penalty = create_penalty
        self.enable_rhythm_queue = enable_rhythm_queue
        self.enable_workload_queue = enable_workload_queue

    def score(
        self, actions: Iterable[LyapunovAction], *, rhythm_debt: float,
        workload_debt: float, qmax: int, qmin: int,
        fedcompass_join_q: int = -1,
    ) -> list[LyapunovAction]:
        scored = []
        h = rhythm_debt / self.rhythm_target if self.enable_rhythm_queue else 0.0
        z = workload_debt if self.enable_workload_queue else 0.0
        utility_norm = math.log1p(qmax / max(qmin, 1))
        trust_cap = (
            min(qmax, math.ceil(self.q_trust_eta * fedcompass_join_q))
            if fedcompass_join_q >= qmin else qmax
        )
        for action in actions:
            reason = ""
            if action.mode == "join" and not action.deadline_safe:
                reason = "deadline_unsafe"
            elif action.mode == "join" and action.holding_wait > self.max_holding_wait:
                reason = "holding_wait_exceeds_cap"
            elif action.mode == "join" and action.q > trust_cap:
                reason = "q_exceeds_trust_region"
            legal = not reason
            sojourn = action.predicted_sojourn / self.rhythm_target
            utility = action.utility / max(utility_norm, 1e-12)
            score = (
                h * sojourn
                - z * action.effective_work
                - self.tradeoff_v * utility
                + (self.create_penalty if action.mode == "create" else 0.0)
            ) if legal else math.inf
            scored.append(LyapunovAction(**{
                **action.__dict__, "score": score, "legal": legal,
                "rejection_reason": reason,
            }))
        return scored

    def choose(self, actions: Iterable[LyapunovAction]) -> LyapunovDecision:
        legal = [action for action in actions if action.legal]
        if not legal:
            return LyapunovDecision(False)
        best = min(legal, key=lambda action: (
            action.score, action.predicted_sojourn,
            0 if action.mode == "join" else 1, -action.q, action.group_id,
        ))
        return LyapunovDecision(True, best, "minimum_drift_plus_penalty")
