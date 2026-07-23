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
    predicted_duration: float
    safe_finish_time: float
    group_frontier_time: float
    latest_arrival_time: float
    deadline_safe: bool
    holding_wait: float
    external_wait: float
    affected_pending_clients: int
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


@dataclass(frozen=True)
class EffectiveServiceSelection:
    decision: LyapunovDecision
    region: str
    reason: str
    joins: int
    creates: int
    best_join: LyapunovAction | None
    best_create: LyapunovAction | None


def choose_effective_service_v2(
    actions: Iterable[LyapunovAction], *, obvious_extension_limit: float,
    obvious_holding_limit: float, create_hysteresis: float,
) -> EffectiveServiceSelection:
    """Pure Region 1/2/3 selector shared by online and replay paths."""
    legal = list(actions)
    joins = [action for action in legal if action.legal and action.mode == "join"]
    creates = [action for action in legal if action.legal and action.mode == "create"]
    best_join = min(joins, key=lambda action: (action.score, action.q, action.group_id)) if joins else None
    best_create = min(creates, key=lambda action: (action.score, action.q)) if creates else None
    obvious = bool(
        best_join is not None
        and best_join.external_wait <= obvious_extension_limit
        and best_join.holding_wait <= obvious_holding_limit
    )
    if obvious:
        selected, region, reason = best_join, "region_1", "obvious_safe_join"
    elif best_join is None:
        selected, region, reason = best_create, "region_3", "no_legal_join"
    elif best_create is not None and best_create.score + create_hysteresis < best_join.score:
        selected, region, reason = best_create, "region_2", "create_wins_hysteresis"
    else:
        selected, region, reason = best_join, "region_2", "join_kept_by_hysteresis"
    decision = LyapunovDecision(
        feasible=selected is not None, action=selected,
        reason=reason if selected is not None else "no_legal_action",
    )
    return EffectiveServiceSelection(
        decision=decision, region=region, reason=reason,
        joins=len(joins), creates=len(creates),
        best_join=best_join, best_create=best_create,
    )


class LyapunovGroupQPolicy:
    """Score legal actions with normalized drift-plus-penalty terms."""

    def __init__(
        self, *, rhythm_target: float, tradeoff_v: float,
        max_holding_wait: float, q_trust_eta: float,
        create_penalty: float, enable_rhythm_queue: bool = True,
        enable_workload_queue: bool = True,
        action_scope: str = "joint_v1", holding_weight: float = 0.0,
        max_holding_ratio: float = math.inf,
        join_cadence_weight: float = 0.0,
    ) -> None:
        if rhythm_target <= 0 or tradeoff_v < 0:
            raise ValueError("rhythm_target must be positive and V non-negative")
        if max_holding_wait < 0 or q_trust_eta < 1.0 or create_penalty < 0:
            raise ValueError("invalid Lyapunov safety configuration")
        if action_scope not in {
            "joint_v1", "join_only_v2", "effective_service_v1",
            "effective_service_v2", "effective_service_v2_1",
        }:
            raise ValueError("invalid Lyapunov action scope")
        if holding_weight < 0 or max_holding_ratio < 0 or join_cadence_weight < 0:
            raise ValueError("holding weight and ratio must be non-negative")
        self.rhythm_target = rhythm_target
        self.tradeoff_v = tradeoff_v
        self.max_holding_wait = max_holding_wait
        self.q_trust_eta = q_trust_eta
        self.create_penalty = create_penalty
        self.enable_rhythm_queue = enable_rhythm_queue
        self.enable_workload_queue = enable_workload_queue
        self.action_scope = action_scope
        self.holding_weight = holding_weight
        self.max_holding_ratio = max_holding_ratio
        self.join_cadence_weight = join_cadence_weight

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
            elif action.mode == "join" and self._extreme_holding(action):
                reason = "extreme_holding_wait"
            elif action.mode == "join" and action.q > trust_cap:
                reason = "q_exceeds_trust_region"
            legal = not reason
            sojourn = action.predicted_sojourn / self.rhythm_target
            holding = action.holding_wait / self.rhythm_target
            extension = action.external_wait / self.rhythm_target
            utility = action.utility / max(utility_norm, 1e-12)
            if self.action_scope in {
                "join_only_v2", "effective_service_v1", "effective_service_v2",
                "effective_service_v2_1",
            } and action.mode == "join":
                # H prices only delay added to the group's existing frontier.
                # Client-only holding is a separate resource-efficiency cost.
                cadence_wait = max(0.0, sojourn - 1.0)
                score = (
                    h * extension * max(action.affected_pending_clients, 1)
                    + self.join_cadence_weight * h * cadence_wait
                    + self.holding_weight * holding
                    - z * action.effective_work
                    - self.tradeoff_v * utility
                ) if legal else math.inf
            else:
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

    def _extreme_holding(self, action: LyapunovAction) -> bool:
        if action.holding_wait <= self.max_holding_wait:
            return False
        if self.action_scope == "joint_v1":
            return True
        duration = max(action.predicted_duration, 1e-12)
        return action.holding_wait / duration > self.max_holding_ratio

    def choose(self, actions: Iterable[LyapunovAction]) -> LyapunovDecision:
        legal = [action for action in actions if action.legal]
        if not legal:
            return LyapunovDecision(False)
        best = min(legal, key=lambda action: (
            action.score, action.predicted_sojourn,
            0 if action.mode == "join" else 1, -action.q, action.group_id,
        ))
        return LyapunovDecision(True, best, "minimum_drift_plus_penalty")
