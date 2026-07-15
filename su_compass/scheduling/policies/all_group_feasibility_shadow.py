"""全量已有组状态可行性Shadow观测。

本模块用于检查FedCompass历史speed是否在候选入口错误排除了状态预测下可行的
arrival group。它枚举所有尚未过期的已有组，不修改控制器、group、Q或事件队列，
只返回FedCompass可行性、状态可行性及旁路推荐事实。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .q_shadow import ShadowQPolicy


VALID_ALL_GROUP_FEASIBILITY_MODES = ("off", "shadow")


@dataclass(frozen=True)
class AllGroupFeasibilityCandidate:
    """一个未过期已有组的双模型可行性事实。"""

    group_id: int
    is_current_group: bool
    expected_arrival_time: float
    latest_arrival_time: float
    fedcompass_reference_q: int
    fedcompass_q_feasible: bool
    state_recommended_q: int
    state_safe_feasible: bool
    predicted_finish_time: float
    safe_finish_time: float
    safe_slack: float
    arrival_deviation: float
    feasibility_class: str
    selected_by_shadow: bool = False
    rejection_reason: str = ""


@dataclass(frozen=True)
class AllGroupFeasibilityRecommendation:
    """一次已有组决策的全量Shadow汇总。"""

    current_group_id: int
    current_group_safe: bool
    recommended_group_id: int
    recommended_q: int
    shadow_action: str
    group_changed: bool
    mismatch_repaired: bool
    reason: str
    candidates: Tuple[AllGroupFeasibilityCandidate, ...]


class AllGroupFeasibilityShadowPolicy:
    """比较历史speed候选集合与状态预测候选集合。"""

    def __init__(self, q_policy: ShadowQPolicy) -> None:
        self.q_policy = q_policy

    def recommend(
        self, *, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], groups: Dict[int, Dict[str, Any]],
        current_group_id: int, qmin: int, qmax: int,
    ) -> AllGroupFeasibilityRecommendation:
        """枚举所有latest尚未过期的组；不以FedCompass合法性提前过滤。"""
        if current_group_id not in groups:
            raise ValueError("current group must exist")

        rows = []
        for group_id, group in groups.items():
            expected = float(group["expected_arrival_time"])
            latest = float(group["latest_arrival_time"])
            if latest <= dispatch_time:
                continue
            q_fc = math.floor((expected - dispatch_time) / max(speed_smoothed, 1e-8))
            fc_ok = expected > dispatch_time and qmin <= q_fc <= qmax
            state = self.q_policy.recommend(
                client_id=client_id, dispatch_time=dispatch_time,
                speed_smoothed=speed_smoothed, runtime_state=runtime_state,
                expected_arrival_time=expected, latest_arrival_time=latest,
                qmin=qmin, qmax=qmax,
            )
            state_ok = state.safe_feasible
            klass = (
                "both_feasible" if fc_ok and state_ok else
                "fedcompass_only" if fc_ok else
                "state_only" if state_ok else "both_infeasible"
            )
            rows.append(AllGroupFeasibilityCandidate(
                group_id=group_id, is_current_group=group_id == current_group_id,
                expected_arrival_time=expected, latest_arrival_time=latest,
                fedcompass_reference_q=q_fc, fedcompass_q_feasible=fc_ok,
                state_recommended_q=state.recommended_q,
                state_safe_feasible=state_ok,
                predicted_finish_time=state.predicted_finish_time,
                safe_finish_time=state.safe_finish_time,
                safe_slack=latest - state.safe_finish_time,
                arrival_deviation=state.expected_deviation,
                feasibility_class=klass,
                rejection_reason="" if state_ok else "no_safe_q_before_latest",
            ))

        current = next(row for row in rows if row.group_id == current_group_id)
        if current.state_safe_feasible:
            chosen = current
            action, reason = "keep_current_group", "current_group_state_safe"
        else:
            alternatives = [row for row in rows if row.state_safe_feasible]
            if alternatives:
                # Shadow排序保持确定性：优先保训练量，再取更大安全余量和更小偏差。
                chosen = min(alternatives, key=lambda row: (
                    -row.state_recommended_q, -row.safe_slack,
                    row.arrival_deviation, row.group_id,
                ))
                action, reason = "switch_existing_group", "other_state_safe_group_found"
            else:
                chosen = current
                action, reason = "consider_create_group", "no_state_safe_existing_group"

        selected = tuple(
            AllGroupFeasibilityCandidate(**{
                **row.__dict__,
                "selected_by_shadow": action != "consider_create_group" and row.group_id == chosen.group_id,
            }) for row in rows
        )
        return AllGroupFeasibilityRecommendation(
            current_group_id=current_group_id,
            current_group_safe=current.state_safe_feasible,
            recommended_group_id=chosen.group_id if action != "consider_create_group" else -1,
            recommended_q=chosen.state_recommended_q,
            shadow_action=action,
            group_changed=action == "switch_existing_group",
            mismatch_repaired=not current.state_safe_feasible and action == "switch_existing_group",
            reason=reason, candidates=selected,
        )
