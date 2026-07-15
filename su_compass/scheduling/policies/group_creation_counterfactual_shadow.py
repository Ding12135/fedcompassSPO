"""FedCompass原始新建组反事实Shadow策略。

Trust-Q判定当前已有group不存在安全Q，并不能自动证明FedCompass原始
``_create_group`` 会产生更安全的新组。本模块严格复算虚拟FedCompass的原始
建组公式，再用同一个状态预测器比较新旧时间窗。

本模块只返回不可变事实：不调用真实_create_group，不修改arrival_group、
client_info、group_counter或事件队列，也不改变Q边界、deadline和聚合机制。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


VALID_GROUP_CREATION_COUNTERFACTUAL_MODES = ("off", "shadow")


@dataclass(frozen=True)
class FedCompassNewGroupPlan:
    """FedCompass原始_create_group公式的只读计算结果。"""

    local_steps: int
    expected_arrival_time: float
    latest_arrival_time: float


@dataclass(frozen=True)
class GroupCreationCounterfactual:
    """一次group mismatch的新旧时间窗反事实比较。"""

    current_group_id: int
    current_q: int
    current_expected_arrival_time: float
    current_latest_arrival_time: float
    current_predicted_finish_time: float
    current_safe_finish_time: float
    current_safe_slack: float
    counterfactual_q: int
    counterfactual_expected_arrival_time: float
    counterfactual_latest_arrival_time: float
    counterfactual_predicted_finish_time: float
    counterfactual_safe_finish_time: float
    counterfactual_safe_slack: float
    counterfactual_safe_feasible: bool
    safe_slack_improvement: float
    expected_delay_increase: float
    q_difference: int
    shadow_action: str
    reason: str


def calculate_fedcompass_new_group_plan(
    *, dispatch_time: float, client_speed: float,
    groups: Dict[int, Dict[str, Any]], client_info: Dict[str, Dict[str, Any]],
    qmin: int, qmax: int, latest_time_factor: float,
) -> FedCompassNewGroupPlan:
    """无副作用复算虚拟FedCompass._create_group中的原始建组公式。"""
    assigned_steps = -1
    for group in groups.values():
        if dispatch_time >= float(group["latest_arrival_time"]):
            continue
        group_clients = group["clients"] + group["arrived_clients"]
        # 原控制器的活跃组应至少保留一个客户端；空组不参与反事实估计。
        if not group_clients:
            continue
        fastest_speed = min(float(client_info[c]["speed"]) for c in group_clients)
        estimated_arrival = float(group["latest_arrival_time"]) + fastest_speed * qmax
        local_steps = math.floor((estimated_arrival - dispatch_time) / client_speed)
        if local_steps <= qmax:
            assigned_steps = max(assigned_steps, local_steps)

    if 0 <= assigned_steps < qmin:
        assigned_steps = qmin
    if assigned_steps < 0:
        assigned_steps = qmax

    expected = dispatch_time + assigned_steps * client_speed
    latest = dispatch_time + assigned_steps * client_speed * latest_time_factor
    return FedCompassNewGroupPlan(assigned_steps, expected, latest)


class GroupCreationCounterfactualShadowPolicy:
    """比较当前不安全已有组与FedCompass原始反事实新组。"""

    def __init__(self, predictor: LatencyPredictor) -> None:
        self.predictor = predictor

    def evaluate(
        self, *, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], groups: Dict[int, Dict[str, Any]],
        client_info: Dict[str, Dict[str, Any]], current_group_id: int,
        current_q: int, current_predicted_finish_time: float,
        current_safe_finish_time: float, qmin: int, qmax: int,
        latest_time_factor: float,
    ) -> GroupCreationCounterfactual:
        """只读生成原始新组计划并用统一预测口径比较安全余量。"""
        current = groups[current_group_id]
        plan = calculate_fedcompass_new_group_plan(
            dispatch_time=dispatch_time, client_speed=speed_smoothed,
            groups=groups, client_info=client_info, qmin=qmin, qmax=qmax,
            latest_time_factor=latest_time_factor,
        )
        # 新旧方案必须使用同一个预测器；否则safe slack差异无法归因于分组。
        prediction = self.predictor.predict(PredictionContext(
            client_id=client_id, dispatch_time=dispatch_time,
            local_steps=plan.local_steps, speed_smoothed=speed_smoothed,
            runtime_state=runtime_state,
        ))
        new_safe_finish = dispatch_time + prediction.safe_duration
        current_slack = float(current["latest_arrival_time"]) - current_safe_finish_time
        new_slack = plan.latest_arrival_time - new_safe_finish
        if new_slack >= 0:
            action, reason = "create_group_candidate", "counterfactual_new_group_safe"
        elif new_slack > current_slack:
            action = "observe_safer_but_unsafe"
            reason = "counterfactual_improves_slack_but_remains_unsafe"
        else:
            action, reason = "keep_current_candidate", "counterfactual_new_group_not_safer"
        return GroupCreationCounterfactual(
            current_group_id=current_group_id, current_q=current_q,
            current_expected_arrival_time=float(current["expected_arrival_time"]),
            current_latest_arrival_time=float(current["latest_arrival_time"]),
            current_predicted_finish_time=current_predicted_finish_time,
            current_safe_finish_time=current_safe_finish_time,
            current_safe_slack=current_slack,
            counterfactual_q=plan.local_steps,
            counterfactual_expected_arrival_time=plan.expected_arrival_time,
            counterfactual_latest_arrival_time=plan.latest_arrival_time,
            counterfactual_predicted_finish_time=prediction.predicted_finish_time,
            counterfactual_safe_finish_time=new_safe_finish,
            counterfactual_safe_slack=new_slack,
            counterfactual_safe_feasible=new_slack >= 0,
            safe_slack_improvement=new_slack - current_slack,
            expected_delay_increase=plan.expected_arrival_time - float(current["expected_arrival_time"]),
            q_difference=plan.local_steps - current_q,
            shadow_action=action, reason=reason,
        )
