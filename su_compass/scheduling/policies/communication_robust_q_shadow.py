"""固定状态新组时间窗下的通信尾部稳健Q Shadow。

历史P90无法可靠预测下一轮独立网络jitter，因此本模块不再判断“尾部是否会
发生”，而是为通信随机尾部预留统一风险预算。在状态新组latest保持不变的
前提下，从原始新组Q向下枚举，选择能够满足风险约束的最大Q：

    dispatch + safe_duration(q) + incremental_comm_reserve <= group_latest

其中 ``incremental_comm_reserve`` 只补 ``beta * communication_std`` 尚未被
预测器总uncertainty覆盖的部分，避免重复计算安全余量。

本策略只输出建议，不修改Q、group、deadline、event queue或聚合机制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


VALID_COMMUNICATION_ROBUST_Q_MODES = ("off", "shadow")


@dataclass(frozen=True)
class CommunicationRobustQRecommendation:
    """一次状态新组候选的稳健Q建议。"""

    original_q: int
    recommended_q: int
    q_reduction: int
    work_retention_ratio: float
    communication_std: float
    risk_beta: float
    target_communication_risk: float
    existing_uncertainty: float
    incremental_communication_reserve: float
    original_robust_slack: float
    recommended_predicted_finish_time: float
    recommended_safe_finish_time: float
    recommended_robust_finish_time: float
    recommended_robust_slack: float
    robust_safe_feasible: bool
    num_q_candidates: int
    shadow_action: str
    reason: str


class CommunicationRobustQShadowPolicy:
    """在固定状态组deadline内选择满足通信风险预算的最大Q。"""

    def __init__(self, predictor: LatencyPredictor, risk_beta: float = 1.645) -> None:
        if risk_beta < 0:
            raise ValueError("risk_beta must be non-negative")
        self.predictor = predictor
        self.risk_beta = risk_beta

    def recommend(
        self, *, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], original_q: int, group_latest_time: float,
        qmin: int, qmax: int,
    ) -> CommunicationRobustQRecommendation:
        if not qmin <= original_q <= qmax:
            raise ValueError("original_q outside original Q bounds")
        communication_std = max(
            0.0, float(getattr(runtime_state, "communication_time_std", 0.0))
        ) if runtime_state is not None else 0.0
        target_risk = self.risk_beta * communication_std

        original = self._predict(
            client_id, dispatch_time, speed_smoothed, runtime_state, original_q
        )
        # uncertainty同时覆盖计算和通信波动；这里只补目标通信风险中尚未覆盖的
        # 部分。这是保守近似，后续由Shadow判断训练量代价是否可接受。
        incremental_reserve = max(0.0, target_risk - original.uncertainty)
        original_robust_slack = (
            group_latest_time
            - (dispatch_time + original.safe_duration + incremental_reserve)
        )

        selected_q = original_q
        selected = original
        feasible = original_robust_slack >= 0
        checked = 1
        if not feasible:
            for q in range(original_q - 1, qmin - 1, -1):
                prediction = self._predict(
                    client_id, dispatch_time, speed_smoothed, runtime_state, q
                )
                checked += 1
                robust_finish = (
                    dispatch_time + prediction.safe_duration + incremental_reserve
                )
                if robust_finish <= group_latest_time:
                    selected_q, selected, feasible = q, prediction, True
                    break

        safe_finish = dispatch_time + selected.safe_duration
        robust_finish = safe_finish + incremental_reserve
        robust_slack = group_latest_time - robust_finish
        if feasible and selected_q == original_q:
            action, reason = "keep_original_q", "original_q_already_robust_safe"
        elif feasible:
            action, reason = "reduce_q_for_comm_reserve", "lower_q_restores_robust_safety"
        else:
            action, reason = "no_robust_q", "qmin_still_cannot_cover_comm_risk"
        return CommunicationRobustQRecommendation(
            original_q=original_q,
            recommended_q=selected_q,
            q_reduction=original_q - selected_q,
            work_retention_ratio=selected_q / max(1, original_q),
            communication_std=communication_std,
            risk_beta=self.risk_beta,
            target_communication_risk=target_risk,
            existing_uncertainty=original.uncertainty,
            incremental_communication_reserve=incremental_reserve,
            original_robust_slack=original_robust_slack,
            recommended_predicted_finish_time=selected.predicted_finish_time,
            recommended_safe_finish_time=safe_finish,
            recommended_robust_finish_time=robust_finish,
            recommended_robust_slack=robust_slack,
            robust_safe_feasible=feasible,
            num_q_candidates=checked,
            shadow_action=action,
            reason=reason,
        )

    def _predict(
        self, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], q: int,
    ):
        return self.predictor.predict(PredictionContext(
            client_id=client_id,
            dispatch_time=dispatch_time,
            local_steps=q,
            speed_smoothed=speed_smoothed,
            runtime_state=runtime_state,
        ))
