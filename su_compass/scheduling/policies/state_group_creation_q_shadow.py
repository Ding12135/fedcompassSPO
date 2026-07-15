"""状态感知新建组Q可行性Shadow。

原始FedCompass新建组使用历史speed同时确定Q、expected arrival和latest arrival。
反事实实验表明，扩大时间窗时同步增大Q可能抵消安全收益。本模块在不改变Q
上下界、latest_time_factor和deadline机制的前提下，枚举全部整数Q，并用状态
预测器判断由原公式构造的对应新组时间窗是否安全。

本模块只回答“是否存在安全Q以及最小改动的安全Q”，不修改真实_create_group、
arrival_group、client_info、事件队列或聚合状态。真实调度仍执行Trust-Q V2。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


VALID_STATE_GROUP_CREATION_Q_MODES = ("off", "shadow")


@dataclass(frozen=True)
class StateGroupCreationQRecommendation:
    """一次mismatch下的新组Q枚举汇总。"""

    original_q: int
    original_safe_feasible: bool
    num_safe_q_candidates: int
    min_safe_q: int
    max_safe_q: int
    recommended_q: int
    recommended_expected_arrival_time: float
    recommended_latest_arrival_time: float
    recommended_predicted_finish_time: float
    recommended_safe_finish_time: float
    recommended_safe_slack: float
    q_difference: int
    work_retention_ratio: float
    expected_delay_difference: float
    shadow_action: str
    reason: str


class StateGroupCreationQShadowPolicy:
    """在原始新组时间窗公式下枚举状态安全Q。"""

    def __init__(self, predictor: LatencyPredictor) -> None:
        self.predictor = predictor

    def recommend(
        self, *, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], original_q: int,
        original_expected_arrival_time: float, original_safe_feasible: bool,
        qmin: int, qmax: int, latest_time_factor: float,
    ) -> StateGroupCreationQRecommendation:
        """枚举Q并选择与原始Q距离最小的安全候选，保证改动尽量局部。"""
        if qmin <= 0 or qmax < qmin:
            raise ValueError("invalid Q range")
        safe_rows = []
        all_rows = []
        for q in range(qmin, qmax + 1):
            # 保留FedCompass原始新组时间窗公式和同一个deadline放大系数。
            expected = dispatch_time + q * speed_smoothed
            latest = dispatch_time + q * speed_smoothed * latest_time_factor
            prediction = self.predictor.predict(PredictionContext(
                client_id=client_id, dispatch_time=dispatch_time,
                local_steps=q, speed_smoothed=speed_smoothed,
                runtime_state=runtime_state,
            ))
            safe_finish = dispatch_time + prediction.safe_duration
            row = (q, expected, latest, prediction.predicted_finish_time,
                   safe_finish, latest - safe_finish)
            all_rows.append(row)
            if safe_finish <= latest:
                safe_rows.append(row)

        if safe_rows:
            # 首要目标是最小化相对原始建组的Q改动；距离相同时保留更多训练量。
            chosen = min(safe_rows, key=lambda row: (abs(row[0] - original_q), -row[0]))
            action = "state_q_candidate"
            reason = (
                "original_q_already_safe" if original_safe_feasible
                else "alternative_q_makes_original_window_safe"
            )
        else:
            # 无安全Q时保留原始Q作为诊断基准，绝不将不安全建议写入真实调度。
            chosen = next(row for row in all_rows if row[0] == original_q)
            action, reason = "no_safe_q_candidate", "all_q_unsafe_under_original_window_formula"

        q, expected, latest, predicted_finish, safe_finish, slack = chosen
        return StateGroupCreationQRecommendation(
            original_q=original_q,
            original_safe_feasible=original_safe_feasible,
            num_safe_q_candidates=len(safe_rows),
            min_safe_q=min((row[0] for row in safe_rows), default=-1),
            max_safe_q=max((row[0] for row in safe_rows), default=-1),
            recommended_q=q,
            recommended_expected_arrival_time=expected,
            recommended_latest_arrival_time=latest,
            recommended_predicted_finish_time=predicted_finish,
            recommended_safe_finish_time=safe_finish,
            recommended_safe_slack=slack,
            q_difference=q - original_q,
            work_retention_ratio=q / original_q if original_q > 0 else 0.0,
            expected_delay_difference=expected - original_expected_arrival_time,
            shadow_action=action,
            reason=reason,
        )
