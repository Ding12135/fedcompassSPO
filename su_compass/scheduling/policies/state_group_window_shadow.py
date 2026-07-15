"""固定Q的状态预测新组时间窗Shadow。

已有新组Q枚举发现，增加Q会同时扩大FedCompass deadline，从而产生贴近边界的
“数学安全”。本模块固定FedCompass原始新组Q，只将完成时间基准从历史
``Q * speed`` 替换为状态预测的mean duration，并沿用原latest_time_factor构造
旁路latest。这样可以单独判断问题是否来自新组时间窗，而不混入Q变化收益。

本模块不修改真实Q、arrival_group、deadline事件、client_info或聚合机制；输出
仅用于Shadow比较，不能视为真实调度收益。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


VALID_STATE_GROUP_WINDOW_MODES = ("off", "shadow")


@dataclass(frozen=True)
class StateGroupWindowRecommendation:
    """相同Q下speed时间窗与状态时间窗的对照事实。"""

    fixed_q: int
    speed_expected_arrival_time: float
    speed_latest_arrival_time: float
    speed_safe_slack: float
    state_expected_arrival_time: float
    state_latest_arrival_time: float
    state_predicted_finish_time: float
    state_safe_finish_time: float
    state_safe_slack: float
    # 保留预测分解，供后续与真实round_time各分量做风险校准。所有字段均为
    # dispatch时刻可见信息，不使用本轮训练完成后的未来事实。
    predicted_duration: float
    uncertainty: float
    predicted_compute_duration: float
    predicted_communication_duration: float
    predicted_spike_duration: float
    predicted_availability_duration: float
    predicted_availability_risk_duration: float
    availability_event_rate: float
    availability_event_count: int
    predictor_used_fallback: bool
    predictor_num_reports: int
    expected_shift: float
    latest_shift: float
    safe_slack_improvement: float
    state_window_safe_feasible: bool
    shadow_action: str
    reason: str


class StateGroupWindowShadowPolicy:
    """固定原始新组Q，只读评估状态预测锚定的时间窗。"""

    def __init__(self, predictor: LatencyPredictor) -> None:
        self.predictor = predictor

    def evaluate(
        self, *, client_id: str, dispatch_time: float, speed_smoothed: float,
        runtime_state: Optional[Any], fixed_q: int,
        speed_expected_arrival_time: float, speed_latest_arrival_time: float,
        qmin: int, qmax: int, latest_time_factor: float,
    ) -> StateGroupWindowRecommendation:
        """用同一Q和同一lambda比较两种时间窗，不改变任何调度状态。"""
        if not qmin <= fixed_q <= qmax:
            raise ValueError("fixed_q outside original Q bounds")
        prediction = self.predictor.predict(PredictionContext(
            client_id=client_id, dispatch_time=dispatch_time,
            local_steps=fixed_q, speed_smoothed=speed_smoothed,
            runtime_state=runtime_state,
        ))
        safe_finish = dispatch_time + prediction.safe_duration
        speed_slack = speed_latest_arrival_time - safe_finish

        # 仅替换持续时间估计，lambda及“从dispatch时刻放大持续时间”的结构不变。
        state_expected = prediction.predicted_finish_time
        state_latest = dispatch_time + prediction.predicted_duration * latest_time_factor
        state_slack = state_latest - safe_finish
        if state_slack >= 0:
            action, reason = "state_window_candidate", "fixed_q_state_window_safe"
        elif state_slack > speed_slack:
            action = "observe_safer_but_unsafe"
            reason = "fixed_q_state_window_improves_but_remains_unsafe"
        else:
            action, reason = "keep_speed_window", "state_window_not_safer"
        return StateGroupWindowRecommendation(
            fixed_q=fixed_q,
            speed_expected_arrival_time=speed_expected_arrival_time,
            speed_latest_arrival_time=speed_latest_arrival_time,
            speed_safe_slack=speed_slack,
            state_expected_arrival_time=state_expected,
            state_latest_arrival_time=state_latest,
            state_predicted_finish_time=prediction.predicted_finish_time,
            state_safe_finish_time=safe_finish,
            state_safe_slack=state_slack,
            predicted_duration=prediction.predicted_duration,
            uncertainty=prediction.uncertainty,
            predicted_compute_duration=prediction.compute_duration,
            predicted_communication_duration=prediction.communication_duration,
            predicted_spike_duration=prediction.spike_duration,
            predicted_availability_duration=prediction.availability_duration,
            predicted_availability_risk_duration=(
                prediction.availability_risk_duration
            ),
            availability_event_rate=prediction.availability_event_rate,
            availability_event_count=prediction.availability_event_count,
            predictor_used_fallback=prediction.used_fallback,
            predictor_num_reports=prediction.num_reports,
            expected_shift=state_expected - speed_expected_arrival_time,
            latest_shift=state_latest - speed_latest_arrival_time,
            safe_slack_improvement=state_slack - speed_slack,
            state_window_safe_feasible=state_slack >= 0,
            shadow_action=action, reason=reason,
        )
