"""根据端侧状态选择组件策略的自适应完成时间预测器。

第一轮改进只针对availability事件，计算、通信和spike仍完全复用已经通过Shadow
验证的DecomposedLatencyPredictor。这样新一轮实验若发生变化，可以明确归因于
不可用事件处理，而不是多个公式同时变化。

该预测器不按profile_type分类，也不识别具体客户端ID；是否启用availability
策略完全由RuntimeState中的事件数量、事件率和条件等待时间决定。
"""

from __future__ import annotations

from typing import Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.predictors.components import AvailabilityEventPredictor
from su_compass.scheduling.predictors.decomposed import DecomposedLatencyPredictor
from su_compass.scheduling.types import LatencyPrediction, PredictionContext


class AdaptiveLatencyPredictor(LatencyPredictor):
    """在时间分解骨架上按状态激活availability事件策略。"""

    def __init__(
        self,
        min_reports: int = 2,
        safety_beta: float = 1.0,
        availability_predictor: Optional[AvailabilityEventPredictor] = None,
    ) -> None:
        self._base = DecomposedLatencyPredictor(
            min_reports=min_reports,
            safety_beta=safety_beta,
        )
        self._availability = availability_predictor or AvailabilityEventPredictor()
        self.safety_beta = safety_beta

    @property
    def name(self) -> str:
        # V2相对V1只改变availability safe risk的概率校准，普通点预测不变。
        return "adaptive_availability_event_v2"

    def predict(self, context: PredictionContext) -> LatencyPrediction:
        base = self._base.predict(context)
        state = context.runtime_state
        if base.used_fallback or state is None:
            return LatencyPrediction(
                **{**base.__dict__, "predictor_name": self.name}
            )

        availability = self._availability.predict(state)
        if not availability.active:
            # 尚未观察到足够不可用事件时完全保持原分解预测，避免一次偶发事件
            # 影响所有客户端，也保证非availability客户端不发生无关改动。
            return LatencyPrediction(
                **{**base.__dict__, "predictor_name": self.name}
            )

        old_availability_mean = max(
            0.0, float(getattr(state, "availability_wait_mean", 0.0))
        )
        duration = (
            base.predicted_duration
            - old_availability_mean
            + availability.point_duration
        )
        # 普通点预测去掉“每轮平摊的不可用均值”；不可用条件尾部只进入安全
        # 时间。后续Q策略可分别使用mean与safe，避免普通轮次被永久压低Q。
        safe_duration = (
            duration
            + self.safety_beta * base.uncertainty
            + availability.risk_duration
        )
        return LatencyPrediction(
            predictor_name=self.name,
            predicted_duration=duration,
            predicted_finish_time=context.dispatch_time + duration,
            uncertainty=base.uncertainty,
            safe_duration=safe_duration,
            compute_duration=base.compute_duration,
            communication_duration=base.communication_duration,
            spike_duration=base.spike_duration,
            availability_duration=availability.point_duration,
            availability_risk_duration=availability.risk_duration,
            availability_event_rate=availability.event_rate,
            availability_event_count=availability.event_count,
            availability_strategy_active=True,
            used_fallback=False,
            num_reports=base.num_reports,
        )
