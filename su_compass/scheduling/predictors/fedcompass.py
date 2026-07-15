"""FedCompass 历史平滑 speed 基线预测器。

该预测器严格复现 FedCompass 当前完成时间假设：下一轮耗时等于平滑单步耗时
乘以本地步数Q。它不用于优化，而是作为状态预测器必须比较的统一baseline。
"""

from __future__ import annotations

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import LatencyPrediction, PredictionContext


class FedCompassLatencyPredictor(LatencyPredictor):
    """使用 `speed_smoothed * Q` 预测下一轮总耗时。"""

    @property
    def name(self) -> str:
        return "fedcompass_speed"

    def predict(self, context: PredictionContext) -> LatencyPrediction:
        duration = max(0.0, context.speed_smoothed) * max(1, context.local_steps)
        return LatencyPrediction(
            predictor_name=self.name,
            predicted_duration=duration,
            predicted_finish_time=context.dispatch_time + duration,
            # baseline 不单独建模风险，safe_duration 与均值预测相同。
            safe_duration=duration,
            num_reports=getattr(context.runtime_state, "num_reports", 0),
        )
