"""自适应完成时间预测器的独立状态组件。

每个组件只负责一种端侧慢因，并分别输出普通点预测与尾部风险。这样某个状态
策略效果不好时可以单独替换，不需要修改计算、通信或调度控制器。
"""

from .availability import AvailabilityEventPrediction, AvailabilityEventPredictor

__all__ = ["AvailabilityEventPrediction", "AvailabilityEventPredictor"]
