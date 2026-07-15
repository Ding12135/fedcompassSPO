"""SU-Compass 调度方法的可复用组件。

本包只放置与具体虚拟控制器解耦的预测器、策略和数据结构。第一阶段仅包含
完成时间预测器；预测结果由 diagnostics 以 shadow 方式评估，不会直接改变
FedCompass 的 Q 或 arrival group。
"""

from .types import LatencyPrediction, PredictionContext

__all__ = ["LatencyPrediction", "PredictionContext"]
