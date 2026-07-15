"""SU-Compass 完成时间预测器集合。

所有预测器均为无副作用对象：输入下发上下文，输出预测结果，不访问或修改
FedCompass 控制器。这样预测方法可以先离线/shadow验证，再由后续Q策略复用。
"""

from .base import LatencyPredictor
from .adaptive import AdaptiveLatencyPredictor
from .decomposed import DecomposedLatencyPredictor
from .fedcompass import FedCompassLatencyPredictor

__all__ = [
    "LatencyPredictor",
    "AdaptiveLatencyPredictor",
    "FedCompassLatencyPredictor",
    "DecomposedLatencyPredictor",
]
