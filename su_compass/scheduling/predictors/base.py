"""完成时间预测器抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from su_compass.scheduling.types import LatencyPrediction, PredictionContext


class LatencyPredictor(ABC):
    """所有完成时间预测方法必须实现的最小接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回稳定的预测器名称，用于trace和论文表格。"""

    @abstractmethod
    def predict(self, context: PredictionContext) -> LatencyPrediction:
        """根据下发时已知信息预测给定Q的完成时间。"""
