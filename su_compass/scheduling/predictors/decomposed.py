"""基于端侧状态分解的完成时间预测器。

本模块解决 FedCompass 把计算、通信、卡顿和不可用等待全部除以Q后压缩成一个
speed 的问题。第一版采用可解释、无需额外训练的时间分解模型：

    T(Q) = Q * compute_step_time_mean
           + communication_time_mean
           + spike_delay_mean
           + availability_wait_mean

其中计算项随Q增长，通信、卡顿和不可用等待作为轮次级开销单独建模。预测器
同时输出不确定性和安全耗时，但第一阶段它们只用于观测，不参与实际Q分配。
"""

from __future__ import annotations

import math

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.predictors.fedcompass import FedCompassLatencyPredictor
from su_compass.scheduling.types import LatencyPrediction, PredictionContext


class DecomposedLatencyPredictor(LatencyPredictor):
    """使用RuntimeState滑动窗口分量预测给定Q的下一轮总耗时。"""

    def __init__(self, min_reports: int = 2, safety_beta: float = 1.0) -> None:
        if min_reports < 1:
            raise ValueError("min_reports must be positive")
        if safety_beta < 0:
            raise ValueError("safety_beta must be non-negative")
        self.min_reports = min_reports
        self.safety_beta = safety_beta
        self._fallback = FedCompassLatencyPredictor()

    @property
    def name(self) -> str:
        return "decomposed_runtime_state"

    def predict(self, context: PredictionContext) -> LatencyPrediction:
        state = context.runtime_state
        num_reports = int(getattr(state, "num_reports", 0)) if state is not None else 0
        if state is None or num_reports < self.min_reports:
            # 冷启动阶段历史样本不足，强制回退到FedCompass，避免用单个随机尖峰
            # 构造看似精确但不可复现的状态预测。
            baseline = self._fallback.predict(context)
            return LatencyPrediction(
                predictor_name=self.name,
                predicted_duration=baseline.predicted_duration,
                predicted_finish_time=baseline.predicted_finish_time,
                safe_duration=baseline.safe_duration,
                used_fallback=True,
                num_reports=num_reports,
            )

        q = max(1, context.local_steps)
        compute_duration = max(0.0, float(state.compute_step_time_mean)) * q
        communication_duration = max(0.0, float(state.communication_time_mean))
        spike_duration = max(0.0, float(state.spike_delay_mean))
        availability_duration = max(0.0, float(state.availability_wait_mean))
        duration = (
            compute_duration
            + communication_duration
            + spike_duration
            + availability_duration
        )

        # 计算波动随Q累积，通信波动是轮次级固定项。两者按独立误差近似合成；
        # spike/availability 第一版缺少独立方差字段，先不伪造额外不确定性。
        compute_uncertainty = max(0.0, float(state.compute_step_time_std)) * q
        communication_uncertainty = max(0.0, float(state.communication_time_std))
        uncertainty = math.sqrt(
            compute_uncertainty ** 2 + communication_uncertainty ** 2
        )
        safe_duration = duration + self.safety_beta * uncertainty
        return LatencyPrediction(
            predictor_name=self.name,
            predicted_duration=duration,
            predicted_finish_time=context.dispatch_time + duration,
            uncertainty=uncertainty,
            safe_duration=safe_duration,
            compute_duration=compute_duration,
            communication_duration=communication_duration,
            spike_duration=spike_duration,
            availability_duration=availability_duration,
            used_fallback=False,
            num_reports=num_reports,
        )
