"""不可用事件的条件预测组件。

availability wait 是零膨胀事件：多数轮次为0，少数轮次出现较长等待。把包含0的
历史均值加入每一轮点预测，会持续高估正常轮次，却仍可能低估真正不可用轮次。

本组件将两类目标分开：
    - point_duration：服务于MAE和expected arrival。若不可用事件概率低于50%，
      MAE意义下更合理的事件点预测为0；
    - risk_duration：服务于safe duration。样本充分时使用事件发生率加权的条件
      尾部等待，使后续Q策略保留风险感知，但不假设下一轮一定发生不可用。

组件仅依据RuntimeState字段激活，不使用profile_type或客户端ID硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AvailabilityEventPrediction:
    """不可用状态对普通完成时间与安全完成时间的独立贡献。"""

    point_duration: float
    risk_duration: float
    event_rate: float
    conditional_wait: float
    event_count: int
    active: bool


class AvailabilityEventPredictor:
    """根据事件概率和条件等待分布预测availability尾部风险。"""

    def __init__(
        self,
        min_event_count: int = 2,
        point_event_threshold: float = 0.5,
        risk_std_weight: float = 1.0,
    ) -> None:
        if min_event_count < 1:
            raise ValueError("min_event_count must be positive")
        if not 0.0 <= point_event_threshold <= 1.0:
            raise ValueError("point_event_threshold must be in [0, 1]")
        if risk_std_weight < 0:
            raise ValueError("risk_std_weight must be non-negative")
        self.min_event_count = min_event_count
        self.point_event_threshold = point_event_threshold
        self.risk_std_weight = risk_std_weight

    def predict(self, state: Any) -> AvailabilityEventPrediction:
        """返回availability的点预测与风险预测，不修改输入状态。"""
        event_rate = _clip01(float(getattr(state, "unavailable_event_rate", 0.0)))
        conditional_wait = max(
            0.0, float(getattr(state, "availability_wait_when_unavailable_mean", 0.0))
        )
        conditional_std = max(
            0.0, float(getattr(state, "availability_wait_when_unavailable_std", 0.0))
        )
        event_count = int(getattr(state, "unavailable_event_count", 0))
        active = event_count >= self.min_event_count and conditional_wait > 0.0

        if not active:
            # 事件不足时不根据单个极端等待建立专门策略；调用方可继续使用旧的
            # availability_wait_mean作为兼容回退。
            return AvailabilityEventPrediction(
                point_duration=0.0,
                risk_duration=0.0,
                event_rate=event_rate,
                conditional_wait=conditional_wait,
                event_count=event_count,
                active=False,
            )

        # 对绝对误差而言，事件发生概率小于50%时，零值是比期望值更合适的点
        # 预测；只有不可用已经成为多数状态时，才把条件等待加入普通预测。
        # 恰好50%时零和条件等待都可能是中位数；为避免小样本早期把Q预测推得
        # 过于保守，这里只有事件严格超过半数时才加入普通点预测。
        point_duration = conditional_wait if event_rate > self.point_event_threshold else 0.0
        # 风险项只进入safe duration。V1直接加入“条件均值+标准差”等价于假设
        # 下一轮一定发生不可用，实验中造成较多deadline误报。V2使用事件率加权，
        # 同时考虑“事件发生概率”和“事件发生后的尾部代价”。
        risk_duration = event_rate * (
            conditional_wait + self.risk_std_weight * conditional_std
        )
        return AvailabilityEventPrediction(
            point_duration=point_duration,
            risk_duration=risk_duration,
            event_rate=event_rate,
            conditional_wait=conditional_wait,
            event_count=event_count,
            active=True,
        )


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
