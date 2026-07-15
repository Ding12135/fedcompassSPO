"""完成时间预测的公共数据结构。

预测器统一使用本模块定义的输入和输出，避免实验运行器、诊断器与某个具体
预测公式绑定。后续增加 EWMA、在线回归或分位数预测器时，只需实现相同接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PredictionContext:
    """一次下发时预测器能够合法使用的全部信息。

    runtime_state 必须是客户端开始本轮训练前已经形成的状态快照，不能使用本轮
    完成后的状态，否则会造成未来信息泄漏。
    """

    client_id: str
    dispatch_time: float
    local_steps: int
    speed_smoothed: float
    runtime_state: Optional[Any] = None


@dataclass(frozen=True)
class LatencyPrediction:
    """预测器返回的统一完成时间预测结果。"""

    predictor_name: str
    predicted_duration: float
    predicted_finish_time: float
    uncertainty: float = 0.0
    safe_duration: float = 0.0
    compute_duration: float = 0.0
    communication_duration: float = 0.0
    spike_duration: float = 0.0
    availability_duration: float = 0.0
    availability_risk_duration: float = 0.0
    availability_event_rate: float = 0.0
    availability_event_count: int = 0
    availability_strategy_active: bool = False
    used_fallback: bool = False
    num_reports: int = 0
