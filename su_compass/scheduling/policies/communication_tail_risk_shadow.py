"""状态新组的通信尾部风险 Shadow。

虚拟端侧每轮传输数据量和基准带宽固定，但网络耗时仍按 ``network_jitter_cv``
独立采样。历史均值适合点预测，却可能漏掉弱网客户端的右尾事件。本策略只读
比较经验 P90/窗口最大值与预测器已有 uncertainty，估计尚未被安全时间覆盖的
“增量通信尾部余量”。

严格约束：
    1. 只消费 dispatch 时刻的 RuntimeState 和状态时间窗结果；
    2. 不修改预测器、Q、group、deadline 或 Admission Gate；
    3. 建议仅写独立 Trace，不能反馈真实调度；
    4. 小样本只做观测，不把经验分位数直接解释为稳定概率保证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


VALID_COMMUNICATION_TAIL_RISK_MODES = ("off", "shadow")


@dataclass(frozen=True)
class CommunicationTailRiskRecommendation:
    """一次状态新组候选的通信尾部校准建议。"""

    num_reports: int
    communication_mean: float
    communication_std: float
    communication_p90: float
    communication_recent_max: float
    existing_uncertainty: float
    p90_tail_excess: float
    max_tail_excess: float
    incremental_p90_margin: float
    incremental_max_margin: float
    original_safe_slack: float
    p90_calibrated_safe_slack: float
    max_calibrated_safe_slack: float
    p90_safe_feasible: bool
    max_safe_feasible: bool
    shadow_action: str
    reason: str


class CommunicationTailRiskShadowPolicy:
    """估计预测器已有 uncertainty 之外的通信尾部风险。"""

    def __init__(self, min_reports: int = 3) -> None:
        if min_reports <= 0:
            raise ValueError("min_reports must be positive")
        self.min_reports = min_reports

    def evaluate(
        self, *, runtime_state: Optional[Any], original_safe_slack: float,
        existing_uncertainty: float,
    ) -> CommunicationTailRiskRecommendation:
        reports = int(getattr(runtime_state, "num_reports", 0)) if runtime_state else 0
        mean = max(0.0, float(getattr(runtime_state, "communication_time_mean", 0.0)))
        std = max(0.0, float(getattr(runtime_state, "communication_time_std", 0.0)))
        p90 = max(mean, float(getattr(runtime_state, "communication_time_p90", mean)))
        recent_max = max(
            p90, float(getattr(runtime_state, "communication_time_recent_max", p90))
        )
        uncertainty = max(0.0, float(existing_uncertainty))
        p90_excess = max(0.0, p90 - mean)
        max_excess = max(0.0, recent_max - mean)

        # safe_duration已经包含一次总 uncertainty，尾部余量只补尚未覆盖部分，
        # 避免把通信标准差重复加入deadline。
        incremental_p90 = max(0.0, p90_excess - uncertainty)
        incremental_max = max(0.0, max_excess - uncertainty)
        p90_slack = original_safe_slack - incremental_p90
        max_slack = original_safe_slack - incremental_max

        if reports < self.min_reports:
            action, reason = "insufficient_history", "communication_tail_history_too_short"
        elif p90_slack >= 0:
            action, reason = "keep_state_window_candidate", "p90_tail_risk_covered"
        else:
            action, reason = "reject_state_window_candidate", "p90_tail_risk_not_covered"
        return CommunicationTailRiskRecommendation(
            num_reports=reports,
            communication_mean=mean,
            communication_std=std,
            communication_p90=p90,
            communication_recent_max=recent_max,
            existing_uncertainty=uncertainty,
            p90_tail_excess=p90_excess,
            max_tail_excess=max_excess,
            incremental_p90_margin=incremental_p90,
            incremental_max_margin=incremental_max,
            original_safe_slack=original_safe_slack,
            p90_calibrated_safe_slack=p90_slack,
            max_calibrated_safe_slack=max_slack,
            p90_safe_feasible=reports >= self.min_reports and p90_slack >= 0,
            max_safe_feasible=reports >= self.min_reports and max_slack >= 0,
            shadow_action=action,
            reason=reason,
        )
