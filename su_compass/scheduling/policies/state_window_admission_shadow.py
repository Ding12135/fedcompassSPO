"""状态时间窗准入组合Shadow Gate。

本模块将已经分别完成Shadow验证的三层事实组合为单一建议：当前已有组是否
安全、是否存在其他安全已有组、固定FedCompass原始新组Q时状态时间窗是否
安全。它只做布尔决策组合，不访问或修改控制器状态，不重新预测Q和完成时间。

Shadow输出用于评估候选覆盖率与误建议率；任何建议都不能反馈真实调度。
"""

from __future__ import annotations

from dataclasses import dataclass


VALID_STATE_WINDOW_ADMISSION_MODES = ("off", "shadow")


@dataclass(frozen=True)
class StateWindowAdmissionDecision:
    """一次已有组决策的组合Shadow建议。"""

    action: str
    current_group_safe: bool
    other_existing_group_safe: bool
    state_new_group_safe: bool
    reason: str


class StateWindowAdmissionShadowPolicy:
    """按硬约束顺序组合已有组与状态新组可行性。"""

    def decide(
        self, *, current_group_safe: bool,
        other_existing_group_safe: bool = False,
        state_new_group_safe: bool = False,
    ) -> StateWindowAdmissionDecision:
        if current_group_safe:
            action, reason = "keep_current_group", "current_group_state_safe"
        elif other_existing_group_safe:
            action, reason = "switch_existing_group", "other_state_safe_group_exists"
        elif state_new_group_safe:
            action = "create_state_window_group"
            reason = "no_safe_existing_group_and_state_new_window_safe"
        else:
            action, reason = "unresolved_mismatch", "no_state_safe_group_resolution"
        return StateWindowAdmissionDecision(
            action=action, current_group_safe=current_group_safe,
            other_existing_group_safe=other_existing_group_safe,
            state_new_group_safe=state_new_group_safe, reason=reason,
        )
