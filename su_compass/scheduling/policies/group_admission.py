"""Risk-Constrained Group Admission（风险约束入组控制）。

本模块只把Trust-Q已经得到的安全可行性转换为“准入/拒绝”决策，不读取或
修改FedCompass控制器状态。shadow模式只记录本应采取的动作；apply模式仅在
已有group不存在安全Q时拒绝入组，使父控制器自然回退到原始 ``_create_group``。
"""

from __future__ import annotations

from dataclasses import dataclass


VALID_GROUP_ADMISSION_MODES = ("shadow", "apply")


@dataclass(frozen=True)
class GroupAdmissionDecision:
    """一次已有group准入判定的不可变结果。"""

    mode: str
    safe_feasible: bool
    admitted: bool
    shadow_action: str
    applied_action: str
    reason: str


class RiskConstrainedGroupAdmissionPolicy:
    """仅拒绝Trust-Q明确判定为无安全Q的已有group。"""

    def __init__(self, mode: str = "shadow") -> None:
        if mode not in VALID_GROUP_ADMISSION_MODES:
            raise ValueError(
                f"group_admission_mode must be one of {VALID_GROUP_ADMISSION_MODES}"
            )
        self.mode = mode

    def decide(self, safe_feasible: bool) -> GroupAdmissionDecision:
        """根据Trust-Q安全性给出准入结果；不重新计算预测或Q。"""
        shadow_action = "join_existing_group" if safe_feasible else "create_group"
        admitted = safe_feasible or self.mode == "shadow"
        applied_action = "join_existing_group" if admitted else "create_group"
        if safe_feasible:
            reason = "trust_q_safe_admit"
        elif self.mode == "shadow":
            reason = "group_mismatch_shadow_keep_join"
        else:
            reason = "group_mismatch_reject_to_fedcompass_create_group"
        return GroupAdmissionDecision(
            mode=self.mode,
            safe_feasible=safe_feasible,
            admitted=admitted,
            shadow_action=shadow_action,
            applied_action=applied_action,
            reason=reason,
        )
