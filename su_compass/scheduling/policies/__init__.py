"""SU-Compass 调度决策策略。

策略模块只消费预测器输出并生成建议，不访问或修改FedCompass控制器。第一阶段
提供Shadow Q策略，后续实际Q控制器可以复用同一接口，避免验证公式与接入公式
不一致。
"""

from .q_shadow import QRecommendation, ShadowQPolicy
from .group_shadow import GroupCandidate, GroupRecommendation, ParetoGroupShadowPolicy
from .group_admission import (
    GroupAdmissionDecision,
    RiskConstrainedGroupAdmissionPolicy,
    VALID_GROUP_ADMISSION_MODES,
)
from .all_group_feasibility_shadow import (
    AllGroupFeasibilityCandidate,
    AllGroupFeasibilityRecommendation,
    AllGroupFeasibilityShadowPolicy,
    VALID_ALL_GROUP_FEASIBILITY_MODES,
)
from .group_creation_counterfactual_shadow import (
    FedCompassNewGroupPlan,
    GroupCreationCounterfactual,
    GroupCreationCounterfactualShadowPolicy,
    VALID_GROUP_CREATION_COUNTERFACTUAL_MODES,
    calculate_fedcompass_new_group_plan,
)
from .state_group_creation_q_shadow import (
    StateGroupCreationQRecommendation,
    StateGroupCreationQShadowPolicy,
    VALID_STATE_GROUP_CREATION_Q_MODES,
)
from .state_group_window_shadow import (
    StateGroupWindowRecommendation,
    StateGroupWindowShadowPolicy,
    VALID_STATE_GROUP_WINDOW_MODES,
)
from .state_window_admission_shadow import (
    StateWindowAdmissionDecision,
    StateWindowAdmissionShadowPolicy,
    VALID_STATE_WINDOW_ADMISSION_MODES,
)
from .communication_tail_risk_shadow import (
    CommunicationTailRiskRecommendation,
    CommunicationTailRiskShadowPolicy,
    VALID_COMMUNICATION_TAIL_RISK_MODES,
)
from .communication_robust_q_shadow import (
    CommunicationRobustQRecommendation,
    CommunicationRobustQShadowPolicy,
    VALID_COMMUNICATION_ROBUST_Q_MODES,
)
from .rup_workload import RUPConfig, RUPDecision, RUPWorkloadPolicy

__all__ = [
    "QRecommendation",
    "ShadowQPolicy",
    "GroupCandidate",
    "GroupRecommendation",
    "ParetoGroupShadowPolicy",
    "GroupAdmissionDecision",
    "RiskConstrainedGroupAdmissionPolicy",
    "VALID_GROUP_ADMISSION_MODES",
    "AllGroupFeasibilityCandidate",
    "AllGroupFeasibilityRecommendation",
    "AllGroupFeasibilityShadowPolicy",
    "VALID_ALL_GROUP_FEASIBILITY_MODES",
    "FedCompassNewGroupPlan",
    "GroupCreationCounterfactual",
    "GroupCreationCounterfactualShadowPolicy",
    "VALID_GROUP_CREATION_COUNTERFACTUAL_MODES",
    "calculate_fedcompass_new_group_plan",
    "StateGroupCreationQRecommendation",
    "StateGroupCreationQShadowPolicy",
    "VALID_STATE_GROUP_CREATION_Q_MODES",
    "StateGroupWindowRecommendation",
    "StateGroupWindowShadowPolicy",
    "VALID_STATE_GROUP_WINDOW_MODES",
    "StateWindowAdmissionDecision",
    "StateWindowAdmissionShadowPolicy",
    "VALID_STATE_WINDOW_ADMISSION_MODES",
    "CommunicationTailRiskRecommendation",
    "CommunicationTailRiskShadowPolicy",
    "VALID_COMMUNICATION_TAIL_RISK_MODES",
    "CommunicationRobustQRecommendation",
    "CommunicationRobustQShadowPolicy",
    "VALID_COMMUNICATION_ROBUST_Q_MODES",
    "RUPConfig",
    "RUPDecision",
    "RUPWorkloadPolicy",
]
