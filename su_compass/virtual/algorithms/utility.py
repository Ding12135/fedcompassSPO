"""
su_compass.virtual.algorithms.utility — Oort 效用适配模块（reason-aware 调度打分）。

本模块是「把 Oort 的效用思想适配到 FedCompass 半异步调度」的唯一计算内核，
不依赖任何调度器内部状态，全部为无副作用纯函数，便于单测、复用与解耦。

────────────────────────────────────────────────────────────────────────
为什么参考 Oort，又为什么不能照搬
────────────────────────────────────────────────────────────────────────
Oort（OSDI'21，开源实现见 github.com/SymbioticLab/Oort 的 `oort.py`
`_training_selector`）核心是「client selection」：为**同步** FL 每轮挑一批
高效用客户端，效用 = 统计效用 × 时间惩罚：

    utility_i = |B_i| * sqrt( mean(loss_i^2) )                 # 统计效用
    if round_duration_i > T:                                    # 时间惩罚
        utility_i *= (T / round_duration_i) ** alpha

FedCompass 是**半异步**的，所有客户端都参与、不做挑选，调度自由度在于：
    1. 给每个客户端分配多少本地步数 Q；
    2. 把哪些客户端编进同一个 arrival group。

因此这里保留 Oort「用效用/代价对客户端排序、惩罚高代价 straggler」的思想，
但把「选或不选」改造成「Q 给多少、能不能进这个组」：

    - Oort 的 round_duration 惩罚  →  SystemPenalty（reason-aware，见下）
    - Oort 的 statistical utility  →  StatisticalUtility（当前占位=1，预留 loss 接入）
    - Oort 挑高效用客户端           →  effective_step_time 决定 Q；risk_score 决定分组

────────────────────────────────────────────────────────────────────────
reason-aware SystemPenalty
────────────────────────────────────────────────────────────────────────
原始 FedCompass 只用一个标量 speed（round_time/steps 的指数平滑）决定 Q，
把「算得慢」和「传得慢 / 爱迟到 / 抖动大」混为一谈。本模块用
RuntimeStateTracker 的多维状态把慢因拆开，折算成一个 >=1 的惩罚系数：

    SystemPenalty_i = clip(
        1 + λ_comm · communication_ratio_mean_i     # 通信占比高 → 惩罚
          + λ_late · late_rate_i                    # 经常迟到   → 惩罚
          + λ_var  · step_time_cv_i                 # 单步抖动大 → 惩罚
          + λ_avail · (1 - availability_rate_i),    # 不可用多   → 惩罚
        1.0, penalty_clip)

    effective_step_time_i = speed_smoothed_i · SystemPenalty_i

语义：通信差 / 迟到多 / 抖动大 / 不稳定的客户端，被视为「有效上更慢」，
Q 分配更保守 → 上传负担更轻、更少触发 deadline / 迟到。

冷启动保护：num_reports < min_reports 时 SystemPenalty=1，退化为原始
FedCompass，避免前几轮无统计数据时乱调。

本模块**只输出分数**，是否真正影响调度由 OortCompass 控制器的 mode 决定
（off / shadow / q_only / q_and_group），从而实现「先影子只算不用、再单点生效」
的安全引入。

读代码时不用关心 FedCompass 的 group 数据结构：本文件只回答
“这个客户端当前状态应该带来多大惩罚/风险”。真正把分数用到 Q 和分组的地方
在 oort_compass.py。
"""

from dataclasses import dataclass
from typing import Any, Optional


# 与 runtime.state 保持一致的极小常数，避免除零。
EPS = 1e-8


@dataclass
class OortConfig:
    """Oort-Compass 效用计算与调度开关配置。

    Attributes:
        mode:            调度作用模式，控制分数是否真正影响调度：
                         - "off":          完全等价原始 FedCompass（不算也不用）。
                         - "shadow":       计算并写 trace，但不改 Q/分组（用于一致性回归）。
                         - "q_only":       仅用 effective_step_time 影响 Q。
                         - "q_and_group":  在 q_only 基础上叠加分组迟到风险过滤。
        lambda_comm:     通信占比惩罚权重。越大，network_poor 客户端越容易被减少 Q。
        lambda_late:     迟到率惩罚权重。越大，经常超过 latest_arrival_time 的客户端越保守。
        lambda_var:      单步耗时变异系数惩罚权重。越大，抖动大的客户端越难拿到高 Q。
        lambda_avail:    不可用惩罚权重（(1-availability_rate) 的系数）。
                         越大，availability_limited 客户端越容易被视作慢。
        penalty_clip:    SystemPenalty 上限，防止个别异常客户端 Q 被压到 0。
                         这是“最多惩罚多少倍”，不是 Q 的上下界。
        min_reports:     冷启动阈值；窗口反馈数不足时不施加惩罚。
                         例如 min_reports=2 表示至少看过两轮 report 才相信状态统计。
        risk_threshold:  分组风险门槛；risk_score 超过它的客户端视为高风险。
                         只在 q_and_group 模式影响是否允许 join 某个 group。
        slack_min_ratio: 高风险客户端可加入 group 的最小时间余量比例
                         （slack / (expected-created) 需 >= 该比例）。
    """

    mode: str = "off"
    lambda_comm: float = 1.0
    lambda_late: float = 1.0
    lambda_var: float = 0.5
    lambda_avail: float = 0.5
    penalty_clip: float = 3.0
    min_reports: int = 2
    risk_threshold: float = 0.5
    # latest_time_factor=1.2 时 group 的典型 slack/span 为 0.2；
    # 门槛需略高于该值，才能对高风险客户端实际触发过滤。
    slack_min_ratio: float = 0.25

    @property
    def active(self) -> bool:
        """是否需要计算 Oort 分数（off 以外都要算，哪怕只写 trace）。"""
        return self.mode != "off"

    @property
    def affects_q(self) -> bool:
        """当前 mode 下 effective_step_time 是否真正参与 Q 计算。"""
        return self.mode in ("q_only", "q_and_group")

    @property
    def affects_group(self) -> bool:
        """当前 mode 下是否启用分组迟到风险过滤。"""
        return self.mode == "q_and_group"


def system_penalty(state: Optional[Any], cfg: OortConfig) -> float:
    """计算客户端的 reason-aware 系统惩罚系数（>= 1）。

    对齐 Oort「对高代价 straggler 施加时间惩罚」的思想，但把单一 duration
    展开成通信 / 迟到 / 抖动 / 可用性多维慢因。

    Args:
        state: RuntimeStateTracker.snapshot() 返回的 ClientRuntimeState；
               None 或反馈不足时按冷启动处理，返回 1.0。
        cfg:   OortConfig。

    Returns:
        clip 到 [1.0, cfg.penalty_clip] 的惩罚系数。
    """
    # 冷启动保护：无状态或窗口样本不足 → 不惩罚，退化为原始 FedCompass。
    if state is None or getattr(state, "num_reports", 0) < cfg.min_reports:
        return 1.0

    # 各维度都来自 RuntimeStateTracker 的滑动窗口，不看单轮尖峰，避免调度抖动过大。
    comm_ratio = max(0.0, getattr(state, "communication_ratio_mean", 0.0))
    late_rate = max(0.0, getattr(state, "late_rate", 0.0))
    step_cv = max(0.0, getattr(state, "step_time_cv", 0.0))
    availability = getattr(state, "availability_rate", 1.0)

    penalty = (
        1.0
        + cfg.lambda_comm * comm_ratio
        + cfg.lambda_late * late_rate
        + cfg.lambda_var * step_cv
        + cfg.lambda_avail * (1.0 - max(0.0, min(1.0, availability)))
    )
    # 下界 1（不奖励，只惩罚）；上界 penalty_clip（防止 Q 被压到 0）。
    return float(min(max(penalty, 1.0), cfg.penalty_clip))


def statistical_utility(state: Optional[Any], cfg: OortConfig) -> float:
    """统计效用占位实现（当前恒为 1.0）。

    Oort 的统计效用来自本地训练 loss（|B|·sqrt(mean(loss^2))）。当前虚拟框架
    的 report 尚未回传 local_loss，故先返回 1.0，保持接口稳定；后续接入 loss
    时只需在此扩展，不影响调用方。
    """
    return 1.0


def effective_step_time(
    speed_smoothed: float,
    state: Optional[Any],
    cfg: OortConfig,
) -> float:
    """把端侧状态折算成「有效单步时间」，作为 Q 计算的分母替代 speed。

    Q = floor(remaining_time / effective_step_time)：
    惩罚越大 → 有效单步越长 → 同样时间窗内分到的 Q 越小。

    Args:
        speed_smoothed: 调度器 speed_momentum 平滑后的 speed（原始 Q 分母）。
        state:          ClientRuntimeState 快照。
        cfg:            OortConfig。

    Returns:
        speed_smoothed × SystemPenalty；冷启动时等于 speed_smoothed。
    """
    # 调度器后续用 remaining_time / effective_step_time 反推 Q；
    # 因此这里返回的是“惩罚后的时间分母”，不是最终 Q。
    return max(speed_smoothed, EPS) * system_penalty(state, cfg)


def risk_score(state: Optional[Any], cfg: OortConfig) -> float:
    """计算客户端的迟到/抖动风险分数，用于分组过滤（q_and_group 模式）。

    对齐文档 §15.3：risk = late_rate + step_time_cv + communication_ratio_mean。
    分数越高，越不适合被塞进时间余量小的 arrival group。冷启动返回 0（不视为高风险）。
    """
    if state is None or getattr(state, "num_reports", 0) < cfg.min_reports:
        return 0.0
    return float(
        max(0.0, getattr(state, "late_rate", 0.0))
        + max(0.0, getattr(state, "step_time_cv", 0.0))
        + max(0.0, getattr(state, "communication_ratio_mean", 0.0))
    )


def oort_score(
    speed_smoothed: float,
    state: Optional[Any],
    cfg: OortConfig,
) -> float:
    """综合 Oort 分数，仅用于 shadow / trace 观测，不直接决定调度。

    定义为 统计效用 / 系统惩罚：值越高代表「高价值、低代价」。
    与 effective_step_time 互补，便于离线分析 Oort 想调整的方向。
    """
    return statistical_utility(state, cfg) / max(system_penalty(state, cfg), EPS)
