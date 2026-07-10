"""
su_compass.virtual.client_runtime — 真实训练与虚拟时间的桥接层。

核心职责：
    1. 调用 APPFLClientAgent 在 GPU 上真实训练，产出 local_model。
    2. 调用 VirtualRuntimeModel 根据端侧画像生成虚拟完成时间。
    3. 用 RuntimeStateTracker 更新客户端运行状态。
    4. 将结果打包为 VirtualEvent(CLIENT_UPLOAD) 返回给事件队列。

真实训练耗时与虚拟 train_time 完全无关；
真实训练只产出模型更新，虚拟时间只驱动调度。

换句话说：这里是“真实模型质量”和“模拟系统时间”的汇合点。
client_agent.train() 决定 local_model 的数值，VirtualRuntimeModel 决定这个模型
在虚拟世界里什么时候上传到 server。
"""

from typing import Dict, Optional, Tuple

import torch

from su_compass.runtime.profile import ClientRuntimeProfile
from su_compass.runtime.state import RuntimeStateTracker, ClientRuntimeState
from su_compass.runtime.virtual_runtime import VirtualRuntimeModel, VirtualRoundResult

from .event import VirtualEvent, EventType


class VirtualClientRuntime:
    """真实训练 + 虚拟端侧时间桥接器。

    每个实验实例创建一个 VirtualClientRuntime，负责管理
    所有客户端的 VirtualRuntimeModel 和 RuntimeStateTracker。

    Attributes:
        profiles:       客户端画像表，key 是 client_id，value 是 (profile_type, profile)。
                        profile_type 只用于 trace 标注，profile 参与虚拟耗时模拟。
        runtime_model:  虚拟运行时间生成器，全部客户端共享同一实例。
                        负责把画像、Q、staleness、目标到达时间转成一轮 finish_time。
        trackers:       每客户端一个 RuntimeStateTracker。
                        用滑动窗口保存迟到率、通信占比、可用性等状态，供 Oort/FedCompass trace 使用。
        qmin:           FedCompass 最小本地训练步数，用于标记 simulate_round 是否命中下界。
        qmax:           FedCompass 最大本地训练步数，用于标记 simulate_round 是否命中上界。
    """

    def __init__(
        self,
        profiles: Dict[str, Tuple[str, ClientRuntimeProfile]],
        base_step_time: float = 0.05,
        model_size_mb: float = 5.0,
        update_size_mb: float = 5.0,
        unavailable_delay_mean: float = 5.0,
        seed: int = 2026,
        window_size: int = 20,
        qmin: int = 1,
        qmax: int = 200,
    ) -> None:
        """初始化桥接器。

        Args:
            profiles:               {client_id: (profile_type, ClientRuntimeProfile)} 映射。
            base_step_time:         计算能力 1.0 时每个 local step 的基础训练耗时。
            model_size_mb:          下发模型大小 (MB)。
            update_size_mb:         上传更新大小 (MB)。
            unavailable_delay_mean: 客户端不可用时的平均等待时间 (秒)。
            seed:                   随机种子。
            window_size:            RuntimeStateTracker 滑动窗口大小。
            qmin:                   最小本地训练步数。
            qmax:                   最大本地训练步数。
        """
        self.profiles = profiles  # {client_id: (profile_type, profile)}，profile_type 只用于 trace 标注
        self.qmin = qmin          # 传给 report，便于分析 Q 是否被 FedCompass/Oort 压到最小值
        self.qmax = qmax          # 传给 report，便于分析 Q 是否仍处在最大训练步数

        # 虚拟运行时间模型——全部客户端共享同一实例
        self.runtime_model = VirtualRuntimeModel(
            base_step_time=base_step_time,
            model_size_mb=model_size_mb,
            update_size_mb=update_size_mb,
            unavailable_delay_mean=unavailable_delay_mean,
            seed=seed,
        )

        # 每个客户端一个状态追踪器
        self.trackers: Dict[str, RuntimeStateTracker] = {
            cid: RuntimeStateTracker(window_size=window_size)
            for cid in profiles
        }

    # ────────────────── 核心方法：训练 + 生成虚拟上传事件 ──────────────────

    def train_and_schedule_upload(
        self,
        client_id: str,
        client_agent,
        dispatch_time: float,
        local_steps: int,
        target_arrival_time: Optional[float] = None,
        latest_arrival_time: Optional[float] = None,
        staleness: int = 0,
    ) -> Tuple[VirtualEvent, VirtualRoundResult, ClientRuntimeState]:
        """执行一轮真实训练并生成虚拟上传事件。

        流程：
            1. 设置 trainer 的 num_local_steps 为指定 Q。
            2. client_agent.train() — 真实 GPU 训练。
            3. client_agent.get_parameters() — 获取真实 local_model。
            4. VirtualRuntimeModel.simulate_round() — 生成虚拟完成时间。
            5. RuntimeStateTracker.update() — 更新端侧运行状态。
            6. 返回 CLIENT_UPLOAD 事件。

        Args:
            client_id:           客户端编号。
            client_agent:        APPFLClientAgent 实例。
            dispatch_time:       本轮虚拟派发时间。
            local_steps:         本轮本地训练步数 Q。
            target_arrival_time: 目标聚合组期望到达时间。
            latest_arrival_time: 目标聚合组最晚到达时间。
            staleness:           模型陈旧度。

        Returns:
            (event, result, state) 三元组：
                event  — CLIENT_UPLOAD 虚拟事件（含 local_model 和 report）。
                result — VirtualRoundResult 详细拆分耗时。
                state  — 更新后的 ClientRuntimeState。
        """
        _, profile = self.profiles[client_id]  # 获取该客户端画像

        # ──── 步骤 1-3：真实训练 ────
        # 注意这里不会 sleep，也不会按虚拟时间等待；真实训练只是为了拿到模型更新。
        client_agent.trainer.train_configs.num_local_steps = local_steps  # 设置本轮 Q
        client_agent.train()                                             # GPU 真实训练
        local_model = client_agent.get_parameters()                      # 取出本地模型
        non_finite = [
            name for name, value in local_model.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if non_finite:
            preview = ", ".join(non_finite[:3])
            raise FloatingPointError(
                f"{client_id} 本地更新出现 NaN/Inf（Q={local_steps}，参数={preview}）；"
                "已中止实验，避免污染全局模型。"
            )

        # ──── 步骤 4：虚拟时间生成 ────
        # simulate_round 会把画像中的计算/网络/可用性因素折算成 finish_time，
        # 该 finish_time 才是事件队列排序使用的时间。
        result = self.runtime_model.simulate_round(
            profile=profile,
            dispatch_time=dispatch_time,
            local_steps=local_steps,
            target_arrival_time=target_arrival_time,
            latest_arrival_time=latest_arrival_time,
            min_local_steps=self.qmin,
            max_local_steps=self.qmax,
            staleness=staleness,
        )

        # ──── 步骤 5：更新端侧状态 ────
        # state 是滑动窗口统计，后续 FedCompass/Oort 的 reason-aware 决策会读取它。
        state = self.trackers[client_id].update(result.report)

        # ──── 步骤 6：构造上传事件 ────
        event = VirtualEvent(
            time=result.report.finish_time,             # 事件虚拟时间 = 完成时间
            event_type=EventType.CLIENT_UPLOAD,
            client_id=client_id,
            payload={
                "local_model": local_model,             # 真实训练产出的模型
                "report": result.report,                # ClientRoundReport
                "runtime_state": state,                 # 更新后的运行状态快照
                "profile_type": self.profiles[client_id][0],  # 画像类型名称
            },
        )

        return event, result, state

    # ────────────────── 辅助方法 ──────────────────

    def get_profile_type(self, client_id: str) -> str:
        """返回客户端对应的画像类型名称。"""
        return self.profiles[client_id][0]

    def get_tracker(self, client_id: str) -> RuntimeStateTracker:
        """返回指定客户端的 RuntimeStateTracker。"""
        return self.trackers[client_id]

    def all_snapshots(self) -> Dict[str, dict]:
        """返回全部客户端最新状态快照（可序列化）。"""
        result = {}
        for cid, tracker in self.trackers.items():
            snap = tracker.snapshot(cid)
            if snap is not None:
                result[cid] = snap.to_dict()
        return result
