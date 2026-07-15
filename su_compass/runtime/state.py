"""Runtime state tracking for SU-Compass client profiles."""  # 模块说明：根据客户端每轮反馈维护运行状态。

import math  # 导入 math，用于平方根和数值计算。
from collections import defaultdict, deque  # 导入 defaultdict/deque，用于按客户端维护滑动窗口。
from dataclasses import dataclass  # 导入 dataclass，用于定义轻量数据结构。
from typing import Deque, Dict, List, Optional  # 导入类型标注，提升后续耦合时的可读性。


EPS = 1e-8  # 定义极小常数，避免除零。


@dataclass  # 使用数据类声明一轮客户端反馈。
class ClientRoundReport:  # 定义客户端每次完成训练并返回后的观测记录。
    """One observed client update used to refresh runtime state."""  # 说明该记录用于更新端侧运行状态。

    client_id: str  # 客户端唯一编号。
    dispatch_time: float  # 服务器下发模型或任务的虚拟/真实时间。
    finish_time: float  # 客户端返回更新的虚拟/真实时间。
    local_steps: int  # 本轮客户端执行的本地训练步数 Q。
    train_time: Optional[float] = None  # 本轮本地训练耗时，用于区分计算慢因。
    download_time: Optional[float] = None  # 本轮模型下载耗时，用于区分通信慢因。
    upload_time: Optional[float] = None  # 本轮模型上传耗时，用于区分通信慢因。
    spike_delay: Optional[float] = None  # 本轮偶发卡顿耗时，用于识别突发系统慢因。
    availability_wait: Optional[float] = None  # 本轮不可用等待耗时，用于识别可用性慢因。
    target_arrival_time: Optional[float] = None  # 客户端目标聚合组的期望到达时间。
    latest_arrival_time: Optional[float] = None  # 客户端目标聚合组允许的最晚到达时间。
    hit_q_min: bool = False  # 本轮 Q 是否命中最小本地训练步数。
    hit_q_max: bool = False  # 本轮 Q 是否命中最大本地训练步数。
    staleness: int = 0  # 本轮更新对应的模型陈旧度。
    available: bool = True  # 本轮客户端是否处于可用状态。

    @property  # 将 round_time 暴露为只读计算属性。
    def round_time(self) -> float:  # 计算客户端本轮总耗时。
        """Return total elapsed time of the client round."""  # 方法说明：返回本轮从派发到完成的耗时。
        return max(0.0, self.finish_time - self.dispatch_time)  # 返回非负总耗时，避免异常时间导致负值。

    @property  # 将 step_time 暴露为只读计算属性。
    def step_time(self) -> float:  # 计算客户端本轮平均单步耗时。
        """Return average time per local training step."""  # 方法说明：返回本轮每个 local step 的平均耗时。
        return self.round_time / max(1, self.local_steps)  # 用 local_steps 归一化总耗时，并避免除零。

    @property  # 将 compute_step_time 暴露为只读计算属性。
    def compute_step_time(self) -> float:  # 计算本轮纯训练平均单步耗时。
        """Return local training time per local step when reported."""  # 方法说明：返回不含通信的训练单步耗时。
        if self.train_time is None:  # 如果端侧没有上报训练拆分耗时。
            return self.step_time  # 回退到原始含通信 step_time，兼容旧反馈。
        return self.train_time / max(1, self.local_steps)  # 用 local_steps 归一化纯训练耗时。

    @property  # 将 communication_time 暴露为只读计算属性。
    def communication_time(self) -> float:  # 计算本轮通信总耗时。
        """Return download plus upload time when reported."""  # 方法说明：返回下载和上传耗时之和。
        return (self.download_time or 0.0) + (self.upload_time or 0.0)  # 缺失字段按 0 处理。

    @property  # 将 communication_ratio 暴露为只读计算属性。
    def communication_ratio(self) -> float:  # 计算本轮通信耗时占总轮次耗时比例。
        """Return the communication share of the round time."""  # 方法说明：衡量慢因中通信占比。
        return self.communication_time / (self.round_time + EPS)  # 避免除零。

    @property  # 将 availability_wait_ratio 暴露为只读计算属性。
    def availability_wait_ratio(self) -> float:  # 计算本轮不可用等待占比。
        """Return the availability-wait share of the round time."""  # 方法说明：衡量慢因中不可用等待占比。
        return (self.availability_wait or 0.0) / (self.round_time + EPS)  # 缺失字段按 0 处理。

    @property  # 将 late 暴露为只读计算属性。
    def late(self) -> bool:  # 判断客户端是否错过最晚到达时间。
        """Return whether the update misses the latest arrival time."""  # 方法说明：返回是否迟到。
        if self.latest_arrival_time is None:  # 如果没有目标组最晚到达时间。
            return False  # 无法判断迟到时默认不记迟到。
        return self.finish_time > self.latest_arrival_time  # 判断完成时间是否晚于最晚到达时间。

    @property  # 将 early_margin 暴露为只读计算属性。
    def early_margin(self) -> float:  # 计算客户端提前到达目标组的时间余量。
        """Return positive time margin before the expected arrival time."""  # 方法说明：返回相对期望到达时间的提前量。
        if self.target_arrival_time is None:  # 如果没有目标组期望到达时间。
            return 0.0  # 无法计算提前量时返回 0。
        return max(self.target_arrival_time - self.finish_time, 0.0)  # 返回正提前量，迟到或刚好到达则为 0。


@dataclass  # 使用数据类声明运行状态快照。
class ClientRuntimeState:  # 定义单个客户端的聚合运行状态。
    """Aggregated runtime state for one client."""  # 说明该类保存客户端运行状态画像。

    step_time_mean: float = 0.0  # 最近窗口内平均单步训练耗时。
    step_time_std: float = 0.0  # 最近窗口内单步训练耗时标准差。
    step_time_cv: float = 0.0  # 最近窗口内单步训练耗时变异系数。
    compute_step_time_mean: float = 0.0  # 最近窗口内纯训练平均单步耗时。
    compute_step_time_std: float = 0.0  # 最近窗口内纯训练单步耗时标准差。
    compute_step_time_cv: float = 0.0  # 最近窗口内纯训练单步耗时变异系数。
    round_time_mean: float = 0.0  # 最近窗口内每轮总耗时平均值。
    round_time_std: float = 0.0  # 最近窗口内每轮总耗时标准差。
    communication_time_mean: float = 0.0  # 最近窗口内下载加上传的平均通信耗时。
    communication_time_std: float = 0.0  # 最近窗口内通信耗时标准差。
    communication_time_p90: float = 0.0  # 最近窗口通信耗时经验P90（nearest-rank）。
    communication_time_recent_max: float = 0.0  # 最近窗口通信耗时最大值，用于尾部压力测试。
    communication_time_cv: float = 0.0  # 最近窗口内通信耗时变异系数。
    communication_ratio_mean: float = 0.0  # 最近窗口内通信耗时占总轮次耗时的平均比例。
    spike_delay_mean: float = 0.0  # 最近窗口内偶发卡顿平均耗时。
    availability_wait_mean: float = 0.0  # 最近窗口内不可用等待平均耗时。
    availability_wait_ratio_mean: float = 0.0  # 最近窗口内不可用等待占总轮次耗时的平均比例。
    unavailable_event_rate: float = 0.0  # 最近窗口内真正发生不可用等待的轮次比例。
    availability_wait_when_unavailable_mean: float = 0.0  # 只在不可用事件发生时统计的条件等待均值。
    availability_wait_when_unavailable_std: float = 0.0  # 不可用事件条件等待标准差，用于安全时间风险估计。
    unavailable_event_count: int = 0  # 当前窗口内不可用事件数量，防止单个事件触发过强策略。
    last_available: bool = True  # 最近一轮是否可用，仅作为状态解释字段，不直接决定下一轮事件。
    rounds_since_last_unavailable: int = -1  # 距最近不可用事件轮数；-1表示窗口内从未发生。
    late_rate: float = 0.0  # 最近窗口内迟到比例。
    early_margin_mean: float = 0.0  # 最近窗口内平均提前到达时间。
    q_min_hit_rate: float = 0.0  # 最近窗口内 Q 命中下界比例。
    q_max_hit_rate: float = 0.0  # 最近窗口内 Q 命中上界比例。
    availability_rate: float = 1.0  # 最近窗口内客户端可用比例。
    last_staleness: int = 0  # 最近一次更新的模型陈旧度。
    num_reports: int = 0  # 已累计观测到的反馈记录数量。

    def to_dict(self) -> Dict[str, float]:  # 定义转字典方法，便于日志和后续主代码接入。
        """Return a serializable state snapshot."""  # 方法说明：返回可序列化状态快照。
        return {  # 构造状态字典。
            "step_time_mean": self.step_time_mean,  # 写入平均单步耗时。
            "step_time_std": self.step_time_std,  # 写入单步耗时标准差。
            "step_time_cv": self.step_time_cv,  # 写入单步耗时变异系数。
            "compute_step_time_mean": self.compute_step_time_mean,  # 写入纯训练平均单步耗时。
            "compute_step_time_std": self.compute_step_time_std,  # 写入纯训练单步耗时标准差。
            "compute_step_time_cv": self.compute_step_time_cv,  # 写入纯训练单步耗时变异系数。
            "round_time_mean": self.round_time_mean,  # 写入每轮总耗时平均值。
            "round_time_std": self.round_time_std,  # 写入每轮总耗时标准差。
            "communication_time_mean": self.communication_time_mean,  # 写入平均通信耗时。
            "communication_time_std": self.communication_time_std,  # 写入通信耗时标准差。
            "communication_time_p90": self.communication_time_p90,  # 写入经验P90。
            "communication_time_recent_max": self.communication_time_recent_max,  # 写入窗口最大值。
            "communication_time_cv": self.communication_time_cv,  # 写入通信耗时变异系数。
            "communication_ratio_mean": self.communication_ratio_mean,  # 写入平均通信耗时占比。
            "spike_delay_mean": self.spike_delay_mean,  # 写入平均偶发卡顿耗时。
            "availability_wait_mean": self.availability_wait_mean,  # 写入平均不可用等待耗时。
            "availability_wait_ratio_mean": self.availability_wait_ratio_mean,  # 写入平均不可用等待占比。
            "unavailable_event_rate": self.unavailable_event_rate,  # 写入不可用事件发生率。
            "availability_wait_when_unavailable_mean": self.availability_wait_when_unavailable_mean,  # 写入事件发生后的条件等待均值。
            "availability_wait_when_unavailable_std": self.availability_wait_when_unavailable_std,  # 写入条件等待标准差。
            "unavailable_event_count": self.unavailable_event_count,  # 写入窗口内事件数量。
            "last_available": int(self.last_available),  # 布尔值转0/1，便于CSV分析。
            "rounds_since_last_unavailable": self.rounds_since_last_unavailable,  # 写入距最近事件轮数。
            "late_rate": self.late_rate,  # 写入迟到率。
            "early_margin_mean": self.early_margin_mean,  # 写入平均提前量。
            "q_min_hit_rate": self.q_min_hit_rate,  # 写入 Q 下界命中率。
            "q_max_hit_rate": self.q_max_hit_rate,  # 写入 Q 上界命中率。
            "availability_rate": self.availability_rate,  # 写入可用率。
            "last_staleness": self.last_staleness,  # 写入最近陈旧度。
            "num_reports": self.num_reports,  # 写入累计反馈次数。
        }  # 结束状态字典。


class RuntimeStateTracker:  # 定义运行状态追踪器，用于维护多个客户端状态。
    """Track per-client runtime states with sliding windows."""  # 说明该追踪器采用滑动窗口更新状态。

    def __init__(self, window_size: int = 20) -> None:  # 初始化追踪器并设置滑动窗口大小。
        """Create a runtime state tracker."""  # 方法说明：创建状态追踪器。
        if window_size <= 0:  # 如果窗口大小无效。
            raise ValueError("window_size must be positive")  # 抛出错误，要求窗口大小为正。
        self.window_size = window_size  # 保存滑动窗口大小。
        self._reports: Dict[str, Deque[ClientRoundReport]] = defaultdict(  # 创建按客户端分组的反馈窗口。
            lambda: deque(maxlen=self.window_size)  # 为每个客户端创建固定长度队列。
        )  # 结束 defaultdict 初始化。
        self._states: Dict[str, ClientRuntimeState] = {}  # 创建按客户端存储的最新状态字典。

    def update(self, report: ClientRoundReport) -> ClientRuntimeState:  # 用一条新反馈更新对应客户端状态。
        """Update one client state from a new round report."""  # 方法说明：根据新反馈刷新状态。
        if report.local_steps <= 0:  # 如果本地训练步数不合法。
            raise ValueError("local_steps must be positive")  # 抛出错误，local_steps 必须为正。
        self._reports[report.client_id].append(report)  # 将新反馈加入该客户端滑动窗口。
        state = self._build_state(report.client_id)  # 基于窗口重新计算该客户端状态。
        self._states[report.client_id] = state  # 保存最新状态快照。
        return state  # 返回最新状态快照。

    def snapshot(self, client_id: str) -> Optional[ClientRuntimeState]:  # 获取指定客户端的最新状态。
        """Return latest state for a client if available."""  # 方法说明：返回客户端最新状态。
        return self._states.get(client_id)  # 如果存在则返回状态，否则返回 None。

    def snapshots(self) -> Dict[str, Dict[str, float]]:  # 获取所有客户端状态字典。
        """Return serializable snapshots for all clients."""  # 方法说明：返回全部客户端可序列化状态。
        return {client_id: state.to_dict() for client_id, state in self._states.items()}  # 将所有状态转换为字典。

    def _build_state(self, client_id: str) -> ClientRuntimeState:  # 根据客户端窗口构建状态。
        """Build a runtime state from recent reports."""  # 方法说明：使用最近窗口反馈计算运行状态。
        reports = list(self._reports[client_id])  # 取出该客户端最近窗口内所有反馈。
        step_times = [report.step_time for report in reports]  # 提取每轮单步耗时序列。
        compute_step_times = [report.compute_step_time for report in reports]  # 提取纯训练单步耗时序列。
        round_times = [report.round_time for report in reports]  # 提取每轮总耗时序列。
        communication_times = [report.communication_time for report in reports]  # 提取通信耗时序列。
        communication_ratios = [report.communication_ratio for report in reports]  # 提取通信耗时占比序列。
        spike_delays = [(report.spike_delay or 0.0) for report in reports]  # 提取偶发卡顿耗时序列。
        availability_waits = [(report.availability_wait or 0.0) for report in reports]  # 提取不可用等待耗时序列。
        # 不可用等待是“多数为0、少数为较大值”的事件型变量，不能只保留包含0的
        # 总体均值。这里额外保留事件发生率与事件发生后的条件等待分布。
        unavailable_waits = [wait for wait in availability_waits if wait > EPS]
        unavailable_flags = [1.0 if wait > EPS else 0.0 for wait in availability_waits]
        unavailable_mean = _mean(unavailable_waits)
        unavailable_std = _std(unavailable_waits, unavailable_mean)
        rounds_since_unavailable = -1
        for distance, wait in enumerate(reversed(availability_waits)):
            if wait > EPS:
                rounds_since_unavailable = distance
                break
        availability_wait_ratios = [report.availability_wait_ratio for report in reports]  # 提取不可用等待占比序列。
        late_flags = [1.0 if report.late else 0.0 for report in reports]  # 提取迟到标记序列。
        early_margins = [report.early_margin for report in reports]  # 提取提前量序列。
        q_min_flags = [1.0 if report.hit_q_min else 0.0 for report in reports]  # 提取 Q 下界命中标记序列。
        q_max_flags = [1.0 if report.hit_q_max else 0.0 for report in reports]  # 提取 Q 上界命中标记序列。
        availability_flags = [1.0 if report.available else 0.0 for report in reports]  # 提取可用标记序列。
        step_mean = _mean(step_times)  # 计算平均单步耗时。
        step_std = _std(step_times, step_mean)  # 计算单步耗时标准差。
        compute_step_mean = _mean(compute_step_times)  # 计算纯训练平均单步耗时。
        compute_step_std = _std(compute_step_times, compute_step_mean)  # 计算纯训练单步耗时标准差。
        round_mean = _mean(round_times)  # 计算每轮总耗时平均值。
        round_std = _std(round_times, round_mean)  # 计算每轮总耗时标准差。
        communication_mean = _mean(communication_times)  # 计算平均通信耗时。
        communication_std = _std(communication_times, communication_mean)  # 计算通信耗时标准差。
        return ClientRuntimeState(  # 构造并返回运行状态快照。
            step_time_mean=step_mean,  # 写入平均单步耗时。
            step_time_std=step_std,  # 写入单步耗时标准差。
            step_time_cv=step_std / (step_mean + EPS),  # 计算并写入单步耗时变异系数。
            compute_step_time_mean=compute_step_mean,  # 写入纯训练平均单步耗时。
            compute_step_time_std=compute_step_std,  # 写入纯训练单步耗时标准差。
            compute_step_time_cv=compute_step_std / (compute_step_mean + EPS),  # 写入纯训练单步耗时变异系数。
            round_time_mean=round_mean,  # 写入每轮总耗时平均值。
            round_time_std=round_std,  # 写入每轮总耗时标准差。
            communication_time_mean=communication_mean,  # 写入平均通信耗时。
            communication_time_std=communication_std,  # 写入通信耗时标准差。
            communication_time_p90=_nearest_rank(communication_times, 0.90),  # 写入经验P90。
            communication_time_recent_max=max(communication_times, default=0.0),  # 写入窗口最大值。
            communication_time_cv=communication_std / (communication_mean + EPS),  # 写入通信耗时变异系数。
            communication_ratio_mean=_mean(communication_ratios),  # 写入平均通信耗时占比。
            spike_delay_mean=_mean(spike_delays),  # 写入平均偶发卡顿耗时。
            availability_wait_mean=_mean(availability_waits),  # 写入平均不可用等待耗时。
            availability_wait_ratio_mean=_mean(availability_wait_ratios),  # 写入平均不可用等待占比。
            unavailable_event_rate=_mean(unavailable_flags),  # 写入窗口内不可用事件比例。
            availability_wait_when_unavailable_mean=unavailable_mean,  # 写入条件等待均值。
            availability_wait_when_unavailable_std=unavailable_std,  # 写入条件等待标准差。
            unavailable_event_count=len(unavailable_waits),  # 写入窗口内事件数。
            last_available=reports[-1].available,  # 写入最近一轮可用状态。
            rounds_since_last_unavailable=rounds_since_unavailable,  # 写入距最近事件轮数。
            late_rate=_mean(late_flags),  # 写入迟到率。
            early_margin_mean=_mean(early_margins),  # 写入平均提前量。
            q_min_hit_rate=_mean(q_min_flags),  # 写入 Q 下界命中率。
            q_max_hit_rate=_mean(q_max_flags),  # 写入 Q 上界命中率。
            availability_rate=_mean(availability_flags),  # 写入可用率。
            last_staleness=reports[-1].staleness,  # 写入最近一次陈旧度。
            num_reports=len(reports),  # 写入当前窗口内反馈数量。
        )  # 结束状态构造。


def _mean(values: List[float]) -> float:  # 定义均值计算函数。
    """Return arithmetic mean for a possibly empty list."""  # 函数说明：计算列表均值。
    if not values:  # 如果列表为空。
        return 0.0  # 空列表均值返回 0。
    return sum(values) / len(values)  # 返回算术平均值。


def _std(values: List[float], mean_value: Optional[float] = None) -> float:  # 定义标准差计算函数。
    """Return population standard deviation for recent observations."""  # 函数说明：计算总体标准差。
    if not values:  # 如果列表为空。
        return 0.0  # 空列表标准差返回 0。
    mean_value = _mean(values) if mean_value is None else mean_value  # 如果未传入均值，则先计算均值。
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)  # 计算总体方差。
    return math.sqrt(max(variance, 0.0))  # 返回非负方差的平方根。


def _nearest_rank(values: List[float], probability: float) -> float:
    """计算保守经验分位数；小窗口不做线性插值，避免淡化真实尾部。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]
