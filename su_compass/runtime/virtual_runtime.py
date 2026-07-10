"""Trace-style virtual runtime generator for SU-Compass client profiles."""  # 模块说明：根据端侧画像生成虚拟运行时间。

import random  # 导入 random，用于生成可复现实验所需的随机波动。
from dataclasses import dataclass  # 导入 dataclass，用于定义单轮模拟结果。
from typing import Optional  # 导入 Optional，用于标注可选参数。

from .profile import ClientRuntimeProfile  # 导入客户端端侧运行画像。
from .state import ClientRoundReport  # 导入客户端每轮反馈记录。


@dataclass  # 使用数据类声明单轮虚拟运行结果。
class VirtualRoundResult:  # 定义虚拟运行模型输出的一轮结果。
    """Detailed virtual runtime outcome for one client round."""  # 说明该结果保存训练、通信和总耗时。

    report: ClientRoundReport  # 用于更新 RuntimeStateTracker 的客户端反馈记录。
    train_time: float  # 本轮虚拟本地训练耗时。
    download_time: float  # 本轮虚拟模型下载耗时。
    upload_time: float  # 本轮虚拟模型上传耗时。
    spike_delay: float  # 本轮虚拟偶发卡顿耗时。
    availability_wait: float  # 本轮虚拟可用性等待耗时。


class VirtualRuntimeModel:  # 定义虚拟端侧运行时间模型。
    """Generate realistic client round times from configurable profiles."""  # 说明该模型从 profile 生成现实风格运行时间。

    def __init__(  # 定义初始化方法。
        self,  # 当前虚拟运行模型实例。
        base_step_time: float = 0.05,  # 计算能力为 1.0 时每个 local step 的基础训练耗时。
        model_size_mb: float = 5.0,  # 服务器下发模型大小，单位 MB。
        update_size_mb: float = 5.0,  # 客户端上传更新大小，单位 MB。
        unavailable_delay_mean: float = 5.0,  # 客户端不可用时的平均等待时间，单位秒。
        seed: int = 2026,  # 随机种子，用于保证模拟可复现。
    ) -> None:  # 初始化方法无返回值。
        """Create a virtual runtime model."""  # 方法说明：创建虚拟端侧运行时间模型。
        if base_step_time <= 0:  # 如果基础单步耗时不合法。
            raise ValueError("base_step_time must be positive")  # 抛出错误，基础单步耗时必须为正。
        if model_size_mb <= 0:  # 如果模型大小不合法。
            raise ValueError("model_size_mb must be positive")  # 抛出错误，模型大小必须为正。
        if update_size_mb <= 0:  # 如果更新大小不合法。
            raise ValueError("update_size_mb must be positive")  # 抛出错误，更新大小必须为正。
        if unavailable_delay_mean < 0:  # 如果不可用等待时间为负。
            raise ValueError("unavailable_delay_mean must be non-negative")  # 抛出错误，不可用等待不能为负。
        self.base_step_time = base_step_time  # 保存基础单步训练耗时。
        self.model_size_mb = model_size_mb  # 保存模型大小。
        self.update_size_mb = update_size_mb  # 保存更新大小。
        self.unavailable_delay_mean = unavailable_delay_mean  # 保存不可用等待平均时间。
        self._rng = random.Random(seed)  # 创建独立随机数生成器，避免影响全局随机状态。

    def simulate_round(  # 定义单轮虚拟运行模拟方法。
        self,  # 当前虚拟运行模型实例。
        profile: ClientRuntimeProfile,  # 参与本轮模拟的客户端端侧画像。
        dispatch_time: float,  # 本轮服务器派发任务的虚拟时间。
        local_steps: int,  # 本轮客户端执行的本地训练步数。
        target_arrival_time: Optional[float] = None,  # 本轮目标聚合组期望到达时间。
        latest_arrival_time: Optional[float] = None,  # 本轮目标聚合组最晚到达时间。
        min_local_steps: Optional[int] = None,  # FedCompass 当前配置的最小本地步数。
        max_local_steps: Optional[int] = None,  # FedCompass 当前配置的最大本地步数。
        staleness: int = 0,  # 本轮模型陈旧度。
    ) -> VirtualRoundResult:  # 返回包含 report 和拆分耗时的虚拟单轮结果。
        """Simulate one client round without sleeping or touching FedCompass."""  # 方法说明：不使用 sleep，独立模拟单轮端侧运行。
        profile.validate()  # 校验客户端画像参数合法性。
        if local_steps <= 0:  # 如果本地训练步数不合法。
            raise ValueError("local_steps must be positive")  # 抛出错误，本地步数必须为正。
        available = self._sample_available(profile)  # 根据可用概率采样本轮客户端是否可用。
        availability_wait = self._sample_availability_wait(available)  # 根据可用状态采样等待时间。
        download_time = self._network_time(  # 计算模型下载耗时。
            size_mb=self.model_size_mb,  # 使用模型大小作为下载数据量。
            bandwidth_mbps=profile.download_bandwidth_mbps,  # 使用客户端下载带宽。
            jitter_cv=profile.network_jitter_cv,  # 使用网络波动系数。
            base_latency=profile.base_latency,  # 使用基础网络延迟。
        )  # 结束下载耗时计算。
        train_time = self._train_time(  # 计算本地训练耗时。
            local_steps=local_steps,  # 使用本轮本地训练步数。
            compute_capacity=profile.compute_capacity,  # 使用客户端计算能力。
            jitter_cv=profile.compute_jitter_cv,  # 使用计算波动系数。
        )  # 结束训练耗时计算。
        upload_time = self._network_time(  # 计算模型更新上传耗时。
            size_mb=self.update_size_mb,  # 使用更新大小作为上传数据量。
            bandwidth_mbps=profile.upload_bandwidth_mbps,  # 使用客户端上传带宽。
            jitter_cv=profile.network_jitter_cv,  # 使用网络波动系数。
            base_latency=profile.base_latency,  # 使用基础网络延迟。
        )  # 结束上传耗时计算。
        spike_delay = self._sample_spike(profile)  # 根据 profile 采样偶发卡顿耗时。
        round_time = download_time + train_time + upload_time + spike_delay + availability_wait  # 汇总本轮虚拟总耗时。
        finish_time = dispatch_time + round_time  # 根据派发时间和总耗时得到完成时间。
        report = ClientRoundReport(  # 构造可被 RuntimeStateTracker 消费的客户端反馈。
            client_id=profile.client_id,  # 写入客户端编号。
            dispatch_time=dispatch_time,  # 写入本轮派发时间。
            finish_time=finish_time,  # 写入本轮完成时间。
            local_steps=local_steps,  # 写入本轮本地训练步数。
            train_time=train_time,  # 写入本轮训练拆分耗时，供服务端识别计算慢因。
            download_time=download_time,  # 写入本轮下载拆分耗时，供服务端识别通信慢因。
            upload_time=upload_time,  # 写入本轮上传拆分耗时，供服务端识别通信慢因。
            spike_delay=spike_delay,  # 写入本轮偶发卡顿耗时，供服务端识别突发慢因。
            availability_wait=availability_wait,  # 写入本轮不可用等待耗时，供服务端识别可用性慢因。
            target_arrival_time=target_arrival_time,  # 写入目标组期望到达时间。
            latest_arrival_time=latest_arrival_time,  # 写入目标组最晚到达时间。
            hit_q_min=(local_steps == min_local_steps) if min_local_steps is not None else False,  # 判断是否命中 Q 下界。
            hit_q_max=(local_steps == max_local_steps) if max_local_steps is not None else False,  # 判断是否命中 Q 上界。
            staleness=staleness,  # 写入模型陈旧度。
            available=available,  # 写入本轮客户端可用状态。
        )  # 结束反馈构造。
        return VirtualRoundResult(  # 返回虚拟单轮结果。
            report=report,  # 写入客户端反馈记录。
            train_time=train_time,  # 写入训练耗时。
            download_time=download_time,  # 写入下载耗时。
            upload_time=upload_time,  # 写入上传耗时。
            spike_delay=spike_delay,  # 写入偶发卡顿耗时。
            availability_wait=availability_wait,  # 写入可用性等待耗时。
        )  # 结束返回结果构造。

    def _train_time(self, local_steps: int, compute_capacity: float, jitter_cv: float) -> float:  # 计算虚拟训练耗时。
        """Return local training time driven by compute capacity."""  # 方法说明：根据计算能力计算训练时间。
        mean_time = local_steps * self.base_step_time / compute_capacity  # 计算无波动情况下的平均训练耗时。
        return self._positive_jitter(mean_time, jitter_cv)  # 施加正值随机波动并返回训练耗时。

    def _network_time(self, size_mb: float, bandwidth_mbps: float, jitter_cv: float, base_latency: float) -> float:  # 计算虚拟通信耗时。
        """Return communication time driven by bandwidth and latency."""  # 方法说明：根据带宽和延迟计算通信时间。
        transfer_time = size_mb * 8.0 / bandwidth_mbps  # 将 MB 转换为 Mbit 后除以 Mbps 得到传输耗时。
        mean_time = transfer_time + base_latency  # 加上基础网络延迟得到平均通信耗时。
        return self._positive_jitter(mean_time, jitter_cv)  # 施加正值随机波动并返回通信耗时。

    def _sample_spike(self, profile: ClientRuntimeProfile) -> float:  # 采样偶发卡顿耗时。
        """Return occasional spike delay from profile settings."""  # 方法说明：根据 profile 生成偶发卡顿。
        if self._rng.random() >= profile.spike_probability:  # 如果本轮没有触发卡顿。
            return 0.0  # 返回 0 卡顿耗时。
        return self._rng.expovariate(1.0 / max(profile.spike_delay_mean, 1e-6))  # 用指数分布生成正卡顿时长。

    def _sample_available(self, profile: ClientRuntimeProfile) -> bool:  # 采样本轮客户端是否可用。
        """Return whether the client is available this round."""  # 方法说明：根据 FLASH 风格可用概率采样。
        return self._rng.random() <= profile.availability_probability  # 返回可用概率采样结果。

    def _sample_availability_wait(self, available: bool) -> float:  # 采样不可用导致的等待耗时。
        """Return extra wait time when a client is temporarily unavailable."""  # 方法说明：模拟设备状态不满足训练条件时的等待。
        if available:  # 如果客户端本轮可用。
            return 0.0  # 不产生额外等待。
        if self.unavailable_delay_mean == 0:  # 如果不可用等待均值为 0。
            return 0.0  # 返回 0 等待。
        return self._rng.expovariate(1.0 / self.unavailable_delay_mean)  # 用指数分布生成不可用等待时长。

    def _positive_jitter(self, mean_value: float, cv: float) -> float:  # 对平均值施加正值随机波动。
        """Return a positive noisy value with approximate coefficient of variation."""  # 方法说明：生成近似指定 CV 的正值波动。
        if cv <= 0:  # 如果波动系数为 0。
            return mean_value  # 直接返回平均值。
        sampled = self._rng.gauss(mean_value, abs(mean_value * cv))  # 用正态分布按均值和标准差采样。
        return max(sampled, mean_value * 0.05)  # 截断为正值，避免出现负耗时。

