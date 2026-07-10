"""Configurable client runtime profiles for SU-Compass experiments."""  # 模块说明：定义可调参数的端侧客户端运行画像。

from dataclasses import dataclass  # 导入 dataclass，用于声明轻量参数对象。
from typing import Dict  # 导入 Dict，用于标注多客户端画像字典类型。


@dataclass  # 使用数据类自动生成初始化、打印和比较方法。
class ClientRuntimeProfile:  # 定义单个客户端的端侧运行画像。
    """A lightweight FedScale/FLASH-style runtime profile."""  # 说明该画像参考 FedScale/FLASH 的端侧异构建模思想。

    client_id: str  # 客户端唯一编号，用于和后续 FedCompass 的 client_id 对齐。
    compute_capacity: float = 1.0  # 客户端计算能力，数值越大表示每步训练越快。
    compute_jitter_cv: float = 0.05  # 计算时间波动系数，参考设备负载/热降频等运行波动。
    upload_bandwidth_mbps: float = 20.0  # 客户端上传带宽，单位 Mbps，参考 FedScale/MobiPerf 网络建模。
    download_bandwidth_mbps: float = 50.0  # 客户端下载带宽，单位 Mbps，参考 FedScale/MobiPerf 网络建模。
    network_jitter_cv: float = 0.10  # 网络延迟/带宽波动系数，用于模拟跨机构网络抖动。
    base_latency: float = 0.05  # 基础往返延迟，单位秒，用于模拟网络传播和协议开销。
    spike_probability: float = 0.0  # 偶发卡顿概率，用于模拟后台抢占、短时拥塞或资源争用。
    spike_delay_mean: float = 0.0  # 偶发卡顿平均时长，单位秒。
    availability_probability: float = 1.0  # 客户端单轮可用概率，参考 FLASH 的可用性状态思想。

    def validate(self) -> None:  # 定义参数校验方法，避免无效 profile 进入模拟。
        """Validate profile values before simulation."""  # 方法说明：在模拟前检查参数合法性。
        if not self.client_id:  # 如果客户端编号为空。
            raise ValueError("client_id must not be empty")  # 抛出错误，要求必须提供客户端编号。
        if self.compute_capacity <= 0:  # 如果计算能力小于等于 0。
            raise ValueError("compute_capacity must be positive")  # 抛出错误，计算能力必须为正数。
        if self.compute_jitter_cv < 0:  # 如果计算波动系数为负数。
            raise ValueError("compute_jitter_cv must be non-negative")  # 抛出错误，波动系数不能为负。
        if self.upload_bandwidth_mbps <= 0:  # 如果上传带宽小于等于 0。
            raise ValueError("upload_bandwidth_mbps must be positive")  # 抛出错误，上传带宽必须为正数。
        if self.download_bandwidth_mbps <= 0:  # 如果下载带宽小于等于 0。
            raise ValueError("download_bandwidth_mbps must be positive")  # 抛出错误，下载带宽必须为正数。
        if self.network_jitter_cv < 0:  # 如果网络波动系数为负数。
            raise ValueError("network_jitter_cv must be non-negative")  # 抛出错误，网络波动系数不能为负。
        if self.base_latency < 0:  # 如果基础延迟为负数。
            raise ValueError("base_latency must be non-negative")  # 抛出错误，基础延迟不能为负。
        if not 0 <= self.spike_probability <= 1:  # 如果偶发卡顿概率不在 [0, 1]。
            raise ValueError("spike_probability must be in [0, 1]")  # 抛出错误，概率必须合法。
        if self.spike_delay_mean < 0:  # 如果偶发卡顿平均时长为负数。
            raise ValueError("spike_delay_mean must be non-negative")  # 抛出错误，卡顿时长不能为负。
        if not 0 <= self.availability_probability <= 1:  # 如果可用概率不在 [0, 1]。
            raise ValueError("availability_probability must be in [0, 1]")  # 抛出错误，可用概率必须合法。

    def to_dict(self) -> Dict[str, float]:  # 定义转字典方法，便于日志记录和后续接入配置。
        """Return a serializable profile dictionary."""  # 方法说明：返回可序列化的画像字典。
        return {  # 返回包含全部 profile 参数的字典。
            "client_id": self.client_id,  # 写入客户端编号。
            "compute_capacity": self.compute_capacity,  # 写入计算能力。
            "compute_jitter_cv": self.compute_jitter_cv,  # 写入计算波动系数。
            "upload_bandwidth_mbps": self.upload_bandwidth_mbps,  # 写入上传带宽。
            "download_bandwidth_mbps": self.download_bandwidth_mbps,  # 写入下载带宽。
            "network_jitter_cv": self.network_jitter_cv,  # 写入网络波动系数。
            "base_latency": self.base_latency,  # 写入基础延迟。
            "spike_probability": self.spike_probability,  # 写入偶发卡顿概率。
            "spike_delay_mean": self.spike_delay_mean,  # 写入偶发卡顿平均时长。
            "availability_probability": self.availability_probability,  # 写入可用概率。
        }  # 结束字典构造。


def build_default_profiles() -> Dict[str, ClientRuntimeProfile]:  # 定义默认多客户端端侧场景。
    """Build a small set of realistic client scenarios for validation."""  # 函数说明：构造少量高质量端侧验证场景。
    profiles = {  # 创建默认客户端画像字典。
        "stable_fast": ClientRuntimeProfile(  # 定义稳定快客户端，作为理想基线端侧设备。
            client_id="stable_fast",  # 设置客户端编号。
            compute_capacity=1.4,  # 设置较高计算能力。
            compute_jitter_cv=0.04,  # 设置较低计算波动。
            upload_bandwidth_mbps=80.0,  # 设置较高上传带宽。
            download_bandwidth_mbps=120.0,  # 设置较高下载带宽。
            network_jitter_cv=0.05,  # 设置较低网络波动。
            base_latency=0.03,  # 设置较低基础延迟。
            spike_probability=0.0,  # 不设置偶发卡顿。
            spike_delay_mean=0.0,  # 不设置卡顿时长。
            availability_probability=1.0,  # 设置为总是可用。
        ),  # 结束稳定快客户端画像。
        "stable_slow": ClientRuntimeProfile(  # 定义稳定慢客户端，模拟计算能力弱但可靠的机构客户端。
            client_id="stable_slow",  # 设置客户端编号。
            compute_capacity=0.35,  # 设置较低计算能力。
            compute_jitter_cv=0.05,  # 设置较低计算波动，表示慢但稳定。
            upload_bandwidth_mbps=40.0,  # 设置中等上传带宽。
            download_bandwidth_mbps=80.0,  # 设置中等下载带宽。
            network_jitter_cv=0.08,  # 设置较低网络波动。
            base_latency=0.05,  # 设置普通基础延迟。
            spike_probability=0.0,  # 不设置偶发卡顿。
            spike_delay_mean=0.0,  # 不设置卡顿时长。
            availability_probability=1.0,  # 设置为总是可用。
        ),  # 结束稳定慢客户端画像。
        "compute_volatile": ClientRuntimeProfile(  # 定义计算波动客户端，模拟共享 GPU/CPU 或后台负载。
            client_id="compute_volatile",  # 设置客户端编号。
            compute_capacity=1.0,  # 设置平均计算能力接近普通客户端。
            compute_jitter_cv=0.45,  # 设置高计算波动。
            upload_bandwidth_mbps=60.0,  # 设置较好上传带宽。
            download_bandwidth_mbps=100.0,  # 设置较好下载带宽。
            network_jitter_cv=0.10,  # 设置普通网络波动。
            base_latency=0.05,  # 设置普通基础延迟。
            spike_probability=0.15,  # 设置一定概率偶发卡顿。
            spike_delay_mean=1.0,  # 设置偶发卡顿平均 1 秒。
            availability_probability=1.0,  # 设置为总是可用。
        ),  # 结束计算波动客户端画像。
        "network_poor": ClientRuntimeProfile(  # 定义网络差客户端，模拟跨机构 VPN 或弱上行网络。
            client_id="network_poor",  # 设置客户端编号。
            compute_capacity=1.0,  # 设置普通计算能力。
            compute_jitter_cv=0.08,  # 设置较低计算波动。
            upload_bandwidth_mbps=4.0,  # 设置低上传带宽。
            download_bandwidth_mbps=12.0,  # 设置低下载带宽。
            network_jitter_cv=0.40,  # 设置高网络波动。
            base_latency=0.25,  # 设置较高基础延迟。
            spike_probability=0.10,  # 设置一定概率网络/系统卡顿。
            spike_delay_mean=1.5,  # 设置偶发卡顿平均 1.5 秒。
            availability_probability=0.95,  # 设置高但非满可用概率。
        ),  # 结束网络差客户端画像。
        "availability_limited": ClientRuntimeProfile(  # 定义可用性受限客户端，模拟 FLASH 风格设备状态变化。
            client_id="availability_limited",  # 设置客户端编号。
            compute_capacity=0.8,  # 设置略低计算能力。
            compute_jitter_cv=0.15,  # 设置中等计算波动。
            upload_bandwidth_mbps=20.0,  # 设置普通上传带宽。
            download_bandwidth_mbps=50.0,  # 设置普通下载带宽。
            network_jitter_cv=0.20,  # 设置中等网络波动。
            base_latency=0.10,  # 设置中等基础延迟。
            spike_probability=0.20,  # 设置较高偶发卡顿概率。
            spike_delay_mean=2.0,  # 设置偶发卡顿平均 2 秒。
            availability_probability=0.75,  # 设置较低可用概率。
        ),  # 结束可用性受限客户端画像。
    }  # 结束默认画像字典。
    for profile in profiles.values():  # 遍历每个默认客户端画像。
        profile.validate()  # 校验画像参数合法性。
    return profiles  # 返回默认画像字典。

