"""Standalone simulator for validating SU-Compass runtime profiles."""  # 模块说明：独立验证端侧画像与运行状态是否符合预期。

import argparse  # 导入 argparse，用于提供命令行参数。
import json  # 导入 json，用于打印结构化状态结果。
from typing import Dict  # 导入 Dict，用于标注画像字典类型。

from .profile import ClientRuntimeProfile, build_default_profiles  # 导入端侧画像和默认场景。
from .state import RuntimeStateTracker  # 导入运行状态追踪器。
from .virtual_runtime import VirtualRuntimeModel  # 导入虚拟运行时间模型。


def simulate_profile(  # 定义单客户端画像模拟函数。
    profile: ClientRuntimeProfile,  # 输入待验证的客户端端侧画像。
    rounds: int,  # 输入模拟轮数。
    local_steps: int,  # 输入每轮本地训练步数。
    min_local_steps: int,  # 输入验证用最小本地训练步数。
    max_local_steps: int,  # 输入验证用最大本地训练步数。
    target_gap: float,  # 输入目标到达时间相对派发时间的间隔。
    latest_factor: float,  # 输入最晚到达时间相对目标间隔的放大系数。
    seed: int,  # 输入随机种子。
) -> Dict[str, float]:  # 返回该客户端最终 runtime 状态字典。
    """Simulate one profile and return its final runtime state."""  # 函数说明：模拟单客户端画像并返回状态。
    runtime_model = VirtualRuntimeModel(seed=seed)  # 创建虚拟运行时间模型。
    tracker = RuntimeStateTracker(window_size=min(20, rounds))  # 创建运行状态追踪器。
    dispatch_time = 0.0  # 初始化虚拟派发时间。
    for round_idx in range(rounds):  # 遍历每一轮模拟。
        target_arrival_time = dispatch_time + target_gap  # 计算本轮目标聚合组期望到达时间。
        latest_arrival_time = dispatch_time + target_gap * latest_factor  # 计算本轮目标聚合组最晚到达时间。
        result = runtime_model.simulate_round(  # 执行单轮虚拟端侧运行。
            profile=profile,  # 传入客户端端侧画像。
            dispatch_time=dispatch_time,  # 传入本轮派发时间。
            local_steps=local_steps,  # 传入本轮本地训练步数。
            target_arrival_time=target_arrival_time,  # 传入目标到达时间。
            latest_arrival_time=latest_arrival_time,  # 传入最晚到达时间。
            min_local_steps=min_local_steps,  # 设置验证用 Q 下界。
            max_local_steps=max_local_steps,  # 设置验证用 Q 上界。
            staleness=round_idx % 3,  # 设置简单循环陈旧度，用于验证字段更新。
        )  # 结束单轮虚拟运行。
        tracker.update(result.report)  # 用本轮反馈更新 runtime 状态。
        dispatch_time = result.report.finish_time  # 将下一轮派发时间设置为本轮完成时间。
    snapshot = tracker.snapshot(profile.client_id)  # 读取该客户端最新状态。
    if snapshot is None:  # 如果没有生成状态。
        raise RuntimeError(f"missing snapshot for {profile.client_id}")  # 抛出错误，说明模拟失败。
    return snapshot.to_dict()  # 返回可序列化状态字典。


def simulate_profiles(  # 定义多客户端画像模拟函数。
    profiles: Dict[str, ClientRuntimeProfile],  # 输入多客户端画像字典。
    rounds: int,  # 输入每个客户端模拟轮数。
    local_steps: int,  # 输入每轮本地训练步数。
    min_local_steps: int,  # 输入验证用最小本地训练步数。
    max_local_steps: int,  # 输入验证用最大本地训练步数。
    target_gap: float,  # 输入目标到达时间间隔。
    latest_factor: float,  # 输入最晚到达时间放大系数。
    seed: int,  # 输入随机种子。
) -> Dict[str, Dict[str, float]]:  # 返回所有客户端最终状态字典。
    """Simulate all profiles and return final runtime states."""  # 函数说明：模拟多个客户端画像。
    states = {}  # 创建结果状态字典。
    for idx, profile in enumerate(profiles.values()):  # 按顺序遍历每个客户端画像。
        states[profile.client_id] = simulate_profile(  # 模拟当前客户端并保存状态。
            profile=profile,  # 传入当前客户端画像。
            rounds=rounds,  # 传入模拟轮数。
            local_steps=local_steps,  # 传入本地训练步数。
            min_local_steps=min_local_steps,  # 传入验证用 Q 下界。
            max_local_steps=max_local_steps,  # 传入验证用 Q 上界。
            target_gap=target_gap,  # 传入目标到达时间间隔。
            latest_factor=latest_factor,  # 传入最晚到达时间放大系数。
            seed=seed + idx,  # 为每个客户端派生不同随机种子。
        )  # 结束当前客户端模拟。
    return states  # 返回所有客户端状态。


def build_single_profile_from_args(args: argparse.Namespace) -> ClientRuntimeProfile:  # 根据命令行参数构造单客户端画像。
    """Build one adjustable profile from command-line arguments."""  # 函数说明：用命令行参数创建可调单客户端。
    profile = ClientRuntimeProfile(  # 创建客户端端侧画像。
        client_id=args.client_id,  # 设置客户端编号。
        compute_capacity=args.compute_capacity,  # 设置计算能力。
        compute_jitter_cv=args.compute_jitter_cv,  # 设置计算波动系数。
        upload_bandwidth_mbps=args.upload_bandwidth_mbps,  # 设置上传带宽。
        download_bandwidth_mbps=args.download_bandwidth_mbps,  # 设置下载带宽。
        network_jitter_cv=args.network_jitter_cv,  # 设置网络波动系数。
        base_latency=args.base_latency,  # 设置基础网络延迟。
        spike_probability=args.spike_probability,  # 设置偶发卡顿概率。
        spike_delay_mean=args.spike_delay_mean,  # 设置偶发卡顿平均时长。
        availability_probability=args.availability_probability,  # 设置客户端可用概率。
    )  # 结束画像构造。
    profile.validate()  # 校验画像参数。
    return profile  # 返回构造好的单客户端画像。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    """Parse simulator command-line arguments."""  # 函数说明：解析模拟器命令行参数。
    parser = argparse.ArgumentParser(description="Validate SU-Compass runtime profiles")  # 创建命令行解析器。
    parser.add_argument("--mode", choices=["single", "default"], default="default")  # 设置模拟模式：单客户端或默认多客户端。
    parser.add_argument("--rounds", type=int, default=30)  # 设置每个客户端模拟轮数。
    parser.add_argument("--local_steps", type=int, default=20)  # 设置每轮本地训练步数。
    parser.add_argument("--min_local_steps", type=int, default=5)  # 设置验证用 Q 下界。
    parser.add_argument("--max_local_steps", type=int, default=100)  # 设置验证用 Q 上界。
    parser.add_argument("--target_gap", type=float, default=2.0)  # 设置目标到达时间间隔。
    parser.add_argument("--latest_factor", type=float, default=1.2)  # 设置最晚到达时间放大系数。
    parser.add_argument("--seed", type=int, default=2026)  # 设置随机种子。
    parser.add_argument("--client_id", type=str, default="custom_client")  # 设置单客户端编号。
    parser.add_argument("--compute_capacity", type=float, default=1.0)  # 设置单客户端计算能力。
    parser.add_argument("--compute_jitter_cv", type=float, default=0.05)  # 设置单客户端计算波动系数。
    parser.add_argument("--upload_bandwidth_mbps", type=float, default=20.0)  # 设置单客户端上传带宽。
    parser.add_argument("--download_bandwidth_mbps", type=float, default=50.0)  # 设置单客户端下载带宽。
    parser.add_argument("--network_jitter_cv", type=float, default=0.10)  # 设置单客户端网络波动系数。
    parser.add_argument("--base_latency", type=float, default=0.05)  # 设置单客户端基础网络延迟。
    parser.add_argument("--spike_probability", type=float, default=0.0)  # 设置单客户端偶发卡顿概率。
    parser.add_argument("--spike_delay_mean", type=float, default=0.0)  # 设置单客户端偶发卡顿平均时长。
    parser.add_argument("--availability_probability", type=float, default=1.0)  # 设置单客户端可用概率。
    return parser.parse_args()  # 返回解析后的参数。


def main() -> None:  # 定义命令行入口函数。
    """Run the standalone runtime profile simulator."""  # 函数说明：运行独立端侧模型模拟器。
    args = parse_args()  # 解析命令行参数。
    if args.mode == "single":  # 如果使用单客户端可调模式。
        profiles = {args.client_id: build_single_profile_from_args(args)}  # 构造只包含一个客户端的画像字典。
    else:  # 如果使用默认多客户端模式。
        profiles = build_default_profiles()  # 构造默认多客户端端侧场景。
    states = simulate_profiles(  # 执行端侧场景模拟。
        profiles=profiles,  # 传入客户端画像字典。
        rounds=args.rounds,  # 传入模拟轮数。
        local_steps=args.local_steps,  # 传入本地训练步数。
        min_local_steps=args.min_local_steps,  # 传入验证用 Q 下界。
        max_local_steps=args.max_local_steps,  # 传入验证用 Q 上界。
        target_gap=args.target_gap,  # 传入目标到达时间间隔。
        latest_factor=args.latest_factor,  # 传入最晚到达时间放大系数。
        seed=args.seed,  # 传入随机种子。
    )  # 结束端侧场景模拟。
    print(json.dumps(states, indent=2, sort_keys=True))  # 打印 JSON 格式状态结果，方便审阅和后续脚本解析。


if __name__ == "__main__":  # 如果该文件作为脚本执行。
    main()  # 调用命令行入口函数。

