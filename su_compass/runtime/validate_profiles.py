"""Generate validation outputs for the 8-client SU-Compass runtime profiles."""  # 模块说明：为 8 客户端端侧画像生成验证数据、图表和汇总结果。

import argparse  # 导入 argparse，用于提供命令行参数。
import csv  # 导入 csv，用于写出逐轮数据和状态轨迹。
import json  # 导入 json，用于写出客户端摘要和总览结果。
from pathlib import Path  # 导入 Path，用于组织输出目录结构。
from typing import Dict, Iterable, List, Tuple  # 导入类型标注，提升可读性。

import matplotlib  # 导入 matplotlib，用于生成验证图表。

matplotlib.use("Agg")  # 使用无界面后端，便于在服务器/终端环境保存图片。
import matplotlib.pyplot as plt  # noqa: E402  # 导入 pyplot，用于绘制单客户端和总览图。
from matplotlib.lines import Line2D  # noqa: E402  # 导入 Line2D，用于构造外侧图例。

from .profile import ClientRuntimeProfile, build_default_profiles  # 导入端侧画像和五类默认场景。
from .state import ClientRuntimeState, RuntimeStateTracker  # 导入运行状态追踪器。
from .virtual_runtime import VirtualRuntimeModel, VirtualRoundResult  # 导入虚拟运行时间模型。


DEFAULT_CLIENT_LAYOUT: Tuple[Tuple[str, str], ...] = (  # 定义 8 客户端到五类画像的默认映射。
    ("client_0", "stable_fast"),  # 稳定快客户端实例 1。
    ("client_1", "stable_fast"),  # 稳定快客户端实例 2。
    ("client_2", "stable_slow"),  # 稳定慢客户端实例。
    ("client_3", "compute_volatile"),  # 计算波动客户端实例 1。
    ("client_4", "compute_volatile"),  # 计算波动客户端实例 2。
    ("client_5", "network_poor"),  # 网络差客户端实例 1。
    ("client_6", "network_poor"),  # 网络差客户端实例 2。
    ("client_7", "availability_limited"),  # 可用性受限客户端实例。
)


ROUND_FIELDS = [  # 定义逐轮上报 CSV 的字段顺序。
    "round_idx",  # 联邦轮次编号。
    "client_id",  # 客户端编号。
    "profile_type",  # 画像类型名称。
    "dispatch_time",  # 本轮派发时间。
    "finish_time",  # 本轮完成时间。
    "local_steps",  # 本轮本地训练步数 Q。
    "train_time",  # 本轮纯训练耗时。
    "download_time",  # 本轮下载耗时。
    "upload_time",  # 本轮上传耗时。
    "spike_delay",  # 本轮偶发卡顿耗时。
    "availability_wait",  # 本轮不可用等待耗时。
    "round_time",  # 本轮总耗时。
    "step_time",  # 本轮平均每步耗时（含通信摊销）。
    "compute_step_time",  # 本轮纯训练平均每步耗时。
    "communication_time",  # 本轮通信总耗时。
    "communication_ratio",  # 本轮通信耗时占比。
    "availability_wait_ratio",  # 本轮不可用等待占比。
    "available",  # 本轮设备是否可用。
    "late",  # 本轮是否迟到。
    "early_margin",  # 本轮相对目标到达时间的提前量。
]


PROFILE_STYLES = {  # 定义汇总图中稳定的 profile 视觉编码，避免在图内反复标注文字。
    "stable_fast": {
        "label": "Stable Fast",
        "color": "#2E8B57",
        "marker": "o",
        "description": "高速、低波动的理想基线客户端",
    },
    "stable_slow": {
        "label": "Stable Slow",
        "color": "#4C78A8",
        "marker": "s",
        "description": "计算受限但行为可预测的稳定慢客户端",
    },
    "compute_volatile": {
        "label": "Compute Volatile",
        "color": "#F58518",
        "marker": "^",
        "description": "本地计算受后台负载或资源争用影响的波动客户端",
    },
    "network_poor": {
        "label": "Network Poor",
        "color": "#E45756",
        "marker": "D",
        "description": "由上传下载开销主导的弱网客户端",
    },
    "availability_limited": {
        "label": "Availability Limited",
        "color": "#B279A2",
        "marker": "P",
        "description": "存在设备不可用等待和尾部延迟风险的客户端",
    },
}


def profile_style(profile_type: str) -> Dict[str, str]:  # 获取画像类型的视觉样式。
    """Return the stable plotting style for one profile type."""
    return PROFILE_STYLES.get(
        profile_type,
        {
            "label": profile_type,
            "color": "#6B7280",
            "marker": "o",
            "description": "自定义客户端画像",
        },
    )


def clone_profile(base_profile: ClientRuntimeProfile, client_id: str) -> ClientRuntimeProfile:  # 复制画像并替换客户端编号。
    """Return a profile copy with a new client id."""  # 函数说明：为 8 客户端布局复制同类型画像。
    return ClientRuntimeProfile(  # 构造新的客户端画像。
        client_id=client_id,  # 写入新的客户端编号。
        compute_capacity=base_profile.compute_capacity,  # 复制计算能力。
        compute_jitter_cv=base_profile.compute_jitter_cv,  # 复制计算波动系数。
        upload_bandwidth_mbps=base_profile.upload_bandwidth_mbps,  # 复制上传带宽。
        download_bandwidth_mbps=base_profile.download_bandwidth_mbps,  # 复制下载带宽。
        network_jitter_cv=base_profile.network_jitter_cv,  # 复制网络波动系数。
        base_latency=base_profile.base_latency,  # 复制基础网络延迟。
        spike_probability=base_profile.spike_probability,  # 复制偶发卡顿概率。
        spike_delay_mean=base_profile.spike_delay_mean,  # 复制偶发卡顿平均时长。
        availability_probability=base_profile.availability_probability,  # 复制设备可用概率。
    )  # 结束画像复制。


def build_8client_profiles() -> Dict[str, Tuple[str, ClientRuntimeProfile]]:  # 构造 8 客户端验证画像字典。
    """Build the default 8-client validation layout."""  # 函数说明：将五类标准画像映射到 8 个客户端实例。
    base_profiles = build_default_profiles()  # 读取五类默认画像。
    profiles: Dict[str, Tuple[str, ClientRuntimeProfile]] = {}  # 创建 8 客户端结果字典。
    for client_id, profile_type in DEFAULT_CLIENT_LAYOUT:  # 遍历默认客户端布局。
        profiles[client_id] = (profile_type, clone_profile(base_profiles[profile_type], client_id))  # 保存画像类型和实例。
    return profiles  # 返回 8 客户端画像映射。


def state_to_row(  # 将滑动窗口状态快照转换为 CSV 行。
    round_idx: int,  # 当前联邦轮次编号。
    client_id: str,  # 客户端编号。
    profile_type: str,  # 画像类型名称。
    state: ClientRuntimeState,  # 当前轮更新后的运行状态。
) -> Dict[str, float]:  # 返回可写入 CSV 的状态行。
    """Convert a runtime state snapshot to a CSV row."""  # 函数说明：导出每轮状态轨迹。
    row = {  # 构造基础字段。
        "round_idx": round_idx,  # 写入轮次编号。
        "client_id": client_id,  # 写入客户端编号。
        "profile_type": profile_type,  # 写入画像类型。
    }  # 结束基础字段。
    row.update(state.to_dict())  # 合并聚合后的 runtime 状态字段。
    return row  # 返回状态 CSV 行。


def result_to_row(  # 将单轮虚拟运行结果转换为 CSV 行。
    round_idx: int,  # 当前联邦轮次编号。
    profile_type: str,  # 画像类型名称。
    result: VirtualRoundResult,  # 单轮虚拟运行结果。
) -> Dict[str, float]:  # 返回可写入 CSV 的逐轮上报行。
    """Convert one virtual round result to a CSV row."""  # 函数说明：导出每轮原始/派生上报数据。
    report = result.report  # 读取客户端每轮反馈记录。
    return {  # 构造逐轮上报字典。
        "round_idx": round_idx,  # 写入轮次编号。
        "client_id": report.client_id,  # 写入客户端编号。
        "profile_type": profile_type,  # 写入画像类型。
        "dispatch_time": report.dispatch_time,  # 写入派发时间。
        "finish_time": report.finish_time,  # 写入完成时间。
        "local_steps": report.local_steps,  # 写入本地训练步数。
        "train_time": result.train_time,  # 写入训练拆分耗时。
        "download_time": result.download_time,  # 写入下载拆分耗时。
        "upload_time": result.upload_time,  # 写入上传拆分耗时。
        "spike_delay": result.spike_delay,  # 写入偶发卡顿耗时。
        "availability_wait": result.availability_wait,  # 写入不可用等待耗时。
        "round_time": report.round_time,  # 写入本轮总耗时。
        "step_time": report.step_time,  # 写入平均每步耗时。
        "compute_step_time": report.compute_step_time,  # 写入纯训练单步耗时。
        "communication_time": report.communication_time,  # 写入通信总耗时。
        "communication_ratio": report.communication_ratio,  # 写入通信占比。
        "availability_wait_ratio": report.availability_wait_ratio,  # 写入不可用等待占比。
        "available": int(report.available),  # 写入可用标记，转为 0/1 便于统计。
        "late": int(report.late),  # 写入迟到标记，转为 0/1 便于统计。
        "early_margin": report.early_margin,  # 写入提前到达余量。
    }  # 结束逐轮上报字典构造。


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Iterable[str]) -> None:  # 将多行数据写入 CSV。
    """Write rows to a CSV file."""  # 函数说明：保存逐轮或汇总表格数据。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    with path.open("w", newline="", encoding="utf-8") as f:  # 以 UTF-8 打开 CSV 文件。
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))  # 创建字典写入器。
        writer.writeheader()  # 写入表头。
        writer.writerows(rows)  # 写入全部数据行。


def write_json(path: Path, payload: Dict[str, object]) -> None:  # 将 JSON 数据写入磁盘。
    """Write JSON payload to disk."""  # 函数说明：保存客户端摘要或总览结果。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    with path.open("w", encoding="utf-8") as f:  # 以 UTF-8 打开 JSON 文件。
        json.dump(payload, f, indent=2, sort_keys=True)  # 写出格式化 JSON。


def simulate_client(  # 模拟单个客户端的多轮虚拟运行。
    profile_type: str,  # 画像类型名称。
    profile: ClientRuntimeProfile,  # 客户端端侧画像。
    rounds: int,  # 模拟联邦轮数。
    local_steps: int,  # 每轮本地训练步数 Q。
    min_local_steps: int,  # 验证用 Q 下界。
    max_local_steps: int,  # 验证用 Q 上界。
    target_gap: float,  # 目标到达时间间隔。
    latest_factor: float,  # 最晚到达时间放大系数。
    seed: int,  # 随机种子。
    window_size: int,  # 滑动窗口大小。
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, object]]:  # 返回逐轮数据、状态轨迹和摘要。
    """Simulate one client and return round rows, state rows, and summary."""  # 函数说明：模拟单客户端并记录每轮变化。
    runtime_model = VirtualRuntimeModel(seed=seed)  # 创建虚拟运行时间模型。
    tracker = RuntimeStateTracker(window_size=min(window_size, rounds))  # 创建运行状态追踪器。
    dispatch_time = 0.0  # 初始化虚拟派发时间。
    round_rows: List[Dict[str, float]] = []  # 保存逐轮上报序列。
    state_rows: List[Dict[str, float]] = []  # 保存逐轮状态轨迹序列。

    for round_idx in range(rounds):  # 遍历每一轮联邦参与。
        target_arrival_time = dispatch_time + target_gap  # 计算目标聚合组期望到达时间。
        latest_arrival_time = dispatch_time + target_gap * latest_factor  # 计算目标聚合组最晚到达时间。
        result = runtime_model.simulate_round(  # 执行单轮虚拟端侧运行。
            profile=profile,  # 传入客户端画像。
            dispatch_time=dispatch_time,  # 传入本轮派发时间。
            local_steps=local_steps,  # 传入本轮本地训练步数。
            target_arrival_time=target_arrival_time,  # 传入目标到达时间。
            latest_arrival_time=latest_arrival_time,  # 传入最晚到达时间。
            min_local_steps=min_local_steps,  # 设置验证用 Q 下界。
            max_local_steps=max_local_steps,  # 设置验证用 Q 上界。
            staleness=round_idx % 3,  # 设置简单循环陈旧度，便于验证状态字段更新。
        )  # 结束单轮虚拟运行。
        state = tracker.update(result.report)  # 用本轮反馈更新滑动窗口状态。
        round_rows.append(result_to_row(round_idx, profile_type, result))  # 记录逐轮上报数据。
        state_rows.append(state_to_row(round_idx, profile.client_id, profile_type, state))  # 记录逐轮状态轨迹。
        dispatch_time = result.report.finish_time  # 将下一轮派发时间设置为本轮完成时间。

    snapshot = tracker.snapshot(profile.client_id)  # 读取该客户端最终状态快照。
    if snapshot is None:  # 如果没有生成最终状态。
        raise RuntimeError(f"missing snapshot for {profile.client_id}")  # 抛出错误，说明模拟失败。

    summary = {  # 构造单客户端摘要信息。
        "client_id": profile.client_id,  # 写入客户端编号。
        "profile_type": profile_type,  # 写入画像类型。
        "profile": profile.to_dict(),  # 写入画像参数。
        "final_state": snapshot.to_dict(),  # 写入最终 runtime 状态。
        "rounds": rounds,  # 写入模拟轮数。
        "local_steps": local_steps,  # 写入每轮本地训练步数。
        "window_size": min(window_size, rounds),  # 写入实际滑动窗口大小。
        "seed": seed,  # 写入随机种子。
    }  # 结束摘要构造。
    return round_rows, state_rows, summary  # 返回逐轮数据、状态轨迹和摘要。


def plot_client_components(  # 绘制单客户端逐轮耗时组成堆叠图。
    client_dir: Path,  # 当前客户端输出目录。
    client_id: str,  # 客户端编号。
    round_rows: List[Dict[str, float]],  # 逐轮上报数据。
) -> None:
    """Plot per-round time components for one client."""  # 函数说明：展示每轮慢因拆分，最直观看单客户端特征。
    x = [row["round_idx"] for row in round_rows]  # 提取轮次横轴。
    components = {  # 定义五类耗时组成。
        "download": [row["download_time"] for row in round_rows],  # 下载耗时序列。
        "train": [row["train_time"] for row in round_rows],  # 训练耗时序列。
        "upload": [row["upload_time"] for row in round_rows],  # 上传耗时序列。
        "spike": [row["spike_delay"] for row in round_rows],  # 偶发卡顿序列。
        "availability_wait": [row["availability_wait"] for row in round_rows],  # 不可用等待序列。
    }  # 结束耗时组成定义。

    fig, ax = plt.subplots(figsize=(12, 5))  # 创建单图坐标轴。
    ax.stackplot(x, components.values(), labels=components.keys(), alpha=0.85)  # 绘制堆叠面积图。
    ax.set_title(f"{client_id} Round Time Components")  # 设置图标题。
    ax.set_xlabel("Round")  # 设置横轴标签。
    ax.set_ylabel("Time (s)")  # 设置纵轴标签。
    ax.legend(loc="upper left")  # 显示图例。
    ax.grid(True, alpha=0.25)  # 添加浅色网格。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(client_dir / "round_time_components.png", dpi=180)  # 保存单客户端慢因堆叠图。
    plt.close(fig)  # 关闭图像释放内存。


def plot_client_state_trace(  # 绘制单客户端状态随轮次变化图。
    client_dir: Path,  # 当前客户端输出目录。
    client_id: str,  # 客户端编号。
    round_rows: List[Dict[str, float]],  # 逐轮上报数据。
    state_rows: List[Dict[str, float]],  # 逐轮状态轨迹数据。
) -> None:
    """Plot per-client runtime state traces."""  # 函数说明：展示单客户端稳定性、速度和慢因事件变化。
    x = [row["round_idx"] for row in round_rows]  # 提取逐轮横轴。
    state_x = [row["round_idx"] for row in state_rows]  # 提取状态轨迹横轴。

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))  # 创建 2x2 子图。

    axes[0][0].plot(x, [row["round_time"] for row in round_rows], label="round_time")  # 绘制单轮总耗时。
    axes[0][0].plot(state_x, [row["round_time_mean"] for row in state_rows], label="round_time_mean")  # 绘制滑动平均总耗时。
    axes[0][0].set_title("Total Round Time")  # 设置子图标题。
    axes[0][0].set_ylabel("Time (s)")  # 设置纵轴。
    axes[0][0].legend()  # 显示图例。

    axes[0][1].plot(x, [row["step_time"] for row in round_rows], label="step_time")  # 绘制含通信摊销的单步耗时。
    axes[0][1].plot(x, [row["compute_step_time"] for row in round_rows], label="compute_step_time")  # 绘制纯训练单步耗时。
    axes[0][1].set_title("Step Time")  # 设置子图标题。
    axes[0][1].set_ylabel("Time (s/step)")  # 设置纵轴。
    axes[0][1].legend()  # 显示图例。

    axes[1][0].plot(x, [row["communication_ratio"] for row in round_rows], label="communication_ratio")  # 绘制单轮通信占比。
    axes[1][0].plot(state_x, [row["communication_ratio_mean"] for row in state_rows], label="rolling_mean")  # 绘制通信占比滑动均值。
    axes[1][0].set_title("Communication Share")  # 设置子图标题。
    axes[1][0].set_ylabel("Ratio")  # 设置纵轴。
    axes[1][0].legend()  # 显示图例。

    axes[1][1].plot(x, [row["spike_delay"] for row in round_rows], label="spike_delay")  # 绘制偶发卡顿事件。
    axes[1][1].plot(x, [row["availability_wait"] for row in round_rows], label="availability_wait")  # 绘制不可用等待事件。
    axes[1][1].set_title("Spike and Availability Events")  # 设置子图标题。
    axes[1][1].set_ylabel("Time (s)")  # 设置纵轴。
    axes[1][1].legend()  # 显示图例。

    for ax_row in axes:  # 遍历全部子图。
        for ax in ax_row:  # 遍历当前行子图。
            ax.set_xlabel("Round")  # 设置横轴标签。
            ax.grid(True, alpha=0.25)  # 添加浅色网格。

    fig.suptitle(f"{client_id} Runtime State Trace")  # 设置总标题。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(client_dir / "state_trace.png", dpi=180)  # 保存单客户端状态轨迹图。
    plt.close(fig)  # 关闭图像释放内存。


def plot_summary_bars(summary_dir: Path, summary_rows: List[Dict[str, float]]) -> None:  # 绘制 8 客户端总览柱状图。
    """Plot major comparison metrics across all clients."""  # 函数说明：对比稳定性、算力、通信和总耗时。
    clients = [row["client_id"] for row in summary_rows]  # 提取客户端标签。
    colors = [profile_style(str(row["profile_type"]))["color"] for row in summary_rows]  # 按画像类型设置颜色。
    metrics = [  # 定义需要对比的关键指标。
        ("compute_step_time_mean", "Compute Step Time (s/step)"),  # 纯计算单步速度。
        ("communication_time_mean", "Communication Time (s)"),  # 平均通信耗时。
        ("communication_ratio_mean", "Communication Share"),  # 平均通信占比。
        ("round_time_mean", "Round Time (s)"),  # 平均每轮总耗时。
        ("step_time_cv", "Step Time CV"),  # 单步耗时变异系数，反映稳定性。
        ("round_time_std", "Round Time Std (s)"),  # 每轮总耗时标准差，反映波动。
    ]  # 结束指标列表。

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # 创建 2x3 子图。
    for ax, (metric, title) in zip(axes.ravel(), metrics):  # 遍历每个指标子图。
        values = [row[metric] for row in summary_rows]  # 提取当前指标值。
        ax.bar(clients, values, color=colors, edgecolor="#263238", linewidth=0.6)  # 绘制按画像着色的柱状图。
        ax.set_title(title)  # 设置子图标题。
        ax.tick_params(axis="x", rotation=35)  # 旋转横轴标签避免重叠。
        ax.grid(True, axis="y", alpha=0.25)  # 添加纵向网格。

    add_profile_legend(fig, summary_rows)  # 在图外侧添加统一图例，减少图内重复标注。
    fig.suptitle("8-Client Runtime Profile Comparison")  # 设置总标题。
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.95])  # 为右侧图例预留空间。
    fig.savefig(summary_dir / "runtime_overview_bar.png", dpi=180, bbox_inches="tight")  # 保存 8 客户端总览柱状图。
    plt.close(fig)  # 关闭图像释放内存。


def plot_speed_stability(summary_dir: Path, summary_rows: List[Dict[str, float]]) -> None:  # 绘制速度-稳定性散点图。
    """Plot speed-stability scatter."""  # 函数说明：展示客户端在快慢与稳定两个维度上的分布。
    fig, ax = plt.subplots(figsize=(10, 6))  # 创建散点图坐标轴。
    for row in summary_rows:  # 遍历每个客户端最终状态。
        style = profile_style(str(row["profile_type"]))  # 获取画像类型视觉样式。
        ax.scatter(  # 绘制速度-稳定性散点，客户端标签放在侧边图例而不是点旁。
            row["step_time_mean"],
            row["step_time_cv"],
            s=95,
            color=style["color"],
            marker=style["marker"],
            edgecolor="#263238",
            linewidth=0.7,
            alpha=0.9,
        )
    ax.axvline(_median([row["step_time_mean"] for row in summary_rows]), color="#90A4AE", linestyle="--", linewidth=1.0)
    ax.axhline(_median([row["step_time_cv"] for row in summary_rows]), color="#90A4AE", linestyle="--", linewidth=1.0)
    ax.set_title("Speed-Stability Space")  # 设置图标题。
    ax.set_xlabel("step_time_mean (s/step)")  # 设置横轴。
    ax.set_ylabel("step_time_cv")  # 设置纵轴。
    ax.grid(True, alpha=0.25)  # 添加网格。
    add_profile_legend(fig, summary_rows)  # 在右侧添加 profile 与客户端对应关系。
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])  # 为右侧图例预留空间。
    fig.savefig(summary_dir / "speed_stability_scatter.png", dpi=180, bbox_inches="tight")  # 保存速度-稳定性散点图。
    plt.close(fig)  # 关闭图像释放内存。


def plot_bottleneck(summary_dir: Path, summary_rows: List[Dict[str, float]]) -> None:  # 绘制计算-通信瓶颈散点图。
    """Plot compute-vs-communication bottleneck scatter."""  # 函数说明：区分计算受限与通信受限客户端。
    fig, ax = plt.subplots(figsize=(10, 6))  # 创建散点图坐标轴。
    for row in summary_rows:  # 遍历每个客户端最终状态。
        style = profile_style(str(row["profile_type"]))  # 获取画像类型视觉样式。
        ax.scatter(  # 绘制瓶颈散点，标签统一放在图外侧。
            row["communication_ratio_mean"],
            row["compute_step_time_mean"],
            s=95,
            color=style["color"],
            marker=style["marker"],
            edgecolor="#263238",
            linewidth=0.7,
            alpha=0.9,
        )
    ax.axvline(_median([row["communication_ratio_mean"] for row in summary_rows]), color="#90A4AE", linestyle="--", linewidth=1.0)
    ax.axhline(_median([row["compute_step_time_mean"] for row in summary_rows]), color="#90A4AE", linestyle="--", linewidth=1.0)
    ax.set_title("Compute vs Communication Bottleneck")  # 设置图标题。
    ax.set_xlabel("communication_ratio_mean")  # 设置横轴。
    ax.set_ylabel("compute_step_time_mean (s/step)")  # 设置纵轴。
    ax.grid(True, alpha=0.25)  # 添加网格。
    add_profile_legend(fig, summary_rows)  # 在右侧添加 profile 与客户端对应关系。
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])  # 为右侧图例预留空间。
    fig.savefig(summary_dir / "bottleneck_scatter.png", dpi=180, bbox_inches="tight")  # 保存瓶颈散点图。
    plt.close(fig)  # 关闭图像释放内存。


def plot_runtime_landscape(summary_dir: Path, summary_rows: List[Dict[str, float]]) -> None:  # 绘制综合运行画像图。
    """Plot a publication-style runtime landscape for all clients."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 创建综合画像图。
    fig.patch.set_facecolor("#F8FAFC")  # 使用浅色背景增强报告质感。

    ax_speed = axes[0][0]  # 速度-稳定性子图。
    ax_bottleneck = axes[0][1]  # 计算-通信瓶颈子图。
    ax_reliability = axes[1][0]  # 可靠性子图。
    ax_composition = axes[1][1]  # 慢因组成子图。

    step_median = _median([row["step_time_mean"] for row in summary_rows])  # 计算速度中位线。
    cv_median = _median([row["step_time_cv"] for row in summary_rows])  # 计算稳定性中位线。
    comm_median = _median([row["communication_ratio_mean"] for row in summary_rows])  # 计算通信占比中位线。
    compute_median = _median([row["compute_step_time_mean"] for row in summary_rows])  # 计算纯训练耗时中位线。

    for row in summary_rows:  # 遍历客户端并绘制前三个诊断视图。
        style = profile_style(str(row["profile_type"]))  # 获取画像类型视觉样式。
        point_size = 95 + row["round_time_mean"] * 4.0  # 用点大小弱编码平均轮次耗时。
        scatter_kwargs = {
            "s": point_size,
            "color": style["color"],
            "marker": style["marker"],
            "edgecolor": "#263238",
            "linewidth": 0.7,
            "alpha": 0.9,
        }
        ax_speed.scatter(row["step_time_mean"], row["step_time_cv"], **scatter_kwargs)
        ax_bottleneck.scatter(row["communication_ratio_mean"], row["compute_step_time_mean"], **scatter_kwargs)
        ax_reliability.scatter(1.0 - row["late_rate"], row["availability_rate"], **scatter_kwargs)

    ax_speed.axvline(step_median, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax_speed.axhline(cv_median, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax_speed.set_title("Performance Regime: Speed vs Variability")
    ax_speed.set_xlabel("Mean step time (s/step, lower is faster)")
    ax_speed.set_ylabel("Step-time CV (lower is steadier)")

    ax_bottleneck.axvline(comm_median, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax_bottleneck.axhline(compute_median, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax_bottleneck.set_title("Bottleneck Regime: Communication vs Compute")
    ax_bottleneck.set_xlabel("Communication share of round time")
    ax_bottleneck.set_ylabel("Compute step time (s/step)")

    ax_reliability.axvline(_median([1.0 - row["late_rate"] for row in summary_rows]), color="#94A3B8", linestyle="--", linewidth=1.0)
    ax_reliability.axhline(_median([row["availability_rate"] for row in summary_rows]), color="#94A3B8", linestyle="--", linewidth=1.0)
    ax_reliability.set_title("Scheduling Reliability")
    ax_reliability.set_xlabel("On-time rate (1 - late_rate)")
    ax_reliability.set_ylabel("Availability rate")
    ax_reliability.set_xlim(-0.05, 1.05)
    ax_reliability.set_ylim(-0.05, 1.05)

    sorted_rows = sorted(summary_rows, key=lambda row: row["round_time_mean"])  # 按总耗时排序，便于观察慢因组成。
    clients = [str(row["client_id"]) for row in sorted_rows]  # 提取排序后的客户端。
    train_values = [row["compute_step_time_mean"] * row["local_steps"] for row in sorted_rows]  # 近似纯训练耗时。
    communication_values = [row["communication_time_mean"] for row in sorted_rows]  # 通信耗时。
    spike_values = [row["spike_delay_mean"] for row in sorted_rows]  # 卡顿耗时。
    availability_values = [row["availability_wait_mean"] for row in sorted_rows]  # 不可用等待耗时。
    left = [0.0] * len(sorted_rows)  # 初始化堆叠条形图左边界。
    component_specs = [
        ("Train", train_values, "#4C78A8"),
        ("Comm", communication_values, "#E45756"),
        ("Spike", spike_values, "#F58518"),
        ("Availability", availability_values, "#B279A2"),
    ]
    for label, values, color in component_specs:  # 绘制慢因堆叠条形图。
        ax_composition.barh(clients, values, left=left, label=label, color=color, edgecolor="white", linewidth=0.5)
        left = [prev + value for prev, value in zip(left, values)]
    ax_composition.set_title("Mean Round-Time Composition")
    ax_composition.set_xlabel("Time (s)")
    ax_composition.legend(loc="lower right", frameon=False)

    for ax in axes.ravel():  # 统一网格和背景。
        ax.grid(True, alpha=0.22)
        ax.set_facecolor("#FFFFFF")

    add_profile_legend(fig, summary_rows)  # 添加外侧 profile/client 图例。
    fig.suptitle("SU-Compass Client Runtime Landscape", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 0.95])
    fig.savefig(summary_dir / "runtime_landscape.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_profile_legend(fig: plt.Figure, summary_rows: List[Dict[str, float]]) -> None:  # 在图右侧添加统一图例。
    """Add an outside legend that maps visual encodings to profile types."""
    profile_types = []  # 按出现顺序收集画像类型，保持图例顺序稳定。
    for row in summary_rows:  # 遍历汇总行。
        profile_type = str(row["profile_type"])  # 读取画像类型。
        if profile_type not in profile_types:  # 如果该画像类型尚未加入图例。
            profile_types.append(profile_type)  # 记录该画像类型。
    handles = []  # 构造图例项。
    labels = []  # 构造图例文字。
    for profile_type in profile_types:  # 按出现顺序输出图例。
        style = profile_style(profile_type)  # 获取样式。
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=style["color"],
                markeredgecolor="#263238",
                markersize=9,
                linestyle="",
            )
        )
        labels.append(style["label"])  # 图例只保留画像类型名称，不重复标注客户端编号。
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=False, title="Runtime Profiles")


def _median(values: List[float]) -> float:  # 计算中位数，用于汇总图分区参考线。
    """Return the median of a non-empty list."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def write_client_insight_report(summary_dir: Path, summary_rows: List[Dict[str, float]]) -> None:  # 写出客户端画像解读报告。
    """Write a high-level Chinese interpretation report for the summary outputs."""
    fastest = min(summary_rows, key=lambda row: row["step_time_mean"])  # 找到最快客户端。
    slowest = max(summary_rows, key=lambda row: row["step_time_mean"])  # 找到最慢客户端。
    most_stable = min(summary_rows, key=lambda row: row["step_time_cv"])  # 找到最稳定客户端。
    most_variable = max(summary_rows, key=lambda row: row["step_time_cv"])  # 找到波动最大客户端。
    most_communication_bound = max(summary_rows, key=lambda row: row["communication_ratio_mean"])  # 找到通信瓶颈最强客户端。
    most_compute_bound = max(summary_rows, key=lambda row: row["compute_step_time_mean"])  # 找到计算瓶颈最强客户端。
    least_available = min(summary_rows, key=lambda row: row["availability_rate"])  # 找到可用性最低客户端。

    lines = [
        "# SU-Compass 端侧运行画像汇总解读",
        "",
        "## 总体结论",
        "",
        _overall_insight_sentence(
            fastest=fastest,
            slowest=slowest,
            most_stable=most_stable,
            most_variable=most_variable,
            most_compute_bound=most_compute_bound,
            most_communication_bound=most_communication_bound,
            least_available=least_available,
        ),
        "",
        (
            "从调度视角看，原始 FedCompass 只能观测 `round_time / local_steps` 形成的整体速度；"
            "当前端侧状态进一步拆解出计算、通信、卡顿和可用性等待，因此可以解释“慢在哪里”和“是否值得等待”。"
        ),
        "",
        "## 关键客户端差异",
        "",
    ]

    for row in summary_rows:
        lines.extend(_client_insight_lines(row))

    lines.extend(
        [
            "## 汇总图阅读建议",
            "",
            "- `runtime_landscape.png` 是主图，建议用于论文/汇报：左上看速度与稳定性，右上看计算-通信瓶颈，左下看按时率与可用性，右下看平均轮次耗时组成。",
            "- `runtime_overview_bar.png` 适合做指标总览，颜色编码与画像类型保持一致。",
            "- `speed_stability_scatter.png` 和 `bottleneck_scatter.png` 不再在点旁重复标注客户端名，客户端映射统一放在右侧图例，视觉上更干净。",
            "",
            "## 调度含义",
            "",
            "- 稳定快客户端适合严格 arrival group，可作为系统吞吐基线。",
            "- 稳定慢客户端主要受计算限制，适合通过较小 Q 或更宽松的 deadline 管理。",
            "- 计算波动客户端平均速度不一定差，但尾部风险更高，调度时应关注 `step_time_cv` 和 `round_time_std`。",
            "- 网络差客户端通信占比极高，简单减少 Q 可能反而放大通信摊销，后续策略应区分通信慢和计算慢。",
            "- 可用性受限客户端的主要问题不是单步计算，而是不可用等待与随机卡顿，不适合放入严格同步等待路径。",
            "",
        ]
    )

    (summary_dir / "client_runtime_insights.md").write_text("\n".join(lines), encoding="utf-8")


def _overall_insight_sentence(
    fastest: Dict[str, float],
    slowest: Dict[str, float],
    most_stable: Dict[str, float],
    most_variable: Dict[str, float],
    most_compute_bound: Dict[str, float],
    most_communication_bound: Dict[str, float],
    least_available: Dict[str, float],
) -> str:  # 生成避免重复客户端名的总体结论。
    """Return a compact high-level conclusion sentence."""
    stability_phrase = (
        f"`{fastest['client_id']}` 同时是最快且最稳定的高效基线"
        if fastest["client_id"] == most_stable["client_id"]
        else f"`{fastest['client_id']}` 是最快客户端，`{most_stable['client_id']}` 是最稳定客户端"
    )
    tail_phrase = (
        f"`{slowest['client_id']}` 是最慢客户端，`{most_variable['client_id']}` 的尾部波动最强"
        if slowest["client_id"] != most_variable["client_id"]
        else f"`{slowest['client_id']}` 同时承担最慢与最高波动的尾部风险"
    )
    return (
        "本组 8 客户端画像形成了清晰的系统异构梯度："
        f"{stability_phrase}；"
        f"`{most_compute_bound['client_id']}` 代表稳定计算瓶颈，"
        f"`{most_communication_bound['client_id']}` 代表通信主导瓶颈；"
        f"{tail_phrase}，"
        f"`{least_available['client_id']}` 则体现可用性受限风险。"
    )


def _client_insight_lines(row: Dict[str, float]) -> List[str]:  # 生成单客户端中文解释。
    """Return narrative insight lines for one client."""
    profile_type = str(row["profile_type"])
    style = profile_style(profile_type)
    bottleneck = _bottleneck_label(row)
    reliability = _reliability_label(row)
    volatility = _volatility_label(row)
    return [
        f"### {row['client_id']} / {style['label']}",
        "",
        (
            f"- 系统画像：{style['description']}；平均每轮 {row['round_time_mean']:.2f}s，"
            f"平均单步 {row['step_time_mean']:.3f}s/step，波动系数 {row['step_time_cv']:.3f}。"
        ),
        (
            f"- 慢因结构：{bottleneck}；通信占比 {row['communication_ratio_mean']:.2%}，"
            f"纯计算单步 {row['compute_step_time_mean']:.3f}s/step。"
        ),
        (
            f"- 调度风险：{reliability}；迟到率 {row['late_rate']:.2%}，"
            f"可用率 {row['availability_rate']:.2%}，{volatility}。"
        ),
        "",
    ]


def _bottleneck_label(row: Dict[str, float]) -> str:  # 根据指标给出瓶颈描述。
    """Return a concise bottleneck label."""
    if row["communication_ratio_mean"] >= 0.75:
        return "通信主导瓶颈"
    if row["compute_step_time_mean"] >= 0.10:
        return "计算主导瓶颈"
    if row["availability_wait_mean"] > 0.5:
        return "可用性等待显著"
    if row["spike_delay_mean"] > 0.1:
        return "偶发卡顿显著"
    return "计算与通信较均衡"


def _reliability_label(row: Dict[str, float]) -> str:  # 根据迟到率和可用率给出可靠性描述。
    """Return a concise reliability label."""
    if row["late_rate"] >= 0.8:
        return "严格 deadline 下高风险"
    if row["availability_rate"] < 0.9:
        return "可用性受限"
    if row["late_rate"] <= 0.05:
        return "按时性稳定"
    return "存在中等迟到风险"


def _volatility_label(row: Dict[str, float]) -> str:  # 根据波动指标给出稳定性描述。
    """Return a concise volatility label."""
    if row["step_time_cv"] >= 0.45:
        return "尾部波动非常明显"
    if row["step_time_cv"] >= 0.20:
        return "运行波动明显"
    if row["step_time_cv"] <= 0.06:
        return "运行状态稳定"
    return "运行波动中等"


def flatten_summary(summary: Dict[str, object]) -> Dict[str, float]:  # 将客户端摘要压平为汇总表行。
    """Flatten client summary into a CSV-friendly row."""  # 函数说明：便于写出 8 客户端对比总表。
    row = {  # 构造基础元信息字段。
        "client_id": summary["client_id"],  # 写入客户端编号。
        "profile_type": summary["profile_type"],  # 写入画像类型。
        "rounds": summary["rounds"],  # 写入模拟轮数。
        "local_steps": summary["local_steps"],  # 写入每轮本地步数。
        "window_size": summary["window_size"],  # 写入滑动窗口大小。
        "seed": summary["seed"],  # 写入随机种子。
    }  # 结束基础字段。
    row.update(summary["final_state"])  # type: ignore[arg-type]  # 合并最终 runtime 状态指标。
    return row  # 返回汇总表行。


def run_validation(args: argparse.Namespace) -> None:  # 运行完整 8 客户端验证实验。
    """Run the full 8-client validation experiment."""  # 函数说明：生成逐客户端输出和总览对比结果。
    output_root = Path(args.output_dir)  # 解析输出根目录。
    experiment_dir = output_root / args.experiment_name  # 构造本次实验目录。
    summary_dir = experiment_dir / "summary"  # 构造总览输出目录。
    summary_dir.mkdir(parents=True, exist_ok=True)  # 创建总览目录。

    profiles = build_8client_profiles()  # 构造 8 客户端画像映射。
    all_round_rows: List[Dict[str, float]] = []  # 汇总全部客户端逐轮数据。
    all_state_rows: List[Dict[str, float]] = []  # 汇总全部客户端状态轨迹。
    summaries: List[Dict[str, object]] = []  # 汇总全部客户端摘要。

    for idx, (client_id, (profile_type, profile)) in enumerate(profiles.items()):  # 遍历 8 个客户端。
        round_rows, state_rows, summary = simulate_client(  # 模拟当前客户端。
            profile_type=profile_type,  # 传入画像类型。
            profile=profile,  # 传入客户端画像。
            rounds=args.rounds,  # 传入模拟轮数。
            local_steps=args.local_steps,  # 传入每轮本地步数。
            min_local_steps=args.min_local_steps,  # 传入 Q 下界。
            max_local_steps=args.max_local_steps,  # 传入 Q 上界。
            target_gap=args.target_gap,  # 传入目标到达间隔。
            latest_factor=args.latest_factor,  # 传入最晚到达放大系数。
            seed=args.seed + idx,  # 为每个客户端派生不同随机种子。
            window_size=args.window_size,  # 传入滑动窗口大小。
        )  # 结束单客户端模拟。

        client_dir = experiment_dir / profile_type / client_id  # 构造单客户端输出目录：状态名/客户端编号。
        client_dir.mkdir(parents=True, exist_ok=True)  # 创建单客户端目录。
        write_csv(client_dir / "round_reports.csv", round_rows, ROUND_FIELDS)  # 保存逐轮上报 CSV。
        write_csv(client_dir / "state_trace.csv", state_rows, state_rows[0].keys())  # 保存逐轮状态轨迹 CSV。
        write_json(client_dir / "client_summary.json", summary)  # 保存单客户端摘要 JSON。
        plot_client_components(client_dir, client_id, round_rows)  # 生成单客户端慢因堆叠图。
        plot_client_state_trace(client_dir, client_id, round_rows, state_rows)  # 生成单客户端状态轨迹图。

        all_round_rows.extend(round_rows)  # 汇总全部逐轮数据。
        all_state_rows.extend(state_rows)  # 汇总全部状态轨迹。
        summaries.append(summary)  # 汇总单客户端摘要。

    summary_rows = [flatten_summary(summary) for summary in summaries]  # 构造 8 客户端对比总表。
    write_csv(summary_dir / "summary_table.csv", summary_rows, summary_rows[0].keys())  # 保存总览对比表。
    write_csv(summary_dir / "all_round_reports.csv", all_round_rows, ROUND_FIELDS)  # 保存全部逐轮上报数据。
    write_csv(summary_dir / "all_state_traces.csv", all_state_rows, all_state_rows[0].keys())  # 保存全部状态轨迹。
    write_json(summary_dir / "summary.json", {"clients": summaries})  # 保存总览 JSON。
    plot_summary_bars(summary_dir, summary_rows)  # 生成 8 客户端总览柱状图。
    plot_speed_stability(summary_dir, summary_rows)  # 生成速度-稳定性散点图。
    plot_bottleneck(summary_dir, summary_rows)  # 生成计算-通信瓶颈散点图。
    plot_runtime_landscape(summary_dir, summary_rows)  # 生成更完整的客户端运行画像综合图。
    write_client_insight_report(summary_dir, summary_rows)  # 生成中文客户端画像解读报告。

    print(f"Validation outputs written to: {experiment_dir}")  # 打印输出目录，便于命令行查看。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    """Parse command-line arguments."""  # 函数说明：解析验证实验参数。
    parser = argparse.ArgumentParser(description="Validate 8 SU-Compass runtime profiles")  # 创建命令行解析器。
    parser.add_argument("--rounds", type=int, default=100)  # 设置每个客户端模拟轮数。
    parser.add_argument("--local_steps", type=int, default=20)  # 设置每轮本地训练步数。
    parser.add_argument("--min_local_steps", type=int, default=5)  # 设置验证用 Q 下界。
    parser.add_argument("--max_local_steps", type=int, default=100)  # 设置验证用 Q 上界。
    parser.add_argument("--target_gap", type=float, default=2.0)  # 设置目标到达时间间隔。
    parser.add_argument("--latest_factor", type=float, default=1.2)  # 设置最晚到达时间放大系数。
    parser.add_argument("--window_size", type=int, default=20)  # 设置滑动窗口大小。
    parser.add_argument("--seed", type=int, default=2026)  # 设置随机种子。
    parser.add_argument("--output_dir", type=str, default="su_compass/output")  # 设置输出根目录。
    parser.add_argument("--experiment_name", type=str, default="runtime_profile_8clients")  # 设置实验目录名。
    return parser.parse_args()  # 返回解析后的参数。


def main() -> None:  # 定义命令行入口函数。
    """Run the validation entrypoint."""  # 函数说明：运行 8 客户端画像验证入口。
    args = parse_args()  # 解析命令行参数。
    run_validation(args)  # 执行完整验证流程。


if __name__ == "__main__":  # 如果该文件作为脚本执行。
    main()  # 调用命令行入口函数。
