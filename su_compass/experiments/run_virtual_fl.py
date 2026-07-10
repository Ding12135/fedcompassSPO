"""
su_compass.experiments.run_virtual_fl — 虚拟时间联邦学习统一实验入口。

三算法共享同一套：
    - 真实训练（APPFLClientAgent + NaiveTrainer）
    - 虚拟端侧画像（VirtualRuntimeModel + RuntimeStateTracker）
    - 统一事件循环（EventQueue）
    - 统一 trace 输出格式（含 aggregation / dispatch_decision / global_eval）

用法：
    python -m su_compass.experiments.run_virtual_fl \\
        --algorithm fedcompass \\
        --server_config examples/config/server_fedcompass_paper_mnist8.yaml \\
        --client_config examples/config/client_mnist.yaml \\
        --num_clients 8 \\
        --num_global_epochs 50 \\
        --output_dir su_compass/output/virtual_fedcompass_mnist8

    --algorithm 可选：fedavg | fedasync | fedcompass
"""

import argparse
import copy
import sys
import os
import random
import time
from typing import Dict, List, Any, Optional

import numpy as np
import torch

from su_compass.experiments.console import (
    ExperimentProgress,
    gather_hardware_info,
    print_experiment_banner,
    print_run_footer,
)

from omegaconf import OmegaConf

# ──── APPFL 主代码复用 ────
from appfl.agent import APPFLClientAgent, APPFLServerAgent
from appfl.aggregator import (
    FedAvgAggregator,
    FedAsyncAggregator,
    FedCompassAggregator,
)
from appfl.misc import create_instance_from_file

# ──── SU-Compass 端侧模型 ────
from su_compass.runtime.validate_profiles import build_8client_profiles

# ──── SU-Compass 虚拟时间框架 ────
from su_compass.virtual.event import EventQueue, EventType
from su_compass.virtual.client_runtime import VirtualClientRuntime
from su_compass.virtual.trace import TraceWriter
from su_compass.virtual.eval import evaluate_global_model
from su_compass.virtual.algorithms.base import Dispatch
from su_compass.virtual.algorithms.fedavg import VirtualFedAvgController
from su_compass.virtual.algorithms.fedasync import VirtualFedAsyncController
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.algorithms.oort_compass import VirtualOortCompassController
from su_compass.virtual.algorithms.utility import OortConfig


# ═══════════════════════════════════════════════════════════════
#  实验主函数
# ═══════════════════════════════════════════════════════════════

def run_experiment(args: argparse.Namespace) -> None:
    """执行一次虚拟时间联邦学习实验。

    流程：
        1. 加载配置。
        2. 创建模型、聚合器、客户端 agent。
        3. 创建 VirtualClientRuntime（桥接真实训练与虚拟时间）。
        4. 创建算法控制器（FedAvg / FedAsync / FedCompass）。
        5. 主事件循环。
        6. 写出 trace。
    """
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    show_ui = not getattr(args, "no_progress", False)
    variant = getattr(args, "algorithm_variant", "") or args.algorithm
    progress: Optional[ExperimentProgress] = None
    last_accuracy: Optional[float] = None

    # ──────────── 步骤 1-2：加载配置、创建模型与聚合器 ────────────────
    # APPFLServerAgent 内部按相对路径加载 model/loss/metric，
    # 需要先切到 examples/ 目录，完成加载后再切回来。
    original_dir = os.getcwd()
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples")
    examples_dir = os.path.normpath(examples_dir)

    # 把 server_config 和 client_config 转为绝对路径，避免 chdir 后找不到
    abs_server_config = os.path.abspath(args.server_config)
    abs_client_config = os.path.abspath(args.client_config)

    os.chdir(examples_dir)  # 切到 examples/ 确保模型/loss/metric 相对路径正确

    server_config = OmegaConf.load(abs_server_config)
    client_config = OmegaConf.load(abs_client_config)

    train_device = "cpu"
    if hasattr(server_config, "server_configs"):
        train_device = str(getattr(server_config.server_configs, "device", "cuda"))

    run_lines = [
        f"算法      {args.algorithm}" + (f" ({args.oort_mode})" if args.algorithm == "oort_compass" else ""),
        f"变体      {variant}",
        f"客户端    {args.num_clients}",
        f"训练预算  {args.num_global_epochs} client updates",
        f"Q 范围    [{args.min_local_steps}, {args.max_local_steps}]",
        f"虚拟种子  {args.seed}",
        f"输出目录  {args.output_dir}",
    ]
    if args.algorithm == "oort_compass":
        run_lines.append(
            f"Oort λ    comm={args.oort_lambda_comm} late={args.oort_lambda_late} "
            f"var={args.oort_lambda_var} avail={args.oort_lambda_avail}"
        )

    hardware = gather_hardware_info(train_device=train_device)
    print_experiment_banner(
        title="SU-Compass 虚拟联邦学习实验",
        run_lines=run_lines,
        hardware=hardware,
        enabled=show_ui,
    )

    progress = ExperimentProgress(args.num_global_epochs, enabled=show_ui)
    progress.start("初始化")
    progress.log("加载配置与创建模型...")

    # 重新覆盖 num_clients（因为重新加载了配置）
    if hasattr(server_config, "server_configs"):
        if hasattr(server_config.server_configs, "scheduler_kwargs"):
            server_config.server_configs.scheduler_kwargs.num_clients = args.num_clients
        if hasattr(server_config.server_configs, "aggregator_kwargs"):
            server_config.server_configs.aggregator_kwargs.num_clients = args.num_clients

    server_agent = APPFLServerAgent(server_agent_config=server_config)
    model = copy.deepcopy(server_agent.model)
    initial_global_model = copy.deepcopy(model.state_dict())

    # 根据算法创建对应的聚合器
    aggregator = _create_aggregator(args.algorithm, model, server_config)

    # ──────────── 步骤 3：创建客户端 agent ────────────────
    # 获取 server 下发的客户端配置
    client_config_from_server = server_agent.get_client_configs()

    client_ids: List[str] = [f"client_{i}" for i in range(args.num_clients)]
    client_agents: Dict[str, APPFLClientAgent] = {}

    for i, cid in enumerate(client_ids):
        cfg = OmegaConf.load(abs_client_config)
        cfg.train_configs.logging_id = f"Client{cid}"
        cfg.data_configs.dataset_kwargs.num_clients = args.num_clients
        cfg.data_configs.dataset_kwargs.client_id = i
        cfg.data_configs.dataset_kwargs.visualization = (i == 0)
        agent = APPFLClientAgent(client_agent_config=cfg)
        agent.load_config(client_config_from_server)
        agent.load_parameters(copy.deepcopy(initial_global_model))
        agent.client_id = cid  # 统一 client_id 而非默认 UUID
        client_agents[cid] = agent

    os.chdir(original_dir)  # 切回原始目录

    progress.log(f"{len(client_agents)} 个客户端 agent 创建完成")

    # ──────────── 步骤 4：创建 VirtualClientRuntime ────────────────
    profiles = build_8client_profiles()

    # 调度参数
    sched_kwargs = {}
    if hasattr(server_config, "server_configs") and hasattr(server_config.server_configs, "scheduler_kwargs"):
        sched_kwargs = OmegaConf.to_container(server_config.server_configs.scheduler_kwargs, resolve=True)
    qmin = sched_kwargs.get("min_local_steps", args.min_local_steps)
    qmax = sched_kwargs.get("max_local_steps", args.max_local_steps)
    # CLI 显式参数覆盖 yaml（正式实验鲁棒性扫描用）
    sched_kwargs["min_local_steps"] = args.min_local_steps
    sched_kwargs["max_local_steps"] = args.max_local_steps
    if getattr(args, "latest_time_factor", None) is not None:
        sched_kwargs["latest_time_factor"] = args.latest_time_factor
    qmin = sched_kwargs["min_local_steps"]
    qmax = sched_kwargs["max_local_steps"]

    client_runtime = VirtualClientRuntime(
        profiles=profiles,
        base_step_time=args.base_step_time,
        model_size_mb=args.model_size_mb,
        update_size_mb=args.update_size_mb,
        unavailable_delay_mean=args.unavailable_delay_mean,
        seed=args.seed,
        window_size=args.window_size,
        qmin=qmin,
        qmax=qmax,
    )

    # ──────────── 步骤 5：创建算法控制器 ────────────────
    controller = _create_controller(args, aggregator, sched_kwargs)

    # ──────────── 步骤 6：创建 trace writer 与评估 agent ────────────────
    trace = TraceWriter(
        output_dir=args.output_dir,
        algorithm=args.algorithm,
        algorithm_variant=getattr(args, "algorithm_variant", "") or args.algorithm,
    )
    eval_agent = client_agents[client_ids[0]]  # 用 client_0 的 val_dataloader 做全局评估

    # ──────────── 步骤 7：主事件循环 ────────────────
    controller.initialize(client_ids, initial_global_model)
    event_queue = EventQueue()
    # 每客户端独立 round 计数，写入 round_reports.client_round_idx
    client_round_counters: Dict[str, int] = {cid: 0 for cid in client_ids}

    progress.set_phase("虚拟时间主循环")
    progress.log("开始事件驱动训练与调度...")

    def _flush_controller_traces(virtual_now: float) -> None:
        """写出控制器产生的 trace，并在每次聚合后做全局评估。

        调用时机：每个虚拟事件（CLIENT_UPLOAD / GROUP_DEADLINE）处理完毕后。
        通过 hasattr 检测控制器是否实现 pop_* 接口，三算法共用同一逻辑。
        """
        if hasattr(controller, "pop_aggregation_traces"):
            for agg in controller.pop_aggregation_traces():
                trace.record_aggregation(agg)
                budget_used = (
                    controller.get_num_client_update_budget_used()
                    if hasattr(controller, "get_num_client_update_budget_used")
                    else controller.get_global_timestamp()
                )
                # training_metrics 仅在聚合时写入，num_client_updates 取真实参与人数
                trace.record_training_metrics(
                    global_timestamp=controller.get_global_timestamp(),
                    virtual_time=virtual_now,
                    num_client_updates=agg.get("num_clients", 1),
                    client_update_budget_used=budget_used,
                )
                if hasattr(controller, "get_current_global_model"):
                    acc, loss, n_samples = evaluate_global_model(
                        eval_agent, controller.get_current_global_model(),
                    )
                    trace.record_global_eval(
                        global_timestamp=controller.get_global_timestamp(),
                        virtual_time=virtual_now,
                        test_accuracy=acc,
                        test_loss=loss,
                        num_val_samples=n_samples,
                    )
                    nonlocal last_accuracy
                    last_accuracy = acc
                    if progress is not None:
                        budget_used = (
                            controller.get_num_client_update_budget_used()
                            if hasattr(controller, "get_num_client_update_budget_used")
                            else controller.get_global_timestamp()
                        )
                        progress.update_budget(
                            budget_used,
                            global_version=controller.get_global_timestamp(),
                            accuracy=acc,
                            virtual_time=virtual_now,
                            phase="聚合+评估",
                        )

        if hasattr(controller, "pop_group_traces"):
            for gt in controller.pop_group_traces():
                trace.record_group(gt)

        if hasattr(controller, "pop_dispatch_decision_traces"):
            for dd in controller.pop_dispatch_decision_traces():
                trace.record_dispatch_decision(dd)

        if hasattr(controller, "pop_oort_traces"):
            for ot in controller.pop_oort_traces():
                trace.record_oort_decision(ot)

    while not controller.training_finished():
        # ──── A. 真实训练阶段：处理待派发任务 ────
        dispatches = controller.next_dispatches()
        for dispatch in dispatches:
            cid = dispatch.client_id
            agent = client_agents[cid]
            client_round_idx = client_round_counters[cid]

            agent.load_parameters(dispatch.global_model)

            if progress is not None:
                progress.on_train_start(cid, dispatch.local_steps, client_round_idx)

            event, result, state = client_runtime.train_and_schedule_upload(
                client_id=cid,
                client_agent=agent,
                dispatch_time=dispatch.dispatch_time,
                local_steps=dispatch.local_steps,
                target_arrival_time=dispatch.target_arrival_time,
                latest_arrival_time=dispatch.latest_arrival_time,
                staleness=dispatch.staleness,
            )
            event.payload["client_round_idx"] = client_round_idx  # 供 upload 处理后 enrich round_reports

            event_queue.push(event)

            # 训练完成即写 round_reports / state_trace（upload 字段稍后 enrich）
            profile_type = client_runtime.get_profile_type(cid)
            trace.record_round_report(
                cid, result.report, result, profile_type, client_round_idx,
                model_version_at_dispatch=dispatch.model_version_at_dispatch,
            )
            trace.record_state_trace(cid, state, profile_type, client_round_idx)
            client_round_counters[cid] += 1

        if not event_queue:
            break

        # ──── B. 虚拟时间推进：处理事件 ────
        event = event_queue.pop()
        virtual_now = event.time

        if event.event_type == EventType.CLIENT_UPLOAD:
            new_events = controller.on_client_upload(event, virtual_now)

            report = event.payload["report"]
            runtime_state = event.payload.get("runtime_state")
            profile_type = event.payload.get("profile_type", "")
            client_round_idx = event.payload.get("client_round_idx", 0)

            # 从控制器取出 upload 元数据（真实 aggregation_staleness、平滑 speed 等）
            upload_meta = (
                controller.pop_upload_result(event.client_id)
                if hasattr(controller, "pop_upload_result")
                else None
            )
            speed_raw = upload_meta["speed_raw"] if upload_meta else report.round_time / max(report.local_steps, 1)
            speed_smoothed = upload_meta["speed_smoothed"] if upload_meta else speed_raw
            dispatch_staleness = upload_meta["dispatch_staleness"] if upload_meta else report.staleness
            aggregation_staleness = upload_meta.get("aggregation_staleness") if upload_meta else None
            upload_group_id = upload_meta.get("upload_group_id", -1) if upload_meta else -1
            next_group_id = upload_meta.get("next_group_id", -1) if upload_meta else -1
            model_version_at_upload = upload_meta.get("model_version_at_upload", 0) if upload_meta else 0

            trace.record_scheduler_event(
                virtual_time=virtual_now,
                client_round_idx=client_round_idx,
                client_id=event.client_id,
                profile_type=profile_type,
                local_steps=report.local_steps,
                speed_raw=speed_raw,
                speed_smoothed=speed_smoothed,
                report=report,
                dispatch_staleness=dispatch_staleness,
                aggregation_staleness=aggregation_staleness,
                global_timestamp=controller.get_global_timestamp(),
                upload_group_id=upload_group_id,
                next_group_id=next_group_id,
                model_version_at_upload=model_version_at_upload,
                runtime_state=runtime_state,
            )

            if upload_meta is not None:
                # 回写 round_reports 中训练阶段无法获知的 upload 侧字段
                trace.enrich_round_report_upload(
                    client_id=event.client_id,
                    client_round_idx=client_round_idx,
                    aggregation_staleness=aggregation_staleness,
                    model_version_at_upload=model_version_at_upload,
                    upload_group_id=upload_group_id,
                    next_group_id=next_group_id,
                    speed_smoothed_at_upload=speed_smoothed,
                )

        elif event.event_type == EventType.FEDCOMPASS_GROUP_DEADLINE:
            new_events = controller.on_timer_event(event, virtual_now)
        else:
            new_events = []

        for new_event in new_events:
            event_queue.push(new_event)

        # 聚合 / group / 决策 trace 与 global_eval 在本轮事件处理末尾统一 flush
        _flush_controller_traces(virtual_now)

    # ──────────── 步骤 8：写出 trace ────────────────
    experiment_config = {
        "algorithm": args.algorithm,
        "num_clients": args.num_clients,
        "num_global_epochs": args.num_global_epochs,
        "min_local_steps": qmin,
        "max_local_steps": qmax,
        "base_step_time": args.base_step_time,
        "model_size_mb": args.model_size_mb,
        "update_size_mb": args.update_size_mb,
        "seed": args.seed,
        "server_config": args.server_config,
        "client_config": args.client_config,
        "hardware": hardware.to_dict(),
        "algorithm_variant": variant,
    }
    if args.algorithm == "oort_compass":
        experiment_config.update({
            "oort_mode": args.oort_mode,
            "oort_lambda_comm": args.oort_lambda_comm,
            "oort_lambda_late": args.oort_lambda_late,
            "oort_lambda_var": args.oort_lambda_var,
            "oort_lambda_avail": args.oort_lambda_avail,
            "oort_risk_threshold": args.oort_risk_threshold,
            "oort_slack_min_ratio": args.oort_slack_min_ratio,
        })
    trace.flush(experiment_config=experiment_config)

    if progress is not None:
        budget_used = (
            controller.get_num_client_update_budget_used()
            if hasattr(controller, "get_num_client_update_budget_used")
            else controller.get_global_timestamp()
        )
        progress.update_budget(
            budget_used,
            global_version=controller.get_global_timestamp(),
            accuracy=last_accuracy,
            phase="完成",
        )
        progress.close()

    elapsed = time.time() - t0
    budget_used = (
        controller.get_num_client_update_budget_used()
        if hasattr(controller, "get_num_client_update_budget_used")
        else controller.get_global_timestamp()
    )
    summary = [
        f"全局模型版本数: {controller.get_global_timestamp()}",
        f"客户端更新贡献: {budget_used} / {args.num_global_epochs}",
    ]
    if last_accuracy is not None:
        summary.append(f"最终验证精度: {last_accuracy:.2f}%")
    summary.append(f"输出目录: {args.output_dir}")
    print_run_footer(success=True, elapsed_s=elapsed, summary_lines=summary, enabled=show_ui)


# ═══════════════════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════════════════

def _create_aggregator(algorithm: str, model, server_config) -> Any:
    """根据算法名创建对应的 APPFL 聚合器。

    直接复用 appfl 主代码的聚合器，不做任何修改。
    """
    agg_kwargs = {}
    if hasattr(server_config.server_configs, "aggregator_kwargs"):
        agg_kwargs = OmegaConf.to_container(
            server_config.server_configs.aggregator_kwargs, resolve=True
        )
    agg_config = OmegaConf.create(agg_kwargs)

    # 使用一个简单的 logger 替代
    logger = _SimpleLogger()

    if algorithm == "fedavg":
        return FedAvgAggregator(model=copy.deepcopy(model), aggregator_config=agg_config, logger=logger)
    elif algorithm == "fedasync":
        return FedAsyncAggregator(model=copy.deepcopy(model), aggregator_config=agg_config, logger=logger)
    elif algorithm in ("fedcompass", "oort_compass"):
        # oort_compass 复用 FedCompass 聚合器，仅调度层引入 Oort 效用
        return FedCompassAggregator(model=copy.deepcopy(model), aggregator_config=agg_config, logger=logger)
    else:
        raise ValueError(f"不支持的算法: {algorithm}，可选 fedavg / fedasync / fedcompass / oort_compass")


def _create_controller(args, aggregator, sched_kwargs):
    """根据算法名创建对应的虚拟调度控制器。"""
    if args.algorithm == "fedavg":
        local_steps = sched_kwargs.get("max_local_steps", args.max_local_steps)
        return VirtualFedAvgController(
            aggregator=aggregator,
            num_clients=args.num_clients,
            local_steps=local_steps,
            num_global_epochs=args.num_global_epochs,
        )
    elif args.algorithm == "fedasync":
        local_steps = sched_kwargs.get("max_local_steps", args.max_local_steps)
        return VirtualFedAsyncController(
            aggregator=aggregator,
            num_clients=args.num_clients,
            local_steps=local_steps,
            num_global_epochs=args.num_global_epochs,
        )
    elif args.algorithm == "fedcompass":
        return VirtualFedCompassController(
            aggregator=aggregator,
            num_clients=args.num_clients,
            min_local_steps=sched_kwargs.get("min_local_steps", args.min_local_steps),
            max_local_steps=sched_kwargs.get("max_local_steps", args.max_local_steps),
            speed_momentum=sched_kwargs.get("speed_momentum", 0.9),
            latest_time_factor=sched_kwargs.get("latest_time_factor", 1.2),
            num_global_epochs=args.num_global_epochs,
        )
    elif args.algorithm == "oort_compass":
        # 与 FedCompass 共享全部调度参数，额外注入 Oort 配置（mode + λ 权重）
        oort_config = OortConfig(
            mode=args.oort_mode,
            lambda_comm=args.oort_lambda_comm,
            lambda_late=args.oort_lambda_late,
            lambda_var=args.oort_lambda_var,
            lambda_avail=args.oort_lambda_avail,
            risk_threshold=args.oort_risk_threshold,
            slack_min_ratio=args.oort_slack_min_ratio,
        )
        return VirtualOortCompassController(
            aggregator=aggregator,
            num_clients=args.num_clients,
            min_local_steps=sched_kwargs.get("min_local_steps", args.min_local_steps),
            max_local_steps=sched_kwargs.get("max_local_steps", args.max_local_steps),
            speed_momentum=sched_kwargs.get("speed_momentum", 0.9),
            latest_time_factor=sched_kwargs.get("latest_time_factor", 1.2),
            num_global_epochs=args.num_global_epochs,
            oort_config=oort_config,
        )
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")


class _SimpleLogger:
    """轻量 logger，替代 APPFL ServerAgentFileLogger 用于虚拟实验。"""
    def log_title(self, title): pass
    def log_content(self, content): pass
    def log(self, msg, level="info"):
        print(f"[{level}] {msg}")


# ═══════════════════════════════════════════════════════════════
#  命令行参数
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="SU-Compass 虚拟时间联邦学习实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ──── 核心参数 ────
    parser.add_argument("--algorithm", type=str, default="fedcompass",
                        choices=["fedavg", "fedasync", "fedcompass", "oort_compass"],
                        help="联邦学习算法")
    parser.add_argument("--server_config", type=str,
                        default="examples/config/server_fedcompass_paper_mnist8.yaml",
                        help="服务端配置文件路径")
    parser.add_argument("--client_config", type=str,
                        default="examples/config/client_mnist.yaml",
                        help="客户端配置文件路径")
    parser.add_argument("--num_clients", type=int, default=8,
                        help="客户端数量")
    parser.add_argument("--num_global_epochs", type=int, default=50,
                        help="目标训练预算；FedCompass 对齐原始实现，按客户端更新贡献数计数")

    # ──── 调度参数 ────
    parser.add_argument("--min_local_steps", type=int, default=40,
                        help="最小本地训练步数 Qmin")
    parser.add_argument("--max_local_steps", type=int, default=200,
                        help="最大本地训练步数 Qmax")
    parser.add_argument("--latest_time_factor", type=float, default=None,
                        help="FedCompass 迟到阈值系数 λ；默认读 server yaml")

    # ──── 虚拟端侧参数 ────
    parser.add_argument("--base_step_time", type=float, default=0.05,
                        help="计算能力 1.0 时每步基础耗时 (秒)")
    parser.add_argument("--model_size_mb", type=float, default=5.0,
                        help="下发模型大小 (MB)")
    parser.add_argument("--update_size_mb", type=float, default=5.0,
                        help="上传更新大小 (MB)")
    parser.add_argument("--unavailable_delay_mean", type=float, default=5.0,
                        help="客户端不可用平均等待 (秒)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="虚拟运行时间随机种子")
    parser.add_argument("--window_size", type=int, default=20,
                        help="RuntimeStateTracker 滑动窗口大小")

    # ──── Oort-Compass 参数（仅 --algorithm oort_compass 生效）────
    parser.add_argument("--oort_mode", type=str, default="shadow",
                        choices=["off", "shadow", "q_only", "q_and_group"],
                        help="Oort 作用模式：off 等价 FedCompass；shadow 只算不用；"
                             "q_only 影响 Q；q_and_group 再叠加分组风险过滤")
    parser.add_argument("--oort_lambda_comm", type=float, default=1.0,
                        help="通信占比惩罚权重")
    parser.add_argument("--oort_lambda_late", type=float, default=1.0,
                        help="迟到率惩罚权重")
    parser.add_argument("--oort_lambda_var", type=float, default=0.5,
                        help="单步耗时变异系数惩罚权重")
    parser.add_argument("--oort_lambda_avail", type=float, default=0.5,
                        help="不可用惩罚权重")
    parser.add_argument("--oort_risk_threshold", type=float, default=0.5,
                        help="分组迟到风险门槛（q_and_group 模式）")
    parser.add_argument("--oort_slack_min_ratio", type=float, default=0.25,
                        help="高风险客户端可加入 group 的最小时间余量比例")

    # ──── 输出 ────
    parser.add_argument("--output_dir", type=str,
                        default="su_compass/output/virtual_fedcompass_mnist8",
                        help="实验输出目录")
    parser.add_argument("--algorithm_variant", type=str, default="",
                        help="算法变体标签（写入 trace，供创新实验对比，如 fedcompass_baseline）")
    parser.add_argument("--no_progress", action="store_true",
                        help="关闭启动横幅与进度条（适合日志重定向或非交互环境）")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
