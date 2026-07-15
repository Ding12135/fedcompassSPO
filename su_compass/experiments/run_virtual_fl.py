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
from dataclasses import asdict
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
from su_compass.virtual.algorithms.state_compass import VirtualStateCompassController
from su_compass.virtual.algorithms.rup_compass import VirtualRUPCompassController
from su_compass.virtual.algorithms.utility import OortConfig
from su_compass.virtual.training import RUPTrainingAdapter
from su_compass.scheduling.policies import RUPConfig
from su_compass.diagnostics import FedCompassProblemObserver


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
    if args.algorithm == "rup_compass":
        run_lines.append(
            f"RUP mode   {args.rup_mode}; state={args.rup_state} trust={args.rup_trust} "
            f"soft={args.rup_soft_boundary} utility={args.rup_utility} "
            f"budget={args.rup_budget} prox={args.rup_prox}"
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

    rup_training_adapter = None
    if args.algorithm == "rup_compass":
        rup_training_adapter = RUPTrainingAdapter(
            utility_enabled=(args.rup_mode != "off" and args.rup_utility == "on"),
            # Shadow must leave both scheduling and optimization unchanged.
            prox_enabled=(args.rup_mode == "apply" and args.rup_prox == "on"),
            prox_mu=args.rup_prox_mu,
            utility_eval_batches=args.rup_utility_eval_batches,
        )
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
        training_adapter=rup_training_adapter,
    )

    # ──────────── 步骤 5：创建算法控制器 ────────────────
    controller = _create_controller(args, aggregator, sched_kwargs)

    # ──────────── 步骤 6：创建 trace writer 与评估 agent ────────────────
    trace = TraceWriter(
        output_dir=args.output_dir,
        algorithm=args.algorithm,
        algorithm_variant=getattr(args, "algorithm_variant", "") or args.algorithm,
    )
    # FedCompass 问题观测器只记录真实决策与结果，不参与 controller 的任何计算。
    # Oort 变体暂不启用该报告，避免把受干预后的决策误写成 baseline 问题证据。
    problem_observer = (
        FedCompassProblemObserver() if args.algorithm == "fedcompass" else None
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
                    if hasattr(controller, "update_global_accuracy"):
                        controller.update_global_accuracy(acc)
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
                if problem_observer is not None:
                    problem_observer.record_dispatch(dd)

        if hasattr(controller, "pop_oort_traces"):
            for ot in controller.pop_oort_traces():
                trace.record_oort_decision(ot)

        if hasattr(controller, "pop_rup_traces"):
            for row in controller.pop_rup_traces():
                row["profile_type"] = client_runtime.get_profile_type(row["client_id"])
                trace.record_rup_decision(row)

        if hasattr(controller, "pop_state_q_traces"):
            for state_q in controller.pop_state_q_traces():
                trace.record_state_q_decision(state_q)

        # RCP-GS当前处于Group Shadow验证阶段：控制器只产生候选与推荐事实，
        # TraceWriter独立落盘，不把推荐结果反馈给FedCompass分组状态。
        if hasattr(controller, "pop_group_candidate_traces"):
            for candidate in controller.pop_group_candidate_traces():
                trace.record_group_candidate_shadow(candidate)

        if hasattr(controller, "pop_group_recommendation_traces"):
            for recommendation in controller.pop_group_recommendation_traces():
                trace.record_group_recommendation_shadow(recommendation)

        if hasattr(controller, "pop_group_admission_traces"):
            for admission in controller.pop_group_admission_traces():
                trace.record_group_admission(admission)

        if hasattr(controller, "pop_all_group_feasibility_traces"):
            for candidate in controller.pop_all_group_feasibility_traces():
                trace.record_all_group_feasibility(candidate)

        if hasattr(controller, "pop_all_group_recommendation_traces"):
            for recommendation in controller.pop_all_group_recommendation_traces():
                trace.record_all_group_recommendation(recommendation)

        if hasattr(controller, "pop_group_creation_counterfactual_traces"):
            for result in controller.pop_group_creation_counterfactual_traces():
                trace.record_group_creation_counterfactual(result)

        if hasattr(controller, "pop_state_group_creation_q_traces"):
            for result in controller.pop_state_group_creation_q_traces():
                trace.record_state_group_creation_q(result)

        if hasattr(controller, "pop_state_group_window_traces"):
            for result in controller.pop_state_group_window_traces():
                trace.record_state_group_window(result)

        if hasattr(controller, "pop_state_window_admission_traces"):
            for result in controller.pop_state_window_admission_traces():
                trace.record_state_window_admission(result)

        if hasattr(controller, "pop_communication_tail_risk_traces"):
            for result in controller.pop_communication_tail_risk_traces():
                trace.record_communication_tail_risk(result)
        if hasattr(controller, "pop_communication_robust_q_traces"):
            for result in controller.pop_communication_robust_q_traces():
                trace.record_communication_robust_q(result)

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

            observation = event.payload.get("rup_training_observation")
            if observation is not None:
                observation_row = observation.to_dict()
                observation_row["client_round_idx"] = client_round_idx
                observation_row["finite"] = int(observation.finite)
                trace.record_rup_training(observation_row)

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
            if problem_observer is not None:
                # 虚拟训练结果在事件入队前已经计算完成，但只有CLIENT_UPLOAD真正
                # 到达当前虚拟时间后，服务端才能合法看到它。必须在这里而不是训练
                # 调用返回时更新observer，否则会提前看到尚未上传客户端的未来状态。
                observed_report = event.payload["report"]
                observed_state = event.payload.get("runtime_state")
                problem_observer.observe_round(
                    client_id=event.client_id,
                    client_round_idx=event.payload.get("client_round_idx", 0),
                    profile_type=event.payload.get("profile_type", ""),
                    report=observed_report,
                    # ClientRoundReport已经包含诊断所需的全部耗时分量，可直接作为
                    # result读取，避免把仅用于观测的对象额外塞入事件payload。
                    result=observed_report,
                )
                # 先完成当前轮误差比较，再公开当前状态。随后控制器产生的下一轮
                # dispatch decision才允许使用这份状态，保证严格的时间因果顺序。
                problem_observer.update_runtime_state(event.client_id, observed_state)
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
        "all_group_feasibility_mode": args.all_group_feasibility_mode,
        "group_creation_counterfactual_mode": args.group_creation_counterfactual_mode,
        "state_group_creation_q_mode": args.state_group_creation_q_mode,
        "state_group_window_mode": args.state_group_window_mode,
        "state_window_admission_mode": args.state_window_admission_mode,
        "communication_tail_risk_mode": args.communication_tail_risk_mode,
        "communication_robust_q_mode": args.communication_robust_q_mode,
        "hardware": hardware.to_dict(),
        "algorithm_variant": variant,
        # Persist the resolved training base, not only the YAML path.  This is
        # essential when comparing convergence runs because optimizer, data
        # partition and BN-buffer behavior can otherwise differ silently.
        "resolved_training_base": {
            "train_configs": OmegaConf.to_container(
                server_config.client_configs.train_configs, resolve=True
            ),
            "dataset_kwargs": OmegaConf.to_container(
                cfg.data_configs.dataset_kwargs, resolve=True
            ),
            "aggregator_kwargs": (
                OmegaConf.to_container(
                    server_config.server_configs.aggregator_kwargs, resolve=True
                )
                if hasattr(server_config.server_configs, "aggregator_kwargs")
                else {}
            ),
        },
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
    if args.algorithm == "state_compass":
        experiment_config["group_admission_mode"] = args.group_admission_mode
    if args.algorithm == "rup_compass":
        experiment_config["rup_config"] = asdict(_rup_config_from_args(args))
        experiment_config["rup_config"].update({
            "prox_requested": args.rup_prox == "on",
            "prox_enabled": args.rup_mode == "apply" and args.rup_prox == "on",
            "prox_mu": args.rup_prox_mu,
            "utility_eval_batches": args.rup_utility_eval_batches,
        })
    if problem_observer is not None:
        trace.set_fedcompass_problem_diagnostics(
            prediction_rows=problem_observer.prediction_rows(),
            report=problem_observer.build_report(trace.group_rows),
            shadow_report=problem_observer.build_shadow_report(),
            q_shadow_report=problem_observer.build_q_shadow_report(),
        )
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

def _rup_config_from_args(args) -> RUPConfig:
    """Build the complete policy config from CLI/programmatic arguments."""
    return RUPConfig(
        mode=getattr(args, "rup_mode", "apply"),
        group_admission_mode=getattr(args, "rup_group_admission", "conservative"),
        group_admission_min_group_size=getattr(args, "rup_group_admission_min_group_size", 3),
        group_admission_late_slack_ratio=getattr(args, "rup_group_admission_late_slack_ratio", 0.05),
        state_enabled=getattr(args, "rup_state", "on") == "on",
        residual_risk_enabled=getattr(args, "rup_residual_risk", "on") == "on",
        trust_region_enabled=getattr(args, "rup_trust", "on") == "on",
        soft_boundary_enabled=getattr(args, "rup_soft_boundary", "on") == "on",
        utility_enabled=getattr(args, "rup_utility", "on") == "on",
        budget_enabled=getattr(args, "rup_budget", "on") == "on",
        state_warmup_reports=getattr(args, "rup_state_warmup_reports", 3),
        residual_min_reports=getattr(args, "rup_residual_min_reports", 5),
        residual_window=getattr(args, "rup_residual_window", 20),
        residual_quantile=getattr(args, "rup_residual_quantile", 0.80),
        trust_fc_lower=getattr(args, "rup_trust_fc_lower", 0.80),
        trust_fc_upper=getattr(args, "rup_trust_fc_upper", 1.20),
        trust_prev_lower=getattr(args, "rup_trust_prev_lower", 0.75),
        trust_prev_upper=getattr(args, "rup_trust_prev_upper", 1.25),
        soft_qmin=getattr(args, "rup_soft_qmin", 50),
        soft_qmax=getattr(args, "rup_soft_qmax", 180),
        utility_beta=getattr(args, "rup_utility_beta", 0.20),
        utility_min_reports=getattr(args, "rup_utility_min_reports", 5),
        utility_lower=getattr(args, "rup_utility_lower", 0.90),
        utility_upper=getattr(args, "rup_utility_upper", 1.10),
        budget_window=getattr(args, "rup_budget_window", 50),
        budget_lower_ratio=getattr(args, "rup_budget_lower_ratio", 0.97),
        budget_upper_ratio=getattr(args, "rup_budget_upper_ratio", 1.03),
        budget_max_adjust_ratio=getattr(args, "rup_budget_max_adjust_ratio", 0.10),
        accuracy_priority_enabled=getattr(args, "rup_accuracy_priority", "on") == "on",
        accuracy_q_floor_ratio=getattr(args, "rup_accuracy_q_floor_ratio", 1.0),
        accuracy_q_boost_ratio=getattr(args, "rup_accuracy_q_boost_ratio", 0.05),
        accuracy_q_boost_min_confidence=getattr(args, "rup_accuracy_q_boost_min_confidence", 0.5),
        accuracy_q_boost_start_accuracy=getattr(args, "rup_accuracy_q_boost_start_accuracy", 50.0),
        risk_gated_floor_enabled=getattr(args, "rup_risk_gated_floor", "off") == "on",
        risk_gated_floor_min_safe_candidates=getattr(args, "rup_risk_gated_floor_min_safe_candidates", 10),
        risk_gated_floor_slack_ratio=getattr(args, "rup_risk_gated_floor_slack_ratio", 0.05),
        q_smoothness_enabled=getattr(args, "rup_q_smoothness", "off") == "on",
        q_smooth_max_increase_ratio=getattr(args, "rup_q_smooth_max_increase_ratio", 0.10),
        q_smooth_max_decrease_ratio=getattr(args, "rup_q_smooth_max_decrease_ratio", 0.20),
    )

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
    elif algorithm in ("fedcompass", "oort_compass", "state_compass", "rup_compass"):
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
    elif args.algorithm == "state_compass":
        return VirtualStateCompassController(
            aggregator=aggregator,
            num_clients=args.num_clients,
            min_local_steps=sched_kwargs.get("min_local_steps", args.min_local_steps),
            max_local_steps=sched_kwargs.get("max_local_steps", args.max_local_steps),
            speed_momentum=sched_kwargs.get("speed_momentum", 0.9),
            latest_time_factor=sched_kwargs.get("latest_time_factor", 1.2),
            num_global_epochs=args.num_global_epochs,
            group_admission_mode=args.group_admission_mode,
            all_group_feasibility_mode=args.all_group_feasibility_mode,
            group_creation_counterfactual_mode=args.group_creation_counterfactual_mode,
            state_group_creation_q_mode=args.state_group_creation_q_mode,
            state_group_window_mode=args.state_group_window_mode,
            state_window_admission_mode=args.state_window_admission_mode,
            communication_tail_risk_mode=args.communication_tail_risk_mode,
            communication_robust_q_mode=args.communication_robust_q_mode,
        )
    elif args.algorithm == "rup_compass":
        return VirtualRUPCompassController(
            aggregator=aggregator,
            num_clients=args.num_clients,
            min_local_steps=sched_kwargs.get("min_local_steps", args.min_local_steps),
            max_local_steps=sched_kwargs.get("max_local_steps", args.max_local_steps),
            speed_momentum=sched_kwargs.get("speed_momentum", 0.9),
            latest_time_factor=sched_kwargs.get("latest_time_factor", 1.2),
            num_global_epochs=args.num_global_epochs,
            rup_config=_rup_config_from_args(args),
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
                        choices=["fedavg", "fedasync", "fedcompass", "oort_compass", "state_compass", "rup_compass"],
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
    parser.add_argument("--group_admission_mode", type=str, default="shadow",
                        choices=["shadow", "apply"],
                        help="StateCompass风险约束入组：shadow只记录；apply在无安全Q时改走原始新建组")
    parser.add_argument("--all_group_feasibility_mode", type=str, default="off",
                        choices=["off", "shadow"],
                        help="全量已有组状态可行性观测：off关闭；shadow只读枚举，不改变调度")
    parser.add_argument("--group_creation_counterfactual_mode", type=str, default="off",
                        choices=["off", "shadow"],
                        help="原始新建组反事实：off关闭；shadow只读复算，不改变真实调度")
    parser.add_argument("--state_group_creation_q_mode", type=str, default="off",
                        choices=["off", "shadow"],
                        help="状态感知新组Q枚举：off关闭；shadow只读建议，不改变真实建组")
    parser.add_argument("--state_group_window_mode", type=str, default="off",
                        choices=["off", "shadow"],
                        help="固定原始新组Q的状态时间窗：off关闭；shadow只读比较")
    parser.add_argument("--state_window_admission_mode", type=str, default="off",
                        choices=["off", "shadow", "apply"],
                        help="状态时间窗组合准入Gate：shadow只记录；apply仅执行安全状态新组")
    parser.add_argument("--communication_tail_risk_mode", type=str, default="off",
                        choices=["off", "shadow"],
                        help="通信尾部风险校准：off关闭；shadow只记录，不改变Admission")
    parser.add_argument("--communication_robust_q_mode", type=str, default="off",
                        choices=["off", "shadow"],
                        help="固定状态组时间窗的通信稳健Q：off关闭；shadow只建议")

    # ──── RUP-Compass 可插拔层（仅 --algorithm rup_compass 生效）────
    parser.add_argument("--rup_mode", choices=["off", "shadow", "apply"], default="apply")
    parser.add_argument("--rup_state", choices=["on", "off"], default="on")
    parser.add_argument("--rup_residual_risk", choices=["on", "off"], default="on")
    parser.add_argument("--rup_trust", choices=["on", "off"], default="on")
    parser.add_argument("--rup_soft_boundary", choices=["on", "off"], default="on")
    parser.add_argument("--rup_utility", choices=["on", "off"], default="on")
    parser.add_argument("--rup_budget", choices=["on", "off"], default="on")
    parser.add_argument(
        "--rup_group_admission", choices=["off", "shadow", "apply", "conservative"],
        default="conservative",
        help="RUP风险约束分组准入：off关闭；shadow只观察；apply激进拒绝；conservative仅在明显迟到且不拆小组时拒绝",
    )
    parser.add_argument("--rup_group_admission_min_group_size", type=int, default=3)
    parser.add_argument("--rup_group_admission_late_slack_ratio", type=float, default=0.05)
    parser.add_argument("--rup_prox", choices=["on", "off"], default="on")
    parser.add_argument("--rup_prox_mu", type=float, default=1e-4)
    parser.add_argument("--rup_state_warmup_reports", type=int, default=3)
    parser.add_argument("--rup_residual_min_reports", type=int, default=5)
    parser.add_argument("--rup_residual_window", type=int, default=20)
    parser.add_argument("--rup_residual_quantile", type=float, default=0.80)
    parser.add_argument("--rup_trust_fc_lower", type=float, default=0.80)
    parser.add_argument("--rup_trust_fc_upper", type=float, default=1.20)
    parser.add_argument("--rup_trust_prev_lower", type=float, default=0.75)
    parser.add_argument("--rup_trust_prev_upper", type=float, default=1.25)
    parser.add_argument("--rup_soft_qmin", type=int, default=50)
    parser.add_argument("--rup_soft_qmax", type=int, default=180)
    parser.add_argument("--rup_utility_beta", type=float, default=0.20)
    parser.add_argument("--rup_utility_min_reports", type=int, default=5)
    parser.add_argument("--rup_utility_lower", type=float, default=0.90)
    parser.add_argument("--rup_utility_upper", type=float, default=1.10)
    parser.add_argument("--rup_utility_eval_batches", type=int, default=1)
    parser.add_argument("--rup_budget_window", type=int, default=50)
    parser.add_argument("--rup_budget_lower_ratio", type=float, default=0.97)
    parser.add_argument("--rup_budget_upper_ratio", type=float, default=1.03)
    parser.add_argument("--rup_budget_max_adjust_ratio", type=float, default=0.10)
    parser.add_argument("--rup_accuracy_priority", choices=["on", "off"], default="on")
    parser.add_argument("--rup_accuracy_q_floor_ratio", type=float, default=1.0)
    parser.add_argument("--rup_accuracy_q_boost_ratio", type=float, default=0.05)
    parser.add_argument("--rup_accuracy_q_boost_min_confidence", type=float, default=0.5)
    parser.add_argument(
        "--rup_accuracy_q_boost_start_accuracy", type=float, default=50.0,
        help="Only enable accuracy boost after the latest global eval reaches this accuracy; floor remains active.",
    )
    parser.add_argument("--rup_risk_gated_floor", choices=["on", "off"], default="off")
    parser.add_argument("--rup_risk_gated_floor_min_safe_candidates", type=int, default=10)
    parser.add_argument("--rup_risk_gated_floor_slack_ratio", type=float, default=0.05)
    parser.add_argument("--rup_q_smoothness", choices=["on", "off"], default="off")
    parser.add_argument("--rup_q_smooth_max_increase_ratio", type=float, default=0.10)
    parser.add_argument("--rup_q_smooth_max_decrease_ratio", type=float, default=0.20)

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
