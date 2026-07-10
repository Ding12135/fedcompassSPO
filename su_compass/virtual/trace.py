"""
su_compass.virtual.trace — 统一 trace 输出模块。

本模块负责虚拟联邦学习实验的全部可观测性输出，三算法（FedAvg / FedAsync /
FedCompass）共用同一套 CSV 结构，便于后续 compare_virtual_algorithms.py 读取对比。

输出文件与写入时机：
    scheduler_trace.csv         — CLIENT_UPLOAD 被调度器处理时（含 upload/next group）
    aggregation_trace.csv       — 每次全局模型聚合后（真实 per-client staleness）
    dispatch_decision_trace.csv — FedCompass Q/group 分配决策（join/create/first_group）
    group_trace.csv               — FedCompass arrival group 生命周期
    training_metrics.csv          — 仅在聚合发生时（num_client_updates 为真实参与人数）
    global_eval_trace.csv         — 每次聚合后在验证集上的全局精度
    client_states/*/round_reports.csv — 每客户端逐轮训练完成时
    client_states/*/state_trace.csv   — 每客户端 RuntimeStateTracker 滑动窗口快照
    experiment_config.json          — 实验结束一次性快照

设计要点：
    - dispatch_staleness：客户端派发时基于的全局版本滞后（写入 round_reports）
    - aggregation_staleness：聚合瞬间的真实陈旧度（由控制器回写 enrich_round_report_upload）
    - client_round_idx：每客户端独立递增，与 global_timestamp 解耦，便于跨表 join

读表顺序建议：
    1. training_metrics.csv / global_eval_trace.csv 看整体“聚合次数-虚拟时间-精度”。
    2. aggregation_trace.csv 看每次聚合吃了哪些客户端、各自 staleness 是多少。
    3. scheduler_trace.csv 看每次上传发生时调度器如何处理它。
    4. FedCompass/Oort 再看 dispatch_decision_trace.csv / oort_trace.csv 解释 Q/group 决策。
    5. client_states/* 追到单个客户端的训练耗时拆分与状态滑窗。
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ──────────────────────── 字段定义 ────────────────────────
# 各 CSV 的列名常量；Runner 与控制器只写值，字段语义在此集中维护。

SCHEDULER_TRACE_FIELDS = [
    "virtual_time",              # 调度器处理 CLIENT_UPLOAD 的虚拟时间
    "client_round_idx",          # 该客户端第几次 dispatch（从 0 递增）
    "global_timestamp",          # 处理完该上传后的全局模型版本号；未触发聚合时可能不变
    "client_id",
    "profile_type",              # 端侧画像类型（stable_fast / network_poor 等）
    "algorithm",
    "upload_group_id",           # 本次上传所属 arrival group（上传前 goa）
    "next_group_id",             # 上传处理后下一轮被安排到的 group；-1 表示无组/非 FedCompass
    "local_steps",               # 本轮客户端执行的 Q
    "speed_raw",                 # round_time / Q（单轮原始速度）
    "speed_smoothed",            # 调度器 speed_momentum 平滑后的速度
    "round_time",                # 本轮虚拟总耗时：下载 + 训练 + 上传 + 不可用等待/尖峰等
    "train_time",                # 纯训练虚拟耗时，不含网络
    "download_time",             # 下发全局模型的虚拟网络耗时
    "upload_time",               # 上传本地更新的虚拟网络耗时
    "communication_ratio",       # (download+upload)/round_time，用于识别网络瓶颈客户端
    "availability_rate",         # RuntimeStateTracker 滑动窗口可用率
    "late",                      # finish_time 是否超过 latest_arrival_time
    "dispatch_staleness",        # 派发时陈旧度（训练所基于的版本滞后）
    "aggregation_staleness",     # 若本轮触发聚合，写真实值；否则空
    "model_version_at_upload",   # 上传时 client_info.timestamp，即本地模型基于的全局版本
]

GROUP_TRACE_FIELDS = [
    "group_id",
    "created_time",              # group 被创建的虚拟时间
    "expected_arrival_time",     # curr + Q * speed
    "latest_arrival_time",       # expected * latest_time_factor（迟到阈值）
    "initial_client_ids",        # 曾分配到该 group 的全部客户端
    "arrived_client_ids",        # 按时到达并参与组聚合的客户端
    "pending_client_ids",        # 聚合时尚未到达的客户端
    "aggregation_time",          # group 实际聚合时间，可能早于 deadline（all_arrived）
    "group_size",                # 实际参与聚合人数（含 general_buffer 合并）
    "late_clients",              # 超过 latest 被转入 general_buffer 的客户端
    "trigger",                   # all_arrived | deadline
    "merged_general_buffer",     # 是否合并了迟到缓冲（0/1）
    "per_client_staleness",      # JSON：组内各客户端聚合 staleness
]

AGGREGATION_TRACE_FIELDS = [
    "aggregation_id",            # 自增序号，便于引用单次聚合
    "global_timestamp_before",   # 聚合前 server 全局版本
    "global_timestamp_after",    # 聚合后 server 全局版本，通常 before + 1
    "virtual_time",              # 聚合发生的虚拟时间
    "trigger",                   # single | group_all_arrived | group_deadline | fedavg_sync | fedasync_single
    "participating_clients",
    "per_client_staleness",      # JSON：聚合权重依据的真实陈旧度
    "per_client_local_steps",
    "per_client_model_version",  # JSON：各客户端训练起始全局版本
    "group_id",                  # FedCompass group；-1 表示非组聚合
    "num_clients",
    "client_update_budget_delta",  # 本次消耗的客户端更新贡献预算
]

DISPATCH_DECISION_TRACE_FIELDS = [
    "virtual_time",              # 做出本次分组/Q 决策的虚拟时间
    "client_id",
    "decision",                  # first_group | join_group | create_group
    "assigned_group",            # 被分配到的 group_id
    "assigned_local_steps",      # 本次决策给客户端下一轮分配的 Q
    "speed_smoothed",            # FedCompass 实际使用的平滑单 step 耗时
    "speed_raw",                 # 本轮刚观测到的原始单 step 耗时
    "remaining_time",            # join 时距 group 期望到达的剩余时间
    "target_arrival_time",
    "latest_arrival_time",
    "qmin",                      # Q 下界，用于判断 assigned_local_steps 是否被截断
    "qmax",                      # Q 上界，用于判断 assigned_local_steps 是否已达最大训练量
    "late_threshold_factor",     # yaml 中的 latest_time_factor（λ）
    "communication_ratio_mean",  # 决策时刻 RuntimeStateTracker 快照
    "late_rate",
    "availability_rate",
]

OORT_TRACE_FIELDS = [
    "virtual_time",              # Oort 计算发生的虚拟时间
    "client_id",
    "decision",                  # first_group | join_group | create_group | join_failed
    "assigned_group",
    "oort_mode",                 # shadow | q_only | q_and_group
    "speed_smoothed",            # 原始 Q 分母（真实速度平滑值）
    "effective_step_time",       # speed_smoothed × system_penalty（Oort Q 分母）
    "system_penalty",            # reason-aware 系统惩罚系数（>=1）
    "risk_score",                # 迟到/抖动/通信综合风险
    "oort_score",                # 统计效用 / 系统惩罚（观测用）
    "q_baseline",                # 若用原始 speed 会分到的 Q
    "q_after_oort",              # 用 effective_step_time 会分到的 Q
    "q_applied",                 # 本次实际写入 Dispatch 的 Q（shadow 下通常等于 baseline）
    "remaining_time",            # join 决策时距 group 期望到达的剩余时间
    "communication_ratio_mean",  # 决策时刻端侧状态快照
    "late_rate",
    "step_time_cv",
    "availability_rate",
    "num_reports",               # 端侧窗口样本数（判断是否冷启动）
]

TRAINING_METRICS_FIELDS = [
    "global_timestamp",          # 当前全局模型版本号
    "virtual_time",              # 本次聚合发生时间
    "algorithm",
    "algorithm_variant",         # 创新实验变体标签（如 fedcompass_baseline / reason_aware_v1）
    "num_global_updates",        # 等于 global_timestamp
    "num_client_updates",        # 本次聚合参与客户端数
    "client_update_budget_used", # FedCompass 累计客户端更新贡献
]

GLOBAL_EVAL_TRACE_FIELDS = [
    "global_timestamp",
    "virtual_time",
    "test_accuracy",             # 0–100，与 APPFL metric/acc.py 一致
    "test_loss",
    "num_val_samples",
]

ROUND_REPORT_FIELDS = [
    "client_round_idx",
    "client_id",
    "profile_type",
    "dispatch_time",             # 客户端本轮收到任务的虚拟时间
    "finish_time",               # 客户端本轮上传完成的虚拟时间，也是 CLIENT_UPLOAD.time
    "local_steps",
    "train_time",                # 训练部分虚拟耗时
    "download_time",             # 模型下发虚拟耗时
    "upload_time",               # 更新上传虚拟耗时
    "spike_delay",               # 画像模拟出的偶发抖动/尖峰延迟
    "availability_wait",         # 客户端不可用导致的等待时间
    "round_time",                # 本轮从 dispatch 到 finish 的总虚拟耗时
    "step_time",                 # round_time/local_steps，调度器观测到的单步总耗时
    "compute_step_time",         # 只看计算部分的单步耗时
    "communication_time",        # download_time + upload_time
    "communication_ratio",       # 通信耗时占 round_time 比例
    "available",                 # 本轮是否无需不可用等待
    "late",                      # 是否晚于 latest_arrival_time
    "dispatch_staleness",        # 派发时版本差：server_global_ts - client_ts
    "aggregation_staleness",     # 聚合时版本差；初始为空，CLIENT_UPLOAD 处理后 enrich 回写
    "model_version_at_dispatch", # 本轮客户端拿到的全局模型版本
    "model_version_at_upload",   # 上传处理时 client_info.timestamp；初始为空，上传后回写
    "target_arrival_time",       # FedCompass 期望该客户端贴近的 group 到达时间
    "latest_arrival_time",       # FedCompass 迟到线；finish_time 超过它则 late=1
    "early_margin",              # latest_arrival_time - finish_time；正数表示提前，负数表示迟到
    "hit_q_min",                 # 本轮 Q 是否被截到下界
    "hit_q_max",                 # 本轮 Q 是否达到上界
    "availability_wait_ratio",   # availability_wait / round_time，衡量不可用等待占比
    "upload_group_id",           # 上传前所属 group；初始为空，上传后回写
    "next_group_id",             # 上传后下一轮被分配到的 group；初始为空，上传后回写
    "speed_smoothed_at_upload",  # 上传处理后调度器看到的平滑 speed
]


# ──────────────────────── TraceWriter ────────────────────────

class TraceWriter:
    """统一 trace 输出管理器。

    实验过程中各 record_* 方法只写内存缓冲；实验结束时调用 flush() 一次性落盘。
    三算法共用同一实例，通过 algorithm / algorithm_variant 区分实验类型。

    Attributes:
        output_dir:           实验输出根目录。
        algorithm:            算法名（fedavg / fedasync / fedcompass）。
        algorithm_variant:    变体标签，供创新实验对比。
        scheduler_rows:       scheduler_trace 缓冲。
        aggregation_rows:     aggregation_trace 缓冲。
        dispatch_decision_rows: dispatch_decision_trace 缓冲。
        group_rows:           group_trace 缓冲。
        training_rows:        training_metrics 缓冲。
        global_eval_rows:     global_eval_trace 缓冲。
        round_report_rows:    按 client_id 分组的 round_reports 缓冲。
        state_trace_rows:     按 client_id 分组的 state_trace 缓冲。
    """

    def __init__(self, output_dir: str, algorithm: str, algorithm_variant: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.algorithm = algorithm
        self.algorithm_variant = algorithm_variant or algorithm

        self.scheduler_rows: List[Dict[str, Any]] = []
        self.group_rows: List[Dict[str, Any]] = []
        self.aggregation_rows: List[Dict[str, Any]] = []
        self.dispatch_decision_rows: List[Dict[str, Any]] = []
        self.oort_rows: List[Dict[str, Any]] = []
        self.training_rows: List[Dict[str, Any]] = []
        self.global_eval_rows: List[Dict[str, Any]] = []
        self.round_report_rows: Dict[str, List[Dict[str, Any]]] = {}
        self.state_trace_rows: Dict[str, List[Dict[str, Any]]] = {}
        self._aggregation_id = 0  # aggregation_trace 自增主键

    def record_scheduler_event(
        self,
        virtual_time: float,
        client_round_idx: int,
        client_id: str,
        profile_type: str,
        local_steps: int,
        speed_raw: float,
        speed_smoothed: float,
        report,
        dispatch_staleness: int,
        aggregation_staleness: Optional[int],
        global_timestamp: int,
        upload_group_id: int = -1,
        next_group_id: int = -1,
        model_version_at_upload: int = 0,
        runtime_state=None,
    ) -> None:
        """记录一条 scheduler_trace 行（CLIENT_UPLOAD 处理时刻）。

        注意：upload_group_id 是上传**前**的 goa；next_group_id 是上传**后**新分配的 goa。
        aggregation_staleness 仅在本轮 upload 触发立即/组聚合时有值。
        """
        # scheduler_trace 是“上传被处理后的事实表”：它不记录训练开始时刻的决策，
        # 而是把训练结果、上传前 group、上传后 next_group 和聚合 staleness 串起来。
        availability_rate = 1.0
        if runtime_state is not None:
            availability_rate = runtime_state.availability_rate

        self.scheduler_rows.append({
            "virtual_time": virtual_time,
            "client_round_idx": client_round_idx,
            "global_timestamp": global_timestamp,
            "client_id": client_id,
            "profile_type": profile_type,
            "algorithm": self.algorithm,
            "upload_group_id": upload_group_id,
            "next_group_id": next_group_id,
            "local_steps": local_steps,
            "speed_raw": speed_raw,
            "speed_smoothed": speed_smoothed,
            "round_time": report.round_time,
            "train_time": report.train_time or 0.0,
            "download_time": report.download_time or 0.0,
            "upload_time": report.upload_time or 0.0,
            "communication_ratio": report.communication_ratio,
            "availability_rate": availability_rate,
            "late": int(report.late),
            "dispatch_staleness": dispatch_staleness,
            "aggregation_staleness": aggregation_staleness if aggregation_staleness is not None else "",
            "model_version_at_upload": model_version_at_upload,
        })

    def record_aggregation(self, row: Dict[str, Any]) -> None:
        """记录一条 aggregation_trace 行。

        row 由控制器构造，需含 per_client_staleness 等 JSON 字段；
        aggregation_id 若未提供则自动递增。
        """
        # aggregation_id 在写入端统一生成，控制器只需要描述一次聚合的业务事实。
        self._aggregation_id += 1
        row.setdefault("aggregation_id", self._aggregation_id)
        self.aggregation_rows.append(row)

    def record_dispatch_decision(self, row: Dict[str, Any]) -> None:
        """记录一条 dispatch_decision_trace 行。"""
        self.dispatch_decision_rows.append(row)

    def record_oort_decision(self, row: Dict[str, Any]) -> None:
        """记录一条 oort_trace 行（仅 Oort-Compass 变体产生）。

        与 dispatch_decision_trace 解耦：baseline FedCompass 不写此表，
        故引入 Oort 不改变原有 trace 结构，便于 shadow 模式下逐字段对比。
        """
        self.oort_rows.append(row)

    def record_group(self, row: Dict[str, Any]) -> None:
        """记录一条 group_trace 行。"""
        self.group_rows.append(row)

    def record_training_metrics(
        self,
        global_timestamp: int,
        virtual_time: float,
        num_client_updates: int,
        client_update_budget_used: int,
    ) -> None:
        """记录一条 training_metrics 行（仅在聚合发生时调用）。"""
        self.training_rows.append({
            "global_timestamp": global_timestamp,
            "virtual_time": virtual_time,
            "algorithm": self.algorithm,
            "algorithm_variant": self.algorithm_variant,
            "num_global_updates": global_timestamp,
            "num_client_updates": num_client_updates,
            "client_update_budget_used": client_update_budget_used,
        })

    def record_global_eval(
        self,
        global_timestamp: int,
        virtual_time: float,
        test_accuracy: float,
        test_loss: float,
        num_val_samples: int,
    ) -> None:
        """记录全局模型评估结果。"""
        self.global_eval_rows.append({
            "global_timestamp": global_timestamp,
            "virtual_time": virtual_time,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "num_val_samples": num_val_samples,
        })

    def record_round_report(
        self,
        client_id: str,
        report,
        result,
        profile_type: str,
        client_round_idx: int,
        model_version_at_dispatch: int = 0,
    ) -> None:
        """记录一条 per-client round_report 行（训练完成、事件入队时）。

        此时仅知 dispatch 侧信息；upload 相关字段（aggregation_staleness 等）
        在 enrich_round_report_upload() 中于 CLIENT_UPLOAD 处理后回写。
        """
        # round_reports 在训练完成时先落一行“dispatch/运行耗时事实”，
        # upload 处理后的 group/staleness 信息稍后由 enrich_round_report_upload 回填。
        if client_id not in self.round_report_rows:
            self.round_report_rows[client_id] = []
        self.round_report_rows[client_id].append({
            "client_round_idx": client_round_idx,
            "client_id": client_id,
            "profile_type": profile_type,
            "dispatch_time": report.dispatch_time,
            "finish_time": report.finish_time,
            "local_steps": report.local_steps,
            "train_time": result.train_time,
            "download_time": result.download_time,
            "upload_time": result.upload_time,
            "spike_delay": result.spike_delay,
            "availability_wait": result.availability_wait,
            "round_time": report.round_time,
            "step_time": report.step_time,
            "compute_step_time": report.compute_step_time,
            "communication_time": report.communication_time,
            "communication_ratio": report.communication_ratio,
            "available": int(report.available),
            "late": int(report.late),
            "dispatch_staleness": report.staleness,
            "aggregation_staleness": "",
            "model_version_at_dispatch": model_version_at_dispatch,
            "model_version_at_upload": "",
            "target_arrival_time": report.target_arrival_time if report.target_arrival_time is not None else "",
            "latest_arrival_time": report.latest_arrival_time if report.latest_arrival_time is not None else "",
            "early_margin": report.early_margin,
            "hit_q_min": int(report.hit_q_min),
            "hit_q_max": int(report.hit_q_max),
            "availability_wait_ratio": report.availability_wait_ratio,
            "upload_group_id": "",
            "next_group_id": "",
            "speed_smoothed_at_upload": "",
        })

    def enrich_round_report_upload(
        self,
        client_id: str,
        client_round_idx: int,
        aggregation_staleness: Optional[int],
        model_version_at_upload: int,
        upload_group_id: int,
        next_group_id: int,
        speed_smoothed_at_upload: float,
    ) -> None:
        """在 CLIENT_UPLOAD 处理后回写 upload 相关字段。

        从 round_reports 末尾向前查找匹配的 client_round_idx，保证与训练阶段
        写入的行一一对应（同一客户端可能交错多轮）。
        """
        rows = self.round_report_rows.get(client_id, [])
        # 同一客户端可能已经排了多轮训练，从后往前找能最快命中刚完成的那一轮。
        for row in reversed(rows):
            if row["client_round_idx"] == client_round_idx:
                if aggregation_staleness is not None:
                    row["aggregation_staleness"] = aggregation_staleness
                row["model_version_at_upload"] = model_version_at_upload
                row["upload_group_id"] = upload_group_id
                row["next_group_id"] = next_group_id
                row["speed_smoothed_at_upload"] = speed_smoothed_at_upload
                break

    def record_state_trace(self, client_id: str, state, profile_type: str, client_round_idx: int) -> None:
        """记录一条 per-client state_trace 行。"""
        if client_id not in self.state_trace_rows:
            self.state_trace_rows[client_id] = []
        row = {"client_round_idx": client_round_idx, "client_id": client_id, "profile_type": profile_type}
        row.update(state.to_dict())
        self.state_trace_rows[client_id].append(row)

    def flush(self, experiment_config: Optional[Dict[str, Any]] = None) -> None:
        """将全部缓冲数据写出到磁盘。

        目录结构：
            output_dir/
                scheduler_trace.csv
                aggregation_trace.csv
                ...
                client_states/client_*/round_reports.csv
                summary/all_round_reports.csv
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 全局级 trace ──
        _write_csv(self.output_dir / "scheduler_trace.csv", self.scheduler_rows, SCHEDULER_TRACE_FIELDS)
        _write_csv(self.output_dir / "aggregation_trace.csv", self.aggregation_rows, AGGREGATION_TRACE_FIELDS)
        _write_csv(
            self.output_dir / "dispatch_decision_trace.csv",
            self.dispatch_decision_rows,
            DISPATCH_DECISION_TRACE_FIELDS,
        )
        _write_csv(self.output_dir / "group_trace.csv", self.group_rows, GROUP_TRACE_FIELDS)
        # oort_trace 只在 Oort-Compass 变体产生；baseline 不写空表，避免误判为启用 Oort。
        if self.oort_rows:
            _write_csv(self.output_dir / "oort_trace.csv", self.oort_rows, OORT_TRACE_FIELDS)
        _write_csv(self.output_dir / "training_metrics.csv", self.training_rows, TRAINING_METRICS_FIELDS)
        _write_csv(self.output_dir / "global_eval_trace.csv", self.global_eval_rows, GLOBAL_EVAL_TRACE_FIELDS)

        # ── 每客户端 trace ──
        for client_id in sorted(self.round_report_rows.keys()):
            client_dir = self.output_dir / "client_states" / client_id
            client_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(client_dir / "round_reports.csv", self.round_report_rows[client_id], ROUND_REPORT_FIELDS)

        for client_id in sorted(self.state_trace_rows.keys()):
            client_dir = self.output_dir / "client_states" / client_id
            client_dir.mkdir(parents=True, exist_ok=True)
            rows = self.state_trace_rows[client_id]
            if rows:
                fieldnames = list(rows[0].keys())
                _write_csv(client_dir / "state_trace.csv", rows, fieldnames)

        summary_dir = self.output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        # summary/all_round_reports.csv 把所有客户端 round_reports 合并，便于一次性画分布图。
        all_reports = []
        for rows in self.round_report_rows.values():
            all_reports.extend(rows)
        if all_reports:
            _write_csv(summary_dir / "all_round_reports.csv", all_reports, ROUND_REPORT_FIELDS)

        if experiment_config is not None:
            experiment_config.setdefault("algorithm_variant", self.algorithm_variant)
            _write_json(self.output_dir / "experiment_config.json", experiment_config)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    """写出 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """写出 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
