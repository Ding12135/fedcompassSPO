"""状态时间窗 Admission 的事件级风险校准分析。

问题背景：
    ``state_window_admission_shadow_trace.csv`` 记录的是派发时的预测事实，
    ``scheduler_trace.csv`` 和 ``group_trace.csv`` 记录的是后续真实结果。仅统计
    总体 late rate 无法回答“预测安全余量多大才可靠”，也无法区分状态新组
    成功、目标客户端迟到和实验结束未闭环。

模块目标：
    只读关联一次 Admission 决策、该客户端的下一次真实上传和新组生命周期，
    输出逐事件校准表及跨种子汇总。分析器不导入控制器、不重放事件，也不修改
    实验输出，因此不会改变 Q、group、deadline 或聚合语义。

使用限制：
    只有 ``applied_action=create_state_window_group`` 的事件才具有真实反事实结果；
    Shadow 建议和 unresolved mismatch 没有执行对应状态 Q/时间窗，不能用于估计
    状态新组的实际迟到概率。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EVENT_FIELDS = [
    "run_dir", "seed", "virtual_time", "client_id", "consecutive_apply_count",
    "current_group_id", "assigned_group_id", "assigned_q",
    "predicted_expected_arrival_time", "predicted_latest_arrival_time",
    "predicted_safe_slack", "actual_finish_time", "actual_deadline_slack",
    "calibration_error", "actual_late", "round_time", "train_time",
    "download_time", "upload_time", "availability_rate", "group_completed",
    "predicted_compute_duration", "predicted_communication_duration",
    "predicted_spike_duration", "predicted_availability_duration",
    "predicted_availability_risk_duration", "predicted_uncertainty",
    "compute_error", "communication_error", "other_duration_error",
    "communication_p90", "communication_recent_max", "incremental_p90_margin",
    "p90_calibrated_safe_slack", "p90_safe_feasible", "tail_shadow_action",
    "group_trigger", "group_size", "target_client_participated",
    "target_client_pending", "merged_general_buffer", "outcome_class",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> Optional[float]:
    if value == "" or value is None:
        return None
    return float(value)


def _split_clients(value: str) -> set[str]:
    return {item for item in value.split(",") if item}


def _nearest_rank(values: Iterable[float], probability: float) -> Optional[float]:
    """返回保守的 nearest-rank 分位数，避免小样本线性插值淡化尾部风险。"""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _seed_from_run(run_dir: Path) -> str:
    name = run_dir.name
    return name[4:] if name.startswith("seed") else name


def analyze_run(run_dir: Path) -> List[Dict[str, Any]]:
    admission = _read_csv(run_dir / "state_window_admission_shadow_trace.csv")
    dispatch = _read_csv(run_dir / "dispatch_decision_trace.csv")
    scheduler = _read_csv(run_dir / "scheduler_trace.csv")
    groups = {
        row["group_id"]: row for row in _read_csv(run_dir / "group_trace.csv")
    }
    window_path = run_dir / "state_group_window_shadow_trace.csv"
    windows = (
        {
            (row["virtual_time"], row["client_id"]): row
            for row in _read_csv(window_path)
        }
        if window_path.exists() else {}
    )
    tail_path = run_dir / "communication_tail_risk_shadow_trace.csv"
    tails = (
        {(row["virtual_time"], row["client_id"]): row for row in _read_csv(tail_path)}
        if tail_path.exists() else {}
    )

    dispatch_by_decision = {
        (row["virtual_time"], row["client_id"]): row for row in dispatch
    }
    uploads_by_client: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in scheduler:
        uploads_by_client[row["client_id"]].append(row)
    for rows in uploads_by_client.values():
        rows.sort(key=lambda row: float(row["virtual_time"]))

    consecutive_by_client: Counter[str] = Counter()
    result: List[Dict[str, Any]] = []
    for gate in admission:
        if gate["applied_action"] != "create_state_window_group":
            # “连续”按该客户端相邻 Admission 决策定义；中间恢复安全即清零。
            consecutive_by_client[gate["client_id"]] = 0
            continue
        key = (gate["virtual_time"], gate["client_id"])
        decision = dispatch_by_decision.get(key)
        if decision is None:
            raise ValueError(f"找不到 Admission 对应的派发决策: {key}")

        client_id = gate["client_id"]
        group_id = decision["assigned_group"]
        decision_time = float(gate["virtual_time"])
        # 同一客户端下一次属于该新组的上传，才是本次状态时间窗的真实结果。
        upload = next(
            (
                row for row in uploads_by_client[client_id]
                if float(row["virtual_time"]) > decision_time
                and row["upload_group_id"] == group_id
            ),
            None,
        )
        lifecycle = groups.get(group_id)
        window = windows.get(key, {})
        tail = tails.get(key, {})
        predicted_latest = float(gate["state_new_group_latest_arrival_time"])
        predicted_slack = float(gate["state_new_group_safe_slack"])
        actual_finish = _as_float(upload["virtual_time"]) if upload else None
        actual_slack = (
            predicted_latest - actual_finish if actual_finish is not None else None
        )
        # 正值表示预测比真实结果乐观，是后续风险余量需要覆盖的量。
        calibration_error = (
            predicted_slack - actual_slack if actual_slack is not None else None
        )
        actual_compute = _as_float(upload["train_time"]) if upload else None
        actual_communication = (
            float(upload["download_time"]) + float(upload["upload_time"])
            if upload else None
        )
        actual_other = (
            float(upload["round_time"]) - actual_compute - actual_communication
            if upload and actual_compute is not None and actual_communication is not None
            else None
        )
        predicted_compute = _as_float(window.get("predicted_compute_duration", ""))
        predicted_communication = _as_float(
            window.get("predicted_communication_duration", "")
        )
        predicted_other = sum(
            _as_float(window.get(field, "")) or 0.0
            for field in (
                "predicted_spike_duration", "predicted_availability_duration"
            )
        ) if window else None

        consecutive_by_client[client_id] += 1
        participated = (
            client_id in _split_clients(lifecycle["arrived_client_ids"])
            if lifecycle else False
        )
        pending = (
            client_id in _split_clients(lifecycle["pending_client_ids"])
            if lifecycle else False
        )
        if upload is None:
            outcome = "incomplete_no_upload"
        elif actual_slack is not None and actual_slack < 0:
            outcome = "target_late"
        elif lifecycle is None:
            outcome = "uploaded_group_incomplete"
        elif pending:
            outcome = "target_pending_at_deadline"
        elif lifecycle["trigger"] == "deadline":
            outcome = "target_ontime_group_deadline"
        else:
            outcome = "target_ontime_all_arrived"

        result.append({
            "run_dir": str(run_dir),
            "seed": _seed_from_run(run_dir),
            "virtual_time": decision_time,
            "client_id": client_id,
            "consecutive_apply_count": consecutive_by_client[client_id],
            "current_group_id": gate["current_group_id"],
            "assigned_group_id": group_id,
            "assigned_q": decision["assigned_local_steps"],
            "predicted_expected_arrival_time": gate[
                "state_new_group_expected_arrival_time"
            ],
            "predicted_latest_arrival_time": predicted_latest,
            "predicted_safe_slack": predicted_slack,
            "actual_finish_time": actual_finish,
            "actual_deadline_slack": actual_slack,
            "calibration_error": calibration_error,
            "actual_late": "" if actual_slack is None else int(actual_slack < 0),
            "round_time": "" if upload is None else upload["round_time"],
            "train_time": "" if upload is None else upload["train_time"],
            "download_time": "" if upload is None else upload["download_time"],
            "upload_time": "" if upload is None else upload["upload_time"],
            "availability_rate": "" if upload is None else upload["availability_rate"],
            "group_completed": int(lifecycle is not None),
            "predicted_compute_duration": window.get("predicted_compute_duration", ""),
            "predicted_communication_duration": window.get(
                "predicted_communication_duration", ""
            ),
            "predicted_spike_duration": window.get("predicted_spike_duration", ""),
            "predicted_availability_duration": window.get(
                "predicted_availability_duration", ""
            ),
            "predicted_availability_risk_duration": window.get(
                "predicted_availability_risk_duration", ""
            ),
            "predicted_uncertainty": window.get("uncertainty", ""),
            "compute_error": (
                "" if predicted_compute is None or actual_compute is None
                else actual_compute - predicted_compute
            ),
            "communication_error": (
                "" if predicted_communication is None or actual_communication is None
                else actual_communication - predicted_communication
            ),
            "other_duration_error": (
                "" if predicted_other is None or actual_other is None
                else actual_other - predicted_other
            ),
            "communication_p90": tail.get("communication_p90", ""),
            "communication_recent_max": tail.get("communication_recent_max", ""),
            "incremental_p90_margin": tail.get("incremental_p90_margin", ""),
            "p90_calibrated_safe_slack": tail.get(
                "p90_calibrated_safe_slack", ""
            ),
            "p90_safe_feasible": tail.get("p90_safe_feasible", ""),
            "tail_shadow_action": tail.get("shadow_action", ""),
            "group_trigger": "" if lifecycle is None else lifecycle["trigger"],
            "group_size": "" if lifecycle is None else lifecycle["group_size"],
            "target_client_participated": int(participated),
            "target_client_pending": int(pending),
            "merged_general_buffer": (
                "" if lifecycle is None else lifecycle["merged_general_buffer"]
            ),
            "outcome_class": outcome,
        })
    return result


def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in events if row["actual_deadline_slack"] is not None]
    errors = [float(row["calibration_error"]) for row in completed]
    actual_slacks = [float(row["actual_deadline_slack"]) for row in completed]
    tail_labeled = [row for row in completed if row["p90_safe_feasible"] != ""]
    tail_confusion = {
        "kept_ontime": sum(
            row["p90_safe_feasible"] == "1" and row["actual_late"] == 0
            for row in tail_labeled
        ),
        "kept_late": sum(
            row["p90_safe_feasible"] == "1" and row["actual_late"] == 1
            for row in tail_labeled
        ),
        "rejected_ontime": sum(
            row["p90_safe_feasible"] == "0" and row["actual_late"] == 0
            for row in tail_labeled
        ),
        "rejected_late": sum(
            row["p90_safe_feasible"] == "0" and row["actual_late"] == 1
            for row in tail_labeled
        ),
    }
    by_seed: Dict[str, Dict[str, Any]] = {}
    for seed in sorted({str(row["seed"]) for row in events}):
        rows = [row for row in events if str(row["seed"]) == seed]
        observed = [row for row in rows if row["actual_deadline_slack"] is not None]
        by_seed[seed] = {
            "apply_events": len(rows),
            "observed_uploads": len(observed),
            "actual_late": sum(int(row["actual_late"]) for row in observed),
            "outcomes": dict(Counter(row["outcome_class"] for row in rows)),
            "clients": dict(Counter(row["client_id"] for row in rows)),
        }
    return {
        "scope": "仅统计真实执行的 create_state_window_group 事件",
        "num_events": len(events),
        "num_observed_uploads": len(completed),
        "num_incomplete": len(events) - len(completed),
        "actual_late_rate": (
            sum(slack < 0 for slack in actual_slacks) / len(actual_slacks)
            if actual_slacks else None
        ),
        "calibration_error": {
            "meaning": "predicted_safe_slack - actual_deadline_slack；正值表示预测乐观",
            "mean": sum(errors) / len(errors) if errors else None,
            "max": max(errors) if errors else None,
            "q80_nearest_rank": _nearest_rank(errors, 0.80),
            "q90_nearest_rank": _nearest_rank(errors, 0.90),
            "q95_nearest_rank": _nearest_rank(errors, 0.95),
        },
        "outcomes": dict(Counter(row["outcome_class"] for row in events)),
        "communication_tail_p90_shadow_confusion": tail_confusion,
        "clients": dict(Counter(row["client_id"] for row in events)),
        "by_seed": by_seed,
        "warning": "样本量很小，分位数只用于设计下一轮Shadow，不可直接作为稳定阈值。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分析状态时间窗的预测/真实安全余量")
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    events: List[Dict[str, Any]] = []
    for run_dir in args.run_dirs:
        events.extend(analyze_run(run_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "state_window_calibration_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)
    with (args.output_dir / "state_window_calibration_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summarize(events), handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
