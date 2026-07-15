"""Produce a high-accuracy TTA report from FedCompass/RUP-Compass traces."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

TTA_THRESHOLDS = (40, 45, 50, 55, 60, 62, 63, 64, 65)
CORE_KEYS = [
    "max_accuracy", "final_accuracy", "last10_accuracy", "last10_std",
    "max_last10_gap", "tta_40", "tta_45", "tta_50", "tta_55", "tta_60",
    "tta_62", "tta_63", "tta_64", "tta_65", "normalized_accuracy_time_auc",
    "late", "late_rate", "deadline_groups", "deadline_rate", "all_arrived_rate",
    "total_groups", "mean_group_size", "mean_staleness", "max_staleness",
    "final_virtual_time", "mean_q", "total_local_steps", "qmin_rate", "qmax_rate",
]


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values, default=None):
    return statistics.mean(values) if values else default


def _rate(count: int, total: int) -> float:
    return count / max(total, 1)


def _cv(values: list[float]) -> float:
    mean = _mean(values, 0.0)
    return statistics.pstdev(values) / mean if values and mean else 0.0


def _gini(values: list[float]) -> float:
    ordered = sorted(max(0.0, value) for value in values)
    total = sum(ordered)
    n = len(ordered)
    if not n or not total:
        return 0.0
    return sum((2 * i - n - 1) * value for i, value in enumerate(ordered, 1)) / (n * total)


def _correlation(xs: list[float], ys: list[float]):
    if len(xs) < 2 or len(ys) != len(xs):
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def _first_tta(times: list[float], accuracy: list[float], threshold: int):
    return next((times[i] for i, value in enumerate(accuracy) if value >= threshold), None)


def summarize_run(directory: Path) -> dict:
    eval_rows = _rows(directory / "global_eval_trace.csv")
    scheduler = _rows(directory / "scheduler_trace.csv")
    groups = _rows(directory / "group_trace.csv")
    aggregations = _rows(directory / "aggregation_trace.csv")
    accuracy = [float(x["test_accuracy"]) for x in eval_rows]
    times = [float(x["virtual_time"]) for x in eval_rows]
    q_values, staleness = [], []
    for row in aggregations:
        for field, target in (("per_client_local_steps", q_values), ("per_client_staleness", staleness)):
            value = ast.literal_eval(row[field])
            target.extend(float(x) for x in (value.values() if isinstance(value, dict) else value))
    group_sizes = [_int(row, "group_size") for row in groups]
    deadline_groups = sum(x["trigger"] == "deadline" for x in groups)
    all_arrived_groups = sum(x["trigger"] == "all_arrived" for x in groups)
    auc = sum(
        (accuracy[i - 1] + accuracy[i]) * 0.5 * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    result = {
        "max_accuracy": max(accuracy), "final_accuracy": accuracy[-1],
        "last10_accuracy": statistics.mean(accuracy[-10:]),
        "last10_std": statistics.pstdev(accuracy[-10:]),
        "max_last10_gap": max(accuracy) - statistics.mean(accuracy[-10:]),
        "final_virtual_time": times[-1],
        "normalized_accuracy_time_auc": auc / max(times[-1] - times[0], 1e-8),
        "late": sum(_int(x, "late") for x in scheduler),
        "late_rate": _rate(sum(_int(x, "late") for x in scheduler), len(scheduler)),
        "deadline_groups": deadline_groups,
        "deadline_rate": _rate(deadline_groups, len(groups)),
        "all_arrived_groups": all_arrived_groups,
        "all_arrived_rate": _rate(all_arrived_groups, len(groups)),
        "total_groups": len(groups),
        "mean_group_size": _mean(group_sizes),
        "mean_staleness": statistics.mean(staleness), "max_staleness": max(staleness),
        "mean_q": statistics.mean(q_values), "total_local_steps": sum(q_values),
        "qmin_rate": sum(x == 40 for x in q_values) / len(q_values),
        "qmax_rate": sum(x == 200 for x in q_values) / len(q_values),
    }
    for threshold in TTA_THRESHOLDS:
        result[f"tta_{threshold}"] = _first_tta(times, accuracy, threshold)
    rup_path = directory / "rup_decision_trace.csv"
    if rup_path.exists():
        decisions = _rows(rup_path)
        baseline_q = [_float(x, "baseline_q") for x in decisions]
        applied_q = [_float(x, "applied_q") for x in decisions]
        recommended_q = [_float(x, "recommended_q") for x in decisions]
        q_delta = [applied_q[i] - baseline_q[i] for i in range(len(decisions))]
        recommendation_delta = [
            recommended_q[i] - baseline_q[i] for i in range(len(decisions))
        ]
        result["rup_observation"] = {
            "decisions": len(decisions),
            "changed_recommended_rate": _rate(sum(x["recommended_q"] != x["baseline_q"] for x in decisions), len(decisions)),
            "changed_applied_rate": _rate(sum(x["applied_q"] != x["baseline_q"] for x in decisions), len(decisions)),
            "no_safe_q": sum(x["fallback_reason"] == "no_safe_q_keep_fedcompass" for x in decisions),
            "fallbacks": dict(Counter(x["fallback_reason"] or "none" for x in decisions)),
            "mean_baseline_q": _mean(baseline_q),
            "mean_applied_q": _mean(applied_q),
            "mean_recommended_q": _mean(recommended_q),
            "mean_applied_minus_baseline_q": _mean(q_delta),
            "mean_recommended_minus_baseline_q": _mean(recommendation_delta),
            "applied_above_baseline_rate": _rate(sum(x > 0 for x in q_delta), len(q_delta)),
            "applied_below_baseline_rate": _rate(sum(x < 0 for x in q_delta), len(q_delta)),
            "accuracy_floor_applied_rate": _rate(sum(_int(x, "accuracy_floor_applied") for x in decisions), len(decisions)),
            "accuracy_boost_applied_rate": _rate(sum(_int(x, "accuracy_boost_applied") for x in decisions), len(decisions)),
            "accuracy_boost_stage_active_rate": _rate(sum(_int(x, "accuracy_boost_stage_active") for x in decisions), len(decisions)),
            "risk_gated_floor_allowed_rate": _rate(sum(_int(x, "risk_gated_floor_allowed", 1) for x in decisions), len(decisions)),
            "q_smooth_applied_rate": _rate(sum(_int(x, "q_smooth_applied") for x in decisions), len(decisions)),
            "mean_pre_accuracy_q": _mean([_float(x, "pre_accuracy_q") for x in decisions]),
            "mean_accuracy_priority_q": _mean([_float(x, "accuracy_priority_q") for x in decisions]),
            "mean_pre_smooth_q": _mean([_float(x, "pre_smooth_q", _float(x, "recommended_q")) for x in decisions]),
            "mean_smooth_q": _mean([_float(x, "smooth_q", _float(x, "recommended_q")) for x in decisions]),
            "mean_current_global_accuracy": _mean([
                _float(x, "current_global_accuracy") for x in decisions
                if x.get("current_global_accuracy") not in ("", None)
            ]),
            "mean_utility_multiplier": _mean([_float(x, "utility_normalized") for x in decisions]),
            "mean_utility_confidence": _mean([_float(x, "utility_confidence") for x in decisions]),
            "mean_budget_ratio_before": _mean([_float(x, "budget_ratio_before") for x in decisions]),
            "cumulative_guard_applied_rate": _rate(sum(_int(x, "cumulative_budget_guard_applied") for x in decisions), len(decisions)),
            "debt_repayment_applied_rate": _rate(sum(_int(x, "debt_repayment_applied") for x in decisions), len(decisions)),
            "final_cumulative_work_ratio": _float(decisions[-1], "cumulative_work_ratio_after", 1.0),
            "final_cumulative_work_debt": _float(decisions[-1], "cumulative_work_debt_after", 0.0),
            "max_cumulative_work_debt": max((_float(x, "cumulative_work_debt_after") for x in decisions), default=0.0),
            "mean_cumulative_work_ratio": _mean([_float(x, "cumulative_work_ratio_after", 1.0) for x in decisions]),
            "baseline_q_safe_rate": _rate(sum(_int(x, "baseline_q_safe", 1) for x in decisions), len(decisions)),
            "applied_q_safe_rate": _rate(sum(_int(x, "applied_q_safe", 1) for x in decisions), len(decisions)),
            "mean_safe_candidate_span": _mean([
                max(0.0, _float(x, "safe_q_max") - _float(x, "safe_q_min"))
                for x in decisions
            ]),
            "mean_predicted_deadline_slack": _mean([_float(x, "predicted_deadline_slack") for x in decisions]),
            "budget_control_reasons": dict(Counter(x.get("budget_control_reason") or "legacy" for x in decisions)),
            "mean_budget_eligible_safe_candidates": _mean([_float(x, "budget_eligible_safe_candidates") for x in decisions]),
            "mean_repayment_headroom_q": _mean([_float(x, "repayment_headroom_q") for x in decisions]),
            "q_delta_utility_correlation": _correlation(
                q_delta, [_float(x, "utility_normalized", 1.0) for x in decisions]
            ),
            "q_delta_deadline_slack_correlation": _correlation(
                q_delta, [_float(x, "predicted_deadline_slack") for x in decisions]
            ),
            "mean_residual_margin": _mean([_float(x, "residual_margin") for x in decisions]),
        }
    training_path = directory / "rup_training_trace.csv"
    if training_path.exists():
        training = _rows(training_path)
        result["rup_training"] = {
            "reports": len(training),
            "non_finite": sum(_int(x, "finite") == 0 for x in training),
            "mean_prox_penalty": _mean([_float(x, "mean_prox_penalty") for x in training]),
            "positive_progress_rate": _rate(sum(_float(x, "loss_delta_per_step") > 0 for x in training), len(training)),
        }
    outcome_path = directory / "rup_outcome_trace.csv"
    if outcome_path.exists():
        outcomes = _rows(outcome_path)
        errors = [_float(x, "prediction_error") for x in outcomes]
        result["rup_outcomes"] = {
            "completed_predicted_dispatches": len(outcomes),
            "mean_prediction_error": _mean(errors),
            "mean_absolute_prediction_error": _mean([abs(x) for x in errors]),
            "p90_absolute_prediction_error": (
                sorted(abs(x) for x in errors)[max(0, math.ceil(0.9 * len(errors)) - 1)]
                if errors else None
            ),
            "safe_duration_exceeded_rate": _rate(sum(_int(x, "safe_duration_exceeded") for x in outcomes), len(outcomes)),
            "deadline_miss_rate": _rate(sum(_int(x, "deadline_miss") for x in outcomes), len(outcomes)),
            "prediction_error_q_correlation": _correlation(
                errors, [_float(x, "applied_q") for x in outcomes]
            ),
        }
    terminal_path = directory / "rup_terminal_state.json"
    if terminal_path.exists():
        result["rup_terminal"] = json.loads(terminal_path.read_text(encoding="utf-8"))

    # Per-client allocation/completion/aggregation closure detects whether debt
    # repayment is concentrated on a few non-IID silos or lost in the run tail.
    client_ids = sorted({x["client_id"] for x in scheduler} | ({x["client_id"] for x in decisions} if rup_path.exists() else set()))
    per_client = {}
    aggregated_work = Counter()
    aggregated_staleness = {}
    for row in aggregations:
        steps = ast.literal_eval(row["per_client_local_steps"])
        stale = ast.literal_eval(row["per_client_staleness"])
        for client_id, value in steps.items():
            aggregated_work[client_id] += float(value)
        for client_id, value in stale.items():
            aggregated_staleness.setdefault(client_id, []).append(float(value))
    for client_id in client_ids:
        dispatched = [x for x in decisions if x["client_id"] == client_id] if rup_path.exists() else []
        completed = [x for x in scheduler if x["client_id"] == client_id]
        client_training = [x for x in training if x["client_id"] == client_id] if training_path.exists() else []
        per_client[client_id] = {
            "dispatched_updates": len(dispatched),
            "dispatched_work": sum(_float(x, "applied_q") for x in dispatched),
            "baseline_work": sum(_float(x, "baseline_q") for x in dispatched),
            "completed_updates": len(completed),
            "completed_work": sum(_float(x, "local_steps") for x in completed),
            "aggregated_work": aggregated_work[client_id],
            "late_rate": _rate(sum(_int(x, "late") for x in completed), len(completed)),
            "mean_staleness": _mean(aggregated_staleness.get(client_id, [])),
            "mean_utility": _mean([_float(x, "utility_normalized") for x in dispatched]),
            "mean_loss_progress_per_step": _mean([_float(x, "loss_delta_per_step") for x in client_training]),
        }
    if per_client:
        work = [x["dispatched_work"] for x in per_client.values()]
        result["rup_client_fairness"] = {
            "per_client": per_client,
            "dispatched_work_cv": _cv(work),
            "dispatched_work_gini": _gini(work),
            "max_to_min_dispatched_work_ratio": max(work) / max(min(work), 1.0),
        }
    admission_path = directory / "group_admission_trace.csv"
    if admission_path.exists():
        admissions = _rows(admission_path)
        mismatches = [x for x in admissions if _int(x, "group_mismatch") == 1]
        rejected = [x for x in mismatches if _int(x, "admitted") == 0]
        severe_late = [x for x in admissions if _int(x, "severe_late_risk") == 1]
        small_group_keeps = [
            x for x in admissions
            if x.get("reason") == "rup_conservative_keep_join_small_group"
        ]
        low_late_risk_keeps = [
            x for x in admissions
            if x.get("reason") == "rup_conservative_keep_join_low_late_risk"
        ]
        result["rup_group_admission"] = {
            "candidates": len(admissions),
            "mismatches": len(mismatches),
            "mismatch_rate": _rate(len(mismatches), len(admissions)),
            "rejected_to_create": len(rejected),
            "rejection_rate": _rate(len(rejected), len(admissions)),
            "mismatch_rejection_rate": _rate(len(rejected), len(mismatches)),
            "severe_late_risk": len(severe_late),
            "severe_late_risk_rate": _rate(len(severe_late), len(admissions)),
            "kept_small_group": len(small_group_keeps),
            "kept_low_late_risk": len(low_late_risk_keeps),
            "mean_current_group_size": _mean([_float(x, "current_group_size") for x in admissions]),
            "mean_lateness_margin": _mean([_float(x, "lateness_margin") for x in admissions]),
            "mean_late_slack": _mean([_float(x, "late_slack") for x in admissions]),
            "rejected_trace_incomplete": sum(
                _int(x, "actual_group_id") < 0 or _int(x, "actual_dispatched_q") < 0
                for x in rejected
            ),
            "reasons": dict(Counter(x["reason"] for x in admissions)),
        }
    return result


def _delta(rup: dict, baseline: dict, keys: list[str]) -> dict:
    return {
        key: (rup[key] - baseline[key]) if rup.get(key) is not None and baseline.get(key) is not None else None
        for key in keys
    }


def _format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _gate_summary(baseline: dict, rup: dict) -> dict:
    work_ratio = rup["total_local_steps"] / max(baseline["total_local_steps"], 1)
    return {
        "tta65_beats_baseline": (
            rup.get("tta_65") is not None
            and baseline.get("tta_65") is not None
            and rup["tta_65"] < baseline["tta_65"]
        ),
        "max_accuracy_at_least_baseline": rup["max_accuracy"] >= baseline["max_accuracy"],
        "max_accuracy_at_least_65": rup["max_accuracy"] >= 65.0,
        "late_rate_below_baseline": rup["late_rate"] < baseline["late_rate"],
        "deadline_rate_below_baseline": rup["deadline_rate"] < baseline["deadline_rate"],
        "final_accuracy_at_least_64_5": rup["final_accuracy"] >= 64.5,
        "last10_std_not_worse": rup["last10_std"] <= baseline["last10_std"],
        "work_ratio_in_0_99_1_02": 0.99 <= work_ratio <= 1.02,
        "tta62_or_tta64_beats_baseline": any(
            rup.get(f"tta_{threshold}") is not None
            and baseline.get(f"tta_{threshold}") is not None
            and rup[f"tta_{threshold}"] < baseline[f"tta_{threshold}"]
            for threshold in (62, 64)
        ),
        "auc_at_least_baseline": (
            rup["normalized_accuracy_time_auc"]
            >= baseline["normalized_accuracy_time_auc"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--rup_dir", required=True)
    parser.add_argument("--output_dir", default="su_compass/output/rup_analysis")
    args = parser.parse_args()
    baseline = summarize_run(Path(args.baseline_dir))
    rup = summarize_run(Path(args.rup_dir))
    delta = _delta(rup, baseline, CORE_KEYS)
    gates = _gate_summary(baseline, rup)
    payload = {
        "baseline": baseline,
        "rup": rup,
        "delta_rup_minus_baseline": delta,
        "success_gates": gates,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RUP-Compass 高精度TTA对比报告",
        "",
        "## 成功判据",
        "",
        "| 判据 | 是否通过 |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| {key} | {int(value)} |")
    lines.extend([
        "",
        "## 核心指标",
        "",
        "| 指标 | FedCompass | RUP | RUP-FedCompass |",
        "|---|---:|---:|---:|",
    ])
    for key in CORE_KEYS:
        lines.append(
            f"| {key} | {_format_value(baseline.get(key))} | "
            f"{_format_value(rup.get(key))} | {_format_value(delta.get(key))} |"
        )
    lines.extend(["", "## RUP观测", "", "```json", json.dumps({
        "decision": rup.get("rup_observation", {}),
        "outcome": rup.get("rup_outcomes", {}),
        "terminal": rup.get("rup_terminal", {}),
        "client_fairness": rup.get("rup_client_fairness", {}),
        "group_admission": rup.get("rup_group_admission", {}),
        "training": rup.get("rup_training", {}),
    }, indent=2, ensure_ascii=False), "```", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
