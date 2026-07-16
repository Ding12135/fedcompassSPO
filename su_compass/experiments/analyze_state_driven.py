"""Automatic summary for State-Driven FedCompass runs."""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str, default=0.0) -> float:
    value = row.get(key, "")
    return default if value in ("", None) else float(value)


def _percentile(values: list[float], quantile: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def _tta(times, accuracy, threshold, mode="first"):
    if mode == "first":
        return next((times[i] for i, value in enumerate(accuracy) if value >= threshold), None)
    if mode == "three":
        return next((times[i] for i in range(2, len(accuracy)) if min(accuracy[i-2:i+1]) >= threshold), None)
    return next((times[i] for i in range(2, len(accuracy)) if statistics.mean(accuracy[i-2:i+1]) >= threshold), None)


def summarize(directory: Path) -> dict:
    config_path = directory / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    qmin = int(config.get("min_local_steps", 40))
    qmax = int(config.get("max_local_steps", 200))
    scheduler = _rows(directory / "scheduler_trace.csv")
    groups = _rows(directory / "group_trace.csv")
    aggregations = _rows(directory / "aggregation_trace.csv")
    evals = _rows(directory / "global_eval_trace.csv")
    reports = _rows(directory / "summary" / "all_round_reports.csv")
    state_time = _rows(directory / "state_time_trace.csv")
    joint = _rows(directory / "joint_group_q_trace.csv")
    creations = _rows(directory / "state_group_creation_trace.csv")

    report_by_id = {row.get("decision_id", ""): row for row in reports if row.get("decision_id")}
    # Prediction error must be measured at the Q that was actually dispatched.
    # In shadow/fallback modes the state recommendation can use a different Q,
    # so evaluating the recommended row against the actual duration is invalid.
    predictions_by_id = {}
    for row in state_time:
        decision_id = row.get("decision_id", "")
        actual = report_by_id.get(decision_id)
        if not actual or int(float(row.get("q", -1))) != int(float(actual.get("local_steps", -2))):
            continue
        current = predictions_by_id.get(decision_id)
        if current is None or row.get("is_state_selected_q") == "1":
            predictions_by_id[decision_id] = row
    applied_predictions = list(predictions_by_id.values())
    prediction_errors = []
    prediction_actuals = []
    safe_covered = []
    for row in applied_predictions:
        actual = report_by_id.get(row.get("decision_id", ""))
        if actual is None:
            continue
        actual_duration = _f(actual, "round_time")
        prediction_errors.append(actual_duration - _f(row, "predicted_duration"))
        prediction_actuals.append(actual_duration)
        safe_covered.append(actual_duration <= _f(row, "safe_duration"))

    alignment = [
        abs(_f(row, "finish_time") - _f(row, "target_arrival_time"))
        for row in reports if row.get("target_arrival_time") not in ("", None)
    ]
    staleness = []
    aggregated_steps = 0.0
    staleness_weighted_steps = 0.0
    for row in aggregations:
        stale = ast.literal_eval(row.get("per_client_staleness", "{}"))
        steps = ast.literal_eval(row.get("per_client_local_steps", "{}"))
        for client_id, value in stale.items():
            staleness.append(float(value))
            q = float(steps.get(client_id, 0.0))
            aggregated_steps += q
            staleness_weighted_steps += q / (1.0 + float(value))

    final_time = _f(evals[-1], "virtual_time") if evals else (
        max((_f(x, "virtual_time") for x in scheduler), default=0.0)
    )
    accuracy = [_f(row, "test_accuracy") for row in evals]
    times = [_f(row, "virtual_time") for row in evals]
    group_sizes = [_f(row, "group_size") for row in groups]
    dispatched_q = [_f(row, "local_steps") for row in reports]
    target_hits = 0
    target_total = 0
    for row in reports:
        if row.get("target_arrival_time") in ("", None) or row.get("latest_arrival_time") in ("", None):
            continue
        expected = _f(row, "target_arrival_time")
        latest = _f(row, "latest_arrival_time")
        tolerance = 0.05 * max(0.0, latest - expected)
        target_total += 1
        target_hits += abs(_f(row, "finish_time") - expected) <= tolerance

    result = {
        "prediction": {
            "count": len(prediction_errors),
            "mae": statistics.mean(abs(x) for x in prediction_errors) if prediction_errors else None,
            "mape": statistics.mean(
                abs(error) / max(actual, 1e-8)
                for error, actual in zip(prediction_errors, prediction_actuals)
            ) if prediction_errors else None,
            "p90_abs_error": _percentile([abs(x) for x in prediction_errors], 0.90),
            "p95_abs_error": _percentile([abs(x) for x in prediction_errors], 0.95),
            "safe_boundary_coverage": sum(safe_covered) / len(safe_covered) if safe_covered else None,
            "fallback_rate": sum(row.get("used_fallback") == "1" for row in applied_predictions) / len(applied_predictions) if applied_predictions else None,
        },
        "scheduling": {
            "mean_actual_alignment_error": statistics.mean(alignment) if alignment else None,
            "median_actual_alignment_error": statistics.median(alignment) if alignment else None,
            "p90_actual_alignment_error": _percentile(alignment, 0.90),
            "target_window_hit_rate": target_hits / target_total if target_total else None,
            "late_rate": sum(int(float(row.get("late", 0))) for row in scheduler) / len(scheduler) if scheduler else None,
            "deadline_trigger_rate": sum(row.get("trigger") == "deadline" for row in groups) / len(groups) if groups else None,
            "group_count": len(groups),
            "mean_group_size": statistics.mean(group_sizes) if group_sizes else None,
            "group_size_std": statistics.pstdev(group_sizes) if group_sizes else None,
            "single_client_group_rate": sum(value == 1 for value in group_sizes) / len(group_sizes) if group_sizes else None,
            "actual_safe_violation_rate": sum(
                _f(row, "finish_time") > _f(row, "latest_arrival_time")
                for row in reports if row.get("latest_arrival_time") not in ("", None)
            ) / max(1, sum(
                row.get("latest_arrival_time") not in ("", None) for row in reports
            )),
        },
        "state_control": {
            "decision_count": len(joint),
            "active_rate": sum(row.get("state_control_active") == "1" for row in joint) / len(joint) if joint else None,
            "fallback_to_fedcompass_rate": sum(row.get("fallback_to_fedcompass") == "1" for row in joint) / len(joint) if joint else None,
            "no_aligned_safe_existing_group_rate": sum(row.get("all_existing_groups_infeasible") == "1" for row in joint) / len(joint) if joint else None,
            "group_change_rate": sum(row.get("group_changed") == "1" for row in joint) / len(joint) if joint else None,
            "q_change_rate": sum(row.get("q_changed") == "1" for row in joint) / len(joint) if joint else None,
            "non_monotonic_curve_rate": sum(row.get("curve_monotonic") == "0" for row in joint) / len(joint) if joint else None,
            "mean_safe_candidates": statistics.mean(_f(row, "num_deadline_safe_candidates") for row in joint) if joint else None,
            "mean_aligned_safe_candidates": statistics.mean(_f(row, "num_target_aligned_candidates") for row in joint) if joint else None,
        },
        "group_creation": {
            "count": len(creations),
            "applied_count": sum(row.get("applied") == "1" for row in creations),
            "fallback_rate": sum(row.get("used_fallback") == "1" for row in creations) / len(creations) if creations else None,
            "safe_window_overflow_count": sum(row.get("safe_window_exceeds_cap") == "1" for row in creations),
            "mean_expected_shift": statistics.mean(_f(row, "expected_shift") for row in creations) if creations else None,
            "mean_latest_shift": statistics.mean(_f(row, "latest_shift") for row in creations) if creations else None,
        },
        "staleness": {
            "mean": statistics.mean(staleness) if staleness else None,
            "median": statistics.median(staleness) if staleness else None,
            "p90": _percentile(staleness, 0.90),
            "max": max(staleness) if staleness else None,
        },
        "workload": {
            "mean_dispatched_q": statistics.mean(dispatched_q) if dispatched_q else None,
            "q_std": statistics.pstdev(dispatched_q) if dispatched_q else None,
            "qmin_rate": sum(value == qmin for value in dispatched_q) / len(dispatched_q) if dispatched_q else None,
            "qmax_rate": sum(value == qmax for value in dispatched_q) / len(dispatched_q) if dispatched_q else None,
            "dispatched_steps": sum(_f(row, "local_steps") for row in reports),
            "completed_steps": sum(_f(row, "local_steps") for row in scheduler),
            "aggregated_steps": aggregated_steps,
            "aggregated_steps_per_virtual_time": aggregated_steps / max(final_time, 1e-8),
            "staleness_weighted_steps": staleness_weighted_steps,
            "staleness_weighted_steps_per_virtual_time": staleness_weighted_steps / max(final_time, 1e-8),
        },
        "learning": {
            "max_accuracy": max(accuracy) if accuracy else None,
            "final_accuracy": accuracy[-1] if accuracy else None,
            "last10_mean": statistics.mean(accuracy[-10:]) if accuracy else None,
            "last10_std": statistics.pstdev(accuracy[-10:]) if accuracy else None,
            "tta_first_hit": {str(t): _tta(times, accuracy, t, "first") for t in (40,45,50,55,60,62,63,64,65)},
            "tta_3_consecutive": {str(t): _tta(times, accuracy, t, "three") for t in (40,45,50,55,60,62,63,64,65)},
            "tta_moving_average": {str(t): _tta(times, accuracy, t, "moving") for t in (40,45,50,55,60,62,63,64,65)},
        },
    }
    return result


def _write_outcomes(directory: Path) -> None:
    reports = _rows(directory / "summary" / "all_round_reports.csv")
    aggregations = _rows(directory / "aggregation_trace.csv")
    contribution = {}
    for row in aggregations:
        decision_ids = ast.literal_eval(row.get("per_client_decision_id", "{}") or "{}")
        stale = ast.literal_eval(row.get("per_client_staleness", "{}") or "{}")
        for client_id, decision_id in decision_ids.items():
            if decision_id:
                contribution[decision_id] = {
                    "aggregated": 1,
                    "aggregation_time": _f(row, "virtual_time"),
                    "aggregation_staleness": stale.get(client_id, ""),
                }
    fields = [
        "decision_id", "client_id", "profile_type", "dispatched_group_id", "dispatched_q",
        "dispatch_time", "actual_finish_time", "actual_duration",
        "target_arrival_time", "latest_arrival_time", "actual_alignment_error",
        "actual_safe_violation", "late", "upload_group_id", "next_group_id",
        "model_version_at_dispatch", "model_version_at_upload",
        "aggregation_staleness", "aggregated", "aggregation_time",
    ]
    rows = []
    for report in reports:
        decision_id = report.get("decision_id", "")
        target = report.get("target_arrival_time", "")
        latest = report.get("latest_arrival_time", "")
        finish = _f(report, "finish_time")
        agg = contribution.get(decision_id, {})
        rows.append({
            "decision_id": decision_id, "client_id": report.get("client_id", ""),
            "profile_type": report.get("profile_type", ""),
            "dispatched_group_id": report.get("upload_group_id", ""),
            "dispatched_q": report.get("local_steps", ""),
            "dispatch_time": report.get("dispatch_time", ""),
            "actual_finish_time": report.get("finish_time", ""),
            "actual_duration": report.get("round_time", ""),
            "target_arrival_time": target, "latest_arrival_time": latest,
            "actual_alignment_error": abs(finish - float(target)) if target not in ("", None) else "",
            "actual_safe_violation": int(finish > float(latest)) if latest not in ("", None) else "",
            "late": report.get("late", ""), "upload_group_id": report.get("upload_group_id", ""),
            "next_group_id": report.get("next_group_id", ""),
            "model_version_at_dispatch": report.get("model_version_at_dispatch", ""),
            "model_version_at_upload": report.get("model_version_at_upload", ""),
            "aggregation_staleness": agg.get("aggregation_staleness", report.get("aggregation_staleness", "")),
            "aggregated": agg.get("aggregated", 0), "aggregation_time": agg.get("aggregation_time", ""),
        })
    if rows:
        with (directory / "state_dispatch_outcome_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)


def write_summary(directory: Path) -> dict:
    _write_outcomes(directory)
    result = summarize(directory)
    (directory / "summary_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def write_ablation_comparison(output_root: Path, presets: list[str], seed: int) -> Path:
    """Write a compact, paper-facing comparison from completed preset summaries."""
    fields = [
        "preset", "seed", "max_accuracy", "final_accuracy", "last10_mean",
        "last10_std", "tta62", "tta64", "tta65", "late_rate",
        "deadline_trigger_rate", "mean_group_size", "group_count",
        "mean_dispatched_q", "aggregated_steps_per_virtual_time",
        "mean_staleness", "prediction_mae", "safe_boundary_coverage",
        "state_active_rate", "state_fallback_rate", "no_feasible_group_rate",
    ]
    rows = []
    for preset in presets:
        directory = output_root / preset / f"seed{seed}"
        summary_path = directory / "summary_metrics.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "preset": preset, "seed": seed,
            "max_accuracy": data["learning"]["max_accuracy"],
            "final_accuracy": data["learning"]["final_accuracy"],
            "last10_mean": data["learning"]["last10_mean"],
            "last10_std": data["learning"]["last10_std"],
            "tta62": data["learning"]["tta_first_hit"]["62"],
            "tta64": data["learning"]["tta_first_hit"]["64"],
            "tta65": data["learning"]["tta_first_hit"]["65"],
            "late_rate": data["scheduling"]["late_rate"],
            "deadline_trigger_rate": data["scheduling"]["deadline_trigger_rate"],
            "mean_group_size": data["scheduling"]["mean_group_size"],
            "group_count": data["scheduling"]["group_count"],
            "mean_dispatched_q": data["workload"]["mean_dispatched_q"],
            "aggregated_steps_per_virtual_time": data["workload"]["aggregated_steps_per_virtual_time"],
            "mean_staleness": data["staleness"]["mean"],
            "prediction_mae": data["prediction"]["mae"],
            "safe_boundary_coverage": data["prediction"]["safe_boundary_coverage"],
            "state_active_rate": data["state_control"]["active_rate"],
            "state_fallback_rate": data["state_control"]["fallback_to_fedcompass_rate"],
            "no_feasible_group_rate": data["state_control"]["no_aligned_safe_existing_group_rate"],
        })
    path = output_root / f"ablation_comparison_seed{seed}.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
