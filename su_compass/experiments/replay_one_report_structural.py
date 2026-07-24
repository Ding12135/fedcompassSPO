"""Offline audit of the one-report structural cold-start prior.

This keeps the completed V2.3 trajectory fixed.  It compares the prediction
available immediately after a client's first report with the actual duration
of its next dispatched job.  It does not replay model accuracy or claim an
Apply/TTA result.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from su_compass.scheduling.policies.one_report_structural import (
    predict_one_report_structural,
)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict, field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else default


def analyze(
    run_dir: Path,
    *,
    communication_ratio_gate: float = 0.95,
    safety_fraction: float = 0.10,
) -> dict:
    scheduler = _read(run_dir / "scheduler_trace.csv")
    by_client: dict[str, list[dict]] = defaultdict(list)
    for row in scheduler:
        by_client[row["client_id"]].append(row)
    for rows in by_client.values():
        rows.sort(key=lambda row: _number(row, "virtual_time"))

    records = []
    for client_id, rows in sorted(by_client.items()):
        if len(rows) < 2:
            continue
        first, second = rows[0], rows[1]
        observed_q = int(_number(first, "local_steps"))
        next_q = int(_number(second, "local_steps"))
        actual = _number(second, "round_time")
        fedcompass = _number(first, "round_time") / observed_q * next_q
        prediction = predict_one_report_structural(
            q=next_q,
            observed_q=observed_q,
            observed_round_duration=_number(first, "round_time"),
            observed_compute_duration=_number(first, "train_time"),
            observed_communication_duration=(
                _number(first, "download_time") + _number(first, "upload_time")
            ),
            num_reports=1,
            communication_ratio_gate=communication_ratio_gate,
            safety_fraction=safety_fraction,
        )
        qmax_prediction = predict_one_report_structural(
            q=200,
            observed_q=observed_q,
            observed_round_duration=_number(first, "round_time"),
            observed_compute_duration=_number(first, "train_time"),
            observed_communication_duration=(
                _number(first, "download_time") + _number(first, "upload_time")
            ),
            num_reports=1,
            communication_ratio_gate=communication_ratio_gate,
            safety_fraction=safety_fraction,
        )
        records.append({
            "client_id": client_id,
            "profile_type": first.get("profile_type", ""),
            "observed_q": observed_q,
            "next_q": next_q,
            "communication_ratio": prediction.communication_ratio,
            "eligible": prediction.eligible,
            "fedcompass_duration": fedcompass,
            "structural_duration": prediction.predicted_duration,
            "structural_safe_duration": prediction.safe_duration,
            "actual_next_duration": actual,
            "fedcompass_abs_error": abs(fedcompass - actual),
            "structural_abs_error": abs(prediction.predicted_duration - actual),
            "fedcompass_safe_hit": fedcompass >= actual,
            "structural_safe_hit": prediction.safe_duration >= actual,
            "point_prediction_better": (
                abs(prediction.predicted_duration - actual)
                < abs(fedcompass - actual)
            ),
            "qmax_candidate": 200,
            "qmax_added_q": 200 - next_q,
            "qmax_added_predicted_duration": (
                qmax_prediction.predicted_duration
                - prediction.predicted_duration
            ),
            "qmax_added_safe_duration": (
                qmax_prediction.safe_duration
                - prediction.safe_duration
            ),
            "qmax_counterfactual_actual_duration": (
                actual
                + qmax_prediction.compute_duration
                - prediction.compute_duration
            ),
        })

    eligible = [row for row in records if row["eligible"]]
    slow_anchor = []
    creations = _read(run_dir / "state_group_creation_trace.csv")
    outcomes = {
        row["decision_id"]: row
        for row in _read(run_dir / "state_dispatch_outcome_trace.csv")
    }
    first_rows = {client_id: rows[0] for client_id, rows in by_client.items()}
    for creation in creations:
        decision_id = creation["decision_id"]
        if int(_number(creation, "num_reports", -1)) != 1:
            continue
        client_id = creation["client_id"]
        first = first_rows.get(client_id)
        outcome = outcomes.get(decision_id)
        if first is None or outcome is None:
            continue
        q = int(_number(creation, "state_assigned_q"))
        prediction = predict_one_report_structural(
            q=q,
            observed_q=int(_number(first, "local_steps")),
            observed_round_duration=_number(first, "round_time"),
            observed_compute_duration=_number(first, "train_time"),
            observed_communication_duration=(
                _number(first, "download_time") + _number(first, "upload_time")
            ),
            num_reports=1,
            communication_ratio_gate=communication_ratio_gate,
            safety_fraction=safety_fraction,
        )
        if not prediction.eligible:
            continue
        now = _number(creation, "virtual_time")
        actual_finish = _number(outcome, "actual_finish_time")
        slow_anchor.append({
            "decision_id": decision_id,
            "client_id": client_id,
            "group_id": int(_number(creation, "new_group_id", -1)),
            "q_unchanged": q,
            "original_expected_time": _number(creation, "state_expected_time"),
            "original_latest_time": _number(creation, "state_latest_time"),
            "structural_expected_time": now + prediction.predicted_duration,
            "structural_latest_time": now + prediction.safe_duration,
            "actual_finish_time": actual_finish,
            "original_late": actual_finish > _number(
                creation, "state_latest_time"
            ),
            "structural_safe_hit": (
                actual_finish <= now + prediction.safe_duration
            ),
            "extra_expected_wait": (
                now + prediction.predicted_duration
                - _number(creation, "state_expected_time")
            ),
            "extra_latest_wait": (
                now + prediction.safe_duration
                - _number(creation, "state_latest_time")
            ),
        })

    def mean(field: str) -> float:
        return statistics.mean(row[field] for row in eligible) if eligible else 0.0

    eligible_qmax = [
        row for row in eligible
        if row["qmax_added_q"] > 0
        and row["qmax_added_predicted_duration"] <= 0.2 * 16.4
        and row["qmax_added_safe_duration"] <= 0.2 * 16.4
    ]
    aggregated_work = 0
    aggregated_slow_work = 0
    slow_ids = {"client_5", "client_6"}
    for aggregation in _read(run_dir / "aggregation_trace.csv"):
        for client_id, q in json.loads(
            aggregation["per_client_local_steps"]
        ).items():
            aggregated_work += int(q)
            if client_id in slow_ids:
                aggregated_slow_work += int(q)
    added_aggregated_q = sum(
        row["qmax_added_q"] for row in eligible_qmax
        if row["client_id"] in slow_ids
    )

    return {
        "scope": (
            "fixed_v2_3_trajectory_one_report_prediction_replay_"
            "not_accuracy_or_tta_replay"
        ),
        "configuration": {
            "communication_ratio_gate": communication_ratio_gate,
            "safety_fraction": safety_fraction,
        },
        "summary": {
            "clients_with_second_report": len(records),
            "eligible_clients": len(eligible),
            "eligible_client_ids": [row["client_id"] for row in eligible],
            "fedcompass_mae": mean("fedcompass_abs_error"),
            "structural_mae": mean("structural_abs_error"),
            "mae_reduction_fraction": (
                1.0 - mean("structural_abs_error") / mean("fedcompass_abs_error")
                if eligible and mean("fedcompass_abs_error") > 0 else 0.0
            ),
            "fedcompass_safe_coverage": (
                statistics.mean(row["fedcompass_safe_hit"] for row in eligible)
                if eligible else 0.0
            ),
            "structural_safe_coverage": (
                statistics.mean(row["structural_safe_hit"] for row in eligible)
                if eligible else 0.0
            ),
            "point_better_rate": (
                statistics.mean(
                    row["point_prediction_better"] for row in eligible
                ) if eligible else 0.0
            ),
        },
        "anchor_counterfactual": {
            "eligible_one_report_anchors": len(slow_anchor),
            "all_structural_safe": all(
                row["structural_safe_hit"] for row in slow_anchor
            ),
            "records": slow_anchor,
            "interpretation": (
                "A corrected long anchor window makes the slow anchor timing "
                "honest. V2.3 cadence would see the long sojourn, but a fixed-"
                "trajectory replay cannot prove downstream join/create choices."
            ),
        },
        "communication_amortized_q": {
            "eligible_records": eligible_qmax,
            "changed_clients": [
                row["client_id"] for row in eligible_qmax
            ],
            "added_aggregated_q": added_aggregated_q,
            "max_added_predicted_duration": max(
                (
                    row["qmax_added_predicted_duration"]
                    for row in eligible_qmax
                ),
                default=0.0,
            ),
            "max_added_safe_duration": max(
                row["qmax_added_safe_duration"]
                for row in eligible_qmax
            ) if eligible_qmax else 0.0,
            "base_slow_work_share": (
                aggregated_slow_work / aggregated_work
                if aggregated_work else 0.0
            ),
            "projected_slow_work_share": (
                (aggregated_slow_work + added_aggregated_q)
                / (aggregated_work + added_aggregated_q)
                if aggregated_work + added_aggregated_q else 0.0
            ),
            "scope": (
                "fixed_trajectory_structural_duration_and_work_accounting_"
                "not_accuracy_or_changed_group_replay"
            ),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--communication_ratio_gate", type=float, default=0.95)
    parser.add_argument("--safety_fraction", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.run_dir,
        communication_ratio_gate=args.communication_ratio_gate,
        safety_fraction=args.safety_fraction,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
