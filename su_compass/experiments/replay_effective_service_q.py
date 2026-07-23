"""Replay controlled state-alignment Q anchors on a completed run.

This is a one-step counterfactual: it validates candidate coverage, Qmax risk,
and timing alignment without claiming to reproduce the changed future grouping
trajectory or model accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _baseline_medians(run_dir: Path) -> dict[str, int]:
    values: dict[str, list[int]] = defaultdict(list)
    for row in _read(run_dir / "dispatch_decision_trace.csv"):
        values[row["client_id"]].append(int(float(row["assigned_local_steps"])))
    return {client: int(round(statistics.median(qs))) for client, qs in values.items()}


def replay(candidate_run: Path, baseline_run: Path, trust_eta: float) -> dict:
    references = _baseline_medians(baseline_run)
    dispatch = {row["decision_id"]: row for row in _read(candidate_run / "dispatch_decision_trace.csv")}
    outcomes = {row["decision_id"]: row for row in _read(candidate_run / "state_dispatch_outcome_trace.csv")}
    curves: dict[str, list[dict]] = defaultdict(list)
    for row in _read(candidate_run / "state_time_trace.csv"):
        curves[row["decision_id"]].append(row)

    records = []
    for decision_id, row in dispatch.items():
        client_id = row["client_id"]
        if "join" not in row["decision"] or decision_id not in outcomes:
            continue
        if client_id not in references or decision_id not in curves:
            continue
        curve = curves[decision_id]
        reliable = [point for point in curve if (
            point["predictor_source"] == "mature_state"
            and not int(float(point["used_fallback"]))
            and int(float(point["curve_monotonic"]))
        )]
        deadline = float(row["latest_arrival_time"])
        safe = [point for point in reliable if float(point["safe_finish_time"]) <= deadline]
        if not safe:
            continue
        target = float(row["target_arrival_time"])
        reference = references[client_id]
        raw = min(safe, key=lambda point: (
            abs(float(point["predicted_finish_time"]) - target), int(point["q"]),
        ))
        cap = min(200, math.ceil(trust_eta * reference))
        bounded = [point for point in safe if int(point["q"]) <= cap]
        if not bounded:
            continue
        controlled = min(bounded, key=lambda point: (
            abs(float(point["predicted_finish_time"]) - target),
            abs(int(point["q"]) - reference), int(point["q"]),
        ))
        outcome = outcomes[decision_id]
        applied_q = max(1, int(float(outcome["dispatched_q"])))
        actual_duration = float(outcome["actual_duration"])
        now = float(row["virtual_time"])

        def timing(point: dict) -> dict:
            # Same linear counterfactual used by the existing native-Q Shadow.
            finish = now + actual_duration / applied_q * int(point["q"])
            return {
                "q": int(point["q"]), "holding": max(0.0, target - finish),
                "extension": max(0.0, finish - target),
                "deadline_violation": finish > deadline,
            }

        records.append({"raw": timing(raw), "controlled": timing(controlled)})

    def summary(key: str) -> dict:
        selected = [record[key] for record in records]
        count = max(len(selected), 1)
        return {
            "mean_q": sum(row["q"] for row in selected) / count,
            "qmax_share": sum(row["q"] == 200 for row in selected) / count,
            "mean_holding": sum(row["holding"] for row in selected) / count,
            "mean_extension": sum(row["extension"] for row in selected) / count,
            "deadline_violation_rate": sum(row["deadline_violation"] for row in selected) / count,
        }

    return {
        "scope": "one_step_timing_counterfactual_not_accuracy_replay",
        "eligible_join_decisions": len(records), "trust_eta": trust_eta,
        "baseline_q_reference": references,
        "raw_state_alignment": summary("raw"),
        "controlled_alignment": summary("controlled"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_run", type=Path, required=True)
    parser.add_argument("--baseline_run", type=Path, required=True)
    parser.add_argument("--trust_eta", type=float, default=1.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.candidate_run, args.baseline_run, args.trust_eta)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
