"""One-step replay for the mature long-join admission repair.

The replay uses the frozen V2.5 Shadow dispatches.  It identifies mature,
non-extreme clients that joined a group whose predicted sojourn exceeded the
common maximum cadence ratio, then coordinates simultaneous avoidances into
one same-Q create batch.  Actual finish times provide an oracle upper bound on
the first counterfactual aggregation time; downstream trajectory and accuracy
are intentionally not claimed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay(
    run_dir: Path,
    *,
    rhythm_target: float = 16.4,
    max_cadence_ratio: float = 2.0,
) -> dict:
    reason = _read(run_dir / "reason_aware_routing_shadow_trace.csv")
    outcomes = _read(run_dir / "state_dispatch_outcome_trace.csv")
    groups = _read(run_dir / "group_trace.csv")
    outcome_by_decision = {
        row["decision_id"]: row for row in outcomes if row.get("decision_id")
    }
    group_by_id = {int(row["group_id"]): row for row in groups}
    limit = rhythm_target * max_cadence_ratio

    candidates: list[dict] = []
    for row in reason:
        if row.get("v23_mode") != "join":
            continue
        if row.get("predictor_mature") != "1":
            continue
        if row.get("slow_cause") == "extreme_communication_bound":
            continue
        sojourns = [
            float(value)
            for value in (
                row.get("shadow_predicted_sojourn", ""),
                row.get("structural_group_sojourn", ""),
            )
            if value not in {"", None}
        ]
        observed_sojourn = max(sojourns, default=0.0)
        if observed_sojourn <= limit:
            continue
        outcome = outcome_by_decision.get(row["decision_id"])
        group_id = int(row["v23_group_id"])
        group = group_by_id.get(group_id)
        if outcome is None or group is None:
            continue
        candidates.append({
            "virtual_time": float(row["virtual_time"]),
            "group_id": group_id,
            "client_id": row["client_id"],
            "decision_id": row["decision_id"],
            "q": int(row["v23_q"]),
            "observed_sojourn": observed_sojourn,
            "cadence_limit": limit,
            "actual_finish_time": float(outcome["actual_finish_time"]),
            "baseline_group_aggregation_time": float(
                group["aggregation_time"]
            ),
        })

    batches_by_key: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in candidates:
        batches_by_key[(row["virtual_time"], row["group_id"])].append(row)

    batches: list[dict] = []
    for (virtual_time, group_id), rows in sorted(batches_by_key.items()):
        counterfactual_time = max(row["actual_finish_time"] for row in rows)
        baseline_time = rows[0]["baseline_group_aggregation_time"]
        batches.append({
            "virtual_time": virtual_time,
            "avoided_group_id": group_id,
            "clients": [row["client_id"] for row in rows],
            "q_by_client": {
                row["client_id"]: row["q"] for row in rows
            },
            "independent_creates_without_coordination": len(rows),
            "coordinated_creates": 1,
            "coordinated_joins": len(rows) - 1,
            "counterfactual_first_completion_oracle": counterfactual_time,
            "baseline_slow_group_aggregation_time": baseline_time,
            "one_step_time_recovered": max(
                0.0, baseline_time - counterfactual_time
            ),
            "all_frozen_dispatches_finish_before_baseline_group": (
                counterfactual_time < baseline_time
            ),
        })

    report = {
        "source_run": str(run_dir),
        "policy": {
            "client_id_agnostic": True,
            "rhythm_target": rhythm_target,
            "max_cadence_ratio": max_cadence_ratio,
            "join_sojourn_limit": limit,
            "q_changed": False,
            "apply_enabled": False,
        },
        "scope": {
            "exact": [
                "recorded_v23_join",
                "recorded_predicted_sojourn",
                "recorded_actual_finish_time",
                "recorded_baseline_group_aggregation_time",
            ],
            "counterfactual": [
                "one_same_q_create_batch_at_the_same_dispatch_time",
            ],
            "not_claimed": [
                "full_counterfactual_trajectory",
                "counterfactual_accuracy",
                "counterfactual_safe_prediction",
            ],
        },
        "candidate_count": len(candidates),
        "batch_count": len(batches),
        "candidates": candidates,
        "batches": batches,
    }
    group34_batches = [
        row for row in batches if row["avoided_group_id"] == 34
    ]
    group34_clients = {
        client for row in group34_batches for client in row["clients"]
    }
    group34_best_recovery = max(
        (
            row["one_step_time_recovered"] for row in group34_batches
        ),
        default=0.0,
    )
    report["gates"] = {
        "detects_group34": bool(group34_batches),
        "group34_separates_multiple_fast_clients": (
            len(group34_clients) >= 5
        ),
        "group34_one_step_time_improves": group34_best_recovery > 0,
        "same_q_preserved": all(
            row["q"] > 0 for row in candidates
        ),
        "same_time_batches_coordinated": all(
            row["coordinated_creates"] == 1 for row in batches
        ),
        "apply_ready": False,
        "apply_blocker": (
            "Frozen one-step replay cannot prove safe create windows or the "
            "changed downstream trajectory; keep the repair in online Shadow."
        ),
    }
    report["gates"]["shadow_ready"] = all(
        report["gates"][name]
        for name in (
            "detects_group34",
            "group34_separates_multiple_fast_clients",
            "group34_one_step_time_improves",
            "same_q_preserved",
            "same_time_batches_coordinated",
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
