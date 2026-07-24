"""Fixed-trajectory replay for bounded communication-amortized local work.

This replay does not claim counterfactual accuracy.  It asks a narrower
question: on already observed mature, healthy and deadline-safe decisions,
how much extra aggregated work could be obtained by increasing Q while keeping
the predicted and safe marginal time inside the existing cadence budget?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else default


def replay(run_dir: Path, *, factors: list[float]) -> dict:
    routing = _rows(run_dir / "reason_aware_routing_shadow_trace.csv")
    outcomes = {
        row["decision_id"]: row
        for row in _rows(run_dir / "state_dispatch_outcome_trace.csv")
    }
    aggregated = [
        row for row in outcomes.values() if int(_number(row, "aggregated")) == 1
    ]
    base_total = sum(int(_number(row, "dispatched_q")) for row in aggregated)
    base_by_client: dict[str, int] = defaultdict(int)
    for row in aggregated:
        base_by_client[row["client_id"]] += int(
            _number(row, "dispatched_q")
        )

    candidates = []
    for row in routing:
        outcome = outcomes.get(row["decision_id"])
        if (
            outcome is None
            or int(_number(outcome, "aggregated")) != 1
            or int(_number(row, "communication_amortized_q_eligible")) != 1
            or int(_number(row, "predictor_mature")) != 1
            or int(_number(row, "system_healthy")) != 1
            or _number(row, "fair_debt_score") <= 0.0
            or int(_number(row, "communication_amortized_deadline_safe")) != 1
        ):
            continue
        base_q = int(_number(row, "v23_q"))
        shadow_q = int(_number(row, "communication_amortized_q"))
        if shadow_q <= base_q:
            continue
        candidates.append((row, outcome, base_q, shadow_q))

    scenarios = []
    for factor in factors:
        extra_total = 0
        extra_by_client: dict[str, int] = defaultdict(int)
        decisions = []
        for row, outcome, base_q, shadow_q in candidates:
            capped_q = min(shadow_q, max(base_q, int(base_q * factor)))
            scale = (capped_q - base_q) / (shadow_q - base_q)
            added_predicted = scale * _number(
                row, "communication_amortized_added_predicted_duration"
            )
            added_safe = scale * _number(
                row, "communication_amortized_added_safe_duration"
            )
            extra = capped_q - base_q
            extra_total += extra
            extra_by_client[row["client_id"]] += extra
            decisions.append({
                "decision_id": row["decision_id"],
                "client_id": row["client_id"],
                "virtual_time": _number(row, "virtual_time"),
                "mode": row["v23_mode"],
                "group_id": int(_number(row, "v23_group_id", -1)),
                "base_q": base_q,
                "bounded_q": capped_q,
                "added_q": extra,
                "added_predicted_duration": added_predicted,
                "added_safe_duration": added_safe,
                "original_aggregated": True,
                "original_aggregation_time": _number(
                    outcome, "aggregation_time"
                ),
                "original_staleness": int(
                    _number(outcome, "aggregation_staleness")
                ),
            })
        projected_total = base_total + extra_total
        slow_clients = {"client_5", "client_6"}
        base_slow = sum(base_by_client.get(client, 0) for client in slow_clients)
        projected_slow = base_slow + sum(
            extra_by_client.get(client, 0) for client in slow_clients
        )
        scenarios.append({
            "q_cap_factor": factor,
            "candidate_count": len(decisions),
            "added_aggregated_q": extra_total,
            "relative_total_work_gain": (
                extra_total / base_total if base_total else 0.0
            ),
            "base_slow_work_share": (
                base_slow / base_total if base_total else 0.0
            ),
            "projected_slow_work_share": (
                projected_slow / projected_total if projected_total else 0.0
            ),
            "max_added_predicted_duration": max(
                (item["added_predicted_duration"] for item in decisions),
                default=0.0,
            ),
            "max_added_safe_duration": max(
                (item["added_safe_duration"] for item in decisions),
                default=0.0,
            ),
            "decisions": decisions,
        })

    return {
        "run_dir": str(run_dir),
        "method": "fixed_trajectory_bounded_communication_amortized_q",
        "scope": (
            "mature + system_healthy + contribution_debt_positive + "
            "originally_aggregated + deadline_safe"
        ),
        "base_aggregated_q": base_total,
        "base_aggregated_q_by_client": dict(sorted(base_by_client.items())),
        "eligible_decisions": len(candidates),
        "scenarios": scenarios,
        "limitations": [
            "Fixed-trajectory replay cannot establish counterfactual accuracy.",
            "Aggregation time and staleness are retained only as observed facts.",
            "An online Shadow/Apply run is required to validate changed arrivals.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--factors", nargs="+", type=float, default=[1.5, 2.0, 3.0]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir, factors=args.factors)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
