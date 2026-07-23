"""Offline feasibility audit for Utility-Paced Effective Service V2.4.

The analyzer keeps the completed V2.3 mode/group trajectory fixed.  It only
tests whether an aged, communication-bound client could carry more local work
inside a small marginal-time budget.  It also audits whether client-level
statistical utility is observable in the completed run.

This is not an accuracy replay and must not be used as an Apply result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from su_compass.scheduling.policies.reason_aware_routing import (
    classify_slow_cause,
)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict, field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else default


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _utility_observability(run_dir: Path) -> dict:
    candidates = [
        run_dir / "rup_local_training_trace.csv",
        run_dir / "local_training_utility_trace.csv",
        run_dir / "client_utility_trace.csv",
    ]
    present = [path.name for path in candidates if path.exists()]
    required = {"client_id", "decision_id", "loss_before", "loss_after"}
    usable = []
    for path in candidates:
        if not path.exists():
            continue
        rows = _read(path)
        if rows and required.issubset(rows[0]):
            usable.append(path.name)
    return {
        "client_statistical_utility_observable": bool(usable),
        "usable_artifacts": usable,
        "related_artifacts_present": present,
        "required_fields": sorted(required),
        "reason": (
            "client_loss_before_after_available"
            if usable
            else "missing_client_loss_before_after_do_not_attribute_global_test_loss"
        ),
    }


def analyze(
    run_dir: Path,
    *,
    rhythm_target: float,
    age_periods: float,
    marginal_time_ratio: float,
    communication_ratio_gate: float,
    cadence_median_ratio: float,
    cadence_max_ratio: float,
    slow_clients: set[str],
) -> dict:
    decisions = sorted(
        _read(run_dir / "lyapunov_decision_trace.csv"),
        key=lambda row: _number(row, "virtual_time"),
    )
    curves: dict[str, list[dict]] = defaultdict(list)
    for row in _read(run_dir / "state_time_trace.csv"):
        curves[row["decision_id"]].append(row)
    q_rows: dict[tuple[str, int], dict] = {}
    for row in _read(run_dir / "effective_service_q_shadow_trace.csv"):
        q_rows[(row["decision_id"], int(_number(row, "group_id", -1)))] = row
    aggregations = sorted(
        _read(run_dir / "aggregation_trace.csv"),
        key=lambda row: _number(row, "virtual_time"),
    )
    dispatches = _read(run_dir / "dispatch_decision_trace.csv")

    age_threshold = age_periods * rhythm_target
    marginal_time_budget = marginal_time_ratio * rhythm_target
    last_service: dict[str, float] = defaultdict(float)
    aggregation_times: list[float] = []
    aggregation_intervals: list[float] = []
    agg_cursor = 0
    records = []
    recommendations: dict[str, int] = {}
    cause_records = []

    for decision in decisions:
        now = _number(decision, "virtual_time")
        while (
            agg_cursor < len(aggregations)
            and _number(aggregations[agg_cursor], "virtual_time") <= now
        ):
            aggregation = aggregations[agg_cursor]
            agg_time = _number(aggregation, "virtual_time")
            if aggregation_times:
                aggregation_intervals.append(agg_time - aggregation_times[-1])
            aggregation_times.append(agg_time)
            for client_id in json.loads(aggregation["per_client_local_steps"]):
                last_service[client_id] = agg_time
            agg_cursor += 1

        decision_id = decision["decision_id"]
        client_id = decision["client_id"]
        mode = decision.get("recommended_mode", "")
        base_q = int(_number(decision, "recommended_q", -1))
        applied = int(_number(decision, "recommendation_applied", 0)) == 1
        points = {int(_number(row, "q")): row for row in curves.get(decision_id, [])}
        base = points.get(base_q)
        if not applied or mode not in {"join", "create"} or base is None:
            continue
        cause = classify_slow_cause(SimpleNamespace(
            predicted_duration=_number(base, "predicted_duration"),
            compute_duration=_number(base, "compute_duration"),
            communication_duration=_number(base, "communication_duration"),
            availability_duration=_number(base, "availability_duration"),
            availability_risk_duration=_number(base, "availability_risk_duration"),
            spike_duration=_number(base, "spike_duration"),
            num_reports=int(_number(base, "num_reports")),
            used_fallback=bool(int(_number(base, "used_fallback", 0))),
            predictor_source=base.get("predictor_source", ""),
        ))
        cause_records.append({
            "decision_id": decision_id,
            "client_id": client_id,
            "slow_cause": cause.label,
            "confidence": cause.confidence,
            "compute_ratio": cause.compute_ratio,
            "communication_ratio": cause.communication_ratio,
            "availability_ratio": cause.availability_ratio,
            "spike_ratio": cause.spike_ratio,
        })

        recent = aggregation_intervals[-4:]
        recent_median = statistics.median(recent) if recent else math.inf
        recent_max = max(recent) if recent else math.inf
        cadence_healthy = bool(
            recent
            and recent_median <= cadence_median_ratio * rhythm_target
            and recent_max <= cadence_max_ratio * rhythm_target
        )
        service_age = max(0.0, now - last_service[client_id])
        communication_ratio = (
            _number(base, "communication_duration")
            / max(_number(base, "predicted_duration"), 1e-12)
        )
        group_id = int(_number(decision, "recommended_group_id", -1))
        group = q_rows.get((decision_id, group_id)) if mode == "join" else None
        deadline = _number(group, "deadline", math.inf) if group else math.inf
        safe_frontier = _number(group, "safe_frontier", -math.inf) if group else -math.inf

        eligible_gate = bool(
            service_age >= age_threshold
            and cadence_healthy
            and communication_ratio >= communication_ratio_gate
            and not bool(int(_number(base, "used_fallback", 0)))
        )
        candidates = []
        if eligible_gate:
            for q, point in points.items():
                if q < base_q or bool(int(_number(point, "used_fallback", 0))):
                    continue
                added_predicted = (
                    _number(point, "predicted_duration")
                    - _number(base, "predicted_duration")
                )
                added_safe = _number(point, "safe_duration") - _number(base, "safe_duration")
                if (
                    added_predicted > marginal_time_budget
                    or added_safe > marginal_time_budget
                ):
                    continue
                if mode == "join" and max(
                    safe_frontier, _number(point, "safe_finish_time")
                ) > deadline:
                    continue
                candidates.append(point)

        selected = max(candidates, key=lambda row: int(_number(row, "q"))) if candidates else base
        recommended_q = int(_number(selected, "q"))
        if recommended_q > base_q:
            recommendations[decision_id] = recommended_q
        records.append({
            "decision_id": decision_id,
            "virtual_time": now,
            "client_id": client_id,
            "mode": mode,
            "group_id": group_id,
            "rhythm_debt": _number(decision, "rhythm_debt"),
            "service_age": service_age,
            "service_age_periods": service_age / rhythm_target,
            "recent_cadence_median": recent_median,
            "recent_cadence_max": recent_max,
            "cadence_healthy": cadence_healthy,
            "communication_ratio": communication_ratio,
            "eligible_gate": eligible_gate,
            "base_q": base_q,
            "recommended_q": recommended_q,
            "added_q": recommended_q - base_q,
            "added_predicted_duration": (
                _number(selected, "predicted_duration")
                - _number(base, "predicted_duration")
            ),
            "added_safe_duration": (
                _number(selected, "safe_duration")
                - _number(base, "safe_duration")
            ),
            "deadline": deadline if math.isfinite(deadline) else None,
            "safe_finish": _number(selected, "safe_finish_time"),
            "deadline_safe": (
                max(safe_frontier, _number(selected, "safe_finish_time")) <= deadline
                if mode == "join" else True
            ),
        })

    aggregated_decisions = set()
    aggregated_work = 0
    aggregated_slow_work = 0
    for aggregation in aggregations:
        qs = json.loads(aggregation["per_client_local_steps"])
        ids = json.loads(aggregation["per_client_decision_id"])
        for client_id, q in qs.items():
            aggregated_work += int(q)
            if client_id in slow_clients:
                aggregated_slow_work += int(q)
            decision_id = ids.get(client_id)
            if decision_id:
                aggregated_decisions.add(decision_id)

    terminal_base_work = aggregated_work
    terminal_base_slow_work = aggregated_slow_work
    terminal_proposed_work = aggregated_work
    terminal_proposed_slow_work = aggregated_slow_work
    pending = []
    for dispatch in dispatches:
        decision_id = dispatch["decision_id"]
        if decision_id in aggregated_decisions:
            continue
        client_id = dispatch["client_id"]
        q = int(_number(dispatch, "assigned_local_steps"))
        proposed_q = recommendations.get(decision_id, q)
        terminal_base_work += q
        terminal_proposed_work += proposed_q
        if client_id in slow_clients:
            terminal_base_slow_work += q
            terminal_proposed_slow_work += proposed_q
        pending.append({
            "decision_id": decision_id,
            "client_id": client_id,
            "group_id": int(_number(dispatch, "assigned_group", -1)),
            "base_q": q,
            "recommended_q": proposed_q,
        })

    changed = [record for record in records if record["added_q"] > 0]
    per_client = {}
    for client_id in sorted({record["client_id"] for record in records}):
        selected = [record for record in records if record["client_id"] == client_id]
        per_client[client_id] = {
            "eligible": sum(record["eligible_gate"] for record in selected),
            "changed": sum(record["added_q"] > 0 for record in selected),
            "added_q": sum(record["added_q"] for record in selected),
            "max_added_predicted_duration": max(
                (record["added_predicted_duration"] for record in selected),
                default=0.0,
            ),
        }
    cause_by_client = {}
    for client_id in sorted({row["client_id"] for row in cause_records}):
        selected = [row for row in cause_records if row["client_id"] == client_id]
        counts: dict[str, int] = defaultdict(int)
        for row in selected:
            counts[row["slow_cause"]] += 1
        cause_by_client[client_id] = {
            "decisions": len(selected),
            "cause_counts": dict(sorted(counts.items())),
            "median_communication_ratio": statistics.median(
                row["communication_ratio"] for row in selected
            ),
            "median_compute_ratio": statistics.median(
                row["compute_ratio"] for row in selected
            ),
        }

    return {
        "scope": "fixed_v2_3_mode_group_one_step_q_counterfactual_not_accuracy_replay",
        "configuration": {
            "rhythm_target": rhythm_target,
            "age_periods": age_periods,
            "age_threshold": age_threshold,
            "marginal_time_ratio": marginal_time_ratio,
            "marginal_time_budget": marginal_time_budget,
            "communication_ratio_gate": communication_ratio_gate,
            "cadence_median_ratio": cadence_median_ratio,
            "cadence_max_ratio": cadence_max_ratio,
            "slow_clients": sorted(slow_clients),
        },
        "statistical_utility": _utility_observability(run_dir),
        "reason_aware_classification": {
            "per_client": cause_by_client,
            "extreme_communication_clients": sorted({
                row["client_id"] for row in cause_records
                if row["slow_cause"] == "extreme_communication_bound"
            }),
            "records": cause_records,
        },
        "counterfactual": {
            "applied_decisions_analyzed": len(records),
            "eligible_decisions": sum(record["eligible_gate"] for record in records),
            "changed_decisions": len(changed),
            "added_q": sum(record["added_q"] for record in changed),
            "max_added_predicted_duration": max(
                (record["added_predicted_duration"] for record in changed),
                default=0.0,
            ),
            "max_added_safe_duration": max(
                (record["added_safe_duration"] for record in changed),
                default=0.0,
            ),
            "all_join_recommendations_deadline_safe": all(
                record["deadline_safe"] for record in changed
            ),
            "per_client": per_client,
        },
        "terminal_accounting": {
            "aggregated_work_observed": aggregated_work,
            "aggregated_slow_work_observed": aggregated_slow_work,
            "aggregated_slow_share_observed": (
                aggregated_slow_work / aggregated_work if aggregated_work else 0.0
            ),
            "terminal_base_work_if_pending_settles": terminal_base_work,
            "terminal_base_slow_work_if_pending_settles": terminal_base_slow_work,
            "terminal_base_slow_share_if_pending_settles": (
                terminal_base_slow_work / terminal_base_work
                if terminal_base_work else 0.0
            ),
            "terminal_proposed_work_if_pending_settles": terminal_proposed_work,
            "terminal_proposed_slow_work_if_pending_settles": terminal_proposed_slow_work,
            "terminal_proposed_slow_share_if_pending_settles": (
                terminal_proposed_slow_work / terminal_proposed_work
                if terminal_proposed_work else 0.0
            ),
            "pending_decisions": pending,
        },
        "changed_records": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--rhythm_target", type=float, default=16.4)
    parser.add_argument("--age_periods", type=float, default=4.0)
    parser.add_argument("--marginal_time_ratio", type=float, default=0.2)
    parser.add_argument("--communication_ratio_gate", type=float, default=0.8)
    parser.add_argument("--cadence_median_ratio", type=float, default=1.25)
    parser.add_argument("--cadence_max_ratio", type=float, default=2.0)
    parser.add_argument("--slow_clients", default="client_5,client_6")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.run_dir,
        rhythm_target=args.rhythm_target,
        age_periods=args.age_periods,
        marginal_time_ratio=args.marginal_time_ratio,
        communication_ratio_gate=args.communication_ratio_gate,
        cadence_median_ratio=args.cadence_median_ratio,
        cadence_max_ratio=args.cadence_max_ratio,
        slow_clients={item.strip() for item in args.slow_clients.split(",") if item.strip()},
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
