"""Unified fixed-trajectory audit for the V2.4 Elastic-Service Shadow.

The report combines every currently replayable component without pretending
that the completed V2.3 trajectory changed:

* reason-aware client-state classification;
* service-age and cadence-health gates;
* one-report structural timing for communication-dominated clients;
* same-Q slow anchor/join opportunities;
* the separately disabled communication-amortized-Q counterfactual.

Only the first four components belong to the proposed routing Shadow.  The Q
counterfactual is retained as an explicitly inactive appendix.
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

from su_compass.experiments.analyze_utility_paced_v24 import (
    analyze as analyze_utility_paced,
)
from su_compass.experiments.replay_one_report_structural import (
    analyze as analyze_one_report,
)
from su_compass.scheduling.policies.reason_aware_routing import (
    classify_slow_cause,
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
    rhythm_target: float = 16.4,
    age_periods: float = 4.0,
    communication_ratio_gate: float = 0.95,
    safety_fraction: float = 0.10,
    cadence_median_ratio: float = 1.25,
    cadence_max_ratio: float = 2.0,
) -> dict:
    structural = analyze_one_report(
        run_dir,
        communication_ratio_gate=communication_ratio_gate,
        safety_fraction=safety_fraction,
    )
    utility = analyze_utility_paced(
        run_dir,
        rhythm_target=rhythm_target,
        age_periods=age_periods,
        marginal_time_ratio=0.2,
        communication_ratio_gate=0.8,
        cadence_median_ratio=cadence_median_ratio,
        cadence_max_ratio=cadence_max_ratio,
        slow_clients={"client_5", "client_6"},
    )

    decisions = sorted(
        _read(run_dir / "lyapunov_decision_trace.csv"),
        key=lambda row: _number(row, "virtual_time"),
    )
    curves: dict[str, list[dict]] = defaultdict(list)
    for row in _read(run_dir / "state_time_trace.csv"):
        curves[row["decision_id"]].append(row)
    aggregations = sorted(
        _read(run_dir / "aggregation_trace.csv"),
        key=lambda row: _number(row, "virtual_time"),
    )

    last_service: dict[str, float] = defaultdict(float)
    aggregation_times: list[float] = []
    intervals: list[float] = []
    cursor = 0
    opportunities = []
    for decision in decisions:
        now = _number(decision, "virtual_time")
        while cursor < len(aggregations) and (
            _number(aggregations[cursor], "virtual_time") <= now
        ):
            row = aggregations[cursor]
            when = _number(row, "virtual_time")
            if aggregation_times:
                intervals.append(when - aggregation_times[-1])
            aggregation_times.append(when)
            for client_id in json.loads(row["per_client_local_steps"]):
                last_service[client_id] = when
            cursor += 1

        if int(_number(decision, "recommendation_applied", 0)) != 1:
            continue
        q = int(_number(decision, "recommended_q", -1))
        point = next((
            row for row in curves.get(decision["decision_id"], [])
            if int(_number(row, "q", -1)) == q
        ), None)
        if point is None:
            continue
        cause = classify_slow_cause(SimpleNamespace(
            predicted_duration=_number(point, "predicted_duration"),
            compute_duration=_number(point, "compute_duration"),
            communication_duration=_number(point, "communication_duration"),
            availability_duration=_number(point, "availability_duration"),
            availability_risk_duration=_number(
                point, "availability_risk_duration"
            ),
            spike_duration=_number(point, "spike_duration"),
            num_reports=int(_number(point, "num_reports")),
            used_fallback=bool(int(_number(point, "used_fallback", 0))),
            predictor_source=point.get("predictor_source", ""),
        ))
        recent = intervals[-4:]
        healthy = bool(
            recent
            and statistics.median(recent)
            <= cadence_median_ratio * rhythm_target
            and max(recent) <= cadence_max_ratio * rhythm_target
        )
        age = max(0.0, now - last_service[decision["client_id"]])
        if cause.label != "extreme_communication_bound":
            continue
        mode = decision["recommended_mode"]
        opportunities.append({
            "decision_id": decision["decision_id"],
            "virtual_time": now,
            "client_id": decision["client_id"],
            "cause": cause.label,
            "service_age_periods": age / rhythm_target,
            "cadence_healthy": healthy,
            "v23_mode": mode,
            "v23_group_id": int(_number(
                decision, "recommended_group_id", -1
            )),
            "q": q,
            "same_q_invariant": True,
            "already_suitable_action": mode in {"create", "join"},
            "aged_elastic_anchor_candidate": bool(
                age >= age_periods * rhythm_target
                and healthy
                and mode != "create"
            ),
        })

    structural_anchors = structural["anchor_counterfactual"]["records"]
    dispatches = _read(run_dir / "dispatch_decision_trace.csv")
    long_window_exposure = []
    for anchor in structural_anchors:
        group_id = anchor["group_id"]
        created = next(
            row for row in _read(run_dir / "state_group_creation_trace.csv")
            if int(_number(row, "new_group_id", -1)) == group_id
        )
        created_at = _number(created, "virtual_time")
        for row in dispatches:
            when = _number(row, "virtual_time")
            if (
                int(_number(row, "assigned_group", -1)) == group_id
                and row["client_id"] != anchor["client_id"]
                and created_at <= when <= anchor["structural_latest_time"]
            ):
                structural_sojourn = max(
                    0.0, anchor["structural_expected_time"] - when
                )
                long_window_exposure.append({
                    "decision_id": row["decision_id"],
                    "client_id": row["client_id"],
                    "group_id": group_id,
                    "dispatch_time": when,
                    "structural_sojourn": structural_sojourn,
                    "structural_cadence_excess": max(
                        0.0, structural_sojourn - rhythm_target
                    ),
                })

    target_clients = set(structural["summary"]["eligible_client_ids"])
    non_target_exposure = [
        row for row in long_window_exposure
        if row["client_id"] not in target_clients
    ]
    routing_ready = bool(
        structural["summary"]["eligible_clients"] > 0
        and structural["summary"]["point_better_rate"] == 1
        and structural["summary"]["structural_safe_coverage"] == 1
        and structural["anchor_counterfactual"]["all_structural_safe"]
        and all(row["same_q_invariant"] for row in opportunities)
    )
    return {
        "scope": (
            "fixed_v2_3_trajectory_unified_elastic_service_audit_"
            "not_changed_trajectory_or_tta_replay"
        ),
        "decision": {
            "routing_shadow_ready": routing_ready,
            "reason": (
                "structural_timing_selective_safe_and_same_q"
                if routing_ready else "offline_gate_failed"
            ),
            "apply_ready": False,
            "apply_blocker": (
                "downstream join_create trajectory and TTA require online Shadow"
            ),
        },
        "active_shadow_candidate": {
            "one_report_structural": structural,
            "mature_reason_aware_opportunities": opportunities,
            "long_anchor_window_exposure": {
                "all_clients": long_window_exposure,
                "non_target_clients": non_target_exposure,
                "interpretation": (
                    "These clients joined the underestimated cold-start group "
                    "on the real path. The corrected window would expose a "
                    "large cadence cost, but offline replay cannot assert the "
                    "alternative action selected by V2.3."
                ),
            },
            "q_changed": False,
        },
        "inactive_appendix": {
            "communication_amortized_q": {
                "enabled_in_shadow": False,
                "reason": (
                    "User scope freezes Q until routing benefit is established"
                ),
                "offline_counterfactual": utility["counterfactual"],
                "terminal_accounting": utility["terminal_accounting"],
            },
            "statistical_utility": utility["statistical_utility"],
        },
        "limitations": [
            "No counterfactual model training or accuracy is synthesized.",
            "Unlogged alternative action scores cannot be reconstructed exactly.",
            "TTA and downstream group composition require an online Shadow run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--rhythm_target", type=float, default=16.4)
    parser.add_argument("--age_periods", type=float, default=4.0)
    parser.add_argument("--communication_ratio_gate", type=float, default=0.95)
    parser.add_argument("--safety_fraction", type=float, default=0.10)
    parser.add_argument("--cadence_median_ratio", type=float, default=1.25)
    parser.add_argument("--cadence_max_ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.run_dir,
        rhythm_target=args.rhythm_target,
        age_periods=args.age_periods,
        communication_ratio_gate=args.communication_ratio_gate,
        safety_fraction=args.safety_fraction,
        cadence_median_ratio=args.cadence_median_ratio,
        cadence_max_ratio=args.cadence_max_ratio,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
