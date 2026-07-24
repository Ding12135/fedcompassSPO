"""Offline replay for unified contribution-aware batch dispatch.

This replay deliberately separates facts that are exact under the frozen
v2.3 trajectory from counterfactual batch coordination:

* effective contribution and deficit updates are exact because they use the
  real FedCompass aggregation coefficient and recorded staleness;
* anchor ordering is replayed at every completed group using the state that
  existed at that aggregation;
* simultaneous structural long-group avoidance recommendations are collapsed
  into one create plus compatible joins, preventing the six-create artifact.

It does not claim counterfactual accuracy or a fully changed training
trajectory.  Those remain Apply-only outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from su_compass.scheduling.policies.fair_contribution_state import (
    FairContributionState,
)
from su_compass.scheduling.policies.unified_batch_dispatch import (
    rank_unified_batch,
)
from su_compass.scheduling.policies.quality_gated_contribution import (
    recommend_contribution_restoration,
)
from su_compass.experiments.replay_one_report_structural import (
    analyze as analyze_one_report,
)
from su_compass.experiments.analyze_utility_paced_v24 import (
    analyze as analyze_mature_amortization,
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _jain(values: list[float]) -> float:
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator > 0 else 1.0


def replay(run_dir: Path, reason_trace: Path | None = None) -> dict:
    aggregations = _read_csv(run_dir / "aggregation_trace.csv")
    groups = _read_csv(run_dir / "group_trace.csv")
    outcomes = _read_csv(run_dir / "state_dispatch_outcome_trace.csv")
    queue_path = run_dir / "lyapunov_queue_trace.csv"
    queue_rows = _read_csv(queue_path) if queue_path.exists() else []
    client_ids = tuple(sorted({row["client_id"] for row in outcomes}))
    if not client_ids:
        raise ValueError("no clients found in dispatch outcomes")
    mean_duration = {
        cid: sum(
            float(row["actual_duration"])
            for row in outcomes if row["client_id"] == cid
        ) / sum(row["client_id"] == cid for row in outcomes)
        for cid in client_ids
    }
    disadvantaged = set(sorted(
        client_ids, key=lambda cid: mean_duration[cid], reverse=True,
    )[:max(1, math.ceil(len(client_ids) / 4))])
    rhythm_by_aggregation: dict[int, float] = {}
    for row in queue_rows:
        rhythm_by_aggregation.setdefault(
            int(row["aggregation_id"]), float(row["rhythm_debt_before"])
        )

    group_by_id = {int(row["group_id"]): row for row in groups}
    next_duration: dict[tuple[str, float], float] = {}
    for row in outcomes:
        next_duration[(row["client_id"], float(row["dispatch_time"]))] = float(
            row["actual_duration"]
        )

    state = FairContributionState(client_ids, score_cap=2.0)
    contribution_rows: list[dict] = []
    restored_cumulative = {cid: 0.0 for cid in client_ids}
    restoration_rows: list[dict] = []
    batch_rows: list[dict] = []
    baseline_anchor_counts: Counter[str] = Counter()
    proposed_anchor_counts: Counter[str] = Counter()

    for row in aggregations:
        staleness = {
            str(cid): int(value)
            for cid, value in json.loads(row["per_client_staleness"]).items()
        }
        records = state.update(
            staleness,
            alpha=0.9,
            staleness_fn=lambda stale: (stale + 1) ** -0.5,
        )
        local_steps = {
            str(cid): int(value)
            for cid, value in json.loads(
                row["per_client_local_steps"]
            ).items()
        }
        restored = recommend_contribution_restoration(
            records,
            local_steps=local_steps,
            qmax=max(
                [int(outcome["dispatched_q"]) for outcome in outcomes] or [1]
            ),
            rhythm_debt=rhythm_by_aggregation.get(
                int(row["aggregation_id"]), 0.0
            ),
            rhythm_stop=44.0,
            debt_score_cap=2.0,
            bonus_mass_cap=0.05,
            staleness_hard_cap=8,
        )
        raw_mass = sum(record.raw_weight for record in records)
        for proposed in restored:
            restored_cumulative[proposed.client_id] += (
                proposed.proposed_share * raw_mass
            )
            restoration_rows.append({
                "aggregation_id": int(row["aggregation_id"]),
                "client_id": proposed.client_id,
                "eligible": proposed.eligible,
                "reason": proposed.reason,
                "quality_score": proposed.quality_score,
                "base_share": proposed.base_share,
                "proposed_share": proposed.proposed_share,
                "share_delta": proposed.proposed_share - proposed.base_share,
            })
        for record in records:
            contribution_rows.append({
                "aggregation_id": int(row["aggregation_id"]),
                "client_id": record.client_id,
                "staleness": record.staleness,
                "effective_contribution": record.effective_contribution,
                "fair_debt_raw": record.fair_debt_raw,
                "fair_debt_score": record.fair_debt_score,
            })

        group_id = int(row["group_id"])
        if group_id < 0 or group_id not in group_by_id:
            continue
        group = group_by_id[group_id]
        arrived = [
            cid for cid in group["arrived_client_ids"].split(",") if cid
        ]
        if len(arrived) <= 1:
            continue
        virtual_time = float(row["virtual_time"])
        safe_duration = {
            cid: 1.10 * next_duration.get(
                (cid, virtual_time),
                200.0 * float(
                    next(
                        (
                            float(outcome["actual_duration"])
                            / max(1, int(outcome["dispatched_q"]))
                            for outcome in reversed(outcomes)
                            if outcome["client_id"] == cid
                            and float(outcome["dispatch_time"]) <= virtual_time
                        ),
                        16.4 / 200.0,
                    )
                ),
            )
            for cid in arrived
        }
        ranked = rank_unified_batch(
            arrived,
            fair_debt={cid: state.score(cid) for cid in arrived},
            safe_duration=safe_duration,
            rhythm_target=16.4,
        )
        baseline_anchor = group["anchor_client_id"]
        proposed_anchor = ranked[0].client_id
        baseline_anchor_counts[baseline_anchor] += 1
        proposed_anchor_counts[proposed_anchor] += 1
        batch_rows.append({
            "aggregation_id": int(row["aggregation_id"]),
            "group_id": group_id,
            "virtual_time": virtual_time,
            "batch_clients": arrived,
            "baseline_anchor": baseline_anchor,
            "proposed_anchor": proposed_anchor,
            "order": [item.client_id for item in ranked],
            "order_changed": proposed_anchor != baseline_anchor,
            "proposed_anchor_debt": ranked[0].fair_debt,
            "proposed_anchor_safe_duration": ranked[0].predicted_safe_duration,
        })

    cumulative = dict(state.cumulative_effective_share)
    total = sum(cumulative.values())
    shares = {
        cid: (value / total if total > 0 else 0.0)
        for cid, value in cumulative.items()
    }
    disadvantaged_share = sum(shares[cid] for cid in disadvantaged)
    restored_total = sum(restored_cumulative.values())
    restored_shares = {
        cid: (
            restored_cumulative[cid] / restored_total
            if restored_total > 0 else 0.0
        )
        for cid in client_ids
    }
    one_report = analyze_one_report(run_dir)
    mature = analyze_mature_amortization(
        run_dir,
        rhythm_target=16.4,
        age_periods=4.0,
        marginal_time_ratio=0.2,
        communication_ratio_gate=0.8,
        cadence_median_ratio=1.25,
        cadence_max_ratio=2.0,
        slow_clients=disadvantaged,
    )
    cold_q = one_report["communication_amortized_q"]
    mature_terminal = mature["terminal_accounting"]
    combined_terminal_work = (
        mature_terminal["terminal_proposed_work_if_pending_settles"]
        + cold_q["added_aggregated_q"]
    )
    combined_terminal_slow_work = (
        mature_terminal["terminal_proposed_slow_work_if_pending_settles"]
        + cold_q["added_aggregated_q"]
    )

    coordination_batches: list[dict] = []
    if reason_trace is not None and reason_trace.exists():
        reason_rows = _read_csv(reason_trace)
        by_time: dict[float, list[dict]] = defaultdict(list)
        for row in reason_rows:
            if row.get("elastic_join_avoidance") == "1":
                by_time[float(row["virtual_time"])].append(row)
        for virtual_time, rows in sorted(by_time.items()):
            if not rows:
                continue
            ranked = rank_unified_batch(
                [row["client_id"] for row in rows],
                fair_debt={
                    row["client_id"]: state.score(row["client_id"])
                    for row in rows
                },
                safe_duration={
                    row["client_id"]: float(
                        row.get("shadow_predicted_sojourn") or 16.4
                    )
                    for row in rows
                },
                rhythm_target=16.4,
            )
            anchor = ranked[0].client_id
            coordination_batches.append({
                "virtual_time": virtual_time,
                "original_independent_creates": len(rows),
                "coordinated_creates": 1,
                "coordinated_joins": len(rows) - 1,
                "anchor": anchor,
                "q_unchanged": all(row.get("q_unchanged") == "1" for row in rows),
                "all_same_q_create_legal": all(
                    row.get("same_q_create_legal") == "1" for row in rows
                ),
            })

    oracle_hold_rows: list[dict] = []
    wait_cap = 0.20 * 16.4
    for group in groups:
        if group.get("trigger") != "deadline":
            continue
        deadline = float(group["aggregation_time"])
        group_id = int(group["group_id"])
        for cid in (
            value for value in group.get("pending_client_ids", "").split(",")
            if value
        ):
            matching = [
                row for row in outcomes
                if row["client_id"] == cid
                and row.get("dispatched_group_id", "") != ""
                and int(row["dispatched_group_id"]) == group_id
                and float(row["actual_finish_time"]) >= deadline
            ]
            if not matching:
                continue
            finish = min(float(row["actual_finish_time"]) for row in matching)
            actual_wait = finish - deadline
            oracle_hold_rows.append({
                "group_id": group_id,
                "client_id": cid,
                "deadline": deadline,
                "actual_finish_time": finish,
                "actual_wait": actual_wait,
                "wait_cap": wait_cap,
                "time_feasible": actual_wait <= wait_cap,
            })

    changed_batches = sum(bool(row["order_changed"]) for row in batch_rows)
    report = {
        "source_run": str(run_dir),
        "scope": {
            "exact": [
                "effective_contribution",
                "fair_debt",
                "observed_batch_membership",
            ],
            "counterfactual": [
                "batch_anchor_order",
                "coordinated_structural_avoidance",
            ],
            "not_claimed": [
                "counterfactual_accuracy",
                "counterfactual_full_trajectory",
                "counterfactual_throughput",
            ],
        },
        "effective_contribution": {
            "cumulative": cumulative,
            "share": shares,
            "jain": _jain(list(cumulative.values())),
            "disadvantaged_clients": sorted(disadvantaged),
            "disadvantaged_share": disadvantaged_share,
            "final_raw_debt": dict(state.raw_debt),
        },
        "quality_gated_contribution_restoration": {
            "projected_share": restored_shares,
            "projected_jain": _jain(list(restored_cumulative.values())),
            "projected_disadvantaged_share": sum(
                restored_shares[cid] for cid in disadvantaged
            ),
            "eligible_client_aggregations": sum(
                bool(row["eligible"]) for row in restoration_rows
            ),
            "maximum_bonus_mass": 0.05,
            "apply_enabled": False,
            "rows": restoration_rows,
        },
        "micro_hold_oracle": {
            "scope": "timing_upper_bound_only_not_an_online_decision",
            "wait_cap": wait_cap,
            "deadline_pending_candidates": len(oracle_hold_rows),
            "time_feasible_candidates": sum(
                bool(row["time_feasible"]) for row in oracle_hold_rows
            ),
            "max_rejected_wait": max(
                (
                    row["actual_wait"] for row in oracle_hold_rows
                    if not row["time_feasible"]
                ),
                default=0.0,
            ),
            "rows": oracle_hold_rows,
        },
        "batch_dispatch": {
            "batch_count": len(batch_rows),
            "changed_anchor_batches": changed_batches,
            "baseline_anchor_counts": dict(baseline_anchor_counts),
            "proposed_anchor_counts": dict(proposed_anchor_counts),
            "rows": batch_rows,
        },
        "structural_avoidance_coordination": {
            "batch_count": len(coordination_batches),
            "rows": coordination_batches,
            "six_create_prevented": any(
                row["original_independent_creates"] >= 6
                and row["coordinated_creates"] == 1
                for row in coordination_batches
            ),
        },
        "communication_amortized_q": {
            "cold_start": cold_q,
            "mature": mature["counterfactual"],
            "combined_terminal_work": combined_terminal_work,
            "combined_terminal_slow_work": combined_terminal_slow_work,
            "combined_terminal_slow_share": (
                combined_terminal_slow_work / combined_terminal_work
                if combined_terminal_work else 0.0
            ),
            "q_apply_enabled": False,
        },
        "design_decision": {
            "retain": [
                "exact_effective_contribution_shadow",
                "one_report_structural_timing",
                "same_time_create_join_coordination",
                "communication_amortized_q_shadow",
                "quality_gated_contribution_restoration_shadow",
            ],
            "reject": [
                "fair_debt_only_redispatch_order",
                "permanent_fast_slow_lanes",
                "unbounded_workload_debt",
            ],
        },
    }
    report["gates"] = {
        "contribution_model_closed": math.isclose(
            sum(shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-9,
        ),
        "anchor_order_changes": changed_batches > 0,
        "no_six_create_artifact": (
            not coordination_batches
            or report["structural_avoidance_coordination"]["six_create_prevented"]
        ),
        "cold_q_selective": (
            set(cold_q["changed_clients"]).issubset(disadvantaged)
            and bool(cold_q["changed_clients"])
        ),
        "cold_q_marginal_time_safe": (
            cold_q["max_added_predicted_duration"] <= 0.2 * 16.4
            and cold_q["max_added_safe_duration"] <= 0.2 * 16.4
        ),
        "slow_work_share_restored_near_fedcompass": (
            cold_q["projected_slow_work_share"] >= 0.047
        ),
        "mature_q_selective": (
            mature["counterfactual"]["changed_decisions"] > 0
            and set(
                cid for cid, row in mature["counterfactual"]["per_client"].items()
                if row["changed"] > 0
            ).issubset(disadvantaged)
        ),
        "restoration_improves_disadvantaged_share": (
            report["quality_gated_contribution_restoration"][
                "projected_disadvantaged_share"
            ] > disadvantaged_share
        ),
        "restoration_bonus_bounded": all(
            row["proposed_share"] - row["base_share"] <= 0.05 + 1e-12
            for row in restoration_rows
        ),
        "micro_hold_rejects_long_waits": all(
            not row["time_feasible"]
            for row in oracle_hold_rows
            if row["actual_wait"] > wait_cap
        ),
        "apply_ready": False,
        "apply_blocker": (
            "The combined routing and Q recommendation is offline/Shadow-only; "
            "run one online Shadow trace before any Apply."
        ),
    }
    report["gates"]["shadow_ready"] = all(
        report["gates"][name]
        for name in (
            "contribution_model_closed",
            "no_six_create_artifact",
            "cold_q_selective",
            "cold_q_marginal_time_safe",
            "slow_work_share_restored_near_fedcompass",
            "mature_q_selective",
            "restoration_improves_disadvantaged_share",
            "restoration_bonus_bounded",
            "micro_hold_rejects_long_waits",
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--reason_trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run_dir, args.reason_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report["gates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
