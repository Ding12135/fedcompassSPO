"""Offline bounds for state-aware just-in-time model binding.

Two counterfactuals are intentionally separated:

* ``availability_bind`` snapshots the model after device-availability wait and
  before the recorded download.  This is implementable without an additional
  transfer under the phase ordering used by the runtime model.
* ``training_start_refresh`` snapshots at the recorded training start.  This
  is an optimistic upper bound because a real system would need an additional
  full or compressed model delta after the original download.

The replay keeps aggregation times fixed.  It estimates staleness and the
FedCompass raw coefficient only; it does not claim counterfactual model
quality, accuracy, or a changed scheduling trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay(run_dir: Path, *, alpha: float = 0.9) -> dict:
    aggregations = _read(run_dir / "aggregation_trace.csv")
    outcomes = _read(run_dir / "state_dispatch_outcome_trace.csv")
    round_rows: list[dict] = []
    client_root = run_dir / "client_states"
    for path in sorted(client_root.glob("*/round_reports.csv")):
        round_rows.extend(_read(path))
    report_by_decision = {
        row["decision_id"]: row for row in round_rows
        if row.get("decision_id")
    }
    client_ids = sorted({row["client_id"] for row in outcomes})
    if not client_ids:
        raise ValueError("no client outcomes found")

    aggregation_times = [float(row["virtual_time"]) for row in aggregations]
    aggregation_versions = [
        int(row["global_timestamp_after"]) for row in aggregations
    ]
    aggregation_by_time = {
        float(row["virtual_time"]): row for row in aggregations
    }

    def version_at(time_value: float) -> int:
        index = bisect_right(aggregation_times, time_value) - 1
        return aggregation_versions[index] if index >= 0 else 0

    phase_totals = defaultdict(lambda: defaultdict(float))
    phase_counts = defaultdict(int)
    scenario_rows: list[dict] = []
    cumulative = {
        name: defaultdict(float)
        for name in ("baseline", "availability_bind", "training_start_refresh")
    }
    staleness_sum = defaultdict(lambda: defaultdict(float))
    staleness_count = defaultdict(lambda: defaultdict(int))

    for outcome in outcomes:
        decision_id = outcome["decision_id"]
        report = report_by_decision.get(decision_id)
        if report is None:
            continue
        cid = outcome["client_id"]
        phase_counts[cid] += 1
        for field in (
            "availability_wait", "download_time", "train_time",
            "upload_time", "spike_delay", "round_time",
        ):
            phase_totals[cid][field] += float(report[field] or 0.0)
        if outcome.get("aggregated") != "1" or not outcome.get("aggregation_time"):
            continue
        aggregation_time = float(outcome["aggregation_time"])
        aggregation = aggregation_by_time.get(aggregation_time)
        if aggregation is None:
            continue
        aggregation_version = int(aggregation["global_timestamp_before"])
        dispatch_time = float(report["dispatch_time"])
        availability = float(report["availability_wait"] or 0.0)
        download = float(report["download_time"] or 0.0)
        baseline_version = int(outcome["model_version_at_dispatch"])
        availability_bind_time = dispatch_time + availability
        training_start_time = availability_bind_time + download
        versions = {
            "baseline": baseline_version,
            "availability_bind": version_at(availability_bind_time),
            "training_start_refresh": version_at(training_start_time),
        }
        stale = {
            name: max(0, aggregation_version - version)
            for name, version in versions.items()
        }
        weights = {
            name: alpha / len(client_ids) * (value + 1) ** -0.5
            for name, value in stale.items()
        }
        for name in weights:
            cumulative[name][cid] += weights[name]
            staleness_sum[name][cid] += stale[name]
            staleness_count[name][cid] += 1
        scenario_rows.append({
            "decision_id": decision_id,
            "client_id": cid,
            "dispatch_time": dispatch_time,
            "availability_bind_time": availability_bind_time,
            "training_start_time": training_start_time,
            "aggregation_time": aggregation_time,
            "aggregation_version": aggregation_version,
            "actual_finish_time": float(outcome["actual_finish_time"]),
            "baseline_version": versions["baseline"],
            "availability_bind_version": versions["availability_bind"],
            "training_start_refresh_version": versions[
                "training_start_refresh"
            ],
            "baseline_staleness": stale["baseline"],
            "availability_bind_staleness": stale["availability_bind"],
            "training_start_refresh_staleness": stale[
                "training_start_refresh"
            ],
            "availability_wait": availability,
            "download_time": download,
            "upload_time": float(report["upload_time"] or 0.0),
            "train_time": float(report["train_time"] or 0.0),
        })

    mean_duration = {
        cid: phase_totals[cid]["round_time"] / phase_counts[cid]
        for cid in client_ids if phase_counts[cid] > 0
    }
    disadvantaged_count = max(1, math.ceil(len(client_ids) / 4))
    disadvantaged = set(sorted(
        mean_duration, key=mean_duration.get, reverse=True,
    )[:disadvantaged_count])

    phase_summary = {}
    for cid in client_ids:
        count = phase_counts[cid]
        if count <= 0:
            continue
        means = {
            field: phase_totals[cid][field] / count
            for field in phase_totals[cid]
        }
        total = max(means["round_time"], 1e-12)
        means.update({
            "availability_ratio": means["availability_wait"] / total,
            "download_ratio": means["download_time"] / total,
            "train_ratio": means["train_time"] / total,
            "upload_ratio": means["upload_time"] / total,
        })
        phase_summary[cid] = means

    scenarios = {}
    for name, values in cumulative.items():
        total = sum(values.values())
        shares = {
            cid: values[cid] / total if total > 0 else 0.0
            for cid in client_ids
        }
        stale_means = {
            cid: (
                staleness_sum[name][cid] / staleness_count[name][cid]
                if staleness_count[name][cid] else 0.0
            )
            for cid in client_ids
        }
        scenarios[name] = {
            "effective_contribution_share": shares,
            "disadvantaged_share": sum(
                shares[cid] for cid in disadvantaged
            ),
            "mean_aggregation_staleness": stale_means,
            "disadvantaged_mean_staleness": sum(
                stale_means[cid] for cid in disadvantaged
            ) / len(disadvantaged),
        }

    compressed_refresh = {}
    baseline_slow_raw = sum(
        cumulative["baseline"][cid] for cid in disadvantaged
    )
    for fraction in (0.05, 0.10, 0.25):
        values = defaultdict(float)
        retained = defaultdict(int)
        attempted = defaultdict(int)
        for row in scenario_rows:
            cid = row["client_id"]
            if cid not in disadvantaged:
                stale = int(row["baseline_staleness"])
                values[cid] += alpha / len(client_ids) * (stale + 1) ** -0.5
                continue
            attempted[cid] += 1
            added_transfer = fraction * float(row["download_time"])
            shifted_finish = float(row["actual_finish_time"]) + added_transfer
            if shifted_finish > float(row["aggregation_time"]):
                continue
            retained[cid] += 1
            stale = int(row["training_start_refresh_staleness"])
            values[cid] += alpha / len(client_ids) * (stale + 1) ** -0.5
        total = sum(values.values())
        slow_raw = sum(values[cid] for cid in disadvantaged)
        compressed_refresh[f"{fraction:.2f}"] = {
            "delta_transfer_fraction_of_full_download": fraction,
            "effective_contribution_share": {
                cid: values[cid] / total if total > 0 else 0.0
                for cid in client_ids
            },
            "disadvantaged_share": (
                slow_raw / total if total > 0 else 0.0
            ),
            "disadvantaged_raw_service_ratio_to_baseline": (
                slow_raw / baseline_slow_raw
                if baseline_slow_raw > 0 else 0.0
            ),
            "retained_same_aggregation": dict(retained),
            "attempted_refreshes": dict(attempted),
            "retained_same_aggregation_rate": (
                sum(retained.values()) / sum(attempted.values())
                if sum(attempted.values()) > 0 else 0.0
            ),
            "scope": (
                "conservative_fixed_trajectory; a shifted update that misses "
                "its original aggregation receives zero service"
            ),
        }

    baseline = scenarios["baseline"]
    availability = scenarios["availability_bind"]
    refresh = scenarios["training_start_refresh"]
    report = {
        "source_run": str(run_dir),
        "disadvantaged_clients": sorted(disadvantaged),
        "phase_summary": phase_summary,
        "scenarios": scenarios,
        "compressed_refresh": compressed_refresh,
        "deltas": {
            "availability_bind": {
                "disadvantaged_share_delta": (
                    availability["disadvantaged_share"]
                    - baseline["disadvantaged_share"]
                ),
                "disadvantaged_staleness_reduction": (
                    baseline["disadvantaged_mean_staleness"]
                    - availability["disadvantaged_mean_staleness"]
                ),
            },
            "training_start_refresh": {
                "disadvantaged_share_delta": (
                    refresh["disadvantaged_share"]
                    - baseline["disadvantaged_share"]
                ),
                "disadvantaged_staleness_reduction": (
                    baseline["disadvantaged_mean_staleness"]
                    - refresh["disadvantaged_mean_staleness"]
                ),
            },
        },
        "scope": {
            "exact": [
                "recorded_phase_durations",
                "global_version_at_recorded_phase_boundaries",
                "fixed_trajectory_staleness_coefficient",
            ],
            "implementable_lower_bound": (
                "availability_bind_without_extra_transfer"
            ),
            "optimistic_upper_bound": (
                "training_start_refresh_with_zero_delta_transfer_cost"
            ),
            "not_claimed": [
                "counterfactual_accuracy",
                "counterfactual_local_model",
                "counterfactual_changed_trajectory",
                "real_delta_transfer_cost",
            ],
        },
        "rows": scenario_rows,
    }
    slow_availability_ratio = sum(
        phase_summary[cid]["availability_ratio"] for cid in disadvantaged
    ) / len(disadvantaged)
    slow_upload_ratio = sum(
        phase_summary[cid]["upload_ratio"] for cid in disadvantaged
    ) / len(disadvantaged)
    report["gates"] = {
        "phase_trace_complete": len(report_by_decision) >= len(outcomes),
        "availability_binding_has_material_gain": (
            report["deltas"]["availability_bind"][
                "disadvantaged_staleness_reduction"
            ] >= 0.5
        ),
        "training_start_refresh_has_material_upper_bound": (
            report["deltas"]["training_start_refresh"][
                "disadvantaged_staleness_reduction"
            ] >= 1.0
        ),
        "five_percent_refresh_retains_most_service": (
            compressed_refresh["0.05"][
                "retained_same_aggregation_rate"
            ] >= 0.8
        ),
        "five_percent_refresh_improves_raw_slow_service": (
            compressed_refresh["0.05"][
                "disadvantaged_raw_service_ratio_to_baseline"
            ] > 1.0
        ),
        "upload_is_dominant_slow_phase": slow_upload_ratio > 0.5,
        "availability_is_material_slow_phase": slow_availability_ratio > 0.1,
    }
    report["decision"] = (
        "implement_availability_late_binding"
        if report["gates"]["availability_binding_has_material_gain"]
        else (
            "do_not_implement_plain_late_binding; evaluate compressed refresh"
            if report["gates"][
                "training_start_refresh_has_material_upper_bound"
            ]
            else "reject_jit_binding_for_this_runtime"
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
    print(json.dumps({
        "decision": report["decision"],
        "deltas": report["deltas"],
        "gates": report["gates"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
