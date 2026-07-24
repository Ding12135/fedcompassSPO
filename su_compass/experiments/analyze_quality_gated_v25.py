"""Summarize the long-run Shadow evidence for quality-gated V2.5."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(run_dir: Path) -> dict:
    contribution = _read(run_dir / "fair_contribution_shadow_trace.csv")
    restoration = _read(
        run_dir / "contribution_restoration_shadow_trace.csv"
    )
    holds = _read(run_dir / "micro_hold_shadow_trace.csv")
    routing = _read(run_dir / "reason_aware_routing_shadow_trace.csv")

    final_by_client: dict[str, dict] = {}
    for row in contribution:
        final_by_client[row["client_id"]] = row
    # The per-row value is epoch service, so accumulate across epochs.
    raw_service = defaultdict(float)
    for row in contribution:
        raw_service[row["client_id"]] += float(row["effective_contribution"])
    total_service = sum(raw_service.values())

    share_delta = defaultdict(float)
    eligible_by_client = Counter()
    total_bonus_by_epoch = defaultdict(float)
    for row in restoration:
        share_delta[row["client_id"]] += float(row["share_delta"])
        if row["eligible"] == "1":
            eligible_by_client[row["client_id"]] += 1
        total_bonus_by_epoch[int(row["aggregation_epoch"])] += float(
            row["allocated_bonus"]
        )

    q_rows = [
        row for row in routing
        if row.get("communication_amortized_q_eligible") == "1"
        and int(float(row.get("communication_amortized_q_added") or 0)) > 0
    ]
    mature_avoidance = [
        row for row in routing
        if row.get("mature_long_join_avoidance") == "1"
        and row.get("elastic_join_avoidance") == "1"
    ]
    hold_reasons = Counter(row["reason"] for row in holds)
    report = {
        "run_dir": str(run_dir),
        "artifacts": {
            "fair_contribution": bool(contribution),
            "contribution_restoration": bool(restoration),
            "micro_hold": bool(holds),
            "reason_aware_routing": bool(routing),
        },
        "effective_contribution": {
            "share": {
                cid: value / total_service if total_service > 0 else 0.0
                for cid, value in sorted(raw_service.items())
            },
            "final_debt": {
                cid: float(row["fair_debt_raw"])
                for cid, row in sorted(final_by_client.items())
            },
        },
        "restoration_shadow": {
            "eligible_by_client": dict(sorted(eligible_by_client.items())),
            "cumulative_share_delta": dict(sorted(share_delta.items())),
            "max_bonus_mass": max(total_bonus_by_epoch.values(), default=0.0),
            "epochs_with_bonus": sum(
                value > 0 for value in total_bonus_by_epoch.values()
            ),
            "applied_rows": sum(row["applied"] == "1" for row in restoration),
        },
        "communication_amortized_q_shadow": {
            "changed_recommendations": len(q_rows),
            "per_client": dict(Counter(row["client_id"] for row in q_rows)),
            "max_added_q": max(
                (
                    int(float(row["communication_amortized_q_added"]))
                    for row in q_rows
                ),
                default=0,
            ),
            "max_added_safe_duration": max(
                (
                    float(
                        row["communication_amortized_added_safe_duration"]
                        or 0
                    )
                    for row in q_rows
                ),
                default=0.0,
            ),
        },
        "mature_join_cadence_shadow": {
            "recommendations": len(mature_avoidance),
            "avoided_real_groups": sorted({
                int(row["v23_group_id"]) for row in mature_avoidance
                if row.get("v23_group_id", "") != ""
            }),
            "max_observed_sojourn": max(
                (
                    float(row["join_observed_sojourn"])
                    for row in mature_avoidance
                    if row.get("join_observed_sojourn", "") != ""
                ),
                default=0.0,
            ),
            "q_changed": any(
                row.get("q_unchanged") != "1" for row in mature_avoidance
            ),
        },
        "micro_hold_shadow": {
            "candidates": len(holds),
            "recommended": sum(row["recommended"] == "1" for row in holds),
            "reason_counts": dict(hold_reasons),
            "max_recommended_safe_wait": max(
                (
                    float(row["safe_wait"])
                    for row in holds
                    if row["recommended"] == "1" and row["safe_wait"] != ""
                ),
                default=0.0,
            ),
            "applied_rows": sum(row["applied"] == "1" for row in holds),
        },
    }
    report["gates"] = {
        "all_shadow_artifacts_present": all(report["artifacts"].values()),
        "v23_path_unchanged": (
            report["restoration_shadow"]["applied_rows"] == 0
            and report["micro_hold_shadow"]["applied_rows"] == 0
        ),
        "restoration_cap_respected": (
            report["restoration_shadow"]["max_bonus_mass"] <= 0.05 + 1e-12
        ),
        "micro_hold_cap_respected": (
            report["micro_hold_shadow"]["max_recommended_safe_wait"]
            <= 0.20 * 16.4 + 1e-12
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
