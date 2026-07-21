"""Prequential replay of RAMP-AC on a completed State-Driven run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from su_compass.scheduling.predictors.regime_calibrated import (
    RegimeCalibratedPredictor,
)


def _float(row, key, default=0.0):
    value = row.get(key, "")
    return default if value in (None, "") else float(value)


def _metrics(rows, prefix):
    errors = [abs(row["actual_duration"] - row[f"{prefix}_duration"]) for row in rows]
    signed = [row["actual_duration"] - row[f"{prefix}_duration"] for row in rows]
    safe_hits = [row["actual_duration"] <= row[f"{prefix}_safe_duration"] for row in rows]
    margins = [row[f"{prefix}_safe_duration"] - row[f"{prefix}_duration"] for row in rows]
    upper_quantile = 0.90
    pinball = []
    for row in rows:
        residual = row["actual_duration"] - row[f"{prefix}_safe_duration"]
        pinball.append(
            upper_quantile * residual if residual >= 0.0
            else (upper_quantile - 1.0) * residual
        )
    ordered = sorted(errors)
    percentile = lambda p: ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]
    return {
        "count": len(rows),
        "mae": sum(errors) / len(errors),
        "median_ae": percentile(0.5),
        "p90_ae": percentile(0.9),
        "p95_ae": percentile(0.95),
        "mean_underprediction": sum(max(0.0, value) for value in signed) / len(signed),
        "safe_coverage": sum(safe_hits) / len(safe_hits),
        "mean_safe_margin": sum(margins) / len(margins),
        "upper_pinball_90": sum(pinball) / len(pinball),
    }


def replay(
    run_dir: Path, output_dir: Path, *, target_coverage: float = 0.85,
    min_observations: int = 5,
) -> dict:
    outcomes = {}
    with (run_dir / "state_dispatch_outcome_trace.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            outcomes[row["decision_id"]] = row

    selected = {}
    with (run_dir / "state_time_trace.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["is_state_selected_q"] == "1":
                selected[(row["decision_id"], int(row["q"]))] = row

    reports = []
    for path in sorted((run_dir / "client_states").glob("*/round_reports.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reports.extend(csv.DictReader(handle))
    reports.sort(key=lambda row: (_float(row, "dispatch_time"), row["client_id"]))

    predictor = RegimeCalibratedPredictor(
        target_coverage=target_coverage, min_observations=min_observations,
    )
    replay_rows = []
    for report in reports:
        decision_id = report["decision_id"]
        q = int(report["local_steps"])
        trace = selected.get((decision_id, q))
        outcome = outcomes.get(decision_id)
        if trace is None or outcome is None:
            continue
        baseline = _float(trace, "predicted_duration")
        baseline_safe = _float(trace, "safe_duration", baseline)
        prediction = predictor.predict(
            client_id=report["client_id"], local_steps=q,
            baseline_duration=baseline, baseline_safe_duration=baseline_safe,
        )
        actual = _float(outcome, "actual_duration")
        row = {
            "decision_id": decision_id,
            "client_id": report["client_id"],
            "profile_type": report["profile_type"],
            "dispatch_time": _float(report, "dispatch_time"),
            "local_steps": q,
            "actual_duration": actual,
            "baseline_duration": prediction.baseline_duration,
            "baseline_safe_duration": prediction.baseline_safe_duration,
            "raw_duration": prediction.raw_duration,
            "raw_safe_duration": prediction.raw_safe_duration,
            "gated_duration": prediction.predicted_duration,
            "gated_safe_duration": prediction.safe_duration,
            "burst_probability": prediction.burst_probability,
            "regime": prediction.regime,
            "conformal_margin": prediction.conformal_margin,
            "expert_weight": prediction.expert_weight,
            "used_candidate": int(prediction.used_candidate),
            "num_observations": prediction.num_observations,
        }
        replay_rows.append(row)
        predictor.observe(
            client_id=report["client_id"], local_steps=q,
            actual_duration=actual,
            compute_duration=_float(report, "train_time"),
            communication_duration=_float(report, "communication_time"),
            spike_duration=_float(report, "spike_delay"),
            availability_duration=_float(report, "availability_wait"),
        )

    if not replay_rows:
        raise RuntimeError("no causally linked selected predictions found")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ramp_ac_replay_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0]))
        writer.writeheader()
        writer.writerows(replay_rows)

    summary = {
        "run_dir": str(run_dir),
        "target_coverage": target_coverage,
        "min_observations": min_observations,
        "num_replayed": len(replay_rows),
        "candidate_apply_rate": sum(row["used_candidate"] for row in replay_rows) / len(replay_rows),
        "overall": {
            name: _metrics(replay_rows, name)
            for name in ("baseline", "raw", "gated")
        },
        "post_warmup": {},
        "by_profile": {},
        "by_client": {},
    }
    warm_rows = [row for row in replay_rows if row["num_observations"] >= 5]
    summary["post_warmup"] = {
        name: _metrics(warm_rows, name) for name in ("baseline", "raw", "gated")
    }
    groups = {"by_profile": defaultdict(list), "by_client": defaultdict(list)}
    for row in replay_rows:
        groups["by_profile"][row["profile_type"]].append(row)
        groups["by_client"][row["client_id"]].append(row)
    for group_name, values in groups.items():
        for key, rows in sorted(values.items()):
            summary[group_name][key] = {
                name: _metrics(rows, name) for name in ("baseline", "raw", "gated")
            }
    with (output_dir / "ramp_ac_replay_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_coverage", type=float, default=0.85)
    parser.add_argument("--min_observations", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(replay(
        args.run_dir, args.output_dir,
        target_coverage=args.target_coverage,
        min_observations=args.min_observations,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
