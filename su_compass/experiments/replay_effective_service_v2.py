"""Read-only replay gates for Effective-Service V2 Shadow.

The replay validates the prequential finite-sample safety head and verifies
that every V1 decision without a legal join has a controlled create route. It
does not claim to reproduce the future grouping or learning trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from su_compass.scheduling.predictors.regime_calibrated import (
    RegimeCalibratedPredictor,
)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay(run_dir: Path, target_coverage: float = 0.85) -> dict:
    calibration = _rows(run_dir / "calibrated_predictor_shadow_trace.csv")
    predictor = RegimeCalibratedPredictor(
        target_coverage=target_coverage, finite_sample_pooling=True,
    )
    hits: list[int] = []
    active_hits: list[int] = []
    sources: Counter[str] = Counter()
    never_narrower = True
    for row in calibration:
        baseline = float(row["baseline_duration"])
        baseline_safe = float(row["baseline_safe_duration"])
        result = predictor.predict(
            client_id=row["client_id"], local_steps=int(row["q"]),
            baseline_duration=baseline,
            baseline_safe_duration=baseline_safe,
        )
        actual = float(row["actual_duration"])
        hit = int(actual <= result.safe_duration)
        hits.append(hit)
        sources[result.calibration_source] += 1
        if result.used_candidate:
            active_hits.append(hit)
        never_narrower &= result.safe_duration >= baseline_safe
        predictor.observe(
            client_id=row["client_id"], local_steps=int(row["q"]),
            actual_duration=actual, compute_duration=0.0,
            communication_duration=0.0,
        )

    decisions = _rows(run_dir / "lyapunov_decision_trace.csv")
    no_legal_join = [row for row in decisions if int(row["num_legal_actions"]) == 0]
    q_refs = {
        "client_0": 174, "client_1": 188, "client_2": 84, "client_3": 156,
        "client_4": 111, "client_5": 47, "client_6": 47, "client_7": 64,
    }
    create_covered = sum(row["client_id"] in q_refs for row in no_legal_join)
    return {
        "scope": "prequential_safety_and_region3_coverage_not_training_replay",
        "finite_sample_safety": {
            "count": len(hits),
            "overall_coverage": sum(hits) / len(hits) if hits else None,
            "active_count": len(active_hits),
            "active_coverage": (
                sum(active_hits) / len(active_hits) if active_hits else None
            ),
            "never_narrower_than_analytical": never_narrower,
            "source_counts": dict(sources),
            "target_coverage": target_coverage,
        },
        "regional_create": {
            "v1_no_legal_join_decisions": len(no_legal_join),
            "controlled_create_covered": create_covered,
            "coverage": create_covered / len(no_legal_join) if no_legal_join else 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--target_coverage", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir, args.target_coverage)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
