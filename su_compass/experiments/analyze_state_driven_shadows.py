"""Summarize completed predictor shadows without optional ML dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value in (None, "") else float(value)


def _calibrated(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    n = len(rows)
    return {
        "count": n,
        "baseline_mae": sum(_f(r, "baseline_abs_error") for r in rows) / n,
        "shadow_mae": sum(_f(r, "shadow_abs_error") for r in rows) / n,
        "baseline_safe_coverage": sum(_f(r, "baseline_safe_hit") for r in rows) / n,
        "shadow_safe_coverage": sum(_f(r, "shadow_safe_hit") for r in rows) / n,
        "baseline_pinball": sum(_f(r, "baseline_pinball") for r in rows) / n,
        "shadow_pinball": sum(_f(r, "shadow_pinball") for r in rows) / n,
        "point_better_rate": sum(_f(r, "point_prediction_better") for r in rows) / n,
        "safe_better_rate": sum(_f(r, "safe_prediction_better") for r in rows) / n,
        "shadow_apply_rate": sum(_f(r, "shadow_used_candidate") for r in rows) / n,
    }


def _native(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    n = len(rows)
    return {
        "count": n,
        "applied_mean_q": sum(_f(r, "applied_q") for r in rows) / n,
        "native_mean_q": sum(_f(r, "native_q") for r in rows) / n,
        "applied_qmax_rate": sum(_f(r, "applied_q") == 200 for r in rows) / n,
        "native_qmax_rate": sum(_f(r, "native_qmax") for r in rows) / n,
        "q_changed_rate": sum(_f(r, "native_q_changed") for r in rows) / n,
        "applied_mae": sum(_f(r, "applied_abs_error") for r in rows) / n,
        "native_counterfactual_mae": sum(_f(r, "native_abs_error") for r in rows) / n,
        "native_safe_coverage": sum(_f(r, "native_safe_hit") for r in rows) / n,
        "native_prediction_better_rate": sum(_f(r, "native_prediction_better") for r in rows) / n,
        "native_reduces_qmax_rate": sum(_f(r, "native_reduces_qmax") for r in rows) / n,
        "native_fallback_rate": sum(_f(r, "native_used_fallback") for r in rows) / n,
    }


def analyze(run_dir: Path) -> dict:
    calibrated_rows = _rows(run_dir / "calibrated_predictor_shadow_trace.csv")
    native_rows = _rows(run_dir / "predictor_native_group_shadow_trace.csv")
    result = {
        "run_dir": str(run_dir),
        "calibrated_predictor": {"overall": _calibrated(calibrated_rows), "by_profile": {}},
        "predictor_native_new_group": {"overall": _native(native_rows), "by_profile": {}},
    }
    for name, rows, metric in (
        ("calibrated_predictor", calibrated_rows, _calibrated),
        ("predictor_native_new_group", native_rows, _native),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("profile_type", "")].append(row)
        result[name]["by_profile"] = {
            profile: metric(group_rows)
            for profile, group_rows in sorted(groups.items())
        }
    output = run_dir / "state_driven_shadow_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
