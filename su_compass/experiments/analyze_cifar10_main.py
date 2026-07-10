"""汇总 CIFAR-10 主实验的精度、TTA 与调度指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fedcompass", "q_only", "q_and_group")
TARGETS = (30.0, 40.0, 50.0, 55.0, 60.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(run_dir: Path) -> dict[str, Any]:
    evaluations = read_csv(run_dir / "global_eval_trace.csv")
    aggregations = read_csv(run_dir / "aggregation_trace.csv")
    training_metrics = read_csv(run_dir / "training_metrics.csv")
    scheduler = read_csv(run_dir / "scheduler_trace.csv")
    groups = read_csv(run_dir / "group_trace.csv")
    curve = [
        (float(row["virtual_time"]), float(row["test_accuracy"]))
        for row in evaluations
    ]
    losses = [float(row["test_loss"]) for row in evaluations]
    accuracies = [accuracy for _, accuracy in curve]
    staleness = [
        int(value)
        for row in aggregations
        for value in json.loads(row["per_client_staleness"]).values()
    ]

    result: dict[str, Any] = {
        "seed": int(run_dir.name.removeprefix("seed")),
        "num_updates": (
            int(training_metrics[-1]["client_update_budget_used"])
            if training_metrics
            else sum(int(row.get("num_clients", 1)) for row in aggregations)
        ),
        "final_accuracy": accuracies[-1],
        "max_accuracy": max(accuracies),
        "last10_accuracy": statistics.mean(accuracies[-10:]),
        "final_virtual_time": curve[-1][0],
        "mean_staleness": statistics.mean(staleness) if staleness else 0.0,
        "late_uploads": sum(row.get("late") == "1" for row in scheduler),
        "deadline_triggers": sum(row.get("trigger") == "deadline" for row in groups),
        "finite": all(math.isfinite(value) for value in accuracies + losses),
    }
    for target in TARGETS:
        result[f"tta_{target:g}"] = next(
            (virtual_time for virtual_time, accuracy in curve if accuracy >= target),
            None,
        )
    return result


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def analyze(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"root": str(root), "methods": {}, "means": {}}
    for method in METHODS:
        method_dir = root / method
        runs = [
            summarize(seed_dir)
            for seed_dir in sorted(method_dir.glob("seed*"))
            if (seed_dir / "experiment_config.json").exists()
        ]
        if not runs:
            continue
        report["methods"][method] = runs
        report["means"][method] = {
            key: mean_or_none([run[key] for run in runs])
            for key in (
                "final_accuracy",
                "max_accuracy",
                "last10_accuracy",
                "mean_staleness",
                "late_uploads",
                "deadline_triggers",
                *(f"tta_{target:g}" for target in TARGETS),
            )
        }
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# CIFAR-10 Oort-Compass 主实验结果", ""]
    for method, runs in report["methods"].items():
        lines.extend(
            [
                f"## {method}",
                "",
                "| seed | updates | final | max | last10 | TTA@40 | TTA@50 | TTA@60 | stale | late | deadline | finite |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for run in runs:
            tta = lambda target: (
                f"{run[f'tta_{target}']:.1f}"
                if run[f"tta_{target}"] is not None
                else "—"
            )
            lines.append(
                f"| {run['seed']} | {run['num_updates']} "
                f"| {run['final_accuracy']:.2f} | {run['max_accuracy']:.2f} "
                f"| {run['last10_accuracy']:.2f} | {tta(40)} | {tta(50)} | {tta(60)} "
                f"| {run['mean_staleness']:.3f} | {run['late_uploads']} "
                f"| {run['deadline_triggers']} | {'是' if run['finite'] else '否'} |"
            )
        lines.append("")

    if report["means"]:
        lines.extend(["## 多种子均值", ""])
        baseline = report["means"].get("fedcompass", {})
        for method, means in report["means"].items():
            delta = ""
            if method != "fedcompass" and baseline:
                delta = (
                    f"，相对 baseline 的 last10 "
                    f"{means['last10_accuracy'] - baseline['last10_accuracy']:+.2f}pp"
                )
            lines.append(
                f"- {method}: final={means['final_accuracy']:.2f}%，"
                f"max={means['max_accuracy']:.2f}%，last10={means['last10_accuracy']:.2f}%"
                f"{delta}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="su_compass/output/cifar10_main")
    args = parser.parse_args()
    root = PROJECT_ROOT / args.output_root
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    report = analyze(root)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    main()
