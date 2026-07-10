"""
su_compass.experiments.analyze_convergence_compare — 收敛对比结果分析。

对比 fedcompass vs q_only 的 accuracy-time、TTA、staleness、最终精度。
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

OUTPUT_ROOT = Path("su_compass/output/convergence_compare")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _tta(curve: List[Dict[str, float]], target: float) -> Optional[float]:
    for pt in curve:
        if pt["acc"] >= target:
            return pt["vtime"]
    return None


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    ge = _read_csv(run_dir / "global_eval_trace.csv")
    agg = _read_csv(run_dir / "aggregation_trace.csv")
    sched = _read_csv(run_dir / "scheduler_trace.csv")
    groups = _read_csv(run_dir / "group_trace.csv")

    import json as _json
    stale = []
    for r in agg:
        stale += list(_json.loads(r["per_client_staleness"]).values())

    curve = [{"vtime": float(r["virtual_time"]), "acc": float(r["test_accuracy"])} for r in ge]
    accs = [p["acc"] for p in curve]

    return {
        "run_dir": str(run_dir),
        "budget": int(_json.loads((run_dir / "experiment_config.json").read_text())["num_global_epochs"]) if (run_dir / "experiment_config.json").exists() else None,
        "num_aggregations": len(agg),
        "num_client_updates": sum(int(a.get("num_clients", 1)) for a in agg),
        "mean_staleness": round(statistics.mean(stale), 4) if stale else 0,
        "max_staleness": max(stale) if stale else 0,
        "late_uploads": sum(1 for r in sched if r.get("late") == "1"),
        "deadline_triggers": sum(1 for g in groups if g.get("trigger") == "deadline"),
        "final_accuracy": round(accs[-1], 2) if accs else 0,
        "max_accuracy": round(max(accs), 2) if accs else 0,
        "final_virtual_time": round(curve[-1]["vtime"], 2) if curve else 0,
        "tta_90": _tta(curve, 90.0),
        "tta_95": _tta(curve, 95.0),
        "accuracy_time_curve": curve,
    }


def analyze(root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"methods": {}}
    for method in ("fedcompass", "q_only"):
        method_dir = root / method
        if not method_dir.exists():
            continue
        runs = []
        for seed_dir in sorted(method_dir.iterdir()):
            if seed_dir.is_dir() and (seed_dir / "experiment_config.json").exists():
                s = summarize_run(seed_dir)
                s["seed"] = int(seed_dir.name.replace("seed", ""))
                runs.append(s)
        report["methods"][method] = runs

        if runs:
            for key in ("final_accuracy", "max_accuracy", "mean_staleness", "late_uploads", "deadline_triggers"):
                vals = [r[key] for r in runs]
                report[f"{method}_{key}_mean"] = round(statistics.mean(vals), 3)
                report[f"{method}_{key}_std"] = round(statistics.stdev(vals), 3) if len(vals) > 1 else 0

    # 相对收益
    if "fedcompass" in report["methods"] and "q_only" in report["methods"]:
        fc = report["methods"]["fedcompass"]
        qo = report["methods"]["q_only"]
        if fc and qo:
            report["delta_final_acc_mean"] = round(
                statistics.mean(r["final_accuracy"] for r in qo)
                - statistics.mean(r["final_accuracy"] for r in fc), 2
            )
            report["delta_staleness_mean"] = round(
                statistics.mean(r["mean_staleness"] for r in fc)
                - statistics.mean(r["mean_staleness"] for r in qo), 4
            )
    return report


def _markdown(report: Dict[str, Any]) -> str:
    lines = ["# FedCompass vs Oort-Compass 收敛对比\n"]
    for method in ("fedcompass", "q_only"):
        runs = report.get("methods", {}).get(method, [])
        if not runs:
            continue
        lines.append(f"## {method}\n")
        lines.append("| seed | budget_used | final_acc | max_acc | TTA@90 | TTA@95 | mean_stale | late | deadline | vtime |")
        lines.append("|------|-------------|-----------|---------|--------|--------|------------|------|----------|-------|")
        for r in runs:
            t90 = f"{r['tta_90']:.1f}" if r["tta_90"] else "—"
            t95 = f"{r['tta_95']:.1f}" if r["tta_95"] else "—"
            lines.append(
                f"| {r['seed']} | {r['num_client_updates']} | {r['final_accuracy']:.1f} | {r['max_accuracy']:.1f} "
                f"| {t90} | {t95} | {r['mean_staleness']:.3f} | {r['late_uploads']} "
                f"| {r['deadline_triggers']} | {r['final_virtual_time']:.1f} |"
            )
        lines.append("")

    d_acc = report.get("delta_final_acc_mean")
    d_stale = report.get("delta_staleness_mean")
    if d_acc is not None:
        lines.append("## 汇总\n")
        lines.append(f"- q_only 相对 fedcompass：final_acc **{d_acc:+.2f}pp**，mean_staleness **{d_stale:+.4f}**（正=改善）")
        if d_acc > 0 and (d_stale or 0) > 0:
            lines.append("- **结论**：精度与调度双改善，值得继续做消融/论文实验。")
        elif d_acc > 0:
            lines.append("- **结论**：精度有提升，调度改善不明显，可继续调 λ 或加 q_and_group。")
        elif (d_stale or 0) > 0:
            lines.append("- **结论**：调度改善但精度未升，检查是否 Q 过保守。")
        else:
            lines.append("- **结论**：收益不明显，需调参或加大 budget 后再判断。")
    return "\n".join(lines)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2] / Path(args.output_root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    report = analyze(root)
    (out / "convergence_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "convergence_report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"[analysis] -> {out}/convergence_report.md")


if __name__ == "__main__":
    main()
