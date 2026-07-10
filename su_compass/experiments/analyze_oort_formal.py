"""
su_compass.experiments.analyze_oort_formal — Oort-Compass 正式实验结果分析。

汇总 Stage A–D 指标，输出 JSON 与 Markdown 报告到 su_compass/output/oort_formal/analysis/
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from su_compass.experiments.oort_formal_common import (
    FORMAL_ROOT,
    SEEDS,
    compare_trace_files,
    summarize_run,
    time_to_accuracy,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate_by_name(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按实验名跨 seed 汇总 mean/std。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[r["name"]].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for name, items in grouped.items():
        metrics = ["mean_staleness", "max_staleness", "late_uploads",
                   "deadline_triggers", "final_accuracy", "final_virtual_time"]
        agg: Dict[str, Any] = {"name": name, "n_seeds": len(items), "seeds": [i["seed"] for i in items]}
        for m in metrics:
            vals = [i[m] for i in items]
            agg[f"{m}_mean"] = round(statistics.mean(vals), 4)
            agg[f"{m}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        # time-to-accuracy @ 40%, 50%
        for target in (40.0, 50.0):
            tta = [time_to_accuracy(i["accuracy_time_curve"], target) for i in items]
            tta_valid = [t for t in tta if t is not None]
            agg[f"tta_{int(target)}_mean"] = (
                round(statistics.mean(tta_valid), 4) if tta_valid else None
            )
        out[name] = agg
    return out


def _q_by_profile(run_dir: Path) -> Dict[str, List[int]]:
    """按 profile_type 汇总 local_steps 分布。"""
    sched = _read_csv(run_dir / "scheduler_trace.csv")
    by_profile: Dict[str, List[int]] = defaultdict(list)
    for row in sched:
        by_profile[row.get("profile_type", "unknown")].append(int(row["local_steps"]))
    return {k: v for k, v in by_profile.items()}


def _oort_q_adjustments(run_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(run_dir / "oort_trace.csv")
    out = []
    for row in rows:
        try:
            penalty = float(row.get("system_penalty", 1))
            qb = row.get("q_baseline")
            qo = row.get("q_after_oort")
            if qb in ("", "-1") or qo in ("", "-1"):
                continue
            qb_i, qo_i = int(float(qb)), int(float(qo))
        except (ValueError, KeyError):
            continue
        if penalty > 1.0001 and qb_i >= 0 and qo_i >= 0:
            out.append({
                "client_id": row["client_id"],
                "profile": row.get("communication_ratio_mean", ""),
                "penalty": penalty,
                "q_baseline": qb_i,
                "q_after_oort": qo_i,
                "delta": qb_i - qo_i,
            })
    return out


def analyze(root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"root": str(root)}

    # Stage A
    baseline = root / "stage_a" / "fedcompass_baseline" / "seed2026"
    shadow = root / "stage_a" / "oort_shadow" / "seed2026"
    if baseline.exists() and shadow.exists():
        report["stage_a_consistency"] = compare_trace_files(baseline, shadow)

    # Collect all runs
    all_runs: List[Dict[str, Any]] = []
    for stage in ("a", "b", "c", "d"):
        stage_dir = root / f"stage_{stage}"
        if not stage_dir.exists():
            continue
        for exp_dir in sorted(stage_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            for seed_dir in sorted(exp_dir.iterdir()):
                if not (seed_dir / "experiment_config.json").exists():
                    continue
                summary = summarize_run(seed_dir)
                seed = int(seed_dir.name.replace("seed", ""))
                summary["name"] = exp_dir.name
                summary["seed"] = seed
                summary["stage"] = stage.upper()
                all_runs.append(summary)

    report["all_runs"] = all_runs

    # Stage B main comparison
    stage_b = [r for r in all_runs if r["stage"] == "B"]
    report["stage_b_aggregate"] = _aggregate_by_name(stage_b)

    # Stage C ablation
    stage_c = [r for r in all_runs if r["stage"] == "C"]
    report["stage_c_ablation"] = {r["name"]: r for r in stage_c}

    # Stage D robustness
    stage_d = [r for r in all_runs if r["stage"] == "D"]
    report["stage_d_robustness"] = {r["name"]: r for r in stage_d}

    # Q by profile (Stage B q_only seed2026 as example)
    example = root / "stage_b" / "q_only" / "seed2026"
    if example.exists():
        report["q_by_profile_q_only_seed2026"] = {
            p: {"mean": round(statistics.mean(v), 2), "n": len(v)}
            for p, v in _q_by_profile(example).items()
        }
        report["oort_q_adjustments_q_only_seed2026"] = _oort_q_adjustments(example)

    return report


def _markdown(report: Dict[str, Any]) -> str:
    lines = ["# Oort-Compass 正式实验分析报告\n"]

    # Stage A
    sa = report.get("stage_a_consistency")
    if sa:
        status = "通过" if sa["passed"] else "未通过"
        lines.append(f"## Stage A：一致性验证（{status}）\n")
        for chk in sa["checks"]:
            mark = "✓" if chk["identical"] else "✗"
            lines.append(f"- {mark} {chk['file']}")
        lines.append("")

    # Stage B table
    sb = report.get("stage_b_aggregate", {})
    if sb:
        lines.append("## Stage B：主实验（3 seed 均值）\n")
        lines.append("| 方法 | mean_staleness | max_staleness | late | deadline | final_acc | final_vtime | TTA@40 |")
        lines.append("|------|----------------|---------------|------|----------|-----------|-------------|--------|")
        for name in ("fedcompass", "q_only", "q_and_group"):
            if name not in sb:
                continue
            r = sb[name]
            lines.append(
                f"| {name} | {r['mean_staleness_mean']:.3f}±{r['mean_staleness_std']:.3f} "
                f"| {r['max_staleness_mean']:.1f} "
                f"| {r['late_uploads_mean']:.1f} "
                f"| {r['deadline_triggers_mean']:.1f} "
                f"| {r['final_accuracy_mean']:.1f}±{r['final_accuracy_std']:.1f} "
                f"| {r['final_virtual_time_mean']:.1f} "
                f"| {r.get('tta_40_mean', 'N/A')} |"
            )
        lines.append("")

    # Stage C
    sc = report.get("stage_c_ablation", {})
    if sc:
        lines.append("## Stage C：消融实验（seed=2026）\n")
        lines.append("| 变体 | mean_staleness | late | deadline | final_acc | q_direction_rate |")
        lines.append("|------|----------------|------|----------|-----------|------------------|")
        for name, r in sc.items():
            qdr = r.get("oort_q_direction_rate")
            qdr_s = f"{qdr:.2%}" if qdr is not None else "N/A"
            lines.append(
                f"| {name} | {r['mean_staleness']:.3f} | {r['late_uploads']} "
                f"| {r['deadline_triggers']} | {r['final_accuracy']:.1f} | {qdr_s} |"
            )
        lines.append("")

    # Stage D
    sd = report.get("stage_d_robustness", {})
    if sd:
        lines.append("## Stage D：鲁棒性实验（seed=2026）\n")
        lines.append("| 变体 | mean_staleness | final_acc | final_vtime |")
        lines.append("|------|----------------|-----------|-------------|")
        for name, r in sd.items():
            lines.append(
                f"| {name} | {r['mean_staleness']:.3f} | {r['final_accuracy']:.1f} "
                f"| {r['final_virtual_time']:.1f} |"
            )
        lines.append("")

    # Q profile
    qp = report.get("q_by_profile_q_only_seed2026")
    if qp:
        lines.append("## Q 分布（q_only, seed=2026, 按 profile）\n")
        for p, v in sorted(qp.items()):
            lines.append(f"- **{p}**: mean Q = {v['mean']} (n={v['n']})")
        lines.append("")

    lines.append("## 结论摘要\n")
    if sb and "fedcompass" in sb and "q_only" in sb:
        base_s = sb["fedcompass"]["mean_staleness_mean"]
        qo_s = sb["q_only"]["mean_staleness_mean"]
        delta = base_s - qo_s
        direction = "降低" if delta > 0 else "升高"
        lines.append(
            f"- **RQ2（调度有效性）**：q_only 相对 fedcompass 平均 staleness {direction} "
            f"{abs(delta):.3f}（{base_s:.3f} → {qo_s:.3f}）。"
        )
    if sa and sa.get("passed"):
        lines.append("- **RQ1（安全性）**：shadow 与 baseline 调度 trace 逐行一致，接入未破坏 FedCompass。")

    return "\n".join(lines)


def main() -> None:
    root = _project_root() / FORMAL_ROOT
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = analyze(root)
    (out_dir / "formal_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "formal_report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"[analysis] 报告已写入 {out_dir}/formal_report.md")


if __name__ == "__main__":
    main()
