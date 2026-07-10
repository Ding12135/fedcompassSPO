"""
su_compass.experiments.run_oort_formal — Oort-Compass 正式实验批量运行器。

按 attached plan 执行 Stage A–D，跳过已存在 experiment_config.json 的 run（可 --force 重跑）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from su_compass.experiments.console import (
    BatchExperimentProgress,
    gather_hardware_info,
    print_experiment_banner,
)
from su_compass.experiments.oort_formal_common import (
    FORMAL_ROOT,
    ExperimentSpec,
    all_specs,
    compare_trace_files,
    stage_a_specs,
    stage_b_specs,
    stage_c_specs,
    stage_d_specs,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_single(
    spec: ExperimentSpec,
    force: bool,
    dry_run: bool,
    batch: Optional[BatchExperimentProgress] = None,
    run_index: int = 0,
    run_total: int = 0,
) -> bool:
    out = _project_root() / spec.output_dir()
    marker = out / "experiment_config.json"
    if marker.exists() and not force:
        msg = f"  ⊘ 跳过（已存在） {spec.name} seed={spec.seed}"
        if batch is not None:
            batch.begin_run(spec.name, spec.seed, run_index)
            batch.log(msg)
            batch.end_run(True, 0.0)
        else:
            print(f"[skip] {spec.name} seed={spec.seed} -> {out}")
        return True

    cmd = [sys.executable, "-m", "su_compass.experiments.run_virtual_fl"] + spec.to_run_args()
    if batch is not None:
        batch.begin_run(spec.name, spec.seed, run_index)
    else:
        print(f"\n[run] {' '.join(cmd)}")
    if dry_run:
        if batch is not None:
            batch.end_run(True, 0.0)
        return True

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_project_root())
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and marker.exists()
    if batch is not None:
        batch.end_run(ok, elapsed)
    else:
        print(f"[{'OK' if ok else 'FAIL'}] {spec.name} seed={spec.seed} ({elapsed:.0f}s)")
    return ok


def _run_stage(
    specs: List[ExperimentSpec],
    stage_name: str,
    force: bool,
    dry_run: bool,
    show_ui: bool,
) -> List[bool]:
    results: List[bool] = []
    with BatchExperimentProgress(len(specs), stage_name, enabled=show_ui) as batch:
        for i, spec in enumerate(specs, start=1):
            results.append(_run_single(spec, force, dry_run, batch, i, len(specs)))
    return results


def _validate_stage_a(root: Path) -> dict:
    baseline = root / "stage_a" / "fedcompass_baseline" / "seed2026"
    shadow = root / "stage_a" / "oort_shadow" / "seed2026"
    result = compare_trace_files(baseline, shadow)
    report_path = root / "stage_a_consistency_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Stage A] consistency passed={result['passed']} -> {report_path}")
    for chk in result["checks"]:
        print(f"  - {chk['file']}: {'OK' if chk['identical'] else 'FAIL'}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oort-Compass 正式实验批量运行")
    p.add_argument("--stage", choices=["A", "B", "C", "D", "all"], default="all")
    p.add_argument("--force", action="store_true", help="重跑已有输出")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    p.add_argument("--no_progress", action="store_true", help="关闭硬件横幅与批量进度条")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    show_ui = not args.no_progress
    root = _project_root() / FORMAL_ROOT
    root.mkdir(parents=True, exist_ok=True)

    hardware = gather_hardware_info()
    print_experiment_banner(
        title="SU-Compass Oort-Compass 正式实验批量运行",
        run_lines=[
            f"Stage      {args.stage}",
            f"输出根目录  {FORMAL_ROOT}",
            f"强制重跑    {args.force}",
            f"仅预览命令  {args.dry_run}",
        ],
        hardware=hardware,
        enabled=show_ui,
    )

    stage_map = {
        "A": stage_a_specs,
        "B": stage_b_specs,
        "C": stage_c_specs,
        "D": stage_d_specs,
    }
    stages = list(stage_map.keys()) if args.stage == "all" else [args.stage]

    all_ok = True
    for st in stages:
        results = _run_stage(stage_map[st](), f"Stage {st}", args.force, args.dry_run, show_ui)
        all_ok = all_ok and all(results)

    if "A" in stages and not args.dry_run:
        baseline_ok = (root / "stage_a" / "fedcompass_baseline" / "seed2026" / "experiment_config.json").exists()
        shadow_ok = (root / "stage_a" / "oort_shadow" / "seed2026" / "experiment_config.json").exists()
        if baseline_ok and shadow_ok:
            validation = _validate_stage_a(root)
            all_ok = all_ok and validation["passed"]

    if not all_ok:
        sys.exit(1)
    print("\n[done] 全部实验运行完成。")


if __name__ == "__main__":
    main()
