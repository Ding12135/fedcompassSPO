"""
su_compass.experiments.oort_formal_common — Oort-Compass 正式实验共享配置与工具。

定义 Stage A–D 实验矩阵、输出目录约定，以及 trace 一致性校验与指标汇总函数。
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# 正式实验根目录（相对项目根 fedcompass/）
FORMAL_ROOT = Path("su_compass/output/oort_formal")

SEEDS: Tuple[int, ...] = (2026, 2027, 2028)
BUDGET = 30  # num_global_epochs，与先前 MNIST 校准实验对齐


@dataclass
class ExperimentSpec:
    """单次实验的配置描述。"""

    name: str
    algorithm: str = "fedcompass"
    oort_mode: str = "shadow"
    seed: int = 2026
    num_global_epochs: int = BUDGET
    algorithm_variant: str = ""
    oort_lambda_comm: float = 1.0
    oort_lambda_late: float = 1.0
    oort_lambda_var: float = 0.5
    oort_lambda_avail: float = 0.5
    oort_risk_threshold: float = 0.5
    min_local_steps: Optional[int] = None
    max_local_steps: Optional[int] = None
    latest_time_factor: Optional[float] = None
    stage: str = "A"

    def output_dir(self, root: Path = FORMAL_ROOT) -> Path:
        return root / f"stage_{self.stage.lower()}" / self.name / f"seed{self.seed}"

    def to_run_args(self) -> List[str]:
        """转为 run_virtual_fl 命令行参数列表。"""
        args = [
            "--algorithm", self.algorithm,
            "--seed", str(self.seed),
            "--num_global_epochs", str(self.num_global_epochs),
            "--num_clients", "8",
            "--output_dir", str(self.output_dir()),
            "--algorithm_variant", self.algorithm_variant or self.name,
        ]
        if self.algorithm == "oort_compass":
            args += [
                "--oort_mode", self.oort_mode,
                "--oort_lambda_comm", str(self.oort_lambda_comm),
                "--oort_lambda_late", str(self.oort_lambda_late),
                "--oort_lambda_var", str(self.oort_lambda_var),
                "--oort_lambda_avail", str(self.oort_lambda_avail),
                "--oort_risk_threshold", str(self.oort_risk_threshold),
            ]
        if self.min_local_steps is not None:
            args += ["--min_local_steps", str(self.min_local_steps)]
        if self.max_local_steps is not None:
            args += ["--max_local_steps", str(self.max_local_steps)]
        if self.latest_time_factor is not None:
            args += ["--latest_time_factor", str(self.latest_time_factor)]
        return args


def stage_a_specs() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="fedcompass_baseline",
            algorithm="fedcompass",
            seed=2026,
            algorithm_variant="fedcompass_baseline",
            stage="A",
        ),
        ExperimentSpec(
            name="oort_shadow",
            algorithm="oort_compass",
            oort_mode="shadow",
            seed=2026,
            algorithm_variant="oort_shadow",
            stage="A",
        ),
    ]


def stage_b_specs() -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    for seed in SEEDS:
        specs.append(ExperimentSpec(
            name="fedcompass",
            algorithm="fedcompass",
            seed=seed,
            algorithm_variant="fedcompass",
            stage="B",
        ))
        specs.append(ExperimentSpec(
            name="q_only",
            algorithm="oort_compass",
            oort_mode="q_only",
            seed=seed,
            algorithm_variant="q_only_full",
            stage="B",
        ))
        specs.append(ExperimentSpec(
            name="q_and_group",
            algorithm="oort_compass",
            oort_mode="q_and_group",
            seed=seed,
            algorithm_variant="q_and_group_full",
            stage="B",
        ))
    return specs


def stage_c_specs() -> List[ExperimentSpec]:
    """消融：单因素 vs full vs q_and_group_full（seed=2026）。"""
    base = dict(algorithm="oort_compass", seed=2026, stage="C")
    return [
        ExperimentSpec(name="q_only_comm_only", oort_mode="q_only",
                       oort_lambda_comm=1.0, oort_lambda_late=0.0,
                       oort_lambda_var=0.0, oort_lambda_avail=0.0,
                       algorithm_variant="q_only_comm_only", **base),
        ExperimentSpec(name="q_only_late_only", oort_mode="q_only",
                       oort_lambda_comm=0.0, oort_lambda_late=1.0,
                       oort_lambda_var=0.0, oort_lambda_avail=0.0,
                       algorithm_variant="q_only_late_only", **base),
        ExperimentSpec(name="q_only_var_only", oort_mode="q_only",
                       oort_lambda_comm=0.0, oort_lambda_late=0.0,
                       oort_lambda_var=0.5, oort_lambda_avail=0.0,
                       algorithm_variant="q_only_var_only", **base),
        ExperimentSpec(name="q_only_full", oort_mode="q_only",
                       algorithm_variant="q_only_full", **base),
        ExperimentSpec(name="q_and_group_full", oort_mode="q_and_group",
                       algorithm_variant="q_and_group_full", **base),
    ]


def stage_d_specs() -> List[ExperimentSpec]:
    """鲁棒性：λ 扫描、latest_time_factor、Q 范围（seed=2026）。"""
    base = dict(algorithm="oort_compass", oort_mode="q_only", seed=2026, stage="D")
    specs: List[ExperimentSpec] = []
    for lam in (0.5, 1.0, 2.0):
        specs.append(ExperimentSpec(
            name=f"lambda_scale_{lam}",
            oort_lambda_comm=lam, oort_lambda_late=lam,
            oort_lambda_var=lam * 0.5, oort_lambda_avail=lam * 0.5,
            algorithm_variant=f"q_only_lambda_{lam}",
            **base,
        ))
    for ltf in (1.0, 1.5):
        specs.append(ExperimentSpec(
            name=f"ltf_{ltf}",
            latest_time_factor=ltf,
            algorithm_variant=f"q_only_ltf_{ltf}",
            **base,
        ))
    specs.append(ExperimentSpec(
        name="qrange_tight",
        min_local_steps=60, max_local_steps=160,
        algorithm_variant="q_only_qrange_tight",
        **base,
    ))
    specs.append(ExperimentSpec(
        name="qrange_wide",
        min_local_steps=30, max_local_steps=240,
        algorithm_variant="q_only_qrange_wide",
        **base,
    ))
    return specs


def all_specs() -> List[ExperimentSpec]:
    return stage_a_specs() + stage_b_specs() + stage_c_specs() + stage_d_specs()


# ──────────────────────── 一致性校验 ────────────────────────

SCHEDULER_SCHED_COLS = (
    "client_round_idx", "upload_group_id", "next_group_id",
    "local_steps", "dispatch_staleness", "aggregation_staleness",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare_trace_files(baseline_dir: Path, shadow_dir: Path) -> Dict[str, Any]:
    """Stage A：baseline vs shadow 逐文件一致性检查。"""
    results: Dict[str, Any] = {"passed": True, "checks": []}

    for fname in ("aggregation_trace.csv", "group_trace.csv", "dispatch_decision_trace.csv"):
        b, s = baseline_dir / fname, shadow_dir / fname
        identical = b.exists() and s.exists() and b.read_text() == s.read_text()
        results["checks"].append({"file": fname, "identical": identical})
        if not identical:
            results["passed"] = False

    # scheduler 调度相关列
    b_rows = _read_csv(baseline_dir / "scheduler_trace.csv")
    s_rows = _read_csv(shadow_dir / "scheduler_trace.csv")
    sched_match = (
        len(b_rows) == len(s_rows)
        and all(
            all(br.get(c, "") == sr.get(c, "") for c in SCHEDULER_SCHED_COLS)
            for br, sr in zip(b_rows, s_rows)
        )
    )
    results["checks"].append({"file": "scheduler_trace (sched cols)", "identical": sched_match})
    if not sched_match:
        results["passed"] = False

    # 每客户端 local_steps
    local_steps_ok = True
    for i in range(8):
        cid = f"client_{i}"
        bp = baseline_dir / "client_states" / cid / "round_reports.csv"
        sp = shadow_dir / "client_states" / cid / "round_reports.csv"
        br = _read_csv(bp)
        sr = _read_csv(sp)
        if len(br) != len(sr) or any(
            a.get("client_round_idx") != b.get("client_round_idx")
            or a.get("local_steps") != b.get("local_steps")
            for a, b in zip(br, sr)
        ):
            local_steps_ok = False
            break
    results["checks"].append({"file": "round_reports.local_steps", "identical": local_steps_ok})
    if not local_steps_ok:
        results["passed"] = False

    # shadow 应产生 oort_trace
    oort_exists = (shadow_dir / "oort_trace.csv").exists()
    baseline_no_oort = not (baseline_dir / "oort_trace.csv").exists()
    results["checks"].append({
        "file": "oort_trace (shadow only)",
        "identical": oort_exists and baseline_no_oort,
    })
    if not (oort_exists and baseline_no_oort):
        results["passed"] = False

    return results


# ──────────────────────── 指标汇总 ────────────────────────

def _staleness_values(agg_rows: List[Dict[str, str]]) -> List[int]:
    vals: List[int] = []
    for row in agg_rows:
        vals.extend(json.loads(row["per_client_staleness"]).values())
    return vals


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    """从单次实验输出目录汇总关键指标。"""
    agg = _read_csv(run_dir / "aggregation_trace.csv")
    sched = _read_csv(run_dir / "scheduler_trace.csv")
    groups = _read_csv(run_dir / "group_trace.csv")
    geval = _read_csv(run_dir / "global_eval_trace.csv")
    oort = _read_csv(run_dir / "oort_trace.csv")

    stale = _staleness_values(agg)
    late_count = sum(1 for r in sched if r.get("late") == "1")
    deadline_triggers = sum(1 for g in groups if g.get("trigger") == "deadline")
    late_in_group = sum(
        len([x for x in g.get("late_clients", "").split(",") if x.strip()])
        for g in groups
    )

    final_acc = float(geval[-1]["test_accuracy"]) if geval else 0.0
    final_vtime = float(geval[-1]["virtual_time"]) if geval else 0.0

    # Oort Q 方向验证：penalty>1 时 q_after_oort <= q_baseline
    q_direction_ok = 0
    q_direction_total = 0
    for row in oort:
        try:
            penalty = float(row.get("system_penalty", 1))
            qb = int(float(row["q_baseline"])) if row.get("q_baseline") not in ("", "-1") else None
            qo = int(float(row["q_after_oort"])) if row.get("q_after_oort") not in ("", "-1") else None
        except (ValueError, KeyError):
            continue
        if penalty > 1.0001 and qb is not None and qo is not None and qb >= 0 and qo >= 0:
            q_direction_total += 1
            if qo <= qb:
                q_direction_ok += 1

    return {
        "run_dir": str(run_dir),
        "num_aggregations": len(agg),
        "num_client_updates": sum(int(a.get("num_clients", 1)) for a in agg),
        "mean_staleness": round(statistics.mean(stale), 4) if stale else 0.0,
        "max_staleness": max(stale) if stale else 0,
        "late_uploads": late_count,
        "deadline_triggers": deadline_triggers,
        "late_clients_in_groups": late_in_group,
        "final_accuracy": round(final_acc, 2),
        "final_virtual_time": round(final_vtime, 4),
        "accuracy_time_curve": [
            {"virtual_time": float(r["virtual_time"]), "test_accuracy": float(r["test_accuracy"])}
            for r in geval
        ],
        "oort_q_direction_rate": (
            round(q_direction_ok / q_direction_total, 4) if q_direction_total else None
        ),
        "oort_q_direction_n": q_direction_total,
    }


def time_to_accuracy(curve: List[Dict[str, float]], target: float) -> Optional[float]:
    """达到 target 准确率所需的最小 virtual time。"""
    for pt in curve:
        if pt["test_accuracy"] >= target:
            return pt["virtual_time"]
    return None
