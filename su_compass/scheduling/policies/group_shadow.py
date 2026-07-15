"""RCP-GS：风险约束Pareto分组的Shadow策略。

RCP-GS（Risk-Constrained Pareto Group Scheduler）以FedCompass原分组决策为
安全基线，只在另一个已有group同时满足以下条件时提出换组建议：

1. 该group也是FedCompass原公式认可的合法候选；
2. 状态预测的safe finish不超过group latest；
3. mean finish落在group已有expected/latest余量内；
4. 候选Q不低于FedCompass基线Q，保护难任务所需本地训练量；
5. 到达偏差不大于基线group；
6. 至少在安全性、Q或到达偏差中的一项严格改善。

多个Pareto候选按“Q更大、到达偏差更小、group更早”的字典序精确选择。
如果没有支配基线的候选，则建议保持FedCompass；如果基线本身不安全且没有
安全候选，则建议走FedCompass原create-group路径。

本模块不访问controller、不修改arrival_group，仅返回候选事实和推荐结果。
Shadow阶段用于证明换组机会是否真实存在，不能当作实际group收益。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


@dataclass(frozen=True)
class GroupCandidate:
    """一个已有group下最优可行Q及其筛选事实。"""

    group_id: int
    fedcompass_reference_q: int
    candidate_q: int
    expected_arrival_time: float
    latest_arrival_time: float
    predicted_finish_time: float
    safe_finish_time: float
    arrival_deviation: float
    safe_slack: float
    original_group_feasible: bool
    safe_feasible: bool
    arrival_window_feasible: bool
    work_preserved: bool
    pareto_dominates_baseline: bool
    rejection_reason: str


@dataclass(frozen=True)
class GroupRecommendation:
    """一次在线dispatch的RCP-GS Shadow推荐。"""

    baseline_group_id: int
    baseline_q: int
    baseline_safe_feasible: bool
    baseline_arrival_deviation: float
    recommended_group_id: int
    recommended_q: int
    action: str
    group_changed: bool
    mismatch_repaired: bool
    reason: str
    candidates: Tuple[GroupCandidate, ...]


class ParetoGroupShadowPolicy:
    """在FedCompass原合法候选子集中寻找Pareto安全改组。"""

    def __init__(self, predictor: LatencyPredictor, q_increase_ratio: float = 0.20) -> None:
        if not 0.0 <= q_increase_ratio <= 1.0:
            raise ValueError("q_increase_ratio must be in [0, 1]")
        self.predictor = predictor
        self.q_increase_ratio = q_increase_ratio

    def recommend(
        self,
        *,
        client_id: str,
        dispatch_time: float,
        speed_smoothed: float,
        runtime_state: Optional[Any],
        groups: Dict[int, Dict[str, Any]],
        baseline_group_id: int,
        baseline_q: int,
        qmin: int,
        qmax: int,
    ) -> GroupRecommendation:
        """枚举已有group并返回Shadow推荐，不修改传入groups。"""
        if baseline_group_id not in groups:
            raise ValueError("baseline group must exist")

        baseline_group = groups[baseline_group_id]
        baseline_prediction = self.predictor.predict(PredictionContext(
            client_id=client_id,
            dispatch_time=dispatch_time,
            local_steps=baseline_q,
            speed_smoothed=speed_smoothed,
            runtime_state=runtime_state,
        ))
        baseline_deviation = abs(
            baseline_prediction.predicted_finish_time
            - float(baseline_group["expected_arrival_time"])
        )
        baseline_safe = (
            dispatch_time + baseline_prediction.safe_duration
            <= float(baseline_group["latest_arrival_time"])
        )

        candidates: List[GroupCandidate] = []
        dominating: List[GroupCandidate] = []
        availability_guard = (
            int(getattr(runtime_state, "unavailable_event_count", 0)) >= 2
            if runtime_state is not None else False
        )

        for group_id, group in groups.items():
            candidate = self._evaluate_group(
                client_id=client_id,
                dispatch_time=dispatch_time,
                speed_smoothed=speed_smoothed,
                runtime_state=runtime_state,
                group_id=group_id,
                group=group,
                baseline_group_id=baseline_group_id,
                baseline_q=baseline_q,
                baseline_safe=baseline_safe,
                baseline_deviation=baseline_deviation,
                qmin=qmin,
                qmax=qmax,
                availability_guard=availability_guard,
            )
            candidates.append(candidate)
            if candidate.pareto_dominates_baseline:
                dominating.append(candidate)

        if dominating:
            # 精确字典序：最大Q、最小到达偏差、更早expected、较小group id。
            chosen = min(
                dominating,
                key=lambda c: (
                    -c.candidate_q,
                    c.arrival_deviation,
                    c.expected_arrival_time,
                    c.group_id,
                ),
            )
            return GroupRecommendation(
                baseline_group_id=baseline_group_id,
                baseline_q=baseline_q,
                baseline_safe_feasible=baseline_safe,
                baseline_arrival_deviation=baseline_deviation,
                recommended_group_id=chosen.group_id,
                recommended_q=chosen.candidate_q,
                action="switch_existing_group",
                group_changed=chosen.group_id != baseline_group_id,
                mismatch_repaired=(not baseline_safe and chosen.safe_feasible),
                reason="pareto_safe_work_preserving_improvement",
                candidates=tuple(candidates),
            )

        action = "keep_fedcompass_group" if baseline_safe else "use_fedcompass_create_group"
        return GroupRecommendation(
            baseline_group_id=baseline_group_id,
            baseline_q=baseline_q,
            baseline_safe_feasible=baseline_safe,
            baseline_arrival_deviation=baseline_deviation,
            recommended_group_id=baseline_group_id if baseline_safe else -1,
            recommended_q=baseline_q,
            action=action,
            group_changed=False,
            mismatch_repaired=False,
            reason=(
                "no_pareto_dominating_group"
                if baseline_safe else "baseline_unsafe_no_safe_existing_group"
            ),
            candidates=tuple(candidates),
        )

    def _evaluate_group(
        self,
        *,
        client_id: str,
        dispatch_time: float,
        speed_smoothed: float,
        runtime_state: Optional[Any],
        group_id: int,
        group: Dict[str, Any],
        baseline_group_id: int,
        baseline_q: int,
        baseline_safe: bool,
        baseline_deviation: float,
        qmin: int,
        qmax: int,
        availability_guard: bool,
    ) -> GroupCandidate:
        """求一个group中满足原规则、Trust-Q和状态安全约束的最大Q。"""
        expected = float(group["expected_arrival_time"])
        latest = float(group["latest_arrival_time"])
        remaining = expected - dispatch_time
        q_fc = math.floor(remaining / max(speed_smoothed, 1e-8))
        original_feasible = remaining > 0 and qmin <= q_fc <= qmax

        if not original_feasible:
            return _rejected_candidate(
                group_id, q_fc, expected, latest, "not_fedcompass_original_candidate"
            )

        upper_q = min(qmax, int(math.floor(q_fc * (1.0 + self.q_increase_ratio))))
        if availability_guard:
            upper_q = min(upper_q, q_fc)

        feasible_rows = []
        saw_safe = False
        saw_window = False
        group_slack = max(0.0, latest - expected)
        for q in range(qmin, upper_q + 1):
            prediction = self.predictor.predict(PredictionContext(
                client_id=client_id,
                dispatch_time=dispatch_time,
                local_steps=q,
                speed_smoothed=speed_smoothed,
                runtime_state=runtime_state,
            ))
            safe_finish = dispatch_time + prediction.safe_duration
            deviation = abs(prediction.predicted_finish_time - expected)
            safe = safe_finish <= latest
            window_ok = deviation <= group_slack
            saw_safe = saw_safe or safe
            saw_window = saw_window or (safe and window_ok)
            if safe and window_ok and q >= baseline_q:
                feasible_rows.append((q, deviation, prediction, safe_finish))

        if not feasible_rows:
            reason = (
                "unsafe_deadline" if not saw_safe
                else "outside_arrival_window" if not saw_window
                else "would_reduce_training_work"
            )
            return GroupCandidate(
                group_id=group_id,
                fedcompass_reference_q=q_fc,
                candidate_q=-1,
                expected_arrival_time=expected,
                latest_arrival_time=latest,
                predicted_finish_time=0.0,
                safe_finish_time=0.0,
                arrival_deviation=float("inf"),
                safe_slack=float("-inf"),
                original_group_feasible=True,
                safe_feasible=saw_safe,
                arrival_window_feasible=saw_window,
                work_preserved=False,
                pareto_dominates_baseline=False,
                rejection_reason=reason,
            )

        # 组内先最大化Q，再最小化到达偏差。
        q, deviation, prediction, safe_finish = min(
            feasible_rows, key=lambda row: (-row[0], row[1])
        )
        strictly_better = (
            (not baseline_safe)
            or q > baseline_q
            or deviation + 1e-12 < baseline_deviation
        )
        dominates = (
            group_id != baseline_group_id
            and deviation <= baseline_deviation + 1e-12
            and q >= baseline_q
            and strictly_better
        )
        return GroupCandidate(
            group_id=group_id,
            fedcompass_reference_q=q_fc,
            candidate_q=q,
            expected_arrival_time=expected,
            latest_arrival_time=latest,
            predicted_finish_time=prediction.predicted_finish_time,
            safe_finish_time=safe_finish,
            arrival_deviation=deviation,
            safe_slack=latest - safe_finish,
            original_group_feasible=True,
            safe_feasible=True,
            arrival_window_feasible=True,
            work_preserved=True,
            pareto_dominates_baseline=dominates,
            rejection_reason="" if dominates else "does_not_pareto_dominate_baseline",
        )


def _rejected_candidate(
    group_id: int,
    q_fc: int,
    expected: float,
    latest: float,
    reason: str,
) -> GroupCandidate:
    return GroupCandidate(
        group_id=group_id,
        fedcompass_reference_q=q_fc,
        candidate_q=-1,
        expected_arrival_time=expected,
        latest_arrival_time=latest,
        predicted_finish_time=0.0,
        safe_finish_time=0.0,
        arrival_deviation=float("inf"),
        safe_slack=float("-inf"),
        original_group_feasible=False,
        safe_feasible=False,
        arrival_window_feasible=False,
        work_preserved=False,
        pareto_dominates_baseline=False,
        rejection_reason=reason,
    )
