"""FedCompass 调度问题的影子观测模块。

本模块用于回答论文问题：FedCompass 使用历史平滑单步耗时预测下一轮完成
时间时，是否在动态端侧环境中产生系统性误差，以及这些误差是否与迟到和
group deadline 相关。

工作方式：
    1. 记录 FedCompass 每次真实下发决策，但不修改该决策；
    2. 客户端下一轮完成后，将下发时预测与实际运行事实配对；
    3. 输出逐轮预测误差，并在实验结束时生成按画像类型汇总的问题报告。

该模块是纯观测层：不持有控制器引用、不写 client_info、不改变 Q/group，
因此启用或关闭它都不应影响 FedCompass 的调度与聚合结果。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from su_compass.scheduling.predictors import (
    AdaptiveLatencyPredictor,
    DecomposedLatencyPredictor,
    FedCompassLatencyPredictor,
    LatencyPredictor,
)
from su_compass.scheduling.types import LatencyPrediction, PredictionContext
from su_compass.scheduling.policies import QRecommendation, ShadowQPolicy


@dataclass(frozen=True)
class _PendingDispatch:
    """保存一次下发时已知的信息，等待下一轮实际返回后完成配对。"""

    client_id: str
    dispatch_time: float
    decision: str
    group_id: int
    local_steps: int
    speed_smoothed: float
    target_arrival_time: Optional[float]
    latest_arrival_time: Optional[float]
    baseline_prediction: LatencyPrediction
    state_prediction: LatencyPrediction
    q_recommendation: QRecommendation


class FedCompassProblemObserver:
    """配对 FedCompass 下发预测与客户端实际完成结果的影子观测器。"""

    def __init__(
        self,
        baseline_predictor: Optional[LatencyPredictor] = None,
        state_predictor: Optional[LatencyPredictor] = None,
    ) -> None:
        # key 使用 client_id + 下发虚拟时间；同一客户端的下一轮 report.dispatch_time
        # 应与该时间一致，因此不依赖容易变化的全局轮次编号。
        self._pending: Dict[Tuple[str, float], _PendingDispatch] = {}
        self._prediction_rows: List[Dict[str, Any]] = []
        # 保存每个客户端“最近一次已完成轮次”的状态。record_dispatch只能读取
        # 此缓存，不能读取下一轮完成后的状态，从结构上避免未来信息泄漏。
        self._latest_states: Dict[str, Any] = {}
        self._baseline_predictor = baseline_predictor or FedCompassLatencyPredictor()
        # 第一轮自适应版本仅对不可用事件采用专门策略，其他分量仍复用已经
        # 验证有效的时间分解预测器。
        self._state_predictor = state_predictor or AdaptiveLatencyPredictor()
        self._q_policy = ShadowQPolicy(self._state_predictor)

    @staticmethod
    def _time_key(value: float) -> float:
        """统一浮点虚拟时间精度，避免 CSV/内存浮点尾差导致配对失败。"""
        return round(float(value), 9)

    def record_dispatch(self, decision: Dict[str, Any]) -> None:
        """记录一次真实 FedCompass 下发决策，不对决策做任何修改。"""
        client_id = str(decision["client_id"])
        dispatch_time = float(decision["virtual_time"])
        target = _optional_float(decision.get("target_arrival_time"))
        latest = _optional_float(decision.get("latest_arrival_time"))
        runtime_state = self._latest_states.get(client_id)
        context = PredictionContext(
            client_id=client_id,
            dispatch_time=dispatch_time,
            local_steps=int(decision["assigned_local_steps"]),
            speed_smoothed=float(decision["speed_smoothed"]),
            runtime_state=runtime_state,
        )
        # 两个预测器看到完全相同的client、Q和dispatch时间，唯一差别是预测公式。
        # 这保证第一阶段比较的是“预测方法”，而不是不同Q造成的结果差异。
        baseline_prediction = self._baseline_predictor.predict(context)
        state_prediction = self._state_predictor.predict(context)
        qmin = int(decision.get("qmin", 1))
        qmax = int(decision.get("qmax", context.local_steps))
        q_recommendation = self._q_policy.recommend(
            client_id=client_id,
            dispatch_time=dispatch_time,
            speed_smoothed=context.speed_smoothed,
            runtime_state=runtime_state,
            expected_arrival_time=target,
            latest_arrival_time=latest,
            qmin=qmin,
            qmax=qmax,
            current_q=context.local_steps,
        )
        pending = _PendingDispatch(
            client_id=client_id,
            dispatch_time=dispatch_time,
            decision=str(decision.get("decision", "")),
            group_id=int(decision.get("assigned_group", -1)),
            local_steps=int(decision["assigned_local_steps"]),
            speed_smoothed=float(decision["speed_smoothed"]),
            target_arrival_time=target,
            latest_arrival_time=latest,
            baseline_prediction=baseline_prediction,
            state_prediction=state_prediction,
            q_recommendation=q_recommendation,
        )
        self._pending[(client_id, self._time_key(dispatch_time))] = pending

    def update_runtime_state(self, client_id: str, runtime_state: Any) -> None:
        """在一轮完成后更新客户端状态，供下一次dispatch预测使用。

        调用顺序必须是 observe_round -> update_runtime_state；这样当前轮的实际结果
        只会用于预测下一轮，而不会泄漏进当前轮预测。
        """
        if runtime_state is not None:
            self._latest_states[str(client_id)] = runtime_state

    def observe_round(
        self,
        *,
        client_id: str,
        client_round_idx: int,
        profile_type: str,
        report: Any,
        result: Any,
    ) -> None:
        """客户端完成后生成一条预测误差事实；冷启动首轮无决策时跳过。"""
        key = (str(client_id), self._time_key(report.dispatch_time))
        pending = self._pending.pop(key, None)
        if pending is None:
            # 初始化时所有客户端直接以 Qmax 启动，尚无历史 speed，不能构造公平的
            # FedCompass 下一轮预测，因此首轮不纳入预测误差统计。
            return

        baseline = pending.baseline_prediction
        state_prediction = pending.state_prediction
        q_recommendation = pending.q_recommendation
        predicted_duration = baseline.predicted_duration
        predicted_finish = baseline.predicted_finish_time
        actual_duration = float(report.round_time)
        actual_finish = float(report.finish_time)
        signed_error = actual_duration - predicted_duration
        absolute_error = abs(signed_error)
        relative_error = absolute_error / max(actual_duration, 1e-8)
        state_signed_error = actual_duration - state_prediction.predicted_duration
        state_absolute_error = abs(state_signed_error)
        state_relative_error = state_absolute_error / max(actual_duration, 1e-8)
        error_reduction = absolute_error - state_absolute_error
        error_reduction_ratio = error_reduction / max(absolute_error, 1e-8)

        # 各慢因直接采用本轮虚拟运行模型产生的事实值，不使用人为权重打分。
        communication_time = float(result.download_time + result.upload_time)
        non_compute_time = (
            communication_time
            + float(result.spike_delay)
            + float(result.availability_wait)
        )
        target_error = (
            actual_finish - pending.target_arrival_time
            if pending.target_arrival_time is not None
            else None
        )
        deadline_margin = (
            pending.latest_arrival_time - actual_finish
            if pending.latest_arrival_time is not None
            else None
        )

        self._prediction_rows.append({
            "client_round_idx": client_round_idx,
            "client_id": client_id,
            "profile_type": profile_type,
            "dispatch_time": pending.dispatch_time,
            "finish_time": actual_finish,
            "decision": pending.decision,
            "group_id": pending.group_id,
            "local_steps": pending.local_steps,
            "speed_smoothed_at_dispatch": pending.speed_smoothed,
            "predicted_duration": predicted_duration,
            "actual_duration": actual_duration,
            "signed_prediction_error": signed_error,
            "absolute_prediction_error": absolute_error,
            "relative_prediction_error": relative_error,
            "predicted_finish_time": predicted_finish,
            "target_arrival_time": _csv_value(pending.target_arrival_time),
            "latest_arrival_time": _csv_value(pending.latest_arrival_time),
            "target_arrival_error": _csv_value(target_error),
            "deadline_margin": _csv_value(deadline_margin),
            "late": int(report.late),
            "train_time": float(result.train_time),
            "communication_time": communication_time,
            "spike_delay": float(result.spike_delay),
            "availability_wait": float(result.availability_wait),
            "non_compute_time": non_compute_time,
            "communication_ratio": float(report.communication_ratio),
            # 以下字段属于第一阶段的状态预测shadow结果。它们不参与实际调度，
            # 只用于和上面的FedCompass baseline在相同Q下进行公平比较。
            "state_predicted_duration": state_prediction.predicted_duration,
            "state_predicted_finish_time": state_prediction.predicted_finish_time,
            "state_signed_prediction_error": state_signed_error,
            "state_absolute_prediction_error": state_absolute_error,
            "state_relative_prediction_error": state_relative_error,
            "state_error_reduction": error_reduction,
            "state_error_reduction_ratio": error_reduction_ratio,
            "state_uncertainty": state_prediction.uncertainty,
            "state_safe_duration": state_prediction.safe_duration,
            "state_compute_duration": state_prediction.compute_duration,
            "state_communication_duration": state_prediction.communication_duration,
            "state_spike_duration": state_prediction.spike_duration,
            "state_availability_duration": state_prediction.availability_duration,
            "state_availability_risk_duration": state_prediction.availability_risk_duration,
            "state_availability_event_rate": state_prediction.availability_event_rate,
            "state_availability_event_count": state_prediction.availability_event_count,
            "state_availability_strategy_active": int(
                state_prediction.availability_strategy_active
            ),
            "state_used_fallback": int(state_prediction.used_fallback),
            "state_num_reports_at_dispatch": state_prediction.num_reports,
            # Shadow Q只记录推荐，不替换pending.local_steps。反事实Q没有真实执行
            # 时间，因此这里只比较预测到达偏差和安全约束，不伪造实际收益。
            "fedcompass_actual_q": pending.local_steps,
            "shadow_recommended_q": q_recommendation.recommended_q,
            "shadow_q_difference": q_recommendation.recommended_q - pending.local_steps,
            "shadow_predicted_duration": q_recommendation.predicted_duration,
            "shadow_predicted_finish_time": q_recommendation.predicted_finish_time,
            "shadow_safe_duration": q_recommendation.safe_duration,
            "shadow_safe_finish_time": q_recommendation.safe_finish_time,
            "shadow_expected_deviation": q_recommendation.expected_deviation,
            "actual_q_state_expected_deviation": (
                abs(state_prediction.predicted_finish_time - pending.target_arrival_time)
                if pending.target_arrival_time is not None else 0.0
            ),
            "shadow_safe_feasible": int(q_recommendation.safe_feasible),
            "shadow_hit_qmin": int(q_recommendation.hit_qmin),
            "shadow_hit_qmax": int(q_recommendation.hit_qmax),
            "shadow_num_safe_candidates": q_recommendation.num_safe_candidates,
            "shadow_recommendation_reason": q_recommendation.reason,
        })

    def prediction_rows(self) -> List[Dict[str, Any]]:
        """返回逐轮诊断行的副本，供 TraceWriter 写出独立 CSV。"""
        return list(self._prediction_rows)

    def shadow_rows(self) -> List[Dict[str, Any]]:
        """返回包含两种预测器公平对照的逐轮shadow记录。"""
        return list(self._prediction_rows)

    def build_report(self, group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成论文问题证据汇总；只汇总事实，不判断新策略是否有效。"""
        rows = self._prediction_rows
        overall = _summarize(rows)
        by_profile: Dict[str, Dict[str, Any]] = {}
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["profile_type"])].append(row)
        for profile, profile_rows in sorted(grouped.items()):
            by_profile[profile] = _summarize(profile_rows)

        deadline_groups = sum(1 for row in group_rows if row.get("trigger") == "deadline")
        all_arrived_groups = sum(1 for row in group_rows if row.get("trigger") == "all_arrived")
        total_groups = len(group_rows)
        return {
            "report_purpose": (
                "观测 FedCompass 历史 speed 的下一轮完成时间预测误差，以及误差与迟到、"
                "group deadline 的关系；本报告不包含任何新策略干预。"
            ),
            "prediction": overall,
            "prediction_by_profile": by_profile,
            "groups": {
                "total": total_groups,
                "deadline_triggered": deadline_groups,
                "all_arrived": all_arrived_groups,
                "deadline_trigger_rate": deadline_groups / total_groups if total_groups else 0.0,
            },
            "interpretation_limits": [
                "预测误差与迟到的相关性可以支持问题存在，但不能单独证明因果关系。",
                "group deadline 比例可以证明分组结果存在失配，但不能证明当时存在更优候选组。",
                "是否存在更优 Q 或 group 必须在后续 shadow 反事实实验中验证。",
            ],
        }

    def build_shadow_report(self) -> Dict[str, Any]:
        """汇总基线与状态预测器在相同真实Q上的误差对照。"""
        rows = self._prediction_rows
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["profile_type"])].append(row)
        return {
            "report_purpose": (
                "在完全相同的FedCompass真实Q上，对比历史speed与端侧状态时间分解"
                "预测；所有状态均冻结于dispatch时刻，本报告不改变任何调度结果。"
            ),
            "overall": _summarize_shadow(rows),
            "by_profile": {
                profile: _summarize_shadow(profile_rows)
                for profile, profile_rows in sorted(grouped.items())
            },
            "acceptance_notes": [
                "先比较MAE/MAPE和有符号误差，不能只用late召回率判断方法有效。",
                "state_used_fallback=1的冷启动记录中，两种预测应完全相同。",
                "本阶段只能证明预测方法是否更准，不能证明shadow Q一定有效。",
            ],
        }

    def build_q_shadow_report(self) -> Dict[str, Any]:
        """汇总Shadow Q的决策形态，不把预测反事实当成实际完成结果。"""
        rows = self._prediction_rows
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["profile_type"])].append(row)
        return {
            "report_purpose": (
                "观察状态预测器推荐的Q是否合理；FedCompass仍执行原Q，本报告不声称"
                "Shadow Q已经产生真实late、deadline或accuracy收益。"
            ),
            "overall": _summarize_q_shadow(rows),
            "by_profile": {
                profile: _summarize_q_shadow(profile_rows)
                for profile, profile_rows in sorted(grouped.items())
            },
            "acceptance_notes": [
                "Qmin命中率过高表示safe约束可能过度保守。",
                "预测偏差改善只用于筛查推荐合理性，必须实际执行Q后才能验证系统收益。",
                "no_safe_q表示当前group时间窗不适合该客户端，后续可能需要group优化。",
            ],
        }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按统一口径汇总一组预测记录。"""
    if not rows:
        return {"count": 0}
    errors = [float(row["signed_prediction_error"]) for row in rows]
    absolute = [float(row["absolute_prediction_error"]) for row in rows]
    relative = [float(row["relative_prediction_error"]) for row in rows]
    late_rows = [row for row in rows if int(row["late"]) == 1]
    ontime_rows = [row for row in rows if int(row["late"]) == 0]
    late_flags = [float(row["late"]) for row in rows]
    return {
        "count": len(rows),
        "mae": _mean(absolute),
        "mape": _mean(relative),
        "mean_signed_error": _mean(errors),
        "underprediction_over_10pct_count": sum(
            float(row["signed_prediction_error"]) > 0.1 * float(row["actual_duration"])
            for row in rows
        ),
        "late_count": len(late_rows),
        "late_rate": len(late_rows) / len(rows),
        "late_mean_signed_error": _mean([
            float(row["signed_prediction_error"]) for row in late_rows
        ]),
        "ontime_mean_signed_error": _mean([
            float(row["signed_prediction_error"]) for row in ontime_rows
        ]),
        "signed_error_late_correlation": _correlation(errors, late_flags),
        "absolute_error_late_correlation": _correlation(absolute, late_flags),
    }


def _summarize_shadow(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """使用相同样本汇总两种预测器，避免样本口径不一致。"""
    if not rows:
        return {"count": 0}
    baseline_abs = [float(row["absolute_prediction_error"]) for row in rows]
    baseline_rel = [float(row["relative_prediction_error"]) for row in rows]
    baseline_signed = [float(row["signed_prediction_error"]) for row in rows]
    state_abs = [float(row["state_absolute_prediction_error"]) for row in rows]
    state_rel = [float(row["state_relative_prediction_error"]) for row in rows]
    state_signed = [float(row["state_signed_prediction_error"]) for row in rows]
    fallback_count = sum(int(row["state_used_fallback"]) for row in rows)
    baseline_mae = _mean(baseline_abs)
    state_mae = _mean(state_abs)
    return {
        "count": len(rows),
        "fallback_count": fallback_count,
        "fedcompass_mae": baseline_mae,
        "state_mae": state_mae,
        "mae_reduction": baseline_mae - state_mae,
        "mae_reduction_ratio": (
            (baseline_mae - state_mae) / baseline_mae if baseline_mae > 0 else 0.0
        ),
        "fedcompass_mape": _mean(baseline_rel),
        "state_mape": _mean(state_rel),
        "fedcompass_mean_signed_error": _mean(baseline_signed),
        "state_mean_signed_error": _mean(state_signed),
        "state_better_count": sum(s < b for s, b in zip(state_abs, baseline_abs)),
        "state_equal_count": sum(abs(s - b) <= 1e-12 for s, b in zip(state_abs, baseline_abs)),
        "state_worse_count": sum(s > b for s, b in zip(state_abs, baseline_abs)),
    }


def _summarize_q_shadow(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总Q推荐分布、边界命中和预测偏差变化。"""
    if not rows:
        return {"count": 0}
    actual_q = [int(row["fedcompass_actual_q"]) for row in rows]
    shadow_q = [int(row["shadow_recommended_q"]) for row in rows]
    actual_dev = [float(row["actual_q_state_expected_deviation"]) for row in rows]
    shadow_dev = [float(row["shadow_expected_deviation"]) for row in rows]
    return {
        "count": len(rows),
        "changed_count": sum(a != s for a, s in zip(actual_q, shadow_q)),
        "changed_rate": sum(a != s for a, s in zip(actual_q, shadow_q)) / len(rows),
        "mean_fedcompass_q": _mean([float(q) for q in actual_q]),
        "mean_shadow_q": _mean([float(q) for q in shadow_q]),
        "mean_q_difference": _mean([float(s - a) for a, s in zip(actual_q, shadow_q)]),
        "qmin_hit_count": sum(int(row["shadow_hit_qmin"]) for row in rows),
        "qmax_hit_count": sum(int(row["shadow_hit_qmax"]) for row in rows),
        "no_safe_q_count": sum(not int(row["shadow_safe_feasible"]) for row in rows),
        "mean_actual_q_predicted_deviation": _mean(actual_dev),
        "mean_shadow_q_predicted_deviation": _mean(shadow_dev),
        "predicted_deviation_reduction": _mean(actual_dev) - _mean(shadow_dev),
        "predicted_deviation_reduction_ratio": (
            (_mean(actual_dev) - _mean(shadow_dev)) / _mean(actual_dev)
            if _mean(actual_dev) > 0 else 0.0
        ),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _correlation(xs: List[float], ys: List[float]) -> float:
    """计算 Pearson 相关系数；常量序列返回 0，避免报告出现 NaN。"""
    if not xs or len(xs) != len(ys):
        return 0.0
    x_mean, y_mean = _mean(xs), _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_energy = sum((x - x_mean) ** 2 for x in xs)
    y_energy = sum((y - y_mean) ** 2 for y in ys)
    denominator = (x_energy * y_energy) ** 0.5
    return numerator / denominator if denominator > 0 else 0.0


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _csv_value(value: Optional[float]) -> Any:
    return "" if value is None else value
