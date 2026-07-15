"""基于完成时间预测的Shadow Q推荐策略。

本策略枚举FedCompass理论边界内的所有整数Q，并使用同一个状态预测器计算：
    - mean finish：用于贴近arrival group expected time；
    - safe finish：用于检查是否超过latest time。

选择规则：
    1. 优先选择safe finish不超过latest的候选；
    2. 在安全候选中最小化mean finish与expected的绝对偏差；
    3. 偏差相同时选择更大的Q，以保留更多本地训练量；
    4. 若没有安全候选，回退Qmin并明确记录原因。

该模块只返回QRecommendation，不写client_info或Dispatch。Shadow阶段FedCompass
仍执行原Q，因此推荐结果只能证明决策形态是否合理，不能当作实际系统收益。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from su_compass.scheduling.predictors.base import LatencyPredictor
from su_compass.scheduling.types import PredictionContext


@dataclass(frozen=True)
class QRecommendation:
    """一次Shadow Q枚举后的完整推荐结果。"""

    recommended_q: int
    predicted_duration: float
    predicted_finish_time: float
    safe_duration: float
    safe_finish_time: float
    expected_deviation: float
    safe_feasible: bool
    hit_qmin: bool
    hit_qmax: bool
    reason: str
    num_safe_candidates: int


class ShadowQPolicy:
    """在Q边界内联合使用mean/safe完成时间生成Shadow推荐。"""

    def __init__(self, predictor: LatencyPredictor) -> None:
        self.predictor = predictor

    def recommend(
        self,
        *,
        client_id: str,
        dispatch_time: float,
        speed_smoothed: float,
        runtime_state: Optional[Any],
        expected_arrival_time: Optional[float],
        latest_arrival_time: Optional[float],
        qmin: int,
        qmax: int,
        current_q: Optional[int] = None,
    ) -> QRecommendation:
        """枚举合法Q并返回推荐；缺少group时间时保守回退Qmax。"""
        if qmin <= 0 or qmax < qmin:
            raise ValueError("invalid Q range")
        if expected_arrival_time is None or latest_arrival_time is None:
            prediction = self.predictor.predict(PredictionContext(
                client_id=client_id,
                dispatch_time=dispatch_time,
                local_steps=qmax,
                speed_smoothed=speed_smoothed,
                runtime_state=runtime_state,
            ))
            return _to_recommendation(
                q=qmax,
                prediction=prediction,
                expected_arrival_time=expected_arrival_time,
                qmin=qmin,
                qmax=qmax,
                safe_feasible=True,
                reason="missing_group_time_keep_qmax",
                num_safe_candidates=0,
            )

        safe_candidates = []
        all_candidates = []
        for q in range(qmin, qmax + 1):
            prediction = self.predictor.predict(PredictionContext(
                client_id=client_id,
                dispatch_time=dispatch_time,
                local_steps=q,
                speed_smoothed=speed_smoothed,
                runtime_state=runtime_state,
            ))
            mean_deviation = abs(
                prediction.predicted_finish_time - expected_arrival_time
            )
            safe_finish = dispatch_time + prediction.safe_duration
            candidate = (mean_deviation, -q, q, prediction)
            all_candidates.append(candidate)
            if safe_finish <= latest_arrival_time:
                safe_candidates.append(candidate)

        if safe_candidates:
            _, _, q, prediction = min(safe_candidates)
            return _to_recommendation(
                q=q,
                prediction=prediction,
                expected_arrival_time=expected_arrival_time,
                qmin=qmin,
                qmax=qmax,
                safe_feasible=True,
                reason="closest_expected_under_safe_deadline",
                num_safe_candidates=len(safe_candidates),
            )

        # 连Qmin都无法通过safe deadline时，减少训练量也解决不了当前group失配。
        # Q-only阶段保持FedCompass原Q，避免“仍然迟到且训练量更少”的双重损失；
        # 后续group策略再根据该标记尝试更晚的group或创建新group。
        q = current_q if current_q is not None else qmin
        q = min(max(int(q), qmin), qmax)
        prediction = next(item[3] for item in all_candidates if item[2] == q)
        return _to_recommendation(
            q=q,
            prediction=prediction,
            expected_arrival_time=expected_arrival_time,
            qmin=qmin,
            qmax=qmax,
            safe_feasible=False,
            reason="no_safe_q_keep_fedcompass" if current_q is not None else "no_safe_q_fallback_qmin",
            num_safe_candidates=0,
        )


def _to_recommendation(
    *,
    q: int,
    prediction,
    expected_arrival_time: Optional[float],
    qmin: int,
    qmax: int,
    safe_feasible: bool,
    reason: str,
    num_safe_candidates: int,
) -> QRecommendation:
    deviation = (
        abs(prediction.predicted_finish_time - expected_arrival_time)
        if expected_arrival_time is not None
        else 0.0
    )
    return QRecommendation(
        recommended_q=q,
        predicted_duration=prediction.predicted_duration,
        predicted_finish_time=prediction.predicted_finish_time,
        safe_duration=prediction.safe_duration,
        safe_finish_time=prediction.predicted_finish_time
        + (prediction.safe_duration - prediction.predicted_duration),
        expected_deviation=deviation,
        safe_feasible=safe_feasible,
        hit_qmin=q == qmin,
        hit_qmax=q == qmax,
        reason=reason,
        num_safe_candidates=num_safe_candidates,
    )
