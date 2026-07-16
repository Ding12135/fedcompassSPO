"""Unified state-aware time model used by State-Driven FedCompass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from su_compass.scheduling.predictors import AdaptiveLatencyPredictor
from su_compass.scheduling.types import PredictionContext


@dataclass(frozen=True)
class QTimeCandidate:
    q: int
    predictor_name: str
    predictor_source: str
    num_reports: int
    used_fallback: bool
    fallback_reason: str
    predicted_duration: float
    safe_duration: float
    predicted_finish_time: float
    safe_finish_time: float
    uncertainty: float
    compute_duration: float
    communication_duration: float
    spike_duration: float
    availability_duration: float
    availability_risk_duration: float

    def to_trace(self) -> dict:
        row = self.__dict__.copy()
        row["used_fallback"] = int(self.used_fallback)
        return row


class StateTimeModel:
    """The only time-query surface used by the state-driven scheduler."""

    version = "state_time_model_v1"

    def __init__(self, predictor=None) -> None:
        self.predictor = predictor or AdaptiveLatencyPredictor()

    @property
    def name(self) -> str:
        return self.predictor.name

    def predict_q(
        self, *, client_id: str, dispatch_time: float, q: int,
        runtime_state: Any, speed_fallback: float,
    ) -> QTimeCandidate:
        pred = self.predictor.predict(PredictionContext(
            client_id=client_id,
            dispatch_time=dispatch_time,
            local_steps=q,
            speed_smoothed=speed_fallback,
            runtime_state=runtime_state,
        ))
        finite = all(math.isfinite(float(value)) for value in (
            pred.predicted_duration, pred.safe_duration,
            pred.predicted_finish_time,
        ))
        invalid = (
            not finite or pred.predicted_duration < 0
            or pred.safe_duration < pred.predicted_duration
        )
        used_fallback = bool(pred.used_fallback or invalid)
        num_reports = int(pred.num_reports)
        if invalid:
            duration = max(0.0, q * speed_fallback)
            safe_duration = duration
            source = "fedcompass_fallback"
            reason = "invalid_state_prediction"
        else:
            duration = float(pred.predicted_duration)
            safe_duration = float(pred.safe_duration)
            source = (
                "fedcompass_fallback" if pred.used_fallback
                else "mature_state" if num_reports >= 5
                else "blended_state"
            )
            reason = "predictor_fallback" if pred.used_fallback else ""
        return QTimeCandidate(
            q=int(q), predictor_name=pred.predictor_name,
            predictor_source=source, num_reports=num_reports,
            used_fallback=used_fallback, fallback_reason=reason,
            predicted_duration=duration, safe_duration=safe_duration,
            predicted_finish_time=dispatch_time + duration,
            safe_finish_time=dispatch_time + safe_duration,
            uncertainty=float(pred.uncertainty),
            compute_duration=float(pred.compute_duration),
            communication_duration=float(pred.communication_duration),
            spike_duration=float(pred.spike_duration),
            availability_duration=float(pred.availability_duration),
            availability_risk_duration=float(pred.availability_risk_duration),
        )

    def predict_curve(
        self, *, client_id: str, dispatch_time: float,
        qs: Iterable[int], runtime_state: Any, speed_fallback: float,
    ) -> tuple[list[QTimeCandidate], bool]:
        curve = [self.predict_q(
            client_id=client_id, dispatch_time=dispatch_time, q=q,
            runtime_state=runtime_state, speed_fallback=speed_fallback,
        ) for q in qs]
        monotonic = all(
            curve[i].predicted_duration <= curve[i + 1].predicted_duration
            and curve[i].safe_duration <= curve[i + 1].safe_duration
            for i in range(len(curve) - 1)
        )
        return curve, monotonic


def state_group_window(point: QTimeCandidate, min_group_slack: float) -> tuple[float, float]:
    """Correct-by-construction state group window."""
    if min_group_slack < 0:
        raise ValueError("min_group_slack must be non-negative")
    expected = point.predicted_finish_time
    latest = max(point.safe_finish_time, expected + min_group_slack)
    return expected, latest
