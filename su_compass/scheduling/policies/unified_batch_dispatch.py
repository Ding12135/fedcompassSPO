"""Batch-level redispatch ordering without permanent speed classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class BatchPriority:
    client_id: str
    fair_debt: float
    predicted_safe_duration: float
    freshness: float
    service_probability: float
    benefit: float
    time_cost: float
    priority: float


def rank_unified_batch(
    client_ids: Iterable[str],
    *,
    fair_debt: Mapping[str, float],
    safe_duration: Mapping[str, float],
    rhythm_target: float,
    freshness: Mapping[str, float] | None = None,
    service_probability: Mapping[str, float] | None = None,
) -> list[BatchPriority]:
    """Rank one simultaneous redispatch batch using a common objective.

    The duration term is normalized by the established cadence target.  It is
    a cost, not a client class: a currently slow client can still lead when
    its effective-contribution deficit justifies the predicted time.
    """

    if rhythm_target <= 0:
        raise ValueError("rhythm_target must be positive")
    freshness = freshness or {}
    service_probability = service_probability or {}
    rows: list[BatchPriority] = []
    for cid in client_ids:
        debt = max(0.0, float(fair_debt.get(cid, 0.0)))
        duration = max(0.0, float(safe_duration.get(cid, rhythm_target)))
        fresh = min(1.0, max(0.0, float(freshness.get(cid, 1.0))))
        probability = min(
            1.0, max(0.0, float(service_probability.get(cid, 1.0)))
        )
        benefit = debt * fresh * probability
        time_cost = duration / rhythm_target
        # A bounded ratio avoids a tiny duration dominating the ordering.
        priority = benefit / (1.0 + time_cost)
        rows.append(
            BatchPriority(
                client_id=cid,
                fair_debt=debt,
                predicted_safe_duration=duration,
                freshness=fresh,
                service_probability=probability,
                benefit=benefit,
                time_cost=time_cost,
                priority=priority,
            )
        )
    return sorted(rows, key=lambda row: (-row.priority, row.client_id))

