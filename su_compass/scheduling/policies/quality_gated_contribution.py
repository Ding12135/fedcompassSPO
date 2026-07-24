"""Bounded contribution restoration for heterogeneous asynchronous FL.

The policy is deliberately client-id agnostic.  It converts an effective
contribution deficit into a small, aggregation-level *shadow* weight bonus,
while retaining the original staleness coefficient as the base weight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from su_compass.scheduling.policies.fair_contribution_state import (
    ContributionRecord,
)


@dataclass(frozen=True)
class ContributionRestoration:
    client_id: str
    eligible: bool
    reason: str
    quality_score: float
    debt_score: float
    base_share: float
    requested_bonus: float
    allocated_bonus: float
    proposed_share: float


def recommend_contribution_restoration(
    records: Iterable[ContributionRecord],
    *,
    local_steps: dict[str, int],
    qmax: int,
    rhythm_debt: float,
    rhythm_stop: float,
    debt_score_cap: float,
    bonus_mass_cap: float,
    staleness_hard_cap: int,
) -> list[ContributionRestoration]:
    """Return a bounded counterfactual allocation for one aggregation.

    Quality is intentionally conservative until online loss telemetry is
    available: useful local work and freshness are both required.  The
    aggregate bonus shrinks continuously to zero as the rhythm queue reaches
    ``rhythm_stop``.
    """
    rows = list(records)
    if qmax <= 0 or debt_score_cap <= 0:
        raise ValueError("qmax and debt_score_cap must be positive")
    if bonus_mass_cap < 0 or rhythm_stop <= 0 or staleness_hard_cap < 0:
        raise ValueError("invalid restoration bound")

    health = max(0.0, 1.0 - max(0.0, rhythm_debt) / rhythm_stop)
    active_cap = bonus_mass_cap * health
    requested: dict[str, float] = {}
    metadata: dict[str, tuple[bool, str, float, float, float]] = {}
    for row in rows:
        debt = min(max(0.0, row.fair_debt_before), debt_score_cap)
        debt_score = debt / debt_score_cap
        step_score = min(1.0, max(0.0, local_steps.get(row.client_id, 0) / qmax))
        stale = row.staleness
        freshness = (
            math.exp(-max(0, stale) / max(1.0, staleness_hard_cap / 2.0))
            if stale >= 0 else 0.0
        )
        quality = step_score * freshness
        if row.raw_weight <= 0:
            eligible, reason = False, "not_participating"
        elif stale > staleness_hard_cap:
            eligible, reason = False, "staleness_hard_cap"
        elif debt_score <= 0:
            eligible, reason = False, "no_contribution_deficit"
        elif active_cap <= 0:
            eligible, reason = False, "rhythm_queue_stop"
        elif quality <= 0:
            eligible, reason = False, "quality_proxy_zero"
        else:
            eligible, reason = True, "bounded_quality_gated_restore"
        demand = (
            row.normalized_weight * debt_score * quality if eligible else 0.0
        )
        requested[row.client_id] = demand
        metadata[row.client_id] = (
            eligible, reason, quality, debt_score, row.normalized_weight,
        )

    total_requested = sum(requested.values())
    scale = (
        min(1.0, active_cap / total_requested)
        if total_requested > 0 else 0.0
    )
    allocated = {cid: value * scale for cid, value in requested.items()}
    total_bonus = sum(allocated.values())
    denominator = 1.0 + total_bonus

    result: list[ContributionRestoration] = []
    for row in rows:
        eligible, reason, quality, debt_score, base_share = metadata[row.client_id]
        bonus = allocated[row.client_id]
        result.append(ContributionRestoration(
            client_id=row.client_id,
            eligible=eligible,
            reason=reason,
            quality_score=quality,
            debt_score=debt_score,
            base_share=base_share,
            requested_bonus=requested[row.client_id],
            allocated_bonus=bonus,
            proposed_share=(base_share + bonus) / denominator,
        ))
    return result
