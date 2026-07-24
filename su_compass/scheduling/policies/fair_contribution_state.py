"""Effective-contribution deficit state for unified client scheduling.

The FedCompass aggregator gives each client a raw coefficient
``alpha / num_clients * staleness_fn(staleness)``.  The scheduler tracks the
normalized share of those coefficients at every aggregation epoch.  This
keeps the fairness state tied to the model update that was actually applied,
instead of using local Q or a binary participation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping


@dataclass(frozen=True)
class ContributionRecord:
    client_id: str
    target_share: float
    raw_weight: float
    normalized_weight: float
    target_effective_contribution: float
    effective_contribution: float
    staleness: int
    fair_debt_before: float
    fair_debt_raw: float
    fair_debt_score: float
    fair_debt_overflow: float


class FairContributionState:
    """Maintain an uncapped deficit and a bounded scheduling score."""

    def __init__(
        self,
        client_ids: Iterable[str],
        *,
        score_cap: float = 2.0,
        target_shares: Mapping[str, float] | None = None,
    ) -> None:
        ids = tuple(str(cid) for cid in client_ids)
        if not ids:
            raise ValueError("client_ids must not be empty")
        if score_cap <= 0:
            raise ValueError("score_cap must be positive")
        self.client_ids = ids
        self.score_cap = float(score_cap)
        if target_shares is None:
            target = 1.0 / len(ids)
            self.target_shares = {cid: target for cid in ids}
        else:
            missing = set(ids) - set(target_shares)
            if missing:
                raise ValueError(f"missing target shares for {sorted(missing)}")
            total = sum(float(target_shares[cid]) for cid in ids)
            if total <= 0:
                raise ValueError("target shares must have positive sum")
            self.target_shares = {
                cid: float(target_shares[cid]) / total for cid in ids
            }
        self.raw_debt: Dict[str, float] = {cid: 0.0 for cid in ids}
        self.cumulative_effective_share: Dict[str, float] = {
            cid: 0.0 for cid in ids
        }
        self.aggregation_epochs = 0

    def score(self, client_id: str) -> float:
        return min(self.raw_debt.get(client_id, 0.0), self.score_cap)

    def update(
        self,
        staleness: Mapping[str, int],
        *,
        alpha: float,
        staleness_fn: Callable[[int], float],
    ) -> list[ContributionRecord]:
        raw_weights = {
            str(cid): max(
                0.0,
                float(alpha)
                / len(self.client_ids)
                * float(staleness_fn(int(stale))),
            )
            for cid, stale in staleness.items()
        }
        total = sum(raw_weights.values())
        normalized = {
            cid: (weight / total if total > 0 else 0.0)
            for cid, weight in raw_weights.items()
        }
        records: list[ContributionRecord] = []
        for cid in self.client_ids:
            before = self.raw_debt[cid]
            target_service = total * self.target_shares[cid]
            service = raw_weights.get(cid, 0.0)
            after = max(0.0, before + target_service - service)
            self.raw_debt[cid] = after
            self.cumulative_effective_share[cid] += service
            score = min(after, self.score_cap)
            records.append(
                ContributionRecord(
                    client_id=cid,
                    target_share=self.target_shares[cid],
                    raw_weight=raw_weights.get(cid, 0.0),
                    normalized_weight=normalized.get(cid, 0.0),
                    target_effective_contribution=target_service,
                    effective_contribution=service,
                    staleness=int(staleness.get(cid, -1)),
                    fair_debt_before=before,
                    fair_debt_raw=after,
                    fair_debt_score=score,
                    fair_debt_overflow=max(0.0, after - self.score_cap),
                )
            )
        self.aggregation_epochs += 1
        return records

    def jain_index(self) -> float:
        values = list(self.cumulative_effective_share.values())
        denominator = len(values) * sum(value * value for value in values)
        if denominator <= 0:
            return 1.0
        return sum(values) ** 2 / denominator
