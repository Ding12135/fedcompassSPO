"""Pure counterfactual helpers reproducing FedCompass speed formulas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FedCompassGroupReference:
    feasible: bool
    group_id: int = -1
    q: int = -1
    expected_time: float = 0.0
    latest_time: float = 0.0


def existing_group_reference(
    *, now: float, speed: float, groups: Mapping[int, dict],
    qmin: int, qmax: int,
) -> FedCompassGroupReference:
    selected = FedCompassGroupReference(False)
    for group_id, group in groups.items():
        remaining = float(group["expected_arrival_time"]) - now
        if remaining <= 0:
            continue
        q = math.floor(remaining / speed)
        if qmin <= q <= qmax and q >= selected.q:
            selected = FedCompassGroupReference(
                True, group_id, q,
                float(group["expected_arrival_time"]),
                float(group["latest_arrival_time"]),
            )
    return selected


def new_group_reference_q(
    *, now: float, client_id: str, speed: float,
    groups: Mapping[int, dict], client_info: Mapping[str, dict],
    qmin: int, qmax: int,
) -> int:
    assigned = -1
    for group in groups.values():
        if now >= float(group["latest_arrival_time"]):
            continue
        clients = list(group.get("clients", [])) + list(group.get("arrived_clients", []))
        speeds = [float(client_info[c]["speed"]) for c in clients if c in client_info]
        if not speeds:
            continue
        estimated = float(group["latest_arrival_time"]) + min(speeds) * qmax
        q = math.floor((estimated - now) / speed)
        if q <= qmax:
            assigned = max(assigned, q)
    if assigned < 0:
        return qmax
    return max(qmin, assigned)


def new_group_reference_window(
    *, now: float, q: int, speed: float, latest_time_factor: float,
) -> tuple[float, float]:
    expected = now + q * speed
    latest = now + q * speed * latest_time_factor
    return expected, latest
