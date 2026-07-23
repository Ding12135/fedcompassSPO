"""Trace-driven sequential replay for the V2.1 stateful Shadow gate.

This replay uses recorded point-time curves and prequential calibration sources.
It validates state evolution and causal blocking, not model accuracy or a full
counterfactual event schedule.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from su_compass.scheduling.policies.controlled_q_candidates import (
    controlled_create_qs,
    controlled_join_qs,
)
from su_compass.scheduling.policies.lyapunov_group_q import (
    LyapunovAction,
    LyapunovGroupQPolicy,
    choose_effective_service_v2,
)
from su_compass.scheduling.state_time_model import QTimeCandidate


REFERENCES = {
    "client_0": 174, "client_1": 188, "client_2": 84, "client_3": 156,
    "client_4": 111, "client_5": 47, "client_6": 47, "client_7": 64,
}


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay(run_dir: Path) -> dict:
    regions = _rows(run_dir / "effective_service_region_shadow_trace.csv")
    calibration = {
        row["decision_id"]: row
        for row in _rows(run_dir / "calibrated_predictor_shadow_trace.csv")
    }
    times: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(run_dir / "state_time_trace.csv"):
        times[row["decision_id"]].append(row)

    groups: dict[int, dict] = {}
    next_group = -1
    settled: list[dict] = []
    counts: Counter[str] = Counter()
    for region in sorted(regions, key=lambda row: float(row["virtual_time"])):
        now = float(region["virtual_time"])
        for group_id in [gid for gid, group in groups.items() if group["deadline"] <= now]:
            group = groups.pop(group_id)
            settled.append({"size": len(group["clients"]), "safe": group["safe"] <= group["deadline"]})
        decision_id = region["decision_id"]
        client_id = region["client_id"]
        cal = calibration.get(decision_id)
        if cal is None or cal.get("calibration_source") == "analytical":
            counts["cold_start_defer"] += 1
            continue
        if any(client_id in group["clients"] for group in groups.values()):
            counts["shadow_dispatch_blocked"] += 1
            continue
        points = times.get(decision_id, [])
        if len(points) < 2:
            counts["missing_curve_defer"] += 1
            continue
        lo, hi = min(points, key=lambda row: int(row["q"])), max(points, key=lambda row: int(row["q"]))
        dq = max(1, int(hi["q"]) - int(lo["q"]))
        slope = (float(hi["predicted_duration"]) - float(lo["predicted_duration"])) / dq
        intercept = float(lo["predicted_duration"]) - slope * int(lo["q"])
        margin = float(cal["selected_margin"])
        curve = []
        for q in range(40, 201):
            duration = max(0.0, intercept + slope * q)
            curve.append(QTimeCandidate(
                q=q, predictor_name="trace_replay", predictor_source="mature_state",
                num_reports=8, used_fallback=False, fallback_reason="",
                predicted_duration=duration, safe_duration=duration + margin,
                predicted_finish_time=now + duration,
                safe_finish_time=now + duration + margin, uncertainty=margin,
                compute_duration=duration, communication_duration=0.0,
                spike_duration=0.0, availability_duration=0.0,
                availability_risk_duration=0.0,
            ))
        candidates = []
        for group_id, group in groups.items():
            qset = controlled_join_qs(
                curve=curve, target_time=group["frontier"], deadline=group["deadline"],
                group_safe_frontier=group["safe"], reference_q=REFERENCES[client_id],
                qmin=40, qmax=200, trust_eta=1.1,
            )
            by_q = {point.q: point for point in curve}
            for q in qset.candidate_qs:
                point = by_q[q]
                frontier = max(group["frontier"], point.predicted_finish_time)
                holding = max(0.0, frontier - point.predicted_finish_time)
                extension = max(0.0, frontier - group["frontier"])
                if not (holding > 85.0 and holding / max(point.predicted_duration, 1e-9) > 6.5):
                    candidates.append((extension * len(group["clients"]) + 0.25 * holding, group_id, point))
        if candidates:
            _, group_id, point = min(candidates, key=lambda item: (item[0], item[2].q))
            group = groups[group_id]
            group["clients"].append(client_id)
            group["frontier"] = max(group["frontier"], point.predicted_finish_time)
            group["safe"] = max(group["safe"], point.safe_finish_time)
            counts["join"] += 1
            continue
        create_qs = controlled_create_qs(
            curve=curve, reference_q=REFERENCES[client_id],
            qmin=40, qmax=200, trust_eta=1.1,
        )
        if not create_qs:
            counts["no_create_candidate"] += 1
            continue
        point = {candidate.q: candidate for candidate in curve}[max(create_qs)]
        recruit_expected = float(region["recruitment_expected"] or 16.4)
        recruit_safe = float(region["recruitment_safe"] or 19.68)
        groups[next_group] = {
            "clients": [client_id],
            "frontier": point.predicted_finish_time + recruit_expected,
            "safe": point.safe_finish_time + recruit_safe,
            "deadline": point.safe_finish_time + recruit_safe,
        }
        next_group -= 1
        counts["create"] += 1

    return {
        "scope": "trace_driven_sequential_group_replay_not_accuracy_or_full_event_replay",
        "decision_counts": dict(counts),
        "settled_groups": len(settled),
        "settled_singleton_rate": (
            sum(group["size"] == 1 for group in settled) / len(settled)
            if settled else None
        ),
        "settled_safe_rate": (
            sum(group["safe"] for group in settled) / len(settled)
            if settled else None
        ),
        "terminal_pending_groups": len(groups),
        "terminal_pending_sizes": sorted(len(group["clients"]) for group in groups.values()),
    }


def _curve_at(now: float, points: list[dict], margin: float) -> list[QTimeCandidate]:
    """Rebase a recorded point-time curve onto the replay's own clock."""
    lo = min(points, key=lambda row: int(row["q"]))
    hi = max(points, key=lambda row: int(row["q"]))
    dq = max(1, int(hi["q"]) - int(lo["q"]))
    slope = (float(hi["predicted_duration"]) - float(lo["predicted_duration"])) / dq
    intercept = float(lo["predicted_duration"]) - slope * int(lo["q"])
    return [
        QTimeCandidate(
            q=q, predictor_name="independent_trace_replay",
            predictor_source="mature_state", num_reports=8,
            used_fallback=False, fallback_reason="",
            predicted_duration=max(0.0, intercept + slope * q),
            safe_duration=max(0.0, intercept + slope * q) + margin,
            predicted_finish_time=now + max(0.0, intercept + slope * q),
            safe_finish_time=now + max(0.0, intercept + slope * q) + margin,
            uncertainty=margin, compute_duration=max(0.0, intercept + slope * q),
            communication_duration=0.0, spike_duration=0.0,
            availability_duration=0.0, availability_risk_duration=0.0,
        )
        for q in range(40, 201)
    ]


def replay_independent(
    run_dir: Path, *, recruit_safe_cap_ratio: float | None = None,
    create_safe_cost: bool | None = None,
    join_cadence_weight: float | None = None,
) -> dict:
    """Run a bounded counterfactual clock after each client's mature branch point.

    Recorded curves/calibration margins remain exogenous.  Dispatch times do not:
    after the branch, only a counterfactual group settlement can make a member
    ready again.  This removes the online Shadow's duplicate-dispatch artifact.
    """
    times: dict[str, list[dict]] = defaultdict(list)
    with (run_dir / "experiment_config.json").open(encoding="utf-8") as handle:
        raw_config = json.load(handle)
    config = raw_config.get("state_driven", raw_config)
    rhythm_target = float(config.get("lyapunov_rhythm_target", 16.4))
    cap_ratio = float(
        config.get("lyapunov_recruit_safe_cap_ratio", 1_000_000.0)
        if recruit_safe_cap_ratio is None else recruit_safe_cap_ratio
    )
    price_safe = bool(
        config.get("lyapunov_create_safe_cost", False)
        if create_safe_cost is None else create_safe_cost
    )
    cadence_weight = float(
        config.get("lyapunov_join_cadence_weight", 0.0)
        if join_cadence_weight is None else join_cadence_weight
    )
    region_extension_ratio = float(config.get("lyapunov_region_extension_ratio", 0.1))
    create_hysteresis = float(config.get("lyapunov_create_hysteresis", 0.1))
    policy = LyapunovGroupQPolicy(
        rhythm_target=rhythm_target,
        tradeoff_v=float(config.get("lyapunov_v", 1.0)),
        max_holding_wait=float(config.get("lyapunov_max_holding_wait", 85.0)),
        q_trust_eta=float(config.get("lyapunov_q_trust_eta", 1.1)),
        create_penalty=float(config.get("lyapunov_create_penalty", 0.25)),
        enable_rhythm_queue=bool(config.get("lyapunov_enable_rhythm_queue", True)),
        enable_workload_queue=False,
        action_scope="effective_service_v2_1",
        holding_weight=float(config.get("lyapunov_holding_weight", 0.25)),
        max_holding_ratio=float(config.get("lyapunov_max_holding_ratio", 6.5)),
        join_cadence_weight=cadence_weight,
    )
    for row in _rows(run_dir / "state_time_trace.csv"):
        times[row["decision_id"]].append(row)
    profiles: dict[str, deque[dict]] = defaultdict(deque)
    skipped_cold = 0
    for row in sorted(
        _rows(run_dir / "effective_service_region_shadow_trace.csv"),
        key=lambda item: float(item["virtual_time"]),
    ):
        if row.get("calibration_source") == "analytical":
            skipped_cold += 1
            continue
        decision_points = times.get(row["decision_id"], [])
        if len(decision_points) < 2:
            continue
        profiles[row["client_id"]].append({
            "region": row, "points": decision_points,
            # This is the decision-time calibrated margin already embedded in
            # the recorded curve.  Do not require a later realized outcome:
            # terminal in-flight dispatches legitimately have no outcome row.
            "margin": max(
                float(point["safe_duration"]) - float(point["predicted_duration"])
                for point in decision_points
            ),
        })

    events: list[tuple[float, int, str, object]] = []
    serial = 0
    for client_id, stream in profiles.items():
        if stream:
            heapq.heappush(events, (
                float(stream[0]["region"]["virtual_time"]), serial,
                "ready", client_id,
            ))
            serial += 1

    groups: dict[int, dict] = {}
    next_group = -1
    settled: list[dict] = []
    counts: Counter[str] = Counter()
    dispatches = 0
    dispatched_q = qmin_count = qmax_count = 0
    terminal_time = 0.0
    rhythm_debt = 0.0
    rhythm_debt_max = 0.0
    last_aggregation_time = min((event[0] for event in events), default=0.0)
    recent_intervals: list[float] = []
    recruit_safe_raw_max = 0.0
    recruit_safe_applied_max = 0.0
    recruit_expected_raw_max = 0.0
    recruit_expected_applied_max = 0.0
    recruit_safe_clipped = 0
    while events:
        now, _, event_type, payload = heapq.heappop(events)
        terminal_time = max(terminal_time, now)
        if event_type == "settle":
            group_id = int(payload)
            group = groups.get(group_id)
            if group is None or group["deadline"] != now:
                continue
            groups.pop(group_id)
            delta_t = max(0.0, now - last_aggregation_time)
            rhythm_debt = max(0.0, rhythm_debt + delta_t - rhythm_target)
            rhythm_debt_max = max(rhythm_debt_max, rhythm_debt)
            last_aggregation_time = now
            recent_intervals.append(delta_t)
            del recent_intervals[:-12]
            settled.append({
                "size": len(group["clients"]),
                "safe": group["safe"] <= group["deadline"],
            })
            for client_id in group["clients"]:
                if profiles[client_id]:
                    heapq.heappush(events, (now, serial, "ready", client_id))
                    serial += 1
            continue

        client_id = str(payload)
        if not profiles[client_id]:
            counts["profile_exhausted"] += 1
            continue
        profile = profiles[client_id].popleft()
        region = profile["region"]
        curve = _curve_at(now, profile["points"], profile["margin"])
        by_q = {point.q: point for point in curve}
        actions: list[LyapunovAction] = []
        for group_id, group in groups.items():
            qset = controlled_join_qs(
                curve=curve, target_time=group["frontier"], deadline=group["deadline"],
                group_safe_frontier=group["safe"], reference_q=REFERENCES[client_id],
                qmin=40, qmax=200, trust_eta=1.1,
            )
            for q in qset.candidate_qs:
                point = by_q[q]
                frontier = max(group["frontier"], point.predicted_finish_time)
                holding = max(0.0, frontier - point.predicted_finish_time)
                extension = max(0.0, frontier - group["frontier"])
                if not (holding > 85.0 and holding / max(point.predicted_duration, 1e-9) > 6.5):
                    actions.append(LyapunovAction(
                        mode="join", group_id=group_id, q=q,
                        predicted_finish_time=point.predicted_finish_time,
                        predicted_duration=point.predicted_duration,
                        safe_finish_time=point.safe_finish_time,
                        group_frontier_time=group["frontier"],
                        latest_arrival_time=group["deadline"],
                        deadline_safe=point.safe_finish_time <= group["deadline"],
                        holding_wait=holding, external_wait=extension,
                        affected_pending_clients=len(group["clients"]),
                        predicted_sojourn=frontier - now,
                        effective_work=q / 200.0,
                        utility=math.log1p(q / 40.0),
                    ))
        create_qs = controlled_create_qs(
            curve=curve, reference_q=REFERENCES[client_id],
            qmin=40, qmax=200, trust_eta=1.1,
        )
        recent = sorted(recent_intervals)
        if recent:
            recruit_expected_raw = recent[len(recent) // 2]
            rank = min(len(recent) - 1, math.ceil(0.85 * (len(recent) + 1)) - 1)
            recruit_safe_raw = recent[rank]
        else:
            recruit_expected_raw = rhythm_target
            recruit_safe_raw = 1.2 * rhythm_target
        recruit_cap = cap_ratio * rhythm_target
        recruit_expected = min(recruit_expected_raw, recruit_cap)
        recruit_safe = max(recruit_expected, min(recruit_safe_raw, recruit_cap))
        recruit_expected_raw_max = max(recruit_expected_raw_max, recruit_expected_raw)
        recruit_expected_applied_max = max(recruit_expected_applied_max, recruit_expected)
        recruit_safe_raw_max = max(recruit_safe_raw_max, recruit_safe_raw)
        recruit_safe_applied_max = max(recruit_safe_applied_max, recruit_safe)
        recruit_safe_clipped += int(
            recruit_expected < recruit_expected_raw
            or recruit_safe < recruit_safe_raw
        )
        for q in create_qs:
            point = by_q[q]
            actions.append(LyapunovAction(
                mode="create", group_id=-1, q=q,
                predicted_finish_time=point.predicted_finish_time,
                predicted_duration=point.predicted_duration,
                safe_finish_time=point.safe_finish_time + recruit_safe,
                group_frontier_time=point.predicted_finish_time + recruit_expected,
                latest_arrival_time=point.safe_finish_time + recruit_safe,
                deadline_safe=True, holding_wait=recruit_expected,
                external_wait=recruit_expected, affected_pending_clients=0,
                predicted_sojourn=point.predicted_duration + (
                    recruit_safe if price_safe else recruit_expected
                ),
                effective_work=q / 200.0, utility=math.log1p(q / 40.0),
            ))
        scored = policy.score(
            actions, rhythm_debt=rhythm_debt, workload_debt=0.0,
            qmax=200, qmin=40, fedcompass_join_q=REFERENCES[client_id],
        )
        selection = choose_effective_service_v2(
            scored,
            obvious_extension_limit=region_extension_ratio * rhythm_target,
            obvious_holding_limit=rhythm_target,
            create_hysteresis=create_hysteresis,
        )
        counts[selection.region] += 1
        selected = selection.decision.action
        if selected is None:
            counts["no_legal_action"] += 1
            continue
        point = by_q[selected.q]
        qmin_count += int(selected.q == 40)
        qmax_count += int(selected.q == 200)
        dispatched_q += selected.q
        if selected.mode == "join":
            group_id = selected.group_id
            group = groups[group_id]
            group["clients"].append(client_id)
            group["frontier"] = max(group["frontier"], point.predicted_finish_time)
            group["safe"] = max(group["safe"], point.safe_finish_time)
            counts["join"] += 1
        else:
            deadline = selected.latest_arrival_time
            groups[next_group] = {
                "clients": [client_id],
                "frontier": selected.group_frontier_time,
                "safe": selected.safe_finish_time,
                "deadline": deadline,
            }
            heapq.heappush(events, (deadline, serial, "settle", next_group))
            serial += 1
            next_group -= 1
            counts["create"] += 1
        dispatches += 1

    return {
        "scope": "independent_counterfactual_clock_system_replay_not_accuracy_replay",
        "branch": {
            "rule": "first_non_analytical_profile_per_client",
            "clients": len(profiles), "cold_profiles_skipped": skipped_cold,
        },
        "dispatches": dispatches,
        "decision_counts": dict(counts),
        "blocked_dispatches": 0,
        "settled_groups": len(settled),
        "settled_mean_size": (
            sum(group["size"] for group in settled) / len(settled) if settled else None
        ),
        "settled_singleton_rate": (
            sum(group["size"] == 1 for group in settled) / len(settled)
            if settled else None
        ),
        "settled_safe_rate": (
            sum(group["safe"] for group in settled) / len(settled)
            if settled else None
        ),
        "terminal_time": terminal_time,
        "dispatched_q": dispatched_q,
        "dispatched_q_per_time": (
            dispatched_q / terminal_time if terminal_time else None
        ),
        "qmin_rate": qmin_count / dispatches if dispatches else None,
        "qmax_rate": qmax_count / dispatches if dispatches else None,
        "rhythm_debt_max": rhythm_debt_max,
        "rhythm_debt_terminal": rhythm_debt,
        "terminal_pending_groups": len(groups),
        "terminal_pending_sizes": sorted(len(group["clients"]) for group in groups.values()),
        "remaining_profiles": sum(len(stream) for stream in profiles.values()),
        "recruitment_guard": {
            "safe_cap_ratio": cap_ratio,
            "create_safe_cost": price_safe,
            "join_cadence_weight": cadence_weight,
            "raw_safe_max": recruit_safe_raw_max,
            "applied_safe_max": recruit_safe_applied_max,
            "raw_expected_max": recruit_expected_raw_max,
            "applied_expected_max": recruit_expected_applied_max,
            "clipped_decisions": recruit_safe_clipped,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("independent", "sequential"), default="independent",
    )
    parser.add_argument("--recruit_safe_cap_ratio", type=float)
    parser.add_argument("--create_safe_cost", action="store_true", default=None)
    parser.add_argument("--join_cadence_weight", type=float)
    args = parser.parse_args()
    result = (
        replay_independent(
            args.run_dir,
            recruit_safe_cap_ratio=args.recruit_safe_cap_ratio,
            create_safe_cost=args.create_safe_cost,
            join_cadence_weight=args.join_cadence_weight,
        )
        if args.mode == "independent" else replay(args.run_dir)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
