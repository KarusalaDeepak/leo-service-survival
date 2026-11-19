#!/usr/bin/env python3
"""Causal delivery-queue replay; no observed EO bytes or physical long passes."""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r32_checkpoint_capacity as r32

r30, r31 = r32.r30, r32.r31
ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("edf", "edf_5min", "frozen", "isotonic")


def stable_int(value):
    return int.from_bytes(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class PublicContact:
    key: str
    sat: str
    gs: str
    start: int
    end: int
    rate_gb_s: float
    frozen: tuple
    isotonic: tuple


def s_at(curve, seconds):
    if seconds <= 0:
        return 1.0
    return float(curve[min(len(curve), math.ceil(seconds / 300)) - 1])


def expected_bytes(contact, now, payload_seconds, overhead, policy):
    curve = getattr(contact, policy)
    denominator = max(s_at(curve, now - contact.start), 1e-6)
    left, right = now + overhead, now + overhead + payload_seconds
    total = 0.0
    while left < right - 1e-9:
        edge = min(right, contact.start + (math.floor((left - contact.start) / 300) + 1) * 300)
        probability = min(1.0, s_at(curve, edge - contact.start) / denominator)
        total += (edge - left) * probability * contact.rate_gb_s
        left = edge
    return total


def make_inputs(instance_path, adjusted, seed):
    instance = r31.synthetic_long_contact_instance(instance_path, 45)
    eligible = r32.add_frozen_curves(adjusted, 12)
    # The common scaffold is geometry-only, independent of assigned outcomes.
    ledger, plans = common_scaffold(instance)
    tasks = {t["id"]: t for t in instance["tasks"]}
    jobs, fixed = [], {}
    for p in plans:
        ready = max([o[3] for o in p.observations] + [r[3] for r in p.relays])
        volumes = {}
        for sat, _gs, _b, _e, _r, amount in p.deliveries:
            volumes[sat] = volumes.get(sat, 0.0) + amount
        for sat, amount in volumes.items():
            jobs.append(dict(id=p.task_id + ":" + sat, task=p.task_id, sat=sat,
                             ready=ready, deadline=tasks[p.task_id]["deadline"],
                             priority=tasks[p.task_id]["priority"], volume=amount))
        for _aid, sat, b, e, _amount in p.observations:
            fixed.setdefault("sat:" + sat, []).append((b, e))
        for sat, other, b, e, _rate, _amount in p.relays:
            for key in ("sat:" + sat, "sat:" + other):
                fixed.setdefault(key, []).append((b, e))
    contacts, hidden = [], {}
    for sat, windows in instance["downlinks"].items():
        for gs, b, original_end, rate in windows:
            identity = (sat, gs, int(b), int(original_end))
            key = json.dumps(identity, separators=(",", ":"))
            minutes = (5, 15, 30, 45)[stable_int(["duration", *identity]) % 4]
            row = r30.keyed_episode(seed, identity, eligible)
            contact = PublicContact(key, sat, gs, int(b), int(b + 60 * minutes), rate / 8,
                                    row["curve_frozen_logistic"], row["curve_temporal_isotonic"])
            contacts.append(contact)
            onset = float("inf") if int(row["observed_survival_target"]) else b + (int(row["duration_bins"]) - 1) * 300
            hidden[key] = onset
    return instance, tasks, jobs, fixed, contacts, hidden


def common_scaffold(instance):
    """Geometry/resource-only common batch acquisition plan, no outcomes."""
    ledger, plans = r30.ms.Ledger(), []
    for task in r30.ms.ordered_tasks(instance):
        choices = r30.ms.candidate_plans(instance, task, ledger, single=False)
        if choices:
            chosen = min(choices, key=lambda p: (p.completion, p.variant))
            r30.ms.commit(chosen, ledger)
            plans.append(chosen)
    r30.ms.validate_schedule(instance, plans)
    return ledger, plans


def free_until(fixed, resource, now, limit):
    for start, end in fixed.get(resource, []):
        if start <= now < end:
            return now
        if now < start < limit:
            limit = start
    return limit


def select_action(job, contacts, now, busy, blocked, fixed, policy, overhead):
    """Policy boundary: receives no event time, target, or future outcome."""
    candidates = []
    for c in contacts:
        if c.sat != job["sat"] or c.key in blocked or not c.start <= now < c.end:
            continue
        keys = ("sat:" + c.sat, "gs:" + c.gs)
        if any(key in busy for key in keys):
            continue
        limit = min(c.end, job["deadline"])
        for key in keys:
            limit = free_until(fixed, key, now, limit)
        maximum = min(900.0, limit - now - overhead, job["remaining"] / c.rate_gb_s)
        if maximum <= 1e-8:
            continue
        sizes = [maximum] if policy == "edf" else ([min(300, maximum)] if policy == "edf_5min" else sorted(set([min(d, maximum) for d in (300, 600, 900)])))
        for duration in sizes:
            end = now + overhead + duration
            if policy in ("edf", "edf_5min"):
                score = (-end, c.key)
            else:
                expected = expected_bytes(c, now, duration, overhead, policy)
                score = (expected / (duration + overhead), -end, c.key)
            candidates.append((score, c, duration))
    if not candidates:
        return None
    _, contact, duration = max(candidates, key=lambda x: x[0])
    return contact, duration


def simulate(tasks, job_specs, fixed, contacts, hidden_onsets, storage, policy,
             load=1.0, overhead=10, capture_actions=False):
    jobs = {s["id"]: {**s, "volume": s["volume"] * load,
                      "remaining": s["volume"] * load, "status": "unready",
                      "available": s["ready"]} for s in job_specs}
    by_id = {c.key: c for c in contacts}
    events, serial = [], 0
    def push(time, kind, key):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (float(time), serial, kind, key))
    for job in jobs.values():
        push(job["ready"], "ready", job["id"])
        push(job["deadline"], "deadline", job["id"])
    for c in contacts:
        push(c.start, "wake", c.key)
        push(c.end, "wake", c.key)
        onset = hidden_onsets[c.key]
        if onset < c.end:
            push(onset + 600, "detect", c.key)
    for slots in fixed.values():
        for b, e in slots:
            push(b, "wake", "fixed")
            push(e, "wake", "fixed")
    active, busy, blocked = {}, {}, set()
    used = {sat: 0.0 for sat in storage}
    occupied = wasted = delivered = 0.0
    count = rejects = 0
    action_log, intervals = [], []
    clock = -float("inf")

    def finish(txid, now, detected=False):
        nonlocal occupied, wasted, delivered
        tx = active.pop(txid, None)
        if tx is None:
            return
        c, job = tx["contact"], jobs[tx["job"]]
        for resource in ("sat:" + c.sat, "gs:" + c.gs):
            if busy.get(resource) != txid:
                raise AssertionError("resource owner mismatch")
            del busy[resource]
            intervals.append((resource, tx["start"], now))
        good_seconds = max(0, min(now, hidden_onsets[c.key]) - tx["payload_start"])
        amount = min(job["remaining"], tx["amount"], good_seconds * c.rate_gb_s)
        job["remaining"] -= amount
        delivered += amount
        used[c.sat] -= amount
        occupied += now - tx["start"]
        wasted += max(0, now - max(tx["start"], hidden_onsets[c.key]))
        job["status"] = "done" if job["remaining"] <= 1e-7 else "queued"
        if detected:
            job["available"] = now + 10
            push(now + 10, "wake", "switch")
        if used[c.sat] < -1e-6:
            raise AssertionError("negative queue storage")

    while events:
        now = events[0][0]
        if now < clock:
            raise AssertionError("clock reversal")
        clock = now
        batch = []
        while events and events[0][0] == now:
            batch.append(heapq.heappop(events))
        # Detection takes precedence over a simultaneous completion, independent
        # of heap insertion order; its switching delay applies in either case.
        newly_detected = {key for _, _, kind, key in batch if kind == "detect"}
        blocked.update(newly_detected)
        for txid, tx in list(active.items()):
            if tx["contact"].key in newly_detected:
                finish(txid, now, True)
        for _, _, kind, key in batch:
            if kind == "finish":
                finish(key, now)
        for _, _, kind, key in batch:
            if kind == "ready":
                j = jobs[key]
                if used[j["sat"]] + j["remaining"] > storage[j["sat"]] + 1e-7:
                    j["status"] = "rejected"
                    rejects += 1
                else:
                    used[j["sat"]] += j["remaining"]
                    j["status"] = "queued"
            elif kind == "deadline":
                j = jobs[key]
                if j["status"] == "queued":
                    used[j["sat"]] -= j["remaining"]
                    j["status"] = "expired"
        queued = sorted((j for j in jobs.values() if j["status"] == "queued" and
                         j["available"] <= now < j["deadline"]),
                        key=lambda j: (j["deadline"], -j["priority"], j["id"]))
        for j in queued:
            choice = select_action(j, contacts, now, busy, blocked, fixed, policy, overhead)
            if choice is None:
                continue
            c, duration = choice
            count += 1
            txid = count
            end = now + overhead + duration
            if not (c.start <= now < end <= min(c.end, j["deadline"]) + 1e-8):
                raise AssertionError("deadline or contact violation")
            active[txid] = dict(job=j["id"], contact=c, start=now,
                                payload_start=now + overhead,
                                amount=min(j["remaining"], duration * c.rate_gb_s))
            for resource in ("sat:" + c.sat, "gs:" + c.gs):
                if resource in busy:
                    raise AssertionError("double resource booking")
                busy[resource] = txid
            j["status"] = "active"
            push(end, "finish", txid)
            if capture_actions:
                action_log.append((now, j["id"], c.key, round(duration, 6)))
    if active or busy:
        raise AssertionError("unfinished engine reservations")
    for resource in {r for r, _, _ in intervals}:
        slots = sorted([(b, e) for r, b, e in intervals if r == resource] + fixed.get(resource, []))
        if any(e > bb + 1e-7 for (_, e), (bb, _) in zip(slots, slots[1:])):
            raise AssertionError("independent overlap check failed")
    task_jobs = {}
    for j in jobs.values():
        if not -1e-6 <= j["remaining"] <= j["volume"] + 1e-6:
            raise AssertionError("byte conservation failed")
        task_jobs.setdefault(j["task"], []).append(j)
    completed = sum(tasks[tid]["priority"] for tid, group in task_jobs.items()
                    if all(j["status"] == "done" for j in group))
    # All released input tasks stay in denominators, not only the scaffold.
    all_volume = sum(a["volume"] for t in tasks.values() for a in t["atoms"]) * load
    result = dict(priority=completed / sum(t["priority"] for t in tasks.values()),
                  goodput=delivered / all_volume, chunks=count, storage_rejections=rejects,
                  occupied_seconds=occupied, post_onset_fraction=wasted / max(occupied, 1))
    if capture_actions:
        result["actions"] = action_log
    return result


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a frozen run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjusted, _, _ = r30.calibrate_and_score(r30.load_frozen_predictions(args.predictions))
    rows = []
    for seed in range(args.seeds):
        instance, tasks, jobs, fixed, contacts, hidden = make_inputs(args.instance, adjusted, seed)
        rate_factor = 1.0
        if args.capacity_profile == "duration_matched":
            _, scaffold = common_scaffold(instance)
            seconds = [amount / (rate / 8) for p in scaffold
                       for _s, _g, _b, _e, rate, amount in p.deliveries]
            rate_factor = float(np.median(seconds)) / 900
            contacts = [dataclasses.replace(c, rate_gb_s=c.rate_gb_s * rate_factor) for c in contacts]
        for load in (1.0, 1.3, 1.5):
            for overhead in (0, 10, 30):
                for policy in POLICIES:
                    row = simulate(tasks, jobs, fixed, contacts, hidden, instance["storage"],
                                   policy, load, overhead)
                    rows.append(dict(seed=seed, load=load, overhead=overhead, policy=policy, **row))
        print(f"completed seed {seed + 1}/{args.seeds}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary = frame.groupby(["load", "overhead", "policy"], as_index=False).mean(numeric_only=True).drop(columns="seed")
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    manifest = dict(protocol_sha256=r31.sha256(args.protocol), script_sha256=r31.sha256(Path(__file__)),
                    prediction_sha256=r31.sha256(args.predictions), overlay_sha256=r31.sha256(args.instance),
                    seeds=args.seeds, evidence="exploratory unsynchronized delivery-queue simulation",
                    source="common Paper 3 acquisition/relay scaffold; mixed constructed windows",
                    capacity_profile=args.capacity_profile, rate_factor=rate_factor,
                    duration_addendum_sha256=r31.sha256(ROOT / "docs/r34_duration_matching_addendum_2026-11-19.md") if args.capacity_profile == "duration_matched" else None,
                    supersedes="R33 recovery effect excluded because future-failure storage initialization")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=ROOT / "results/reviewer_prediction_release/wetlinks_all_pipeline_horizon_predictions.csv.gz")
    p.add_argument("--instance", type=Path, default=r30.PAPER3 / "data/external/eos_bench/eos_s1_overlay.json")
    p.add_argument("--protocol", type=Path, default=ROOT / "docs/r34_causal_chunk_protocol_2026-11-19.md")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--capacity-profile", choices=("nominal", "duration_matched"), default="nominal")
    run(p.parse_args())
