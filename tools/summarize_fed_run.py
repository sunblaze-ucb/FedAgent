#!/usr/bin/env python
"""Summarize fedagent.fed.run_fed output dirs: per-round reward, and compare conditions.

Reads each run's per-step metrics (critic/rewards/mean by default) and reports per round
the mean-over-clients of the round's mean and max step reward. With multiple LABEL=DIR args
it prints a comparison
table -- e.g. the A/B/C decomposition:
    catalog_split (env+task het) vs task_disjoint (task het) vs homogeneous (IID).
A-B isolates the env-heterogeneity effect, B-C the task-heterogeneity effect.

Beyond the per-round table this also reports, for the COMPOUNDING question:
  * per-condition TREND -- least-squares slope of round-mean reward vs round index, plus
    the first->last delta (does federated reward climb over rounds, or stay flat?);
  * the ASYMMETRY TRAJECTORY -- A-B (env effect) and B-C (task effect) at EVERY round and
    their slopes (does env heterogeneity degrade CUMULATIVELY relative to task/IID, the
    Input-Dynamics Asymmetry, or is the gap round-independent?).

The 3 conditions for the decomposition default to labels A,B,C; override with
    --decomp=ENVLABEL,TASKLABEL,IIDLABEL
so the real run labels can be used, e.g.:
    --decomp=envhet,task,homog

Run on the node where the logs live (compute node /tmp):
    python summarize_fed_run.py A=/tmp/.../scaled_env B=/tmp/.../scaled_task C=/tmp/.../scaled_homog
    python summarize_fed_run.py envhet=/tmp/...envhet task=/tmp/...task homog=/tmp/...homog \
        --decomp=envhet,task,homog
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import fedagent
from fedagent.fed.metrics_logger import parse_training_log  # noqa: E402

KEY = "critic/rewards/mean"


def _plan_clients(round_dir):
    """The round's selected client ids, in plan order (the order they were trained)."""
    plan = os.path.join(round_dir, "persistent_plan.json")
    if os.path.isfile(plan):
        try:
            with open(plan) as f:
                return [int(s["client"]) for s in json.load(f)]
        except Exception:
            pass
    ids = []
    for d in sorted(glob.glob(os.path.join(round_dir, "client_*"))):
        m = re.search(r"client_(\d+)$", d)
        if m:
            ids.append(int(m.group(1)))
    return ids


def _split_by_client(entries, client_ids):
    """Split a shared per-round metrics stream into one segment per client.

    The persistent path (fed/run_fed.py:1758) writes ONE round_<k>/json_logs/metrics.json for
    the whole round: the clients' steps are concatenated in plan order, each restarting its
    step counter at 1, so a DECREASE in `step` marks the next client. Cross-round runs share a
    single launch log, so that file is cumulative over rounds 1..k -- the round's own clients
    are then the LAST len(client_ids) segments (run_fed.py:1750-1751)."""
    segs, cur, last = [], [], None
    for e in entries:
        s = int(e.get("step", 0))
        if last is not None and s <= last and cur:
            segs.append(cur)
            cur = []
        cur.append(e)
        last = s
    if cur:
        segs.append(cur)
    if client_ids and len(segs) > len(client_ids):
        segs = segs[-len(client_ids):]              # cross-round cumulative log -> this round
    ids = client_ids if len(client_ids) == len(segs) else list(range(len(segs)))
    return dict(zip(ids, segs))


def _round_entries(round_dir):
    """{client -> [metric entries]} for one round, from whichever layout the run wrote.

    Source order mirrors how run_fed writes metrics: per-client json_logs (subprocess path,
    run_fed.py:1437), the shared per-round json_logs (persistent path, :1758), per-lane
    json_logs (lane path, :1642), and finally the raw client training.log (the original
    source -- kept so pre-json_logs runs still parse)."""
    out = {}
    for cj in sorted(glob.glob(os.path.join(round_dir, "client_*", "json_logs", "metrics.json"))):
        m = re.search(r"client_(\d+)", cj)
        if not m:
            continue
        with open(cj) as f:
            out[int(m.group(1))] = json.load(f)
    if out:
        return out
    ids = _plan_clients(round_dir)
    shared = os.path.join(round_dir, "json_logs", "metrics.json")
    if os.path.isfile(shared):
        with open(shared) as f:
            return _split_by_client(json.load(f), ids)
    lanes = sorted(glob.glob(os.path.join(round_dir, "json_logs_lane*", "metrics.json")))
    if lanes:
        for i, lj in enumerate(lanes):
            with open(lj) as f:
                out[ids[i] if i < len(ids) else i] = json.load(f)
        return out
    for log in sorted(glob.glob(os.path.join(round_dir, "client_*", "training.log"))):
        m = re.search(r"client_(\d+)", log)
        if m:
            out[int(m.group(1))] = parse_training_log(log)
    return out


def run_rounds(run_dir, key):
    """round -> {client -> (mean_reward_over_steps, max_reward)}"""
    rounds = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "round_*"))):
        m = re.search(r"round_(\d+)$", d)
        if not m or not os.path.isdir(d):
            continue
        rnd = int(m.group(1))
        for cl, entries in _round_entries(d).items():
            vals = [e["metrics"][key] for e in entries if key in e.get("metrics", {})]
            if vals:
                rounds.setdefault(rnd, {})[cl] = (sum(vals) / len(vals), max(vals))
    return rounds


def round_mean(rr, r):
    """mean over clients of the round's step-averaged reward, or None if absent."""
    v = rr.get(r, {})
    if not v:
        return None
    means = [x[0] for x in v.values()]
    return sum(means) / len(means)


def lin_slope(pairs):
    """least-squares slope of y vs x for [(x,y), ...]; None if <2 points."""
    pts = [(x, y) for x, y in pairs if y is not None]
    n = len(pts)
    if n < 2:
        return None
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    key = KEY
    decomp = ("A", "B", "C")
    for a in sys.argv[1:]:
        if a.startswith("--key="):
            key = a.split("=", 1)[1]
        elif a.startswith("--decomp="):
            parts = a.split("=", 1)[1].split(",")
            if len(parts) == 3:
                decomp = tuple(p.strip() for p in parts)
    runs = {}
    for a in args:
        label, _, d = a.partition("=")
        runs[label] = run_rounds(d, key)

    all_rounds = sorted({r for rr in runs.values() for r in rr})
    print(f"\nmetric = {key}   (per round: mean[max] of clients' step-averaged reward)\n")
    header = "round | " + " | ".join(f"{lbl:>18}" for lbl in runs)
    print(header)
    print("-" * len(header))
    for r in all_rounds:
        cells = []
        for lbl, rr in runs.items():
            if r in rr:
                means = [v[0] for v in rr[r].values()]
                maxes = [v[1] for v in rr[r].values()]
                cells.append(f"{sum(means)/len(means):.3f}[{max(maxes):.2f}]")
            else:
                cells.append("-")
        print(f"{r:>5} | " + " | ".join(f"{c:>18}" for c in cells))

    # per-condition compounding trend: slope of round-mean over rounds + first->last
    print("\ncompounding trend (round-mean reward vs round):")
    for lbl, rr in runs.items():
        rs = sorted(rr)
        if not rs:
            print(f"  {lbl:>18}: (no data)")
            continue
        slope = lin_slope([(r, round_mean(rr, r)) for r in rs])
        first, last = round_mean(rr, rs[0]), round_mean(rr, rs[-1])
        slope_s = f"{slope:+.4f}/round" if slope is not None else "n/a"
        print(f"  {lbl:>18}: R{rs[0]}={first:.3f} -> R{rs[-1]}={last:.3f}  "
              f"(delta {last-first:+.3f}, slope {slope_s})")

    # asymmetry trajectory (env effect A-B, task effect B-C) at EVERY round + slopes
    Lenv, Ltask, Liid = decomp
    if {Lenv, Ltask, Liid} <= set(runs):
        A, B, C = runs[Lenv], runs[Ltask], runs[Liid]
        print(f"\nasymmetry trajectory  [env={Lenv} (A), task={Ltask} (B), iid={Liid} (C)]")
        print(f"  A-B = env-het effect (neg => env heterogeneity HURTS under FedAvg)")
        print(f"  B-C = task-het effect (~0 => task heterogeneity is FedAvg-robust)\n")
        print("  round |     A-B |     B-C")
        print("  ------+---------+--------")
        ab_pairs, bc_pairs = [], []
        for r in all_rounds:
            a, b, c = round_mean(A, r), round_mean(B, r), round_mean(C, r)
            ab = (a - b) if (a is not None and b is not None) else None
            bc = (b - c) if (b is not None and c is not None) else None
            ab_pairs.append((r, ab))
            bc_pairs.append((r, bc))
            ab_s = f"{ab:+.3f}" if ab is not None else "   -  "
            bc_s = f"{bc:+.3f}" if bc is not None else "   -  "
            print(f"  {r:>5} | {ab_s:>7} | {bc_s:>7}")
        ab_slope, bc_slope = lin_slope(ab_pairs), lin_slope(bc_pairs)
        ab_sl_s = f"{ab_slope:+.4f}/round" if ab_slope is not None else "n/a"
        bc_sl_s = f"{bc_slope:+.4f}/round" if bc_slope is not None else "n/a"
        def interp_env(s):
            if s is None:
                return "(need >=2 rounds with all 3 conditions)"
            return ("env-het gap WIDENS over rounds (cumulative degradation -- the asymmetry)"
                    if s < -1e-4 else "env-het gap NARROWS over rounds"
                    if s > 1e-4 else "env-het gap ~flat across rounds")
        def interp_task(s):
            if s is None:
                return "(need >=2 rounds with all 3 conditions)"
            return ("task-het gap ~flat => FedAvg-robust" if abs(s) <= 1e-3
                    else "task-het gap drifts (check noise/budget)")
        print(f"\n  env-effect (A-B) slope over rounds = {ab_sl_s}   <- {interp_env(ab_slope)}")
        print(f"  task-effect (B-C) slope over rounds = {bc_sl_s}   <- {interp_task(bc_slope)}")
        print("\n  NOTE: at 4 steps/round the per-round reward is noisy; trust the SIGN + the\n"
              "  full-8-round slope, not 2-3 round deltas. A-B<0 with |slope| growing is the signal.")


if __name__ == "__main__":
    main()
