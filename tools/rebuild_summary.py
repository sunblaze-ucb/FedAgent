#!/usr/bin/env python
"""Rebuild a run's ``federated_summary.json`` from its on-disk artifacts.

``run_fed`` writes the summary ONCE, in its teardown (run_fed.py:2490). A run that is
killed mid-flight -- preempted worker, OOM, ctrl-C -- therefore leaves a complete set of
per-round eval dumps and metrics on disk but NO summary, and every downstream reader
(the paper figures, tools/plot_training_dynamics.py) plots straight from that file.

This reconstructs it with the SAME fold-in logic the teardown uses:

    val_curve    <- round_<k>/eval/val_samples/*.jsonl                (run_fed.py:2436-2446)
    client_curve <- round_<k>/client_<c>/eval/val_samples/*.jsonl     (run_fed.py:2450-2461)
    rounds       <- round_<k>/persistent_plan.json + json_logs/metrics.json

Only fields the live orchestrator alone knows (model paths, critic provenance, mode) are
absent; everything a curve is drawn from is recovered. The output is marked
``"_reconstructed": true`` so a rebuilt summary is never mistaken for the run's own.

Round dirs are addressed as exact ``round_<k>`` paths for k in 0..max, so quarantined
attempts under ``_stale_rounds/`` can never leak in (same invariant as
quarantine_stale_rounds).

Usage:
    python tools/rebuild_summary.py <run_dir> [--out FILE] [--epochs-per-round N] [--print]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import fedagent
# The SAME reader the federated loop uses (fed/eval_dumps.py), not a copy: which dump file wins
# and which keys are averaged must never drift between the live run and the offline rebuild.
# Both modules are dependency-free, so this tool still runs under a bare interpreter.
from fedagent.fed.eval_dumps import summarize_val_dump  # noqa: E402
from fedagent.fed.metrics_logger import parse_training_log  # noqa: E402

REWARD_KEY = "critic/rewards/mean"


def round_dirs(out: Path):
    """Exact round_<k> dirs present, k ascending (never a glob into _stale_rounds)."""
    ks = []
    for d in out.iterdir():
        m = re.fullmatch(r"round_(\d+)", d.name)
        if m and d.is_dir():
            ks.append(int(m.group(1)))
    return sorted(ks)


def train_metrics(rdir: Path):
    """Per-round training metrics -> {steps, reward_mean, reward_max, clients}.

    Sources, in the order run_fed writes them: the persistent path's shared
    round_<k>/json_logs/metrics.json (all of the round's clients concatenated, step counters
    restarting per client), the lane path's json_logs_lane*/, and the subprocess path's
    per-client client_<c>/json_logs/metrics.json."""
    entries, per_client = [], {}
    shared = rdir / "json_logs" / "metrics.json"
    if shared.is_file():
        entries = json.loads(shared.read_text())
    else:
        for lane in sorted(rdir.glob("json_logs_lane*/metrics.json")):
            entries += json.loads(lane.read_text())
    for cj in sorted(rdir.glob("client_*/json_logs/metrics.json")):
        try:
            c = int(cj.parent.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        per_client[c] = json.loads(cj.read_text())
        if not shared.is_file():
            entries += per_client[c]
    for log in sorted(rdir.glob("client_*/training.log")):   # last resort: the raw log
        if not entries:
            entries += parse_training_log(log)
    vals = [e["metrics"][REWARD_KEY] for e in entries if REWARD_KEY in e.get("metrics", {})]
    if not vals:
        return None
    return {"train_steps": len(entries), "train_reward_mean": round(sum(vals) / len(vals), 4),
            "train_reward_max": round(max(vals), 4)}


def plan_clients(rdir: Path):
    """The round's selected client ids: the plan if present, else the client_<c> dirs."""
    plan = rdir / "persistent_plan.json"
    if plan.is_file():
        try:
            return [int(s["client"]) for s in json.loads(plan.read_text())]
        except Exception:
            pass
    ids = []
    for d in sorted(rdir.glob("client_*")):
        try:
            ids.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def infer_env_kind(val_curve_dirs) -> str:
    """webshop rows carry task_score; alfworld rows carry task_type."""
    for d in val_curve_dirs:
        files = sorted(Path(d).glob("*.jsonl"))
        if not files:
            continue
        with open(files[-1]) as f:
            line = f.readline()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("task_score") is not None:
            return "webshop"
        if row.get("task_type") is not None:
            return "alfworld"
    return "unknown"


def rebuild(out: Path, epochs_per_round=None) -> dict:
    ks = round_dirs(out)
    if not ks:
        raise SystemExit(f"no round_<k> dirs under {out}")
    val_curve, client_curve, rounds, dump_dirs = [], [], [], []
    for k in ks:
        rdir = out / f"round_{k}"
        d = rdir / "eval" / "val_samples"
        m = summarize_val_dump(d)
        if m:
            dump_dirs.append(d)
            val_curve.append({"round": k, "model": "base" if k == 0 else "aggregated", **m})
        for cd in sorted(rdir.glob("client_*/eval/val_samples")):
            try:
                c = int(cd.parent.parent.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            cm = summarize_val_dump(cd)
            if cm:
                client_curve.append({"round": k, "client": c, **cm})
        if k == 0:
            continue                     # round_0 is the base-model eval point, not a train round
        rec = {"round": k, "clients": plan_clients(rdir)}
        tm = train_metrics(rdir)
        if tm:
            rec.update(tm)
        rounds.append(rec)

    trained = [r for r in rounds if r.get("clients")]
    cpr = max((len(r["clients"]) for r in trained), default=0)
    summary = {
        "total_clients": None,           # not derivable from artifacts (it is a config value)
        "clients_per_round": cpr or None,
        "total_rounds": max(ks),
        "epochs_per_round": epochs_per_round,
        "env_kind": infer_env_kind(dump_dirs),
        "partition_strategy": None,
        **({"val_curve": val_curve} if val_curve else {}),
        **({"client_curve": client_curve} if client_curve else {}),
        "rounds": rounds,
        "_reconstructed": True,
        "_note": ("rebuilt from on-disk artifacts by tools/rebuild_summary.py (the run ended "
                  "before run_fed's teardown wrote its own summary). val_curve/client_curve use "
                  "the orchestrator's own summarize_val_dump; config-only fields are null."),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="default: <run_dir>/federated_summary.json")
    ap.add_argument("--epochs-per-round", type=int, default=None,
                    help="config value E (not recoverable from artifacts); recorded as-is")
    ap.add_argument("--print", dest="show", action="store_true", help="print the val curve")
    args = ap.parse_args()

    out = Path(args.run_dir)
    summary = rebuild(out, args.epochs_per_round)
    dest = Path(args.out) if args.out else out / "federated_summary.json"
    if dest.exists():
        raise SystemExit(f"refusing to overwrite an existing {dest} (pass --out to redirect)")
    dest.write_text(json.dumps(summary, indent=2))
    vc = summary.get("val_curve", [])
    print(f"wrote {dest}: {len(vc)} val points, "
          f"{len(summary.get('client_curve', []))} client circles, {len(summary['rounds'])} rounds")
    if args.show and vc:
        for v in vc:
            print(f"  r{v['round']:>3} success={v['success_rate']} reward={v['reward_mean']} n={v['n']}")


if __name__ == "__main__":
    main()
