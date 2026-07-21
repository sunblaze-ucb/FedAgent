"""Post-process a verl client training.log into json_logs/metrics.json.

verl 0.8's stock Tracking has no JSON backend (verl-agent had added one), and the
thin overlay does NOT fork verl. But verl's console logger already prints the full
per-step metric dict to stdout (captured in each client's training.log) as:

    ... step:<N> - global_seqlen/mean:23379.5 - actor/entropy:1.17 - ... - critic/rewards/mean:0.026 - ...

So we just parse those lines into the SAME schema the FedAgent plots/loaders expect
(tools/plot_training_dynamics.py; originally the verl-agent-0.3.1
core/fed/client_runner._load_metrics_from_json — see the paper-reproduce branch):

    metrics.json = [ {"step": int, "metrics": {"<key>": float, ...}}, ... ]

This keeps measurability identical to the 0.3.1 baseline with no verl modification.
run_fed calls write_metrics_json() after each client round, writing
round_<r>/client_<c>/json_logs/metrics.json.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

_STEP_RE = re.compile(r"step:(\d+) - (.+)")
_WRAP_RE = re.compile(r"^(?:np\.float64|np\.float32|np\.int64|np\.int32|tensor)\((.*)\)$")


def _parse_value(s: str):
    s = s.strip()
    m = _WRAP_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_training_log(log_path) -> List[Dict[str, Any]]:
    """Extract per-step metric entries from a verl training.log."""
    entries: List[Dict[str, Any]] = []
    p = Path(log_path)
    if not p.is_file():
        return entries
    with open(p, errors="ignore") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            metrics: Dict[str, float] = {}
            for tok in m.group(2).split(" - "):
                if ":" not in tok:
                    continue
                k, v = tok.split(":", 1)
                val = _parse_value(v)
                if val is not None:
                    metrics[k.strip()] = val
            # real metric dumps carry many keys; this filters stray "step:N" mentions
            if len(metrics) >= 5:
                entries.append({"step": step, "metrics": metrics})
    return entries


def write_metrics_json(log_path, out_dir) -> Path:
    """Parse log_path and write <out_dir>/metrics.json (FedAgent schema). Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_training_log(log_path)
    out = out_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(entries, f, indent=2)
    return out


def summarize(entries: List[Dict[str, Any]], keys=("critic/rewards/mean", "critic/score/mean")) -> str:
    """One-line per-step reward summary for logs."""
    if not entries:
        return "(no metric steps parsed)"
    parts = []
    for e in entries:
        rk = next((k for k in keys if k in e["metrics"]), None)
        if rk is not None:
            parts.append(f"s{e['step']}:{e['metrics'][rk]:.3f}")
    return " ".join(parts) if parts else f"{len(entries)} steps, keys={list(entries[0]['metrics'])[:3]}..."


def main():
    import argparse
    ap = argparse.ArgumentParser(description="parse a verl training.log -> metrics.json")
    ap.add_argument("log_path")
    ap.add_argument("--out-dir", default=None, help="default: <log dir>/json_logs")
    args = ap.parse_args()
    out_dir = args.out_dir or (Path(args.log_path).parent / "json_logs")
    entries = parse_training_log(args.log_path)
    path = write_metrics_json(args.log_path, out_dir)
    print(f"wrote {path}  ({len(entries)} steps)")
    print("reward:", summarize(entries))


if __name__ == "__main__":
    main()
