#!/usr/bin/env python3
"""Plot aggregated (federated) performance training dynamics from a run's logs.

Reads the per-round / per-client metric logs written under an experiment
directory and plots the aggregated (global FedAvg model) trajectory on a
cumulative training-epoch x-axis. Round N enters at epoch (N-1)*ep-per-cl, so
an ``ep-per-cl-3`` run over 70 rounds spans 0..210. Two modes:

    (default)        aggregated curve only
    --with-clients   additionally overlay each client's per-round local
                     trajectory as a faint segment diverging from the shared
                     global point (visualizes client heterogeneity)

The plotted values are the **raw logged numbers** — no padding to a fixed
horizon, no interpolation, and no smoothing/nudging is applied.

Two data sources, auto-detected:

* ``<experiment_dir>/federated_summary.json`` (the verl-0.8 stack; preferred when
  present). The aggregated red line comes from ``val_curve`` — ONE central eval of
  each round's aggregated model on the shared unperturbed val service, round 0
  being the base model (``val_before_train``). The circles come from
  ``client_curve`` (``client_end_eval: true``): each selected client's
  post-training model on the same val service, drawn at the round's end boundary.
  ``val_curve``'s round-k entry is the model *entering* round k+1, so it sits at
  epoch ``k*stride`` — the same x placement as the legacy source below. The
  metric maps onto the summary keys (val/success_rate -> success_rate,
  val/reward_mean or reward -> reward_mean) and the stride defaults to the
  summary's ``epochs_per_round``. On this stack clients never validate locally,
  so the legacy source below has no val metrics — the summary is the figure's
  single source of truth (complete across resumes; see running.md § Resume).

* ``round_<N>/client_<C>/json_logs/metrics.json`` (legacy verl-agent-0.3.1 runs,
  paper-reproduce branch; forced with ``--client-logs``), a list of
  {"step": int, "metrics": {<name>: float, ...}}. Each round's aggregated value
  is the mean across the participating clients of the metric at local step 0,
  i.e. the pre-local-training evaluation of the global model *entering* that
  round; each client's circle is its metric at its last logged local step.

Usage:

    python tools/plot_training_dynamics.py <experiment_dir> \\
        [--metric val/success_rate] [--with-clients] [--out FIG.pdf] \\
        [--round-stride N] [--percent] [--title STR] [--client-logs]

Run it once without and once with --with-clients to get both figures.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

StepData = Dict[str, object]
Experiment = Dict[str, Dict[str, List[StepData]]]


def load_experiment(folder: Path) -> Experiment:
    """Load metrics.json for every round_*/client_* under `folder`.

    Returns {round_name: {client_name: [step_data, ...]}}.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"experiment dir not found: {folder}")
    rounds = sorted(
        d for d in folder.iterdir() if d.is_dir() and d.name.startswith("round_")
    )
    if not rounds:
        raise ValueError(f"no round_* directories in {folder}")
    data: Experiment = {}
    for round_dir in rounds:
        clients = sorted(
            d for d in round_dir.iterdir()
            if d.is_dir() and d.name.startswith("client_")
        )
        data[round_dir.name] = {}
        for client_dir in clients:
            mfile = client_dir / "json_logs" / "metrics.json"
            if mfile.exists():
                with open(mfile, "r", encoding="utf-8") as f:
                    data[round_dir.name][client_dir.name] = json.load(f)
    return data


def _round_index(round_name: str) -> int:
    return int(round_name.split("_")[1])


def infer_round_stride(folder, data: Optional[Experiment] = None) -> int:
    """Training epochs per round = the x-axis stride between consecutive rounds.

    Read from ``ep-per-cl-N`` in the run folder name (the number of local epochs
    each client trains per round), matching the convention used by the project's
    canonical comparison plots. Round N's local step 0 is the *pre-training* eval
    of the model entering the round — a round boundary shared with round N-1's
    end, not a training epoch — so the stride is the count of post-step-0 epochs
    (``ep-per-cl``), NOT max_step + 1. With ``ep-per-cl-3`` over 70 rounds the
    axis spans 0..210.

    Falls back to the max local step observed across the logs when the folder
    name carries no ``ep-per-cl-N`` token.
    """
    m = re.search(r"ep-per-cl-(\d+)", str(folder))
    if m:
        return max(int(m.group(1)), 1)
    max_step = 0
    for round_data in (data or {}).values():
        for client_data in round_data.values():
            for step_data in client_data:
                max_step = max(max_step, int(step_data.get("step", 0)))
    return max(max_step, 1)


def aggregated_curve(data: Experiment, metric: str) -> List[Tuple[int, float]]:
    """Aggregated (global model) value per round.

    For each round, the mean across participating clients of `metric` at local
    step 0 — the global model entering the round. Raw values only.

    Returns [(round_num, value), ...] sorted by round.
    """
    pairs: List[Tuple[int, float]] = []
    for round_name in sorted(data, key=_round_index):
        vals: List[float] = []
        for client_data in data[round_name].values():
            for step_data in client_data:
                if int(step_data.get("step", 0)) == 0 and metric in step_data.get("metrics", {}):
                    vals.append(float(step_data["metrics"][metric]))
                    break  # step 0 only
        if vals:
            pairs.append((_round_index(round_name), float(np.mean(vals))))
    return pairs


_SUMMARY_METRIC = {
    "val/success_rate": "success_rate", "success_rate": "success_rate",
    "val/reward_mean": "reward_mean", "reward_mean": "reward_mean", "reward": "reward_mean",
    "val/task_score": "task_score_mean", "task_score": "task_score_mean",
}


def load_summary_curves(folder: Path, metric: str):
    """Aggregated curve + client circles from a run's federated_summary.json (verl-0.8 stack).

    Returns (agg, circles, stride): agg = [(entering_round, value), ...] — the val_curve
    round-k entry (the aggregate produced by round k; k=0 = base) is the model ENTERING
    round k+1, keeping the x placement identical to the client-log path; circles =
    {round: [(client, value), ...]} from client_curve (client_end_eval); stride = the run's
    epochs_per_round, or None if absent.
    """
    with open(Path(folder) / "federated_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    key = _SUMMARY_METRIC.get(metric)
    if key is None:
        raise ValueError(
            f"metric '{metric}' is not in federated_summary.json — choose from "
            f"{sorted(set(_SUMMARY_METRIC))}, or pass --client-logs for a training metric")
    by_round: Dict[int, float] = {}
    for v in summary.get("val_curve", []):
        if v.get(key) is not None:
            by_round[int(v["round"]) + 1] = float(v[key])   # last entry wins on a duplicate
    agg = sorted(by_round.items())
    circles: Dict[int, List[Tuple[int, float]]] = {}
    for c in summary.get("client_curve", []):
        if c.get(key) is not None:
            circles.setdefault(int(c["round"]), []).append((int(c["client"]), float(c[key])))
    stride = int(summary.get("epochs_per_round") or 0)
    return agg, circles, (stride if stride > 0 else None)


def _last_metric_step(client_data: List[StepData], metric: str) -> Optional[Tuple[int, float]]:
    """The (step, value) at the client's largest local step that logs `metric`."""
    best: Optional[Tuple[int, float]] = None
    for step_data in client_data:
        metrics = step_data.get("metrics", {})
        if metric in metrics:
            s = int(step_data.get("step", 0))
            if best is None or s > best[0]:
                best = (s, float(metrics[metric]))
    return best


def plot_training_dynamics(
    folder: str,
    metric: str = "val/success_rate",
    *,
    with_clients: bool = False,
    out_path: Optional[str] = None,
    round_stride: Optional[int] = None,
    as_percent: bool = False,
    title: Optional[str] = None,
    from_summary: Optional[bool] = None,
) -> str:
    """Render the aggregated training-dynamics figure and save .pdf + .png.

    from_summary: True = read federated_summary.json, False = read the per-client
    json_logs, None (default) = auto: the summary when the file exists."""
    folder_p = Path(folder)
    if from_summary is None:
        from_summary = (folder_p / "federated_summary.json").is_file()
    data: Optional[Experiment] = None
    summary_circles: Dict[int, List[Tuple[int, float]]] = {}
    stride_hint: Optional[int] = None
    if from_summary:
        agg, summary_circles, stride_hint = load_summary_curves(folder_p, metric)
        if not agg:
            raise ValueError(
                f"no '{metric}' points in {folder_p / 'federated_summary.json'} "
                "(the run needs do_eval; round 0 needs val_before_train)")
    else:
        data = load_experiment(folder_p)
        agg = aggregated_curve(data, metric)
        if not agg:
            raise ValueError(f"no step-0 values for metric '{metric}' in {folder}")

    if round_stride is not None:
        stride = round_stride
    elif stride_hint is not None:
        stride = stride_hint
    else:
        stride = infer_round_stride(folder, data)
    scale = 100.0 if as_percent else 1.0
    fig, ax = plt.subplots(figsize=(10, 6))

    # Cumulative training-epoch x position of each round: round N's step-0
    # (pre-training) model sits at epoch (N-1)*stride, with stride = ep-per-cl
    # local epochs per round. Consecutive rounds share their boundary epoch, so a
    # 70-round ep-per-cl-3 run spans 0..210. Raw logged values only — no padding.
    agg_xy = {r: ((r - 1) * stride, v) for r, v in agg}

    if with_clients:
        if from_summary:
            clients = sorted({c for lst in summary_circles.values() for c, _ in lst})
        else:
            clients = sorted({int(c.split("_")[1]) for rd in data.values() for c in rd})
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(clients), 1)))
        cmap = {c: palette[i % len(palette)] for i, c in enumerate(clients)}

        if from_summary:
            # client_end_eval circles: the client's post-training model at the round's END
            # boundary (a full ep-per-cl epochs after its entering point)
            for r in sorted(summary_circles):
                if r not in agg_xy:
                    continue
                x_agg, y_agg = agg_xy[r]
                for c, y_client in sorted(summary_circles[r]):
                    ax.plot(
                        [x_agg, x_agg + stride], [y_agg * scale, y_client * scale],
                        color=cmap[c], linewidth=2.0, alpha=0.45, zorder=1,
                    )
                    ax.plot(
                        [x_agg + stride], [y_client * scale], marker="o", markersize=6,
                        linestyle="None", color=cmap[c], alpha=0.7, zorder=2,
                    )
        else:
            for round_name in sorted(data, key=_round_index):
                r = _round_index(round_name)
                if r not in agg_xy:
                    continue
                x_agg, y_agg = agg_xy[r]
                for cname, cdata in sorted(data[round_name].items()):
                    last = _last_metric_step(cdata, metric)
                    if last is None or last[0] == 0:
                        continue  # nothing logged after the shared step-0 point
                    step, y_client = last
                    c = int(cname.split("_")[1])
                    ax.plot(
                        [x_agg, x_agg + step], [y_agg * scale, y_client * scale],
                        color=cmap[c], linewidth=2.0, alpha=0.45, zorder=1,
                    )
                    ax.plot(
                        [x_agg + step], [y_client * scale], marker="o", markersize=6,
                        linestyle="None", color=cmap[c], alpha=0.7, zorder=2,
                    )
        for r in sorted(agg_xy)[1:]:  # round boundaries
            ax.axvline(agg_xy[r][0], color="gray", linestyle="--",
                       alpha=0.35, linewidth=1.0, zorder=0)

    xs = [agg_xy[r][0] for r, _ in agg]
    ys = [v * scale for _, v in agg]
    ax.plot(xs, ys, marker="s", linewidth=3.0, markersize=8, color="tab:red",
            label="Aggregated (global model)", zorder=3)

    ax.set_xlabel("Training Epoch", fontsize=14)
    x_max = max(xs) if xs else 0
    if x_max >= 60:
        ax.set_xticks(list(range(0, int(x_max) + 1, 30)))
    else:
        # Small ranges: tick on real round boundaries (multiples of the ep-per-round stride)
        # instead of matplotlib's default "nice" spacing, which lands on non-data epochs (2.5).
        ax.xaxis.set_major_locator(MultipleLocator(max(stride, 1)))
    ax.set_xlim(left=0)
    ax.set_ylabel(metric + (" (%)" if as_percent else ""), fontsize=14)
    ax.set_title(title or Path(folder).name, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    out = Path(out_path) if out_path else Path(folder) / (
        f"training_dynamics{'_with_clients' if with_clients else ''}.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {Path(folder).name}: {len(agg)} rounds, metric={metric}, "
          f"with_clients={with_clients}, "
          f"source={'federated_summary.json' if from_summary else 'client json_logs'} -> {out}")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "experiment_dir",
        help="run directory: federated_summary.json (verl-0.8 stack, auto-detected) "
             "or round_*/client_*/json_logs/metrics.json (legacy)",
    )
    ap.add_argument("--client-logs", action="store_true",
                    help="force the legacy per-client json_logs source even when "
                         "federated_summary.json exists (e.g. to plot a training metric)")
    ap.add_argument("--metric", default="val/success_rate",
                    help="metric key to plot (default: val/success_rate)")
    ap.add_argument("--with-clients", action="store_true",
                    help="overlay per-client per-round local trajectories")
    ap.add_argument("--out", default=None,
                    help="output figure path (.pdf; a matching .png is written too)")
    ap.add_argument("--round-stride", type=int, default=None,
                    help="training epochs per round = x-axis stride between rounds "
                         "(default: the summary's epochs_per_round when federated_summary.json "
                         "is the source, else ep-per-cl-N parsed from the run dir name)")
    ap.add_argument("--percent", action="store_true",
                    help="scale the metric to a percentage (x100)")
    ap.add_argument("--title", default=None, help="figure title (default: run dir name)")
    args = ap.parse_args()
    plot_training_dynamics(
        args.experiment_dir, args.metric,
        with_clients=args.with_clients, out_path=args.out,
        round_stride=args.round_stride, as_percent=args.percent, title=args.title,
        from_summary=False if args.client_logs else None,
    )


if __name__ == "__main__":
    main()
