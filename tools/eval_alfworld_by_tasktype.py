#!/usr/bin/env python
"""ALFWorld per-task-type eval breakdown (the paper's Pick/Look/Clean/Heat/Cool/Pick2 + All).

Scores ONE model on the unperturbed in-distribution eval games, broken down by the 6 ALFWorld
task types.

TWO estimators -- they measure different quantities, do not mix them:

* ``--mode single`` (DEFAULT; the paper-table estimator). ONE unfiltered eval pass over the
  val set, then the dumped rows are grouped by their per-row ``task_type`` tag (written by the
  agent loop from the episode's gamefile). Each per-type rate is that type's share of the ONE
  pass (denominator = how many of the val episodes are that type -- small: single digits to
  low tens), and ``All`` is the pooled pass, so the per-type numbers recombine to ``All``
  exactly. This is how the paper's table columns were produced (they satisfy that pooling
  identity row by row); only this mode can reproduce them.

* ``--mode per-type-passes`` (legacy; a DIFFERENT, higher-sample estimator). Seven separate
  eval passes: one unfiltered (``All``) plus one per task type with the val service filtered
  to that type (``ALFWORLD_TASK_TYPES``). With the filter, the pool becomes that type's WHOLE
  split and the pass still runs the full val ``n_envs`` episodes, so games repeat (seed ->
  ``gfs[seed % len(gfs)]``); every number has denominator ``n_envs`` and ``All`` is an
  independent pass, NOT the pooled six. More samples per type, but it cannot reproduce the
  paper tables even in principle (2026-07 audit; see docs/bugfixes.md).

Run it after a federated/centralized/local run on the final aggregated model. NOTE the val-64
composition changed with the 2026-07-26 machine-independent game ordering: at HEAD the first
64 of ``valid_seen`` comprise (Pick 15, Look 4, Clean 14, Heat 7, Cool 10, Pick2 14), not the
paper's (14, 8, 12, 9, 11, 10) -- per-type denominators from a fresh run differ from the
tables' accordingly.

Usage (inside fedagent-verl08, on the GPU node):
    python -m tools.eval_alfworld_by_tasktype \
        --config fedagent/config/examples/alfworld/paper.yaml \
        --model  /path/to/final/aggregated/hf \
        --output /path/to/alfworld_tasktype_breakdown.json [--n-gpus 4] [--mode single]
"""
import argparse
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedagent.fed.eval_dumps import summarize_val_dump_by_tasktype  # noqa: E402
from fedagent.fed.run_fed import (  # noqa: E402
    DEFAULTS,
    eval_global,
    start_val_service,
    stop_services,
    verl_cfg_dir,
)

# AlfredTWEnv task-type ID -> (canonical name == the per-row task_type tag, paper label)
TASK_TYPES = {
    1: ("pick_and_place_simple", "Pick"),
    2: ("look_at_obj_in_light", "Look"),
    3: ("pick_clean_then_place_in_recep", "Clean"),
    4: ("pick_heat_then_place_in_recep", "Heat"),
    5: ("pick_cool_then_place_in_recep", "Cool"),
    6: ("pick_two_obj_and_place", "Pick2"),
}
NAME_TO_LABEL = {name: lbl for name, lbl in TASK_TYPES.values()}
PKG_DIR = REPO_ROOT / "fedagent"


def _resolve_pkg_paths(cfg):
    for k in ("env_spec", "val_env_spec", "custom_cls_path", "agent_config_path",
              "alfworld_run_service", "webshop_run_service"):
        v = cfg.get(k)
        if v and not os.path.isabs(str(v)):
            cfg[k] = str(PKG_DIR / str(v))


def run_single_pass(cfg, model, env_base, url):
    """ONE unfiltered pass -> group the dump rows by task_type (the paper-table estimator)."""
    cfg.alfworld_task_types = ""
    svc = None
    try:
        svc = start_val_service(cfg, env_base)
        eval_global(cfg, model, 0, env_base, url)
    finally:
        stop_services([svc] if svc else [])
    dump_dir = Path(cfg.output_dir) / "round_0" / "eval" / "val_samples"
    grouped = summarize_val_dump_by_tasktype(dump_dir)
    if grouped is None:
        raise RuntimeError(f"no readable val dump under {dump_dir}")
    if set(grouped["by_type"]) == {"untagged"}:
        raise RuntimeError(
            "val dump rows carry no task_type tag (dump predates the tagging agent loop?) -- "
            "cannot break down a single pass; re-run on this codebase or use --mode per-type-passes")
    # canonical name -> paper label; anything unexpected keeps its raw name
    results = {"All": grouped["All"]}
    for name, stats in grouped["by_type"].items():
        results[NAME_TO_LABEL.get(name, name)] = stats
    return results


def run_per_type_passes(cfg, model, env_base, url):
    """Legacy 7-pass estimator (per-type full pools, games repeat; see module docstring)."""
    passes = [("All", "", 0)] + [(lbl, str(tid), tid) for tid, (_n, lbl) in TASK_TYPES.items()]
    results = {}
    for label, ids, rid in passes:
        cfg.alfworld_task_types = ids
        print(f"\n=== eval task type: {label} (ids={ids or 'all'}) ===", flush=True)
        svc = None
        try:
            svc = start_val_service(cfg, env_base)
            m = eval_global(cfg, model, rid, env_base, url)
            results[label] = m
        except Exception as e:
            print(f"[warn] {label}: {e}", flush=True)
            results[label] = None
        finally:
            stop_services([svc] if svc else [])
    return results


def main():
    ap = argparse.ArgumentParser(description="ALFWorld per-task-type eval breakdown")
    ap.add_argument("--config", required=True, help="a fed YAML (for model/env/rollout settings)")
    ap.add_argument("--model", required=True, help="HF model dir to evaluate (e.g. the final aggregated model)")
    ap.add_argument("--output", default=None, help="breakdown JSON path (default: <output_dir>/alfworld_tasktype_breakdown.json)")
    ap.add_argument("--n-gpus", type=int, default=None)
    ap.add_argument("--mode", choices=("single", "per-type-passes"), default="single",
                    help="single: ONE pass grouped by the per-row task_type tag (the paper-table "
                         "estimator; per-type numbers pool to All). per-type-passes: legacy 7 "
                         "independent passes (higher per-type sample size, different quantity).")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.create(dict(DEFAULTS)), OmegaConf.load(args.config))
    cfg.env_kind = "alfworld"
    if not cfg.get("val_env_spec"):
        cfg.val_env_spec = "config/envs/alfworld_val.yaml"
    cfg.alfworld_val_split = "eval_in_distribution"   # the 274-game in-distribution eval set
    if args.n_gpus is not None:
        cfg.n_gpus_per_node = args.n_gpus
    # the breakdown writes under its own dir so eval rounds don't collide with a training run
    cfg.output_dir = str(Path(cfg.output_dir) / "tasktype_eval")
    _resolve_pkg_paths(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO_ROOT}:{env_base.get('PYTHONPATH', '')}".rstrip(":")
    env_base["VERL_CFG"] = verl_cfg_dir()
    env_base.pop("FEDPROX_MU", None)

    url = f"http://localhost:{cfg.alfworld_val_port}"
    if args.mode == "single":
        results = run_single_pass(cfg, args.model, env_base, url)
    else:
        results = run_per_type_passes(cfg, args.model, env_base, url)

    out = args.output or str(Path(cfg.output_dir) / "alfworld_tasktype_breakdown.json")
    payload = {"model": args.model, "config": args.config, "mode": args.mode,
               "by_task_type": results}
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"\n=== ALFWorld task-type breakdown (success_rate, mode={args.mode}) ===", flush=True)
    order = ["All"] + [lbl for _tid, (_n, lbl) in TASK_TYPES.items()]
    for label in order + [k for k in results if k not in order]:
        m = results.get(label)
        if m:
            n = f" (n={m['n']})" if isinstance(m, dict) and m.get("n") is not None else ""
            print(f"  {label:6s}: {m['success_rate']}{n}", flush=True)
        elif label in results:
            print(f"  {label:6s}: FAILED", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
