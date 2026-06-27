#!/usr/bin/env python
"""ALFWorld per-task-type eval breakdown (the paper's Pick/Look/Clean/Heat/Cool/Pick2 + All).

Scores ONE model on the unperturbed in-distribution eval games, broken down by the 6 ALFWorld
task types. Reuses run_fed's eval machinery verbatim: for each task type it starts the shared
UNPERTURBED ALFWorld val service filtered to that type (ALFWORLD_TASK_TYPES), runs a verl
val-only pass, and reads the success rate from the dumped val samples. "All" runs once with no
filter (== the headline ALFWorld success used in the round->success curve).

This is decoupled from the training loop (run it after a federated/centralized/local run on the
final aggregated model) and from verl's val-metrics internals (each type is a separate pass on
that type's games -> robust, no per-sample tagging needed).

Usage (inside fedagent-verl08, on the GPU node):
    python -m tools.verl08_migration.eval_alfworld_by_tasktype \
        --config fedagent/config/examples/alfworld/paper.yaml \
        --model  /path/to/final/aggregated/hf \
        --output /path/to/alfworld_tasktype_breakdown.json [--n-gpus 4]
"""
import argparse
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedagent.fed.run_fed import (  # noqa: E402
    DEFAULTS,
    eval_global,
    start_val_service,
    stop_services,
    verl_cfg_dir,
)

# AlfredTWEnv task-type ID -> (canonical name, paper label)
TASK_TYPES = {
    1: ("pick_and_place_simple", "Pick"),
    2: ("look_at_obj_in_light", "Look"),
    3: ("pick_clean_then_place_in_recep", "Clean"),
    4: ("pick_heat_then_place_in_recep", "Heat"),
    5: ("pick_cool_then_place_in_recep", "Cool"),
    6: ("pick_two_obj_and_place", "Pick2"),
}
PKG_DIR = REPO_ROOT / "fedagent"


def _resolve_pkg_paths(cfg):
    for k in ("env_spec", "val_env_spec", "custom_cls_path", "agent_config_path",
              "alfworld_run_service", "webshop_run_service"):
        v = cfg.get(k)
        if v and not os.path.isabs(str(v)):
            cfg[k] = str(PKG_DIR / str(v))


def main():
    ap = argparse.ArgumentParser(description="ALFWorld per-task-type eval breakdown")
    ap.add_argument("--config", required=True, help="a fed YAML (for model/env/rollout settings)")
    ap.add_argument("--model", required=True, help="HF model dir to evaluate (e.g. the final aggregated model)")
    ap.add_argument("--output", default=None, help="breakdown JSON path (default: <output_dir>/alfworld_tasktype_breakdown.json)")
    ap.add_argument("--n-gpus", type=int, default=None)
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
    # (round-id used only to give each pass a distinct eval dir): 0=All, 1..6=task types
    passes = [("All", "", 0)] + [(lbl, str(tid), tid) for tid, (_n, lbl) in TASK_TYPES.items()]
    results = {}
    for label, ids, rid in passes:
        cfg.alfworld_task_types = ids
        print(f"\n=== eval task type: {label} (ids={ids or 'all'}) ===", flush=True)
        svc = None
        try:
            svc = start_val_service(cfg, env_base)
            m = eval_global(cfg, args.model, rid, env_base, url)
            results[label] = m
        except Exception as e:
            print(f"[warn] {label}: {e}", flush=True)
            results[label] = None
        finally:
            stop_services([svc] if svc else [])

    out = args.output or str(Path(cfg.output_dir) / "alfworld_tasktype_breakdown.json")
    payload = {"model": args.model, "config": args.config, "by_task_type": results}
    Path(out).write_text(json.dumps(payload, indent=2))
    print("\n=== ALFWorld task-type breakdown (success_rate) ===", flush=True)
    for label, _ids, _rid in passes:
        m = results.get(label)
        sr = m["success_rate"] if m else "FAILED"
        print(f"  {label:6s}: {sr}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
