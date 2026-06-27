#!/usr/bin/env python
"""Generate the WebShop hardness trajectories file (task_id -> success) for the Hardness(xi')
task-heterogeneity arm.

The Hardness partition (fedagent/hetero/webshop_hardness.py) buckets train goals into
easy/hard by a per-goal success label from a reference policy, then Beta-allocates easy goals
across clients. That label file is REQUIRED and has no default; this script produces it the
faithful way: run a (zero-shot) REFERENCE model over the train goals on the UNPERTURBED full
catalog, recording per-goal (asin -> success).

Mechanism (reuses the overlay end to end, no new rollout code):
  - start a WebShop service: TRAIN split, full catalog, FEDAGENT_LOG_GOAL_ID=1 so /reset also
    returns each goal's TASK_ID -- computed from the env's REAL server.goals via the same
    formula hardness_for_client/hardness_partition key on: f"{asin}_{md5(goal_options)}" (or
    asin+instruction_text hash, else asin). The service and the partition both derive this from
    server.goals, so the labels match by construction (no asin-vs-options-hash drift);
  - run a verl val-only pass of the reference model over a spec of N train goals; the agent
    loop tags every sample with goal_id (a string -> kept in verl's validation dump, skipped
    by metric aggregation), so the dump is a list of (goal_id, traj_success);
  - aggregate per task_id -> success = (mean success >= threshold) and write
    {"trajectories": [{"task_info": {"task_id": tid}, "traj_info": {"success": bool}}, ...]}.

Run ONCE per backbone (the labels depend on the reference policy), then point the Hardness
configs' trajectories_file at the output. Greedy reference by default (deterministic labels).

Usage (inside fedagent-verl08, on the GPU node):
    python -m tools.verl08_migration.gen_hardness_trajectories \
        --config fedagent/config/examples/webshop/scaled/hardness.yaml \
        --model  /path/to/reference/hf  --num-goals 6410 \
        --output fedagent/data/hardness/webshop_trajectories_qwen1.5b.json [--n-gpus 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedagent.fed.run_fed import DEFAULTS, log, stream, verl_cfg_dir  # noqa: E402

PKG_DIR = REPO_ROOT / "fedagent"


def main():
    ap = argparse.ArgumentParser(description="WebShop hardness trajectories generator")
    ap.add_argument("--config", required=True, help="a WebShop fed YAML (for model/rollout settings)")
    ap.add_argument("--model", default=None, help="reference HF model (default: config model_path)")
    ap.add_argument("--num-goals", type=int, default=512,
                    help="# train goals to label (6410 = whole train pool; smaller = smoke)")
    ap.add_argument("--output", default=None, help="trajectories JSON path")
    ap.add_argument("--port", type=int, default=8095, help="labelling service port")
    ap.add_argument("--n-gpus", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0, help="reference sampling temp (0 = greedy)")
    ap.add_argument("--threshold", type=float, default=0.5, help="mean-success >= threshold -> easy(label=success)")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.create(dict(DEFAULTS)), OmegaConf.load(args.config))
    model = args.model or cfg.model_path
    if not model:
        raise SystemExit("no --model and config has no model_path")
    if args.n_gpus is not None:
        cfg.n_gpus_per_node = args.n_gpus
    # resolve package-relative paths
    for k in ("custom_cls_path", "agent_config_path", "webshop_run_service"):
        v = cfg.get(k)
        if v and not os.path.isabs(str(v)):
            cfg[k] = str(PKG_DIR / str(v))

    out_dir = Path(cfg.output_dir) / "hardness_labelling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "webshop_trajectories.json"

    # spec enumerating N train goals (seeds 0..N-1 -> contiguous train goals[500:500+N])
    spec = {"envs": [{"name": "WebShop", "n_envs": int(args.num_goals), "max_turns": 15,
                      "agent_name": "gym_text", "config": {"timeout": 180.0}}]}
    spec_path = out_dir / "label_spec.yaml"
    OmegaConf.save(OmegaConf.create(spec), spec_path)

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO_ROOT}:{env_base.get('PYTHONPATH', '')}".rstrip(":")
    env_base["VERL_CFG"] = verl_cfg_dir()
    env_base.pop("FEDPROX_MU", None)

    # 1) start the labelling service: TRAIN split, UNPERTURBED full catalog, goal-id logging on
    svc_env = dict(env_base)
    svc_env.update({
        "WEBSHOP_PORT": str(args.port),
        "WEBSHOP_POOL_SIZE": str(cfg.webshop_pool_size),
        "WEBSHOP_SEARCH_RETURN_N": str(cfg.get("search_return_n", 200)),
        "WEBSHOP_SPLIT": "train",
        "PARTITION_STRATEGY": "",            # unperturbed full catalog
        "CLIENT_ID": "0", "CLIENT_NUM": "1",
        "FEDAGENT_LOG_GOAL_ID": "1",         # /reset returns each goal's asin
    })
    svc_log = out_dir / "label_service.log"
    lf = open(svc_log, "w")
    log(f"starting WebShop labelling service on :{args.port} (TRAIN, full catalog, goal-id ON)")
    svc = subprocess.Popen(["bash", str(cfg.webshop_run_service)], env=svc_env,
                           stdout=lf, stderr=subprocess.STDOUT)
    try:
        url = f"http://localhost:{args.port}/health"
        up = False
        for _ in range(int(cfg.service_health_timeout / 3)):
            if svc.poll() is not None:
                raise RuntimeError(f"labelling service DIED; see {svc_log}")
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    json.loads(r.read())
                up = True
                break
            except Exception:
                time.sleep(3)
        if not up:
            raise RuntimeError(f"labelling service health timeout; see {svc_log}")

        # 2) verl val-only pass of the reference model over the N goals
        dump_dir = out_dir / "val_samples"
        cmd = [
            sys.executable, "-m", "fedagent.main_ppo_fed",
            f"data.train_files={spec_path}",
            f"data.val_files={spec_path}",
            f"data.custom_cls.path={cfg.custom_cls_path}",
            f"actor_rollout_ref.model.path={model}",
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            f"actor_rollout_ref.rollout.agent.agent_loop_config_path={cfg.agent_config_path}",
            f"trainer.default_local_dir={out_dir / 'ckpt'}",
            f"trainer.n_gpus_per_node={cfg.n_gpus_per_node}",
            "trainer.val_only=true",
            "trainer.val_before_train=true",
            "algorithm.adv_estimator=grpo",
            f"trainer.validation_data_dir={dump_dir}",
            "actor_rollout_ref.rollout.n=1",                 # one reference trajectory per goal
            f"actor_rollout_ref.rollout.val_kwargs.temperature={args.temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={'true' if args.temperature > 0 else 'false'}",
            "trainer.project_name=fedagent_hardness_label",
            "trainer.experiment_name=label",
        ]
        # the script forces rollout.n=1 (ONE greedy reference trajectory per goal); drop any
        # rollout.n from client_overrides so a training group size (e.g. 8) can't clobber it
        # back -- with temperature=0/do_sample=false the extra trajectories are identical
        # duplicates, pure wasted compute.
        cmd += [str(o) for o in (cfg.client_overrides or [])
                if not str(o).startswith("actor_rollout_ref.rollout.n=")]
        run_env = dict(env_base)
        run_env["WEBSHOP_SERVICE_URL"] = f"http://localhost:{args.port}"
        rc = stream(cmd, run_env, out_dir / "label.log", tag="label")
        if rc != 0:
            raise SystemExit(f"labelling val pass FAILED (rc={rc}); see {out_dir / 'label.log'}")
    finally:
        try:
            svc.terminate(); svc.wait(timeout=15)
        except Exception:
            try:
                svc.kill()
            except Exception:
                pass
        lf.close()

    # 3) aggregate per asin -> success label
    files = sorted((out_dir / "val_samples").glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no val dump produced under {out_dir / 'val_samples'}")
    by_goal = defaultdict(list)
    n_rows = 0
    with open(files[-1]) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            gid = r.get("goal_id")
            if gid is None:
                continue
            val = r.get("traj_success", r.get("score"))
            if val is not None:
                by_goal[gid].append(float(val))
                n_rows += 1
    if not by_goal:
        raise SystemExit("dump has no goal_id fields -- is FEDAGENT_LOG_GOAL_ID wired in the service?")

    trajectories = []
    n_success = 0
    for gid, vals in by_goal.items():
        ok = (sum(vals) / len(vals)) >= args.threshold
        n_success += int(ok)
        trajectories.append({"task_info": {"task_id": gid}, "traj_info": {"success": bool(ok)}})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"trajectories": trajectories}, indent=2))
    log(f"labelled {len(by_goal)} goals ({n_rows} samples): "
        f"{n_success}/{len(by_goal)} success ({100*n_success/len(by_goal):.1f}%)")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
