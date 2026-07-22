#!/usr/bin/env python
"""Generate the ALFWorld hardness trajectories file (task_id -> success) for the Hardness(xi')
task-heterogeneity arm -- the ALFWorld twin of tools/gen_hardness_trajectories.py.

The Hardness partition (partition_strategy.py::hardness_partition, ALFWorld branch) buckets train
GAMES into easy/hard by a per-game success label from a reference policy, keyed by the task_id
derived from the game's tw-pddl path: f"alfworld_{grandparent}_{parent}_game". This script produces
that label file the faithful way: roll the reference model over the 3,553 train games (one greedy
windowed episode each) and record per-game (task_id -> success).

Mechanism (reuses the overlay end to end, no new rollout code):
  - an ALFWorld service on the TRAIN split, UNPERTURBED (PARTITION_STRATEGY=""), with
    ALFWORLD_SEED_IS_INDEX=1 so /reset treats `seed` as a direct game index -> contiguous seeds
    0..num_games-1 cover EVERY train game exactly once (the default seeded-shuffle map is only
    pseudo-random, so full coverage would otherwise need ~N ln N episodes);
  - the AlfworldEnv client derives the hardness task_id from each episode's /reset gamefile and
    surfaces it as goal_id (+ task_type) in step info; the windowed/concat agent loop keeps it in
    verl's validation dump (skipped by metric aggregation);
  - run a verl val-only pass of the reference over a spec of N games; aggregate per task_id ->
    success = (mean success >= threshold) and write
    {"trajectories": [{"task_info": {"task_id": tid}, "traj_info": {"success": bool}}, ...]}.

Rollout-mode faithfulness: labelled in the SAME mode run_fed uses (cfg.rollout_mode, DEFAULT
windowed = the paper per-turn rollout, history_length=2, prompt 2048 / response 512 /
max_model_len 2560). A windowed-trained reference rolled out in concat mode measures near
zero-shot and the easy/hard split degenerates -- pass the paper (or a paper-budget) config.

Chunked/resumable: FEDAGENT_SEED_OFFSET=K shifts every row's seed so the run labels the disjoint
game window games[K : K+N]; merge chunk outputs afterwards. num_games wraps mod N.

Usage (service in verl-agent-alfworld already running, or launched by this tool):
    python -m tools.gen_alfworld_hardness_trajectories \
        --config fedagent/config/paper/task_heterogeneity/grpo/alfworld/fed_alfworld_grpo_*hardness*std-1.yaml \
        --model /path/to/reference/hf --num-goals 3553 \
        --output data/hardness/qwen2.5-1.5b_alfworld_trajectories.json \
        [--service-url http://127.0.0.1:8091] [--n-gpus 1]
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedagent.fed.run_fed import (  # noqa: E402
    DEFAULTS, log, stream, verl_cfg_dir, inject_rollout_mode, history_length_env,
)

PKG_DIR = REPO_ROOT / "fedagent"


def main():
    ap = argparse.ArgumentParser(description="ALFWorld hardness trajectories generator")
    ap.add_argument("--config", required=True, help="an ALFWorld fed YAML (for model/rollout settings)")
    ap.add_argument("--model", default=None, help="reference HF model (default: config model_path)")
    ap.add_argument("--num-goals", type=int, default=3553,
                    help="# train games to label (3553 = whole train pool; smaller = smoke)")
    ap.add_argument("--output", default=None, help="trajectories JSON path")
    ap.add_argument("--port", type=int, default=8091, help="labelling service port")
    ap.add_argument("--service-url", default=None,
                    help="reuse an ALREADY-RUNNING index-mode service at this URL (skip launch)")
    ap.add_argument("--pool-size", type=int, default=None, help="override alfworld_pool_size")
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
    pool_size = args.pool_size or int(cfg.get("alfworld_pool_size", 8))
    for k in ("custom_cls_path", "agent_config_path", "alfworld_run_service"):
        v = cfg.get(k)
        if v and not os.path.isabs(str(v)):
            cfg[k] = str(PKG_DIR / str(v))

    out_dir = Path(cfg.output_dir) / "hardness_labelling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "alfworld_trajectories.json"

    # spec enumerating N games (seeds 0..N-1 -> games[0:N] via index mode)
    spec = {"envs": [{"name": "ALFWorld", "n_envs": int(args.num_goals), "max_turns": 50,
                      "agent_name": "gym_text", "config": {"timeout": 180.0}}]}
    spec_path = out_dir / "label_spec.yaml"
    OmegaConf.save(OmegaConf.create(spec), spec_path)

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO_ROOT}:{env_base.get('PYTHONPATH', '')}".rstrip(":")
    env_base["VERL_CFG"] = verl_cfg_dir()
    env_base.pop("FEDPROX_MU", None)

    svc = None
    port = args.port
    if args.service_url:
        service_url = args.service_url
        log(f"reusing running ALFWorld service at {service_url}")
    else:
        # launch the labelling service: TRAIN split, unperturbed, INDEX MODE (seed == game idx)
        svc_env = dict(env_base)
        svc_env.update({
            "ALFWORLD_PORT": str(port),
            "ALFWORLD_POOL_SIZE": str(pool_size),
            "ALFWORLD_TRAIN_EVAL": str(cfg.get("alfworld_train_eval", "train")),
            "PARTITION_STRATEGY": "",             # unperturbed full train pool
            "CLIENT_ID": "0", "CLIENT_NUM": "1",
            "ALFWORLD_SEED_IS_INDEX": "1",        # contiguous seeds -> bijective game coverage
            "ALFWORLD_DATA": os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld")),
        })
        svc_log = out_dir / "label_service.log"
        lf = open(svc_log, "w")
        log(f"starting ALFWorld labelling service on :{port} (TRAIN, unperturbed, INDEX MODE)")
        svc = subprocess.Popen(["bash", str(cfg.alfworld_run_service)], env=svc_env,
                               stdout=lf, stderr=subprocess.STDOUT)
        service_url = f"http://localhost:{port}"
        url = f"{service_url}/health"
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

    try:
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
            "actor_rollout_ref.rollout.n=1",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={args.temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={'true' if args.temperature > 0 else 'false'}",
            "trainer.project_name=fedagent_alfworld_hardness_label",
            "trainer.experiment_name=label",
        ]
        cmd += [str(o) for o in (cfg.client_overrides or [])
                if not str(o).startswith("actor_rollout_ref.rollout.n=")]
        inject_rollout_mode(cmd, cfg)
        run_env = dict(env_base)
        run_env.update(history_length_env(cfg))
        run_env["ALFWORLD_SERVICE_URL"] = service_url
        rc = stream(cmd, run_env, out_dir / "label.log", tag="label")
        if rc != 0:
            raise SystemExit(f"labelling val pass FAILED (rc={rc}); see {out_dir / 'label.log'}")
    finally:
        if svc is not None:
            try:
                svc.terminate(); svc.wait(timeout=15)
            except Exception:
                try:
                    svc.kill()
                except Exception:
                    pass
            lf.close()

    # aggregate per task_id -> success label
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
        raise SystemExit("dump has no goal_id fields -- is the AlfworldEnv goal_id wiring present?")

    trajectories = []
    n_success = 0
    for gid, vals in by_goal.items():
        ok = (sum(vals) / len(vals)) >= args.threshold
        n_success += int(ok)
        trajectories.append({"task_info": {"task_id": gid}, "traj_info": {"success": bool(ok)}})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"trajectories": trajectories}, indent=2))
    log(f"labelled {len(by_goal)} games ({n_rows} samples): "
        f"{n_success}/{len(by_goal)} success ({100*n_success/len(by_goal):.1f}%)")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
