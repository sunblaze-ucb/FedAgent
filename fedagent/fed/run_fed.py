#!/usr/bin/env python
"""Lean verl-0.8 federated runner for FedAgent (thin overlay).

Closes the FedAgent federated loop on STOCK verl 0.8 the FedAgent way -- one
training SUBPROCESS per (client, round), then FedAvg the resulting FSDP
checkpoints and re-enter the next round from the aggregated model. The
orchestration is verl-agnostic: a client is just ``python -m fedagent.main_ppo_fed``
(the Phase-1 entry), so this driver never imports verl.

Round r:
    model_r = base_model                       (r == 1)
            = round_{r-1}/aggregated/hf        (r > 1)   # merged FedAvg'd shards
    for each selected client c (SEQUENTIAL):
        python -m fedagent.main_ppo_fed ...
            actor_rollout_ref.model.path=model_r
            trainer.default_local_dir=round_r/client_c/checkpoints
            trainer.total_epochs=E
          env  FEDAGENT_BASE_SEED=base_seed+c     # distinct env instances per client
        -> round_r/client_c/checkpoints/global_step_K/actor   (FSDP shards, ws=n_gpus)
    FedAvg: torchrun --nproc_per_node=ws tools/.../aggregate_fedavg_fsdp.py
            --client-actor-dirs <c0>,<c1> --output-actor-dir round_r/aggregated/.../actor
    merge : python -m verl.model_merger merge --backend fsdp
            --local_dir <agg actor> --target_dir round_r/aggregated/hf

The loop is "closed" when round 2 trains from round-1's aggregated model and a
final aggregated model exists.

Why not reuse core/custom_fed_server.py: that orchestrator regex-rewrites the
verl-agent 0.3.1 base bash script (core/fed/script_builder.py) and assumes a
``config['verl']/['data_preprocess']`` schema + ``model_world_size_1`` single-rank
checkpoints -- none of which the thin overlay uses. This is its verl-0.8 successor.

Usage (run inside the fedagent-verl08 env, on the GPU node):
    python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml
CLI flags override the YAML: --model-path --output-dir --rounds --clients.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

# fedagent/fed/run_fed.py -> PKG_DIR=fedagent/ , REPO_ROOT=repo root
PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parent
AGGREGATOR = REPO_ROOT / "tools" / "verl08_migration" / "aggregate_fedavg_fsdp.py"

DEFAULTS = {
    "model_path": "",                       # "" => auto-discover Qwen2.5-0.5B-Instruct
    "env_spec": str(PKG_DIR / "config" / "envs" / "tiny_guess.yaml"),
    "custom_cls_path": str(PKG_DIR / "data" / "agentic_dataset.py"),
    "agent_config_path": str(PKG_DIR / "config" / "agent.yaml"),
    "output_dir": "/tmp/xbb9020_fedagent_fed_tinyguess",
    "total_clients": 2,
    "clients_per_round": 2,
    "total_rounds": 2,
    "epochs_per_round": 1,
    "base_seed": 42,
    "n_gpus_per_node": 2,
    "total_training_steps": 1,              # cap per client-round (keep the smoke fast)
    "save_freq": 1,
    "weights": "",                          # "" => uniform FedAvg
    "wait_between_clients": 5,              # seconds; let Ray/GPU fully release
    "client_overrides": [],                 # extra `key=value` Hydra overrides per client (env-specific)
    # --- env_kind=webshop: per-client remote env services + Catalog-Split heterogeneity ---
    "env_kind": "tinyguess",                # "tinyguess" (in-process) | "webshop" | "alfworld" (remote services)
    "webshop_run_service": str(PKG_DIR / "envs" / "webshop" / "service" / "run_service.sh"),
    "webshop_base_port": 8080,              # client c's service -> webshop_base_port + c
    "webshop_pool_size": 8,                 # env pool per service (must be >= gen_batch)
    "search_return_n": 200,                 # WEBSHOP_SEARCH_RETURN_N: BM25 top-K (paper=200; engine default 50 drops targets under env-het filtering)
    # --- env_kind=alfworld: per-client remote ALFWorld services + game-shard heterogeneity ---
    "alfworld_run_service": str(PKG_DIR / "envs" / "alfworld" / "service" / "run_service.sh"),
    "alfworld_base_port": 8200,             # client c's service -> alfworld_base_port + c
    "alfworld_pool_size": 4,                # textworld env pool per service (must be >= gen_batch)
    "alfworld_train_eval": "train",         # game split: train | eval_in_distribution | eval_out_of_distribution
    "alfworld_task_types": "",               # "" => all 6 types; else comma-sep IDs (1=Pick..6=Pick2) for the eval breakdown
    "partition_strategy": "",               # "" | catalog_split/task_disjoint (env) | preference/coverage/hardness (task) | bm25_field_subset/bm25_reweight/lookalike/rank_wrapper (env variants 2-5)
    "env_div": 0.7,                         # catalog-split heterogeneity strength
    "keep_ratio": 0.7,                      # catalog-split distractor density
    "omega": 0.5,                           # preference (task-het) Dirichlet spread
    "size_std": 1.0,                        # coverage (task-het) Beta dispersion (xi)
    "success_std": 1.0,                     # hardness (task-het) Beta dispersion (xi')
    "variant_n": 0,                         # env-variant arms (bm25/lookalike/rank): # variants in pool (0 => fn default 2/4)
    "trajectories_file": "",                # hardness: REQUIRED task_id->success labels file
    "min_goals_per_client": 100,
    "service_health_timeout": 900,          # seconds to wait for a service /health
    "fedprox_mu": 0.0,                       # >0 => FedProx proximal term (else FedAvg)
    "cleanup_checkpoints": True,             # delete consumed FSDP shards after each merge (disk hygiene)
    "adv_estimator": "grpo",                 # "grpo" (no critic) | "gae" (PPO: federate the critic too)
    # --- baseline modes (vs the default N-client federated loop) ---
    #   Federated  : default (total_clients=N>1, local_client_id<0)  -> FedAvg across clients.
    #   Centralized: total_clients=1 (+ partition_strategy="")        -> one model on the pooled
    #                data; the per-round FedAvg of a single client is the identity, so the loop
    #                is just T*E epochs of continued centralized training.
    #   Local      : local_client_id=k>=0 (+ total_clients=N)         -> the paper's "Local Agent
    #                Training": pin ONE client (its slice of the N-way partition) every round and
    #                train it alone, no federation (== original 'uniform_single').
    "local_client_id": -1,                   # >=0 => Local baseline: train only this client of total_clients
    # --- unperturbed global-model validation/eval (paper: test_freq=5, val_before_train, temp 0.4) ---
    "val_env_spec": "",                      # "" => NO eval (back-compat); else the UNPERTURBED val env-spec
    "test_freq": 5,                          # eval the aggregated global model every K rounds (+ final round)
    "val_before_train": True,               # also eval the base model before round 1 (the round-0 point)
    "val_temperature": 0.4,                 # val sampling temperature (paper val_kwargs.temperature=0.4)
    "webshop_val_port": 8090,               # shared unperturbed WebShop val service port
    "alfworld_val_port": 8290,              # shared unperturbed ALFWorld val service port
    "alfworld_val_split": "eval_in_distribution",  # ALFWorld val games (274 in-distribution eval set)
}


# ----------------------------------------------------------------- logging
def log(msg: str):
    print(f"[fed] {msg}", flush=True)


def banner(msg: str):
    bar = "=" * 78
    print(f"\n{bar}\n[fed] {msg}\n{bar}", flush=True)


# ----------------------------------------------------------------- helpers
def discover_model() -> str:
    """Locate a local Qwen2.5-0.5B-Instruct snapshot (offline), mirroring run_smoke.sh."""
    for base in ("/projects/b1222/.cache/huggingface", os.path.expanduser("~/.cache/huggingface")):
        hub = Path(base) / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots"
        if hub.is_dir():
            snaps = sorted(hub.glob("*/"))
            if snaps:
                return str(snaps[0]).rstrip("/")
    raise FileNotFoundError("No local Qwen2.5-0.5B-Instruct snapshot found")


def verl_cfg_dir() -> str:
    """verl's stock ppo_trainer config dir (for hydra.searchpath); env wins if set."""
    if os.environ.get("VERL_CFG"):
        return os.environ["VERL_CFG"]
    import verl  # noqa
    return str(Path(verl.__file__).resolve().parent / "trainer" / "config")


def latest_actor_dir(ckpt_root: Path) -> Optional[Path]:
    """Newest ``global_step_K/actor`` under ckpt_root that actually holds FSDP shards."""
    if not ckpt_root.is_dir():
        return None
    steps = []
    for d in ckpt_root.iterdir():
        if d.is_dir() and d.name.startswith("global_step_"):
            try:
                steps.append((int(d.name.split("_")[2]), d))
            except (ValueError, IndexError):
                continue
    for _, d in sorted(steps, reverse=True):
        actor = d / "actor"
        if actor.is_dir() and list(actor.glob("model_world_size_*_rank_*.pt")):
            return actor
    return None


def critic_dir_for(actor_dir: Optional[Path]) -> Optional[Path]:
    """The critic FSDP-shard dir saved alongside an actor dir (PPO/gae), or None (GRPO).

    verl saves PPO's value model to ``global_step_K/critic`` next to ``actor`` (same world
    size, same ``model_world_size_*_rank_*.pt`` layout + a ``huggingface/`` config that
    serializes as ``...ForCausalLM`` with a scalar value head -- so it FedAvgs + merges with the
    exact actor machinery). GRPO writes no critic, so this returns None and the critic path is
    skipped entirely."""
    if actor_dir is None:
        return None
    critic = actor_dir.parent / "critic"
    if critic.is_dir() and list(critic.glob("model_world_size_*_rank_*.pt")):
        return critic
    return None


def world_size_of(actor_dir: Path) -> int:
    """world_size of the saved shards (== aggregator nproc): fsdp_config.json, else filename."""
    cfg = actor_dir / "fsdp_config.json"
    if cfg.is_file():
        try:
            return int(json.loads(cfg.read_text())["world_size"])
        except (ValueError, KeyError, OSError):
            pass
    ws = {int(p.name.split("model_world_size_")[1].split("_rank_")[0])
          for p in actor_dir.glob("model_world_size_*_rank_*.pt")}
    if len(ws) != 1:
        raise RuntimeError(f"Cannot determine a unique world_size in {actor_dir}: {ws}")
    return ws.pop()


def stream(cmd: List[str], env: dict, log_path: Path, tag: str) -> int:
    """Run cmd, tee combined stdout/stderr to console (tagged) and to log_path. Return rc."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}")
    log(f"  (log: {log_path})")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(f"  [{tag}] {line}")
            sys.stdout.flush()
            lf.write(line)
        proc.wait()
    return proc.returncode


# ----------------------------------------------------------------- webshop services
def webshop_service_url(cfg, client_id: int) -> str:
    return f"http://localhost:{cfg.webshop_base_port + client_id}"


def start_webshop_services(cfg, env_base: dict, client_ids: Optional[List[int]] = None) -> List[dict]:
    """Launch ONE WebShop remote service per client in ``client_ids`` (Design A: one service
    == one client's environment / hidden transition kernel). Each service builds its whole
    env pool with that client's Catalog-Split variant (via env-var bridge). Returns handles
    for teardown. Raises if any service fails to come up healthy.

    ``client_ids`` defaults to ALL participating clients; run_fed passes only the ROUND's
    selected clients (lazy per-round startup), so an N=100 config never starts 100 services
    at once -- at most ``clients_per_round`` are alive at a time. The per-client shard is a
    function of CLIENT_ID/CLIENT_NUM (round-independent), so lazy startup is reproducible."""
    import urllib.request

    if client_ids is None:
        client_ids = participating_client_ids(cfg)
    # Launch + health-wait inside a try so a PARTIAL-startup failure (a later service in this
    # batch dies / times out) tears down the ones already Popen'd -- otherwise the caller's
    # `round_services = start_*_services(...)` never binds (the call raised), its finally sees
    # [] and stop_services([]) is a no-op, leaking uvicorn procs + their bound ports.
    services = []
    try:
        for c in client_ids:
            port = cfg.webshop_base_port + c
            env = dict(env_base)
            env.update({
                "WEBSHOP_PORT": str(port),
                "WEBSHOP_POOL_SIZE": str(cfg.webshop_pool_size),
                "WEBSHOP_SEARCH_RETURN_N": str(cfg.get("search_return_n", 200)),
                "PARTITION_STRATEGY": cfg.partition_strategy or "",
                "CLIENT_ID": str(c),
                "CLIENT_NUM": str(cfg.total_clients),
                "ENV_DIV": str(cfg.env_div),
                "KEEP_RATIO": str(cfg.keep_ratio),
                "OMEGA": str(cfg.get("omega", 0.5)),
                "SIZE_STD": str(cfg.get("size_std", 1.0)),
                "SUCCESS_STD": str(cfg.get("success_std", 1.0)),
                "VARIANT_N": (str(cfg.get("variant_n")) if cfg.get("variant_n", 0) else ""),
                "TRAJECTORIES_FILE": str(cfg.get("trajectories_file", "")),
                "MIN_GOALS_PER_CLIENT": str(cfg.min_goals_per_client),
            })
            log_path = Path(cfg.output_dir) / f"webshop_service_client{c}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_path, "w")
            log(f"starting WebShop service client {c} on :{port} "
                f"(pool={cfg.webshop_pool_size}, partition={cfg.partition_strategy or 'none'}, "
                f"env_div={cfg.env_div})  (log: {log_path})")
            proc = subprocess.Popen(["bash", str(cfg.webshop_run_service)], env=env,
                                    stdout=lf, stderr=subprocess.STDOUT)
            services.append({"client_id": c, "port": port, "proc": proc, "log": log_path, "lf": lf})

        # wait for each to report /health (pool warmup of WebShop envs takes minutes)
        for s in services:
            url = f"http://localhost:{s['port']}/health"
            deadline_polls = int(cfg.service_health_timeout / 3)
            up = False
            for _ in range(deadline_polls):
                if s["proc"].poll() is not None:
                    raise RuntimeError(f"WebShop service client {s['client_id']} DIED; see {s['log']}")
                try:
                    with urllib.request.urlopen(url, timeout=3) as r:
                        import json as _json
                        d = _json.loads(r.read())
                    log(f"WebShop service client {s['client_id']} healthy on :{s['port']} "
                        f"(partition={d.get('partition')}, catalog_size={d.get('catalog_size')})")
                    up = True
                    break
                except Exception:
                    time.sleep(3)
            if not up:
                raise RuntimeError(f"WebShop service client {s['client_id']} health timeout; see {s['log']}")
    except BaseException:
        stop_services(services)   # no orphaned uvicorn/port survives a partial-startup failure
        raise
    return services


def stop_services(services: List[dict]):
    for s in services or []:
        try:
            s["proc"].terminate()
        except Exception:
            pass
    for s in services or []:
        try:
            s["proc"].wait(timeout=15)
        except Exception:
            try:
                s["proc"].kill()
            except Exception:
                pass
        try:
            s["lf"].close()
        except Exception:
            pass
    if services:
        log(f"stopped {len(services)} remote env service(s)")


# ----------------------------------------------------------------- alfworld services
def alfworld_service_url(cfg, client_id: int) -> str:
    return f"http://localhost:{cfg.alfworld_base_port + client_id}"


def start_alfworld_services(cfg, env_base: dict, client_ids: Optional[List[int]] = None) -> List[dict]:
    """Launch ONE ALFWorld remote service per client in ``client_ids`` (Design A: one service
    == one client's game shard / hidden transition kernel). Each service builds its textworld
    env pool from that client's slice of the train games (via the env-var bridge). Mirrors
    start_webshop_services, incl. lazy per-round startup: ``client_ids`` defaults to all
    participating clients; run_fed passes only the round's selected clients. Raises on failure."""
    import urllib.request

    if client_ids is None:
        client_ids = participating_client_ids(cfg)
    # partial-startup teardown (see start_webshop_services): a later service failing must not
    # leak the ones already launched, since the caller's finally would otherwise see [].
    services = []
    try:
        for c in client_ids:
            port = cfg.alfworld_base_port + c
            env = dict(env_base)
            env.update({
                "ALFWORLD_PORT": str(port),
                "ALFWORLD_POOL_SIZE": str(cfg.alfworld_pool_size),
                "ALFWORLD_TRAIN_EVAL": str(cfg.get("alfworld_train_eval", "train")),
                "PARTITION_STRATEGY": cfg.partition_strategy or "uniform",
                "CLIENT_ID": str(c),
                "CLIENT_NUM": str(cfg.total_clients),
                "MIN_GOALS_PER_CLIENT": str(cfg.min_goals_per_client),
                # task-het knobs (the service forwards only the ones its strategy needs ->
                # AlfredTWEnv -> partition_dataset): preference(omega)/coverage(size_std)/
                # hardness(success_std,trajectories_file). uniform/env_disjoint ignore them.
                "OMEGA": str(cfg.get("omega", 0.5)),
                "SIZE_STD": str(cfg.get("size_std", 1.0)),
                "SUCCESS_STD": str(cfg.get("success_std", 1.0)),
                "TRAJECTORIES_FILE": str(cfg.get("trajectories_file", "")),
            })
            log_path = Path(cfg.output_dir) / f"alfworld_service_client{c}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_path, "w")
            log(f"starting ALFWorld service client {c} on :{port} "
                f"(pool={cfg.alfworld_pool_size}, partition={cfg.partition_strategy or 'uniform'}, "
                f"split={cfg.get('alfworld_train_eval', 'train')})  (log: {log_path})")
            proc = subprocess.Popen(["bash", str(cfg.alfworld_run_service)], env=env,
                                    stdout=lf, stderr=subprocess.STDOUT)
            services.append({"client_id": c, "port": port, "proc": proc, "log": log_path, "lf": lf})

        # wait for each to report /health (ALFWorld collects game files -> can take minutes)
        for s in services:
            url = f"http://localhost:{s['port']}/health"
            deadline_polls = int(cfg.service_health_timeout / 3)
            up = False
            for _ in range(deadline_polls):
                if s["proc"].poll() is not None:
                    raise RuntimeError(f"ALFWorld service client {s['client_id']} DIED; see {s['log']}")
                try:
                    with urllib.request.urlopen(url, timeout=3) as r:
                        import json as _json
                        d = _json.loads(r.read())
                    log(f"ALFWorld service client {s['client_id']} healthy on :{s['port']} "
                        f"(partition={d.get('partition')}, num_games={d.get('num_games')})")
                    up = True
                    break
                except Exception:
                    time.sleep(3)
            if not up:
                raise RuntimeError(f"ALFWorld service client {s['client_id']} health timeout; see {s['log']}")
    except BaseException:
        stop_services(services)   # no orphaned uvicorn/port survives a partial-startup failure
        raise
    return services


# ----------------------------------------------------------------- unperturbed eval
def val_service_url(cfg) -> str:
    base = cfg.webshop_val_port if cfg.env_kind == "webshop" else cfg.alfworld_val_port
    return f"http://localhost:{base}"


def start_val_service(cfg, env_base: dict) -> Optional[dict]:
    """Start the ONE shared UNPERTURBED validation service (full env, held-out val goal/game
    split), used to score the aggregated GLOBAL model every test_freq rounds so all arms are
    measured on the same fixed set. Returns None when eval is off (val_env_spec unset) or the
    env is in-process (tinyguess). Mirrors the per-client starters' health-wait."""
    import urllib.request

    if not cfg.get("val_env_spec"):
        return None
    if cfg.env_kind == "webshop":
        port = cfg.webshop_val_port
        run_service = cfg.webshop_run_service
        env = dict(env_base)
        env.update({
            "WEBSHOP_PORT": str(port),
            "WEBSHOP_POOL_SIZE": str(cfg.webshop_pool_size),
            "WEBSHOP_SEARCH_RETURN_N": str(cfg.get("search_return_n", 200)),
            "WEBSHOP_SPLIT": "val",          # held-out goals[0:VAL_SIZE]
            "PARTITION_STRATEGY": "",        # UNPERTURBED (no catalog/goal/variant skew)
            "CLIENT_ID": "0", "CLIENT_NUM": "1",
        })
        tag, health_extra = "WebShop", "split"
    elif cfg.env_kind == "alfworld":
        port = cfg.alfworld_val_port
        run_service = cfg.alfworld_run_service
        env = dict(env_base)
        env.update({
            "ALFWORLD_PORT": str(port),
            "ALFWORLD_POOL_SIZE": str(cfg.alfworld_pool_size),
            "ALFWORLD_TRAIN_EVAL": str(cfg.get("alfworld_val_split", "eval_in_distribution")),
            "ALFWORLD_TASK_TYPES": str(cfg.get("alfworld_task_types", "")),  # "" => all; else the eval-breakdown subset
            "PARTITION_STRATEGY": "uniform",  # UNPERTURBED (full game set, no client shard)
            "CLIENT_ID": "0", "CLIENT_NUM": "1",
        })
        tag, health_extra = "ALFWorld", "num_games"
    else:
        return None  # tinyguess runs in-process; no remote val service

    log_path = Path(cfg.output_dir) / f"{cfg.env_kind}_val_service.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_path, "w")
    log(f"starting {tag} VAL service (UNPERTURBED) on :{port}  (log: {log_path})")
    proc = subprocess.Popen(["bash", str(run_service)], env=env, stdout=lf, stderr=subprocess.STDOUT)
    s = {"client_id": "val", "port": port, "proc": proc, "log": log_path, "lf": lf}
    url = f"http://localhost:{port}/health"
    try:  # mirror the per-client starters: a death/timeout here must not orphan the uvicorn/port
        for _ in range(int(cfg.service_health_timeout / 3)):
            if proc.poll() is not None:
                raise RuntimeError(f"{tag} VAL service DIED; see {log_path}")
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    import json as _json
                    d = _json.loads(r.read())
                log(f"{tag} VAL service healthy on :{port} ({health_extra}={d.get(health_extra)})")
                return s
            except Exception:
                time.sleep(3)
        raise RuntimeError(f"{tag} VAL service health timeout; see {log_path}")
    except BaseException:
        stop_services([s])   # caller hasn't received the handle yet -> clean up here before propagating
        raise


def summarize_val_dump(dump_dir: Path) -> Optional[dict]:
    """Read verl's validation_data_dir JSONL dump -> {n, success_rate, reward_mean}. The
    agent loop tags each val sample with traj_success (1.0 if the episode succeeded) and
    score (episode reward), so the mean over samples is the val success / reward."""
    files = sorted(Path(dump_dir).glob("*.jsonl"))
    if not files:
        return None
    rows = []
    with open(files[-1]) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None

    def mean(key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {"n": len(rows), "success_rate": mean("traj_success"), "reward_mean": mean("score")}


def eval_global(cfg, model_path: str, round_num: int, env_base: dict, val_url: str) -> Optional[dict]:
    """Score the GLOBAL model (base on round 0, else the round's aggregated HF) on the shared
    unperturbed val service via a verl val-only pass (no training, no critic, val temp from
    cfg). Returns the parsed val metrics, or None on failure (a failed eval never aborts the
    federated run -- it is measurement, not the loop)."""
    eval_dir = Path(cfg.output_dir) / (f"round_{round_num}" if round_num > 0 else "round_0") / "eval"
    dump_dir = eval_dir / "val_samples"
    cmd = [
        sys.executable, "-m", "fedagent.main_ppo_fed",
        f"data.train_files={cfg.val_env_spec}",
        f"data.val_files={cfg.val_env_spec}",
        f"data.custom_cls.path={cfg.custom_cls_path}",
        f"actor_rollout_ref.model.path={model_path}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={cfg.agent_config_path}",
        f"trainer.default_local_dir={eval_dir / 'ckpt'}",
        f"trainer.n_gpus_per_node={cfg.n_gpus_per_node}",
        "trainer.val_only=true",
        "trainer.val_before_train=true",
        "algorithm.adv_estimator=grpo",                  # eval = generate+score only; no critic regardless of train algo
        f"trainer.validation_data_dir={dump_dir}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={cfg.val_temperature}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "trainer.project_name=fedagent_fed_eval",
        f"trainer.experiment_name=round{round_num}_eval",
    ]
    cmd += [str(o) for o in (cfg.client_overrides or [])]   # reuse rollout shape (prompt/response/n/mem)
    env = dict(env_base)
    env.pop("FEDPROX_MU", None)                              # eval must never enable the proximal term
    if cfg.env_kind == "webshop":
        env["WEBSHOP_SERVICE_URL"] = val_url
    elif cfg.env_kind == "alfworld":
        env["ALFWORLD_SERVICE_URL"] = val_url
    log_path = eval_dir / "eval.log"
    rc = stream(cmd, env, log_path, tag=f"eval-r{round_num}")
    if rc != 0:
        log(f"[warn] eval round {round_num} FAILED (rc={rc}); see {log_path} (continuing)")
        return None
    metrics = summarize_val_dump(dump_dir)
    if metrics:
        log(f"round {round_num} VAL (unperturbed): success={metrics['success_rate']} "
            f"reward={metrics['reward_mean']} (n={metrics['n']})")
    return metrics


# ----------------------------------------------------------------- stages
def participating_client_ids(cfg) -> List[int]:
    """Client ids that ever train -> the env services to launch. Local mode needs only the
    one pinned client; federated/centralized need all of them."""
    lid = int(cfg.get("local_client_id", -1))
    if lid >= 0:
        return [lid]
    return list(range(cfg.total_clients))


def select_clients(round_num: int, total: int, per_round: int, base_seed: int,
                   local_client_id: int = -1) -> List[int]:
    """Deterministic per-round client selection (seed = base_seed + round - 1), matching
    core/fed/round_orchestrator.select_clients so the loop is reproducible on resume.

    Local baseline (local_client_id>=0): pin that single client every round (no sampling,
    no federation) -- the original's 'uniform_single' "Local Agent Training" baseline."""
    if local_client_id >= 0:
        return [local_client_id]
    if per_round >= total:
        return list(range(total))
    import random
    rng = random.Random(base_seed + round_num - 1)
    return sorted(rng.sample(range(total), per_round))


def run_client(cfg, round_num: int, client_id: int, model_path: str,
               env_base: dict, critic_model_path: Optional[str] = None) -> tuple:
    """Train one client for one round; return (actor_dir, critic_dir_or_None).

    GRPO: critic_dir is None. PPO (cfg.adv_estimator=="gae"): the value model trains from
    ``critic_model_path`` (base model on round 1, previous round's aggregated critic after),
    and its checkpoint dir is returned for FedAvg alongside the actor."""
    ckpt_root = Path(cfg.output_dir) / f"round_{round_num}" / f"client_{client_id}" / "checkpoints"
    cmd = [
        sys.executable, "-m", "fedagent.main_ppo_fed",
        f"data.train_files={cfg.env_spec}",
        f"data.val_files={cfg.env_spec}",
        f"data.custom_cls.path={cfg.custom_cls_path}",
        f"actor_rollout_ref.model.path={model_path}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={cfg.agent_config_path}",
        f"trainer.default_local_dir={ckpt_root}",
        f"trainer.n_gpus_per_node={cfg.n_gpus_per_node}",
        f"trainer.total_epochs={cfg.epochs_per_round}",
        f"trainer.save_freq={cfg.save_freq}",
        "trainer.val_before_train=false",
        # Federation owns "resume" at the ROUND level (each client starts from the merged HF via
        # model.path). Disable verl's per-run auto-resume so a re-run never silently resumes a
        # crashed in-flight round's partial per-client checkpoint and FedAvgs pre-crash weights.
        "trainer.resume_mode=disable",
        f"trainer.project_name=fedagent_fed",
        f"trainer.experiment_name=round{round_num}_client{client_id}",
    ]
    # total_training_steps>0 caps steps/round (smokes); <=0 => emit null so verl runs the FULL
    # E epochs (paper configs) via len(dataloader)*total_epochs (ray_trainer.py:438-441).
    # MUST emit explicitly (not omit): omitting would let the base config's value (a smoke cap)
    # silently leak into paper runs. NOTE: passing 0 would make verl treat step 0 as the last
    # step (no training), so <=0 maps to null, never 0.
    if int(cfg.total_training_steps) > 0:
        cmd.append(f"trainer.total_training_steps={cfg.total_training_steps}")
    else:
        cmd.append("trainer.total_training_steps=null")
    cmd += [str(o) for o in (cfg.client_overrides or [])]   # env-specific Hydra overrides
    # PPO (gae): enable the critic and load it from critic_model_path. Appended AFTER
    # client_overrides so the per-round critic path wins; adv_estimator=gae alone flips
    # need_critic on (critic.enable defaults to null). GRPO leaves the cmd byte-identical
    # to the verified path (no extra overrides).
    if str(cfg.get("adv_estimator", "grpo")).lower() == "gae":
        cmd += ["algorithm.adv_estimator=gae"]
        if critic_model_path:
            cmd += [f"critic.model.path={critic_model_path}"]
    # NOTE: FedProx is injected via sitecustomize.py (repo root, on PYTHONPATH for the
    # client + its Ray workers), NOT via a Ray runtime_env worker_process_setup_hook. The
    # cluster-level runtime_env hook clobbered verl's per-worker CUDA_VISIBLE_DEVICES
    # assignment -> "Duplicate GPU detected: rank N and rank 0 both on CUDA device". The
    # sitecustomize path runs at interpreter startup in every process (gated on FEDPROX_MU)
    # without touching the runtime_env, so GPU isolation is preserved.
    env = dict(env_base)
    # distinct, reproducible env instances per client (AgenticDataset reads this);
    # round-invariant so a client's task distribution is stable across rounds.
    # Round-threaded data seed: like the ORIGINAL fed sampler, seed per (round, client) so each
    # client re-draws goals from its FIXED shard every round (covering the shard over T rounds).
    # (The original used base_seed + round*1000 + client*100; this overlay uses the smaller stride
    # on the line below -- same round-threading intent, different constants.) Without the round
    # term every client would train on the SAME goals every round. Stride 100 (round) + client_id
    # (<100) is collision-free and keeps AgenticDataset's seed*100000 < 2**32.
    env["FEDAGENT_BASE_SEED"] = str(cfg.base_seed + round_num * 100 + client_id)
    if cfg.env_kind == "webshop":
        # talk to THIS client's WebShop service (its disjoint Catalog-Split env)
        env["WEBSHOP_SERVICE_URL"] = webshop_service_url(cfg, client_id)
    elif cfg.env_kind == "alfworld":
        # talk to THIS client's ALFWorld service (its disjoint game shard)
        env["ALFWORLD_SERVICE_URL"] = alfworld_service_url(cfg, client_id)
    if cfg.get("fedprox_mu", 0) and cfg.fedprox_mu > 0:
        # FedProx: sitecustomize.py reads this at startup in each worker and patches
        # FSDPEngine.optimizer_step to add the proximal term (mu>0 => FedProx, else FedAvg).
        env["FEDPROX_MU"] = str(cfg.fedprox_mu)

    log_path = ckpt_root.parent / "training.log"
    rc = stream(cmd, env, log_path, tag=f"r{round_num}c{client_id}")
    if rc != 0:
        raise RuntimeError(f"client {client_id} round {round_num} FAILED (rc={rc}); see {log_path}")

    actor = latest_actor_dir(ckpt_root)
    if actor is None:
        raise RuntimeError(
            f"client {client_id} round {round_num}: no checkpoint produced under {ckpt_root}; "
            f"see {log_path}"
        )
    critic = critic_dir_for(actor)
    log(f"client {client_id} round {round_num} OK -> {actor}"
        + (f" (+critic {critic})" if critic else ""))
    # measurability: post-process the training.log into json_logs/metrics.json
    # (FedAgent plot format) and echo the per-step reward curve.
    try:
        from fedagent.fed.metrics_logger import parse_training_log, summarize, write_metrics_json
        write_metrics_json(log_path, ckpt_root.parent / "json_logs")
        log(f"client {client_id} round {round_num} reward: {summarize(parse_training_log(log_path))}")
    except Exception as e:
        log(f"[warn] metrics parse r{round_num}c{client_id}: {e}")
    return actor, critic


def fedavg(cfg, round_num: int, client_dirs: List[Path], env_base: dict,
           kind: str = "actor") -> Path:
    """FedAvg the clients' ``kind`` shards (actor|critic) under a matched-ws PG; return the
    aggregated dir. The aggregator is name-agnostic (it averages whatever shard dir it is
    given and copies fsdp_config.json + huggingface/), so the critic uses the same path."""
    ws = world_size_of(client_dirs[0])
    for a in client_dirs[1:]:
        if world_size_of(a) != ws:
            raise RuntimeError(f"{kind} world_size mismatch across clients: {a} != ws {ws}")

    agg = (Path(cfg.output_dir) / f"round_{round_num}" / "aggregated"
           / "checkpoints" / "global_step_0" / kind)
    cmd = [
        "torchrun", f"--nproc_per_node={ws}", str(AGGREGATOR),
        "--phase", "aggregate",
        "--client-actor-dirs", ",".join(str(a) for a in client_dirs),
        "--output-actor-dir", str(agg),
        "--global-step", "0",
    ]
    if cfg.weights:
        cmd += ["--weights", cfg.weights]
    log_path = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / f"aggregate_{kind}.log"
    rc = stream(cmd, env_base, log_path, tag=f"agg-{kind}-r{round_num}")
    if rc != 0:
        raise RuntimeError(f"FedAvg {kind} round {round_num} FAILED (rc={rc}); see {log_path}")
    if not list(agg.glob("model_world_size_*_rank_*.pt")):
        raise RuntimeError(f"FedAvg {kind} round {round_num}: no aggregated shards in {agg}")
    if not (agg / "huggingface").is_dir():
        raise RuntimeError(f"FedAvg {kind} round {round_num}: missing huggingface/ config in {agg}")
    log(f"FedAvg {kind} round {round_num} OK (ws={ws}) -> {agg}")
    return agg


def merge_to_hf(cfg, round_num: int, agg_dir: Path, env_base: dict,
                kind: str = "actor") -> Path:
    """Merge aggregated FSDP shards -> a complete HF model dir for the next round's model.path
    (actor) or critic.model.path (critic). The merger auto-detects the architecture from the
    shard's huggingface/config.json (both serialize as ...ForCausalLM; the value model just
    carries an extra scalar value head), so no per-kind flag is needed."""
    sub = "hf" if kind == "actor" else f"{kind}_hf"
    hf_dir = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / sub
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(agg_dir),
        "--target_dir", str(hf_dir),
    ]
    log_path = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / f"merge_{kind}.log"
    rc = stream(cmd, env_base, log_path, tag=f"merge-{kind}-r{round_num}")
    if rc != 0:
        raise RuntimeError(f"model_merger {kind} round {round_num} FAILED (rc={rc}); see {log_path}")
    if not (hf_dir / "config.json").is_file():
        raise RuntimeError(f"merge {kind} round {round_num}: no config.json in {hf_dir}")
    weights = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("*.bin"))
    if not weights:
        raise RuntimeError(f"merge {kind} round {round_num}: no weight files in {hf_dir}")
    log(f"merge {kind} round {round_num} OK -> {hf_dir} ({len(weights)} weight file(s))")
    return hf_dir


def cleanup_round_checkpoints(cfg, round_num: int):
    """Disk hygiene: once a round's shards are merged to HF, the heavy FSDP checkpoints
    (per-client + the aggregated actor) are consumed -- only round r's aggregated/hf is
    needed for r+1. Delete those shard dirs (KEEP every training.log + the HF) so peak
    disk stays ~one round instead of growing to ~40GB×rounds (an 8-round run was 367GB and
    filled the compute node's /tmp). Gated by cfg.cleanup_checkpoints (default on)."""
    import shutil

    if not cfg.get("cleanup_checkpoints", True):
        return
    rdir = Path(cfg.output_dir) / f"round_{round_num}"
    targets = list(rdir.glob("client_*/checkpoints")) + [rdir / "aggregated" / "checkpoints"]
    freed = 0
    for t in targets:
        if t.is_dir():
            try:
                shutil.rmtree(t)
                freed += 1
            except Exception as e:
                log(f"cleanup round {round_num}: could not remove {t} ({e})")
    if freed:
        log(f"cleanup round {round_num}: removed {freed} consumed checkpoint dir(s) (kept HF + logs)")


# ----------------------------------------------------------------- driver
def run(cfg) -> dict:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_model = cfg.model_path or discover_model()
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO_ROOT}:{env_base.get('PYTHONPATH', '')}".rstrip(":")
    env_base["VERL_CFG"] = verl_cfg_dir()

    banner(f"FedAgent federated loop  |  {cfg.total_clients} clients, "
           f"{cfg.clients_per_round}/round, {cfg.total_rounds} rounds, "
           f"E={cfg.epochs_per_round}")
    log(f"base model : {base_model}")
    log(f"output dir : {out}")
    log(f"aggregator : {AGGREGATOR}")
    if not AGGREGATOR.is_file():
        raise FileNotFoundError(f"aggregator not found: {AGGREGATOR}")

    # Per-client env services start LAZILY, one round at a time (only that round's selected
    # clients) -- NOT all N upfront. An N=100 config would otherwise try to start 100 services
    # / N*pool envs before round 1 for the 2 clients it actually uses (and collide ports). The
    # shared unperturbed VAL service is the exception: started once, stays up the whole run.
    # (An LRU service pool could keep recently-used clients warm; lazy per-round is the simplest
    # correct policy since M-of-N random selection rarely repeats a client.)
    if cfg.env_kind in ("webshop", "alfworld"):
        log(f"env_kind={cfg.env_kind} -> per-client services start LAZILY each round "
            f"(<= clients_per_round={cfg.clients_per_round} alive at a time; "
            f"partition={cfg.partition_strategy or ('uniform' if cfg.env_kind == 'alfworld' else 'none')})")

    # shared unperturbed validation service + the round->val-metrics curve (off unless val_env_spec set)
    do_eval = bool(cfg.get("val_env_spec"))
    val_url = val_service_url(cfg) if do_eval else None
    val_history: List[dict] = []
    val_services: List[dict] = []
    if do_eval:
        # guard: the always-on val service port must NOT fall inside the per-client service band
        # [base, base+total_clients) or that client's service can't bind -> a confusing multi-minute
        # health hang instead of a clear error. (Generated paper configs are safe; this catches
        # hand-written/default-port configs, e.g. webshop_base_port=8080 + client 10 == val 8090.)
        if cfg.env_kind in ("webshop", "alfworld"):
            _base = int(cfg.webshop_base_port if cfg.env_kind == "webshop" else cfg.alfworld_base_port)
            _vp = int(cfg.webshop_val_port if cfg.env_kind == "webshop" else cfg.alfworld_val_port)
            if _base <= _vp < _base + int(cfg.total_clients):
                raise ValueError(
                    f"{cfg.env_kind}_val_port={_vp} collides with the per-client service band "
                    f"[{_base}, {_base + int(cfg.total_clients)}); move {cfg.env_kind}_val_port or "
                    f"{cfg.env_kind}_base_port so the val service and client services use disjoint ports.")
        log(f"eval ON: unperturbed val every test_freq={cfg.test_freq} rounds "
            f"(val_before_train={cfg.val_before_train}, temp={cfg.val_temperature}) -> {cfg.val_env_spec}")
        vs = start_val_service(cfg, env_base)
        if vs:
            val_services.append(vs)

    is_ppo = str(cfg.get("adv_estimator", "grpo")).lower() == "gae"
    if is_ppo:
        log("adv_estimator=gae -> PPO: federating the critic (value model) alongside the actor "
            "each round (round-1 critic = base model)")
    lid = int(cfg.get("local_client_id", -1))
    if lid >= 0:
        mode = "local"      # paper "Local Agent Training": one pinned client, no federation
    elif cfg.total_clients <= 1:
        mode = "centralized"  # one model on the pooled data (FedAvg of 1 client == identity)
    else:
        mode = "federated"
    if mode != "federated":
        log(f"baseline mode = {mode.upper()}"
            + (f" (pinned client {lid} of {cfg.total_clients})" if mode == "local" else ""))
    try:
        history = []
        current_model = base_model
        # PPO round-1 value model starts from the base (random value head on the backbone),
        # mirroring the original's critic.model.path=<base>; thereafter the aggregated critic.
        current_critic = base_model if is_ppo else None

        # round-0 point: the base model on the unperturbed val set (paper val_before_train)
        if do_eval and cfg.val_before_train:
            m = eval_global(cfg, base_model, 0, env_base, val_url)
            if m:
                val_history.append({"round": 0, "model": "base", **m})

        for r in range(1, cfg.total_rounds + 1):
            selected = select_clients(r, cfg.total_clients, cfg.clients_per_round, cfg.base_seed,
                                      local_client_id=lid)
            banner(f"ROUND {r}/{cfg.total_rounds}  |  clients={selected}  |  "
                   f"model={'BASE' if r == 1 else 'round %d aggregated' % (r - 1)}")
            log(f"round {r} starting model: {current_model}"
                + (f"  |  critic: {current_critic}" if is_ppo else ""))

            # lazy per-round services: start ONLY this round's selected clients' env services,
            # train, then tear them down (services aren't needed for aggregation/merge/eval).
            round_services: List[dict] = []
            client_actors, client_critics = [], []
            try:
                if cfg.env_kind == "webshop":
                    round_services = start_webshop_services(cfg, env_base, client_ids=selected)
                elif cfg.env_kind == "alfworld":
                    round_services = start_alfworld_services(cfg, env_base, client_ids=selected)
                for c in selected:
                    actor, critic = run_client(cfg, r, c, current_model, env_base,
                                               critic_model_path=current_critic)
                    client_actors.append(actor)
                    if critic is not None:
                        client_critics.append(critic)
                    if c != selected[-1] and cfg.wait_between_clients > 0:
                        time.sleep(cfg.wait_between_clients)
            finally:
                stop_services(round_services)   # free this round's services before aggregation

            agg_actor = fedavg(cfg, r, client_actors, env_base, kind="actor")
            hf_dir = merge_to_hf(cfg, r, agg_actor, env_base, kind="actor")

            critic_hf = None
            if is_ppo:
                if len(client_critics) != len(selected):
                    raise RuntimeError(
                        f"round {r}: adv_estimator=gae but only {len(client_critics)}/"
                        f"{len(selected)} clients produced a critic checkpoint; cannot FedAvg "
                        f"the value model (check critic.checkpoint.save_contents includes 'model')")
                agg_critic = fedavg(cfg, r, client_critics, env_base, kind="critic")
                critic_hf = merge_to_hf(cfg, r, agg_critic, env_base, kind="critic")

            cleanup_round_checkpoints(cfg, r)  # free consumed FSDP shards (keep HF + logs)

            history.append({
                "round": r, "clients": selected,
                "started_from": current_model,
                "client_actors": [str(a) for a in client_actors],
                "aggregated_actor": str(agg_actor),
                "aggregated_hf": str(hf_dir),
                **({"started_critic_from": current_critic,
                    "client_critics": [str(a) for a in client_critics],
                    "aggregated_critic_hf": str(critic_hf)} if is_ppo else {}),
            })
            current_model = str(hf_dir)   # <-- the loop closes here: next round trains from this
            if critic_hf is not None:
                current_critic = str(critic_hf)   # ...and the federated value model carries forward

            # score the aggregated GLOBAL model on the unperturbed val set every test_freq
            # rounds (+ the final round), giving the round->success curve the paper reports.
            if do_eval and (r % int(cfg.test_freq) == 0 or r == cfg.total_rounds):
                m = eval_global(cfg, current_model, r, env_base, val_url)
                if m:
                    val_history.append({"round": r, "model": "aggregated", **m})
    finally:
        stop_services(val_services)   # round services are torn down per-round; only val remains

    summary = {
        "total_clients": cfg.total_clients,
        "clients_per_round": cfg.clients_per_round,
        "total_rounds": cfg.total_rounds,
        "epochs_per_round": cfg.epochs_per_round,
        "env_kind": cfg.env_kind,
        "mode": mode,
        "adv_estimator": "gae" if is_ppo else "grpo",
        **({"local_client_id": lid} if mode == "local" else {}),
        "partition_strategy": cfg.partition_strategy or "none",
        "base_model": base_model,
        "final_model": current_model,
        **({"final_critic": current_critic} if is_ppo else {}),
        **({"val_curve": val_history} if do_eval else {}),
        "rounds": history,
    }
    (out / "federated_summary.json").write_text(json.dumps(summary, indent=2))

    banner("FEDERATED LOOP CLOSED ✅")
    for h in history:
        from_lbl = "BASE" if h["started_from"] == base_model else "prev-aggregated"
        log(f"round {h['round']}: clients {h['clients']} trained from {from_lbl} "
            f"-> aggregated -> {h['aggregated_hf']}")
    if val_history:
        log("unperturbed val success curve: "
            + ", ".join(f"r{v['round']}={v['success_rate']}" for v in val_history))
    log(f"final aggregated model: {current_model}")
    log(f"summary: {out / 'federated_summary.json'}")
    return summary


def load_cfg(args) -> "OmegaConf":
    cfg = OmegaConf.create(dict(DEFAULTS))
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.rounds is not None:
        cfg.total_rounds = args.rounds
    if args.clients is not None:
        cfg.total_clients = args.clients
        cfg.clients_per_round = min(cfg.clients_per_round, args.clients)
    if getattr(args, "n_gpus", None) is not None:
        cfg.n_gpus_per_node = args.n_gpus
    if getattr(args, "base_seed", None) is not None:
        cfg.base_seed = args.base_seed
    if getattr(args, "port_base", None) is not None:
        cfg.webshop_base_port = args.port_base
    if getattr(args, "fedprox_mu", None) is not None:
        cfg.fedprox_mu = args.fedprox_mu
    if getattr(args, "local_client_id", None) is not None:
        cfg.local_client_id = args.local_client_id
    # resolve package-relative paths (so configs can use e.g. config/envs/webshop.yaml)
    for key in ("env_spec", "val_env_spec", "custom_cls_path", "agent_config_path",
                "webshop_run_service", "alfworld_run_service", "trajectories_file"):
        v = cfg.get(key)
        if v and not os.path.isabs(str(v)):
            cfg[key] = str(PKG_DIR / str(v))
    if str(cfg.get("partition_strategy", "")).strip().lower() == "hardness":
        if not cfg.get("trajectories_file"):
            raise ValueError(
                "partition_strategy=hardness requires trajectories_file "
                "(see fedagent/data/hardness/README.md)."
            )
        if not Path(str(cfg.trajectories_file)).is_file():
            raise FileNotFoundError(f"hardness trajectories_file not found: {cfg.trajectories_file}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="FedAgent verl-0.8 federated runner")
    ap.add_argument("--config", help="federated YAML config")
    ap.add_argument("--model-path", default=None, help="base HF model dir (round 1)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--clients", type=int, default=None)
    ap.add_argument("--n-gpus", type=int, default=None, help="override n_gpus_per_node (e.g. 4 for a single 4-GPU run)")
    ap.add_argument("--base-seed", type=int, default=None, help="override base_seed (for seed sweeps)")
    ap.add_argument("--port-base", type=int, default=None, help="override webshop_base_port (concurrent runs)")
    ap.add_argument("--fedprox-mu", type=float, default=None, help=">0 enables the FedProx proximal term")
    ap.add_argument("--local-client-id", type=int, default=None,
                    help="Local baseline: train only this client of --clients (no federation)")
    args = ap.parse_args()

    cfg = load_cfg(args)
    run(cfg)


if __name__ == "__main__":
    main()
