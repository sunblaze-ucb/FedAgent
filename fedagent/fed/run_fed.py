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
    python -m fedagent.fed.run_fed --config fedagent/config/fed_tinyguess_2cl_2rd.yaml
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
    "env_kind": "tinyguess",                # "tinyguess" (in-process env) | "webshop" (remote service)
    "webshop_run_service": str(PKG_DIR / "webshop_service" / "run_service.sh"),
    "webshop_base_port": 8080,              # client c's service -> webshop_base_port + c
    "webshop_pool_size": 8,                 # env pool per service (must be >= gen_batch)
    "partition_strategy": "",               # "" | "catalog_split" (env-level heterogeneity)
    "env_div": 0.7,                         # catalog-split heterogeneity strength
    "keep_ratio": 0.7,                      # catalog-split distractor density
    "omega": 0.5,                           # preference (task-het) Dirichlet spread
    "min_goals_per_client": 100,
    "service_health_timeout": 900,          # seconds to wait for a service /health
    "fedprox_mu": 0.0,                       # >0 => FedProx proximal term (else FedAvg)
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


def start_webshop_services(cfg, env_base: dict) -> List[dict]:
    """Launch ONE WebShop remote service per client (Design A: one service == one
    client's environment / hidden transition kernel). Each service builds its whole
    env pool with that client's Catalog-Split variant (via env-var bridge). Returns
    handles for teardown. Raises if any service fails to come up healthy."""
    import urllib.request

    services = []
    for c in range(cfg.total_clients):
        port = cfg.webshop_base_port + c
        env = dict(env_base)
        env.update({
            "WEBSHOP_PORT": str(port),
            "WEBSHOP_POOL_SIZE": str(cfg.webshop_pool_size),
            "PARTITION_STRATEGY": cfg.partition_strategy or "",
            "CLIENT_ID": str(c),
            "CLIENT_NUM": str(cfg.total_clients),
            "ENV_DIV": str(cfg.env_div),
            "KEEP_RATIO": str(cfg.keep_ratio),
            "OMEGA": str(cfg.get("omega", 0.5)),
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
    return services


def stop_webshop_services(services: List[dict]):
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
        log(f"stopped {len(services)} WebShop service(s)")


# ----------------------------------------------------------------- stages
def select_clients(round_num: int, total: int, per_round: int, base_seed: int) -> List[int]:
    """Deterministic per-round client selection (seed = base_seed + round - 1), matching
    core/fed/round_orchestrator.select_clients so the loop is reproducible on resume."""
    if per_round >= total:
        return list(range(total))
    import random
    rng = random.Random(base_seed + round_num - 1)
    return sorted(rng.sample(range(total), per_round))


def run_client(cfg, round_num: int, client_id: int, model_path: str,
               env_base: dict) -> Path:
    """Train one client for one round; return its latest actor checkpoint dir."""
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
        f"trainer.total_training_steps={cfg.total_training_steps}",
        f"trainer.save_freq={cfg.save_freq}",
        "trainer.val_before_train=false",
        f"trainer.project_name=fedagent_fed",
        f"trainer.experiment_name=round{round_num}_client{client_id}",
    ]
    cmd += [str(o) for o in (cfg.client_overrides or [])]   # env-specific Hydra overrides
    if cfg.get("fedprox_mu", 0) and cfg.fedprox_mu > 0:
        # run FedProx's patch in every Ray worker (incl. the actor-engine worker, which is
        # a separate process from the agent-loop workers); gated on the FEDPROX_MU env var.
        cmd.append("+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=fedagent.fedprox.worker_setup")
    env = dict(env_base)
    # distinct, reproducible env instances per client (AgenticDataset reads this);
    # round-invariant so a client's task distribution is stable across rounds.
    env["FEDAGENT_BASE_SEED"] = str(cfg.base_seed + client_id)
    if cfg.env_kind == "webshop":
        # talk to THIS client's WebShop service (its disjoint Catalog-Split env)
        env["WEBSHOP_SERVICE_URL"] = webshop_service_url(cfg, client_id)
    if cfg.get("fedprox_mu", 0) and cfg.fedprox_mu > 0:
        # FedProx: the worker_process_setup_hook reads this and patches optimizer_step
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
    log(f"client {client_id} round {round_num} OK -> {actor}")
    # measurability: post-process the training.log into json_logs/metrics.json
    # (FedAgent plot format) and echo the per-step reward curve.
    try:
        from fedagent.fed.metrics_logger import parse_training_log, summarize, write_metrics_json
        write_metrics_json(log_path, ckpt_root.parent / "json_logs")
        log(f"client {client_id} round {round_num} reward: {summarize(parse_training_log(log_path))}")
    except Exception as e:
        log(f"[warn] metrics parse r{round_num}c{client_id}: {e}")
    return actor


def fedavg(cfg, round_num: int, client_actors: List[Path], env_base: dict) -> Path:
    """FedAvg the clients' actor shards under a matched-ws PG; return the aggregated actor dir."""
    ws = world_size_of(client_actors[0])
    for a in client_actors[1:]:
        if world_size_of(a) != ws:
            raise RuntimeError(f"world_size mismatch across clients: {a} != ws {ws}")

    agg_actor = (Path(cfg.output_dir) / f"round_{round_num}" / "aggregated"
                 / "checkpoints" / "global_step_0" / "actor")
    cmd = [
        "torchrun", f"--nproc_per_node={ws}", str(AGGREGATOR),
        "--phase", "aggregate",
        "--client-actor-dirs", ",".join(str(a) for a in client_actors),
        "--output-actor-dir", str(agg_actor),
        "--global-step", "0",
    ]
    if cfg.weights:
        cmd += ["--weights", cfg.weights]
    log_path = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / "aggregate.log"
    rc = stream(cmd, env_base, log_path, tag=f"agg-r{round_num}")
    if rc != 0:
        raise RuntimeError(f"FedAvg round {round_num} FAILED (rc={rc}); see {log_path}")
    if not list(agg_actor.glob("model_world_size_*_rank_*.pt")):
        raise RuntimeError(f"FedAvg round {round_num}: no aggregated shards in {agg_actor}")
    if not (agg_actor / "huggingface").is_dir():
        raise RuntimeError(f"FedAvg round {round_num}: missing huggingface/ config in {agg_actor}")
    log(f"FedAvg round {round_num} OK (ws={ws}) -> {agg_actor}")
    return agg_actor


def merge_to_hf(cfg, round_num: int, agg_actor: Path, env_base: dict) -> Path:
    """Merge aggregated FSDP shards -> a complete HF model dir for the next round's model.path."""
    hf_dir = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / "hf"
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(agg_actor),
        "--target_dir", str(hf_dir),
    ]
    log_path = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / "merge.log"
    rc = stream(cmd, env_base, log_path, tag=f"merge-r{round_num}")
    if rc != 0:
        raise RuntimeError(f"model_merger round {round_num} FAILED (rc={rc}); see {log_path}")
    if not (hf_dir / "config.json").is_file():
        raise RuntimeError(f"merge round {round_num}: no config.json in {hf_dir}")
    weights = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("*.bin"))
    if not weights:
        raise RuntimeError(f"merge round {round_num}: no weight files in {hf_dir}")
    log(f"merge round {round_num} OK -> {hf_dir} ({len(weights)} weight file(s))")
    return hf_dir


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

    services = []
    if cfg.env_kind == "webshop":
        log(f"env_kind=webshop -> starting {cfg.total_clients} per-client services "
            f"(partition={cfg.partition_strategy or 'none'}, env_div={cfg.env_div})")
        services = start_webshop_services(cfg, env_base)

    try:
        history = []
        current_model = base_model
        for r in range(1, cfg.total_rounds + 1):
            selected = select_clients(r, cfg.total_clients, cfg.clients_per_round, cfg.base_seed)
            banner(f"ROUND {r}/{cfg.total_rounds}  |  clients={selected}  |  "
                   f"model={'BASE' if r == 1 else 'round %d aggregated' % (r - 1)}")
            log(f"round {r} starting model: {current_model}")

            client_actors = []
            for c in selected:
                client_actors.append(run_client(cfg, r, c, current_model, env_base))
                if c != selected[-1] and cfg.wait_between_clients > 0:
                    time.sleep(cfg.wait_between_clients)

            agg_actor = fedavg(cfg, r, client_actors, env_base)
            hf_dir = merge_to_hf(cfg, r, agg_actor, env_base)

            history.append({
                "round": r, "clients": selected,
                "started_from": current_model,
                "client_actors": [str(a) for a in client_actors],
                "aggregated_actor": str(agg_actor),
                "aggregated_hf": str(hf_dir),
            })
            current_model = str(hf_dir)   # <-- the loop closes here: next round trains from this
    finally:
        stop_webshop_services(services)

    summary = {
        "total_clients": cfg.total_clients,
        "clients_per_round": cfg.clients_per_round,
        "total_rounds": cfg.total_rounds,
        "epochs_per_round": cfg.epochs_per_round,
        "env_kind": cfg.env_kind,
        "partition_strategy": cfg.partition_strategy or "none",
        "base_model": base_model,
        "final_model": current_model,
        "rounds": history,
    }
    (out / "federated_summary.json").write_text(json.dumps(summary, indent=2))

    banner("FEDERATED LOOP CLOSED ✅")
    for h in history:
        from_lbl = "BASE" if h["started_from"] == base_model else "prev-aggregated"
        log(f"round {h['round']}: clients {h['clients']} trained from {from_lbl} "
            f"-> aggregated -> {h['aggregated_hf']}")
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
    if getattr(args, "base_seed", None) is not None:
        cfg.base_seed = args.base_seed
    if getattr(args, "port_base", None) is not None:
        cfg.webshop_base_port = args.port_base
    if getattr(args, "fedprox_mu", None) is not None:
        cfg.fedprox_mu = args.fedprox_mu
    # resolve package-relative paths (so configs can use e.g. config/envs/webshop.yaml)
    for key in ("env_spec", "custom_cls_path", "agent_config_path", "webshop_run_service"):
        v = cfg.get(key)
        if v and not os.path.isabs(str(v)):
            cfg[key] = str(PKG_DIR / str(v))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="FedAgent verl-0.8 federated runner")
    ap.add_argument("--config", help="federated YAML config")
    ap.add_argument("--model-path", default=None, help="base HF model dir (round 1)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--clients", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=None, help="override base_seed (for seed sweeps)")
    ap.add_argument("--port-base", type=int, default=None, help="override webshop_base_port (concurrent runs)")
    ap.add_argument("--fedprox-mu", type=float, default=None, help=">0 enables the FedProx proximal term")
    args = ap.parse_args()

    cfg = load_cfg(args)
    run(cfg)


if __name__ == "__main__":
    main()
