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
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf, open_dict

# fedagent/fed/run_fed.py -> PKG_DIR=fedagent/ , REPO_ROOT=repo root
PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parent
AGGREGATOR = REPO_ROOT / "tools" / "verl08_migration" / "aggregate_fedavg_fsdp.py"

# Per-PROCESS tag for the FSDP->vLLM weight-transfer IPC socket namespace. verl derives that
# socket path (/tmp/rl-colocate-zmq-<job_id>-...) from the Ray job id to keep concurrent jobs
# disjoint -- but every ISOLATED Ray cluster (one per client/eval subprocess, RAY_TMPDIR-separated)
# assigns the SAME first job id (01000000), so concurrent clients/eval on one node compute the
# SAME path on the shared /tmp and the weight sync DEADLOCKS (GPU-confirmed). We export a unique
# VERL_RAY_JOB_ID per launched verl subprocess (honored by the small verl patch in vllm_rollout.py
# + vllm_async_server.py) so each job's socket path is disjoint. Unique per process (this tag) AND
# per launch (role/client/round suffix below) -> safe for client-parallel and eval||train alike.
_RUN_TAG = uuid.uuid4().hex[:8]

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
    # rollout_mode: "windowed" (faithful per-turn = the paper, DEFAULT) | "concat" (stock full-history,
    # 1 sample/episode, opt-in). windowed auto-injects WindowedAgentLoopManager into train+eval cmds;
    # the env spec must pair it with agent_name=gym_text_windowed + config.history_length>=1 (concat
    # uses gym_text + history_length=0). If client_overrides already sets a manager_class, that wins
    # (back-compat with the explicit smoke configs).
    "rollout_mode": "windowed",
    "windowed_history_length": 2,           # FEDAGENT_HISTORY_LENGTH for windowed (paper=2); concat=0
    # --- env_kind=webshop: per-client remote env services + Catalog-Split heterogeneity ---
    "env_kind": "tinyguess",                # "tinyguess" (in-process) | "webshop" | "alfworld" (remote services)
    "webshop_run_service": str(PKG_DIR / "envs" / "webshop" / "service" / "run_service.sh"),
    "webshop_base_port": 8080,              # client c's service -> webshop_base_port + c
    "webshop_pool_size": 8,                 # env pool per service (must be >= gen_batch)
    "search_return_n": 200,                 # WEBSHOP_SEARCH_RETURN_N: BM25 top-K (paper=200; engine default 50 drops targets under env-het filtering)
    # --- env_kind=alfworld: per-client remote ALFWorld services + game-shard heterogeneity ---
    "alfworld_run_service": str(PKG_DIR / "envs" / "alfworld" / "service" / "run_service.sh"),
    "alfworld_base_port": 8200,             # client c's service -> alfworld_base_port + c*replicas + j
    "alfworld_pool_size": 4,                # textworld env pool per CLIENT, TOTAL across replicas (must be >= gen_batch)
    # Env-service REPLICA SHARDING (Tier-1 lever, docs/acceleration.md): run K identical service
    # processes per client (same game/goal shard -> same episode distribution; sessions spread
    # round-robin client-side via the comma-separated URL list). Kills the per-PROCESS env
    # serialization floor: ALFWorld's _TW_LOCK (tatsu PDDL parser is a process-global singleton)
    # serializes ALL env steps in one service -- measured 86ms/step x ~3200 steps = ~73% of a
    # 4-GPU training step. K replicas = K independent locks. WebShop has no lock but is one
    # GIL-bound process -- same sharding applies. 1 = today's single-service behavior.
    "alfworld_replicas": 1,
    "webshop_replicas": 1,
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
    # lever #2 (docs/acceleration.md): overlap round r+1's env-service pool warmup (minutes) with
    # round r's FedAvg/merge/eval by LAUNCHING r+1's services early + health-checking at adoption.
    # Pure scheduling, zero numerical impact. Default off until A/B-validated; flip on for paper runs.
    "prewarm_next_round_services": False,
    "fedprox_mu": 0.0,                       # >0 => FedProx proximal term (else FedAvg)
    "cleanup_checkpoints": True,             # delete consumed FSDP shards after each merge (disk hygiene)
    # lever #4 (docs/acceleration.md §7): train a ROUND's clients in ONE persistent process
    # (init_workers once, fit-per-client w/ in-process reset) instead of a subprocess per client.
    # GPU-validated equivalent (max|Δ|~1e-6) + ~37% faster on a smoke. IN-PROCESS envs only
    # (tinyguess): webshop/alfworld need per-client service routing to the shared rollout workers.
    "persistent": False,
    # lever #4 extended (docs/acceleration.md §7.2): cross_round=true keeps ONE persistent process
    # alive across ALL rounds (not just one round's clients), paying the cold-start exactly ONCE for
    # the whole run. Between rounds the worker idles (holding GPUs) while the SAME external
    # FedAvg/merge runs (byte-identical => equivalence preserved), then resets to the merged model.
    # Implies persistent. IN-PROCESS envs only (tinyguess) until per-client service routing lands.
    "cross_round": False,
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
    "test_freq": 5,                          # verl WITHIN-job step cadence (client-end marks); NOT the global eval gate
    "val_before_train": True,               # eval the base model before round 1 (the round-0 red-line point)
    "client_end_eval": False,               # also eval EACH client's post-training model on the unperturbed val
                                            #   service -> the paper's per-client "circle" marks; costs +C evals/round
    "val_temperature": 0.4,                 # val sampling temperature (paper val_kwargs.temperature=0.4)
    # --- eval/training GPU sharing (docs/acceleration.md §7.7). How eval uses the node's GPUs: ---
    #   inline  : (default) eval is a BLOCKING subprocess after merge, on cfg.n_gpus_per_node GPUs.
    #             Training is paused, so eval gets the whole node -> already "full GPU use" when
    #             training saturates the node (paper n_gpus=4). Cost = eval cold-start, on the path.
    #   parallel: eval runs CONCURRENTLY on a DISJOINT GPU subset (CUDA_VISIBLE_DEVICES) while the
    #             next round trains on the rest. Free async when training uses < node GPUs (e.g. 2 of
    #             4). Since eval is read-only on a checkpoint, this is bit-equivalent to serial eval.
    #   shared  : eval coexists on the SAME (cross-round-held) GPUs at a reduced gpu_memory_utilization
    #             that fits the leftover VRAM -> no OOM, keeps cross_round's speed (eval still serial).
    "eval_mode": "inline",                   # inline | parallel | shared
    "eval_gpus": 2,                          # parallel: # GPUs eval gets (train gets n_gpus_per_node; sum <= node)
    "eval_gpu_mem_util": 0.3,                # shared: eval vLLM gpu_memory_utilization (fits leftover VRAM)
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
            lf.flush()   # else the inner log buffers (0 bytes) until exit -> blind during a run or hang
        proc.wait()
    return proc.returncode


class BgProc:
    """A long-lived subprocess whose combined stdout/stderr is tee'd to console (tagged) + log by
    a daemon reader thread. Used by cross-round persistence (lever #4 extended): the orchestrator
    keeps ONE training process alive across all rounds and drives it with signal files, instead of
    blocking on a fresh `stream()` per round."""

    def __init__(self, cmd: List[str], env: dict, log_path: Path, tag: str):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"$ {' '.join(cmd)}")
        log(f"  (log: {log_path})  [long-lived cross-round process]")
        self.tag = tag
        self._lf = open(log_path, "w", buffering=1)   # line-buffered: per-round metrics can read it mid-run
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        for line in self.proc.stdout:
            sys.stdout.write(f"  [{self.tag}] {line}")
            sys.stdout.flush()
            self._lf.write(line)
            self._lf.flush()   # so write_metrics_json (mid-run, between rounds) sees complete lines
        self._lf.flush()

    def flush(self) -> None:
        try:
            self._lf.flush()
        except Exception:
            pass

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait(self, timeout: Optional[float] = None) -> int:
        self.proc.wait(timeout=timeout)
        self._t.join(timeout=5)
        try:
            self._lf.flush(); self._lf.close()
        except Exception:
            pass
        return self.proc.returncode


def _wait_signal(path: Path, proc: BgProc, what: str, poll_s: float = 2.0,
                 timeout: Optional[float] = None) -> None:
    """Block until `path` appears; raise if the long-lived process dies first or we time out."""
    waited = 0.0
    while not path.exists():
        if not proc.alive():
            raise RuntimeError(
                f"persistent cross-round process died while waiting for {what} "
                f"(rc={proc.proc.returncode}); see its log")
        time.sleep(poll_s)
        waited += poll_s
        if timeout and waited > timeout:
            raise RuntimeError(f"timed out ({timeout:.0f}s) waiting for {what}")


def stop_persistent_cross_round(xstate: Optional[dict]) -> None:
    """End the long-lived cross-round worker: touch `stop`, wait for a clean exit, else terminate.
    Idempotent; safe to call in a finally even if nothing was launched."""
    if not xstate or xstate.get("proc") is None:
        return
    proc, xdir = xstate["proc"], xstate["xdir"]
    if proc.alive():
        (xdir / "stop").write_text("stop")
        log("cross-round: sent STOP to the long-lived worker; waiting for exit")
        try:
            log(f"cross-round: worker exited rc={proc.wait(timeout=180)}")
        except Exception:
            log("[warn] cross-round worker did not exit in 180s; terminating")
            proc.proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.proc.kill()
    xstate["proc"] = None


def _wait_services_healthy(cfg, services: List[dict], tag: str) -> None:
    """Block until each service reports /health (or DIES / times out). Extracted from
    start_*_services so the lever-#2 prewarm path can LAUNCH services without waiting
    (wait=False) and defer this health-wait to ADOPTION next round -- overlapping the env-pool
    warmup (minutes) with the previous round's FedAvg/merge/eval. Raises on death/timeout; the
    caller tears down."""
    import urllib.request
    import json as _json
    for s in services:
        url = f"http://localhost:{s['port']}/health"
        up = False
        for _ in range(int(cfg.service_health_timeout / 3)):
            if s["proc"].poll() is not None:
                raise RuntimeError(f"{tag} service client {s['client_id']} DIED; see {s['log']}")
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    d = _json.loads(r.read())
                extra = ", ".join(f"{k}={d.get(k)}" for k in ("partition", "catalog_size", "num_games")
                                  if k in d)
                log(f"{tag} service client {s['client_id']} healthy on :{s['port']} ({extra})")
                up = True
                break
            except Exception:
                time.sleep(3)
        if not up:
            raise RuntimeError(f"{tag} service client {s['client_id']} health timeout; see {s['log']}")


# ----------------------------------------------------------------- webshop services
def webshop_service_url(cfg, client_id: int) -> str:
    """Client c's service URL. With webshop_replicas=K>1 this is a COMMA-SEPARATED list of K
    replica URLs (ports base + c*K + j); the env client binds each episode to one replica
    round-robin (fedagent/envs/base.py::_pick_replica). K=1 reduces to the legacy base+c."""
    r = int(cfg.get("webshop_replicas", 1) or 1)
    return ",".join(f"http://localhost:{cfg.webshop_base_port + client_id * r + j}" for j in range(r))


def start_webshop_services(cfg, env_base: dict, client_ids: Optional[List[int]] = None,
                           wait: bool = True) -> List[dict]:
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
    # Replica sharding (Tier-1): K identical services per client, SAME shard env-vars (CLIENT_ID/
    # CLIENT_NUM unchanged -> identical goal/catalog slice -> same episode distribution); the pool
    # is split ~evenly (+2 slack for round-robin imbalance across agent-loop workers).
    reps = int(cfg.get("webshop_replicas", 1) or 1)
    per_pool = (-(-int(cfg.webshop_pool_size) // reps) + 2) if reps > 1 else int(cfg.webshop_pool_size)
    try:
        for c in client_ids:
            for j in range(reps):
                port = cfg.webshop_base_port + c * reps + j
                env = dict(env_base)
                env.update({
                    "WEBSHOP_PORT": str(port),
                    "WEBSHOP_POOL_SIZE": str(per_pool),
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
                suffix = f"_r{j}" if reps > 1 else ""
                log_path = Path(cfg.output_dir) / f"webshop_service_client{c}{suffix}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                lf = open(log_path, "w")
                log(f"starting WebShop service client {c}{suffix} on :{port} "
                    f"(pool={per_pool}, partition={cfg.partition_strategy or 'none'}, "
                    f"env_div={cfg.env_div})  (log: {log_path})")
                proc = subprocess.Popen(["bash", str(cfg.webshop_run_service)], env=env,
                                        stdout=lf, stderr=subprocess.STDOUT)
                services.append({"client_id": f"{c}{suffix}", "port": port, "proc": proc,
                                 "log": log_path, "lf": lf})

        if wait:
            # health-wait extracted to _wait_services_healthy so the prewarm path can defer it
            # (wait=False) and overlap pool warmup with the previous round's FedAvg/merge/eval.
            _wait_services_healthy(cfg, services, "WebShop")
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
    """Client c's service URL; comma-separated replica list when alfworld_replicas=K>1
    (see webshop_service_url / envs/base.py::_pick_replica). K=1 reduces to legacy base+c."""
    r = int(cfg.get("alfworld_replicas", 1) or 1)
    return ",".join(f"http://localhost:{cfg.alfworld_base_port + client_id * r + j}" for j in range(r))


def client_service_url(cfg, client_id: int) -> Optional[str]:
    """Per-client env-service URL for the given env_kind, or None for in-process envs (tinyguess).
    Used by the persistent path to route each client (sharing ONE process) to its own service."""
    if cfg.env_kind == "webshop":
        return webshop_service_url(cfg, client_id)
    if cfg.env_kind == "alfworld":
        return alfworld_service_url(cfg, client_id)
    return None


def start_alfworld_services(cfg, env_base: dict, client_ids: Optional[List[int]] = None,
                            wait: bool = True) -> List[dict]:
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
    # Replica sharding (Tier-1): K identical services per client over the SAME game shard
    # (CLIENT_ID/CLIENT_NUM unchanged) -> K independent _TW_LOCKs. Pool split evenly (+2 slack).
    reps = int(cfg.get("alfworld_replicas", 1) or 1)
    per_pool = (-(-int(cfg.alfworld_pool_size) // reps) + 2) if reps > 1 else int(cfg.alfworld_pool_size)
    try:
        for c in client_ids:
            for j in range(reps):
                port = cfg.alfworld_base_port + c * reps + j
                env = dict(env_base)
                env.update({
                    "ALFWORLD_PORT": str(port),
                    "ALFWORLD_POOL_SIZE": str(per_pool),
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
                suffix = f"_r{j}" if reps > 1 else ""
                log_path = Path(cfg.output_dir) / f"alfworld_service_client{c}{suffix}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                lf = open(log_path, "w")
                log(f"starting ALFWorld service client {c}{suffix} on :{port} "
                    f"(pool={per_pool}, partition={cfg.partition_strategy or 'uniform'}, "
                    f"split={cfg.get('alfworld_train_eval', 'train')})  (log: {log_path})")
                proc = subprocess.Popen(["bash", str(cfg.alfworld_run_service)], env=env,
                                        stdout=lf, stderr=subprocess.STDOUT)
                services.append({"client_id": f"{c}{suffix}", "port": port, "proc": proc,
                                 "log": log_path, "lf": lf})

        if wait:
            # health-wait extracted to _wait_services_healthy so the prewarm path can defer it
            # (wait=False) and overlap pool warmup with the previous round's FedAvg/merge/eval.
            _wait_services_healthy(cfg, services, "ALFWorld")
    except BaseException:
        stop_services(services)   # no orphaned uvicorn/port survives a partial-startup failure
        raise
    return services


def prewarm_next_round_services(cfg, env_base: dict, round_num: int) -> List[dict]:
    """Lever #2 (docs/acceleration.md): LAUNCH round (round_num+1)'s selected-client env services
    WITHOUT health-waiting, so their multi-minute env-pool warmup overlaps round_num's
    FedAvg/merge/eval instead of blocking the next round's start. The handles are adopted (and
    health-checked) at the top of the next round. Returns [] -- falling back to the normal lazy
    per-round start -- when:
      * prewarm is disabled (default), there is no next round, or env_kind has no services;
      * a client is selected in BOTH this round and the next (its port base_port+client_id is
        still bound, so an early relaunch would collide) -- the next round restarts it fresh;
      * launch fails (graceful degradation -- never abort the run for a scheduling optimization).
    Pure scheduling: zero numerical impact."""
    if not cfg.get("prewarm_next_round_services", False):
        return []
    if round_num >= cfg.total_rounds:
        return []
    if cfg.env_kind not in ("webshop", "alfworld"):
        return []
    lid = int(cfg.get("local_client_id", -1))
    cur = select_clients(round_num, cfg.total_clients, cfg.clients_per_round, cfg.base_seed,
                         local_client_id=lid)
    nxt = select_clients(round_num + 1, cfg.total_clients, cfg.clients_per_round, cfg.base_seed,
                         local_client_id=lid)
    overlap = set(cur) & set(nxt)
    if overlap:
        log(f"prewarm: client(s) {sorted(overlap)} repeat across rounds {round_num}->{round_num + 1}; "
            f"skipping prewarm (port reuse) -- round {round_num + 1} starts those services fresh")
        return []
    try:
        log(f"prewarm: launching round {round_num + 1} env services for clients {nxt} "
            f"(warmup overlaps round {round_num}'s FedAvg/merge/eval; health-checked at adoption)")
        if cfg.env_kind == "webshop":
            return start_webshop_services(cfg, env_base, client_ids=nxt, wait=False)
        return start_alfworld_services(cfg, env_base, client_ids=nxt, wait=False)
    except Exception as e:
        log(f"[warn] prewarm round {round_num + 1} failed ({e}); round {round_num + 1} starts fresh")
        return []


# ----------------------------------------------------------------- unperturbed eval
def val_service_url(cfg) -> str:
    """Shared val-service URL; comma-separated replica list when the env's *_replicas=K>1
    (val ports val_port..val_port+K-1). The env client picks a replica per episode."""
    base = cfg.webshop_val_port if cfg.env_kind == "webshop" else cfg.alfworld_val_port
    reps = int(cfg.get(f"{cfg.env_kind}_replicas", 1) or 1)
    return ",".join(f"http://localhost:{base + j}" for j in range(reps))


def start_val_service(cfg, env_base: dict) -> List[dict]:
    """Start the shared UNPERTURBED validation service (full env, held-out val goal/game split),
    used to score the aggregated GLOBAL model every test_freq rounds so all arms are measured on
    the same fixed set. Returns [] when eval is off (val_env_spec unset) or the env is in-process
    (tinyguess). With *_replicas=K>1 starts K identical val services (same split -> same val
    distribution; the client spreads the n_envs sessions across them). Mirrors the per-client
    starters' health-wait."""
    if not cfg.get("val_env_spec"):
        return []
    reps = int(cfg.get(f"{cfg.env_kind}_replicas", 1) or 1) if cfg.env_kind in ("webshop", "alfworld") else 1
    if cfg.env_kind == "webshop":
        base_port = cfg.webshop_val_port
        run_service = cfg.webshop_run_service
        pool = int(cfg.webshop_pool_size)
        def mk_env(port):
            env = dict(env_base)
            env.update({
                "WEBSHOP_PORT": str(port),
                "WEBSHOP_POOL_SIZE": str((-(-pool // reps) + 2) if reps > 1 else pool),
                "WEBSHOP_SEARCH_RETURN_N": str(cfg.get("search_return_n", 200)),
                "WEBSHOP_SPLIT": "val",          # held-out goals[0:VAL_SIZE]
                "PARTITION_STRATEGY": "",        # UNPERTURBED (no catalog/goal/variant skew)
                "CLIENT_ID": "0", "CLIENT_NUM": "1",
            })
            return env
        tag = "WebShop"
    elif cfg.env_kind == "alfworld":
        base_port = cfg.alfworld_val_port
        run_service = cfg.alfworld_run_service
        pool = int(cfg.alfworld_pool_size)
        def mk_env(port):
            env = dict(env_base)
            env.update({
                "ALFWORLD_PORT": str(port),
                "ALFWORLD_POOL_SIZE": str((-(-pool // reps) + 2) if reps > 1 else pool),
                "ALFWORLD_TRAIN_EVAL": str(cfg.get("alfworld_val_split", "eval_in_distribution")),
                "ALFWORLD_TASK_TYPES": str(cfg.get("alfworld_task_types", "")),  # "" => all; else the eval-breakdown subset
                "PARTITION_STRATEGY": "uniform",  # UNPERTURBED (full game set, no client shard)
                "CLIENT_ID": "0", "CLIENT_NUM": "1",
            })
            return env
        tag = "ALFWorld"
    else:
        return []  # tinyguess runs in-process; no remote val service

    services: List[dict] = []
    try:  # mirror the per-client starters: a death/timeout here must not orphan the uvicorn/port
        for j in range(reps):
            port = base_port + j
            suffix = f"_r{j}" if reps > 1 else ""
            log_path = Path(cfg.output_dir) / f"{cfg.env_kind}_val_service{suffix}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_path, "w")
            log(f"starting {tag} VAL service (UNPERTURBED) on :{port}  (log: {log_path})")
            proc = subprocess.Popen(["bash", str(run_service)], env=mk_env(port),
                                    stdout=lf, stderr=subprocess.STDOUT)
            services.append({"client_id": f"val{suffix}", "port": port, "proc": proc,
                             "log": log_path, "lf": lf})
        _wait_services_healthy(cfg, services, f"{tag} VAL")
        return services
    except BaseException:
        stop_services(services)   # caller hasn't received the handles yet -> clean up here
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


WINDOWED_MANAGER_FQN = "fedagent.agent_loops.windowed_manager.WindowedAgentLoopManager"


def inject_rollout_mode(cmd: List[str], cfg) -> None:
    """Append the rollout-mode selector. rollout_mode=windowed (default) injects the
    WindowedAgentLoopManager (faithful per-turn rollout = the paper); concat injects nothing
    (stock 1-sample/episode). No-op if client_overrides already pins a manager_class (explicit
    configs win — back-compat). Applied to BOTH train and eval cmds so eval uses the same rollout."""
    if any("agent_loop_manager_class=" in str(o) for o in (cfg.client_overrides or [])):
        return
    if str(cfg.get("rollout_mode", "windowed")).lower() == "windowed":
        cmd.append(f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={WINDOWED_MANAGER_FQN}")


def history_length_env(cfg) -> dict:
    """FEDAGENT_HISTORY_LENGTH for the env, AUTHORITATIVE over the spec: windowed ->
    windowed_history_length (paper 2), concat -> 0. Lets ONE shared env spec drive both modes
    faithfully (alfworld/webshop_env read this; concat=0 => the GymTextAgentLoop owns history).
    Skipped when client_overrides pins a manager_class (explicit configs drive it via their spec)."""
    if any("agent_loop_manager_class=" in str(o) for o in (cfg.client_overrides or [])):
        return {}
    if str(cfg.get("rollout_mode", "windowed")).lower() == "windowed":
        return {"FEDAGENT_HISTORY_LENGTH": str(cfg.get("windowed_history_length", 2))}
    return {"FEDAGENT_HISTORY_LENGTH": "0"}


def _build_eval(cfg, model_path: str, round_num: int, env_base: dict, val_url: str,
                gpu_ids: Optional[str] = None, mem_util: Optional[float] = None,
                client_id: Optional[int] = None):
    """Build the (cmd, env, log_path, dump_dir) for a verl val-only pass. ``gpu_ids`` (e.g. "2,3")
    pins this eval to a DISJOINT GPU subset (parallel mode) and sets n_gpus to that count; ``mem_util``
    overrides the eval vLLM's gpu_memory_utilization (shared mode -> fit the leftover VRAM).
    ``client_id`` set => the client-end eval: dump to round_<r>/client_<c>/eval (the per-client circle),
    not the aggregated round_<r>/eval."""
    rdir = Path(cfg.output_dir) / (f"round_{round_num}" if round_num > 0 else "round_0")
    eval_dir = (rdir / f"client_{client_id}" / "eval") if client_id is not None else (rdir / "eval")
    dump_dir = eval_dir / "val_samples"
    exp = f"round{round_num}" + (f"_client{client_id}" if client_id is not None else "") + "_eval"
    n_gpus = len(gpu_ids.split(",")) if gpu_ids else cfg.n_gpus_per_node
    cmd = [
        sys.executable, "-m", "fedagent.main_ppo_fed",
        f"data.train_files={cfg.val_env_spec}",
        f"data.val_files={cfg.val_env_spec}",
        f"data.custom_cls.path={cfg.custom_cls_path}",
        f"actor_rollout_ref.model.path={model_path}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={cfg.agent_config_path}",
        f"trainer.default_local_dir={eval_dir / 'ckpt'}",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.val_only=true",
        "trainer.val_before_train=true",
        "algorithm.adv_estimator=grpo",                  # eval = generate+score only; no critic regardless of train algo
        f"trainer.validation_data_dir={dump_dir}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={cfg.val_temperature}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "trainer.project_name=fedagent_fed_eval",
        f"trainer.experiment_name={exp}",
    ]
    cmd += [str(o) for o in (cfg.client_overrides or [])]   # reuse rollout shape (prompt/response/n/mem)
    if mem_util is not None:                                 # shared mode: shrink eval's KV pool to fit
        cmd.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={mem_util}")
    inject_rollout_mode(cmd, cfg)                            # windowed (default) -> WindowedAgentLoopManager
    env = dict(env_base)
    env.pop("FEDPROX_MU", None)                              # eval must never enable the proximal term
    env["VERL_RAY_JOB_ID"] = (f"{_RUN_TAG}-eval-r{round_num}"
                              + (f"-c{client_id}" if client_id is not None else ""))  # disjoint socket
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids               # pin to a disjoint subset (parallel mode)
    if cfg.env_kind == "webshop":
        env["WEBSHOP_SERVICE_URL"] = val_url
    elif cfg.env_kind == "alfworld":
        env["ALFWORLD_SERVICE_URL"] = val_url
    return cmd, env, eval_dir / "eval.log", dump_dir


def eval_global(cfg, model_path: str, round_num: int, env_base: dict, val_url: str,
                mem_util: Optional[float] = None) -> Optional[dict]:
    """Score the GLOBAL model on the shared unperturbed val service via a BLOCKING verl val-only
    pass (inline / shared modes). Returns parsed val metrics or None on failure (a failed eval never
    aborts the federated run -- it is measurement, not the loop)."""
    cmd, env, log_path, dump_dir = _build_eval(cfg, model_path, round_num, env_base, val_url,
                                               mem_util=mem_util)
    rc = stream(cmd, env, log_path, tag=f"eval-r{round_num}")
    if rc != 0:
        log(f"[warn] eval round {round_num} FAILED (rc={rc}); see {log_path} (continuing)")
        return None
    metrics = summarize_val_dump(dump_dir)
    if metrics:
        log(f"round {round_num} VAL (unperturbed): success={metrics['success_rate']} "
            f"reward={metrics['reward_mean']} (n={metrics['n']})")
    return metrics


def eval_client(cfg, round_num: int, client_id: int, actor_dir, env_base: dict, val_url: str,
                mem_util: Optional[float] = None) -> Optional[dict]:
    """Client-end eval = the paper's per-client circle mark (docs §7.4): merge client c's POST-training
    actor to round_<r>/client_<c>/hf, then score it on the SAME unperturbed val service the aggregated
    red line uses (blocking val-only pass). Routes to the val service via _build_eval (not the client's
    training service), so it scores on the benchmark not the client env -- which the within-job
    _validate could not do (the env can't tell train from val rollouts). Dumps to
    round_<r>/client_<c>/eval (survives cleanup_round_checkpoints: it is not under checkpoints/).
    Read-only -> zero equivalence risk. Returns metrics or None on failure (never aborts the run)."""
    client_hf = Path(cfg.output_dir) / f"round_{round_num}" / f"client_{client_id}" / "hf"
    try:
        merge_to_hf(cfg, round_num, Path(actor_dir), env_base, kind="actor", out_hf=client_hf)
    except Exception as e:
        log(f"[warn] client-end eval r{round_num} c{client_id}: merge failed ({e}); skipping circle")
        return None
    cmd, env, log_path, dump_dir = _build_eval(cfg, str(client_hf), round_num, env_base, val_url,
                                               mem_util=mem_util, client_id=client_id)
    rc = stream(cmd, env, log_path, tag=f"eval-r{round_num}c{client_id}")
    if rc != 0:
        log(f"[warn] client-end eval r{round_num} c{client_id} FAILED (rc={rc}); continuing")
        return None
    m = summarize_val_dump(dump_dir)
    if m:
        log(f"client-end eval r{round_num} c{client_id}: success={m['success_rate']} "
            f"reward={m['reward_mean']} (n={m['n']})")
    return m


def launch_eval_async(cfg, model_path: str, round_num: int, env_base: dict, val_url: str,
                      gpu_ids: str) -> dict:
    """parallel mode: launch eval as a NON-BLOCKING BgProc pinned to ``gpu_ids`` (disjoint from the
    training GPUs), so eval(model_r) overlaps train(round r+1). Returns a handle for collect_eval."""
    cmd, env, log_path, dump_dir = _build_eval(cfg, model_path, round_num, env_base, val_url,
                                               gpu_ids=gpu_ids)
    log(f"eval round {round_num} -> async on GPU(s) {gpu_ids} (overlaps next round's training)")
    proc = BgProc(cmd, env, log_path, tag=f"eval-r{round_num}")
    return {"round": round_num, "proc": proc, "dump_dir": dump_dir, "log": log_path}


def collect_eval(handle: Optional[dict]) -> Optional[tuple]:
    """Join an async eval BgProc and parse its metrics. Returns (round_num, metrics) or None."""
    if not handle:
        return None
    rc = handle["proc"].wait()
    if rc != 0:
        log(f"[warn] async eval round {handle['round']} FAILED (rc={rc}); see {handle['log']} (continuing)")
        return (handle["round"], None)
    metrics = summarize_val_dump(handle["dump_dir"])
    if metrics:
        log(f"round {handle['round']} VAL (unperturbed): success={metrics['success_rate']} "
            f"reward={metrics['reward_mean']} (n={metrics['n']})")
    return (handle["round"], metrics)


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
    inject_rollout_mode(cmd, cfg)                            # windowed (default) -> WindowedAgentLoopManager
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
    env["VERL_RAY_JOB_ID"] = f"{_RUN_TAG}-train-c{client_id}-r{round_num}"  # disjoint weight-xfer socket
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


PERSISTENT_MAIN = "fedagent.fed.persistent_main"


def run_round_persistent(cfg, round_num: int, selected: List[int], model_path: str,
                         env_base: dict, critic_model_path: Optional[str] = None,
                         xstate: Optional[dict] = None) -> dict:
    """Lever #4: train ALL of this round's selected clients in ONE persistent process
    (init_workers once; per-client in-process reset), instead of a subprocess per client --
    paying the ~76% cold-start ONCE per round instead of once per client. Writes a plan JSON,
    launches ``fedagent.fed.persistent_main`` with FEDAGENT_PERSISTENT=1, then returns
    ``{client_id: (actor_dir, critic_dir_or_None)}`` by scanning each client's ckpt dir (same
    layout as run_client, so FedAvg/merge is byte-identical downstream).

    ``cross_round`` (xstate given): keep the SAME process alive across rounds -- pay the cold-start
    ONCE for the whole run. The first call launches a long-lived ``BgProc`` (FEDAGENT_CROSS_ROUND=1)
    and waits for its ``done_<r>`` signal; later calls just publish ``plan_round_<r>.json`` + touch
    ``go_<r>`` and wait for ``done_<r>``. Between rounds the worker idles (holding GPUs) while the
    orchestrator runs the SAME external FedAvg/merge (byte-identical => equivalence preserved).

    Per-client env seed flows via the plan (persistent_task_runner sets FEDAGENT_BASE_SEED
    driver-side before each client's dataset build -- matching run_client's
    base_seed+round*100+client). IN-PROCESS envs only: webshop/alfworld would need per-client
    service-URL routing to the shared rollout workers (the persistent process can't give each
    in-process client a distinct WEBSHOP_SERVICE_URL via process env), tracked as a follow-up."""
    round_dir = Path(cfg.output_dir) / f"round_{round_num}"
    plan = [{
        "client": c,
        "model_path": str(model_path),
        "critic_path": str(critic_model_path) if critic_model_path else None,
        "seed": int(cfg.base_seed + round_num * 100 + c),   # == run_client (run_fed.py seed)
        "out_dir": str(round_dir / f"client_{c}" / "checkpoints"),
        "exp": f"round{round_num}_client{c}",
        # per-client env-service URL (webshop/alfworld); None for in-process tinyguess. The worker
        # rewrites FEDAGENT_SERVICE_URL_FILE with this before each client's fit() (see _route_service).
        "service_url": client_service_url(cfg, c),
    } for c in selected]
    plan_path = round_dir / "persistent_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2))

    cross = bool(cfg.get("cross_round", False))
    log_path = round_dir / "persistent_training.log"

    if cross and xstate is not None and xstate.get("proc") is not None:
        # --- subsequent round: the long-lived worker is already up; publish this round's plan ----
        # (merged model from the previous round). Touch go_<r> AFTER the plan is fully written so
        # the worker (polling for both) never reads a half-written plan.
        xdir = xstate["xdir"]
        (xdir / f"plan_round_{round_num}.json").write_text(json.dumps(plan, indent=2))
        (xdir / f"go_{round_num}").write_text("go")
        log(f"cross-round: published round {round_num} plan (model={model_path}"
            + (f", critic={critic_model_path}" if critic_model_path else "") + "); awaiting worker")
        _wait_signal(xdir / f"done_{round_num}", xstate["proc"], f"round {round_num} done")
    else:
        # --- launch (per-round path, or the FIRST round of a cross-round run) --------------------
        cmd = [
            sys.executable, "-m", PERSISTENT_MAIN,
            f"data.train_files={cfg.env_spec}",
            f"data.val_files={cfg.env_spec}",
            f"data.custom_cls.path={cfg.custom_cls_path}",
            f"actor_rollout_ref.model.path={model_path}",
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            f"actor_rollout_ref.rollout.agent.agent_loop_config_path={cfg.agent_config_path}",
            f"trainer.default_local_dir={plan[0]['out_dir']}",   # the plan overrides this per client
            f"trainer.n_gpus_per_node={cfg.n_gpus_per_node}",
            f"trainer.total_epochs={cfg.epochs_per_round}",
            f"trainer.save_freq={cfg.save_freq}",
            "trainer.val_before_train=false",
            "trainer.resume_mode=disable",
            "trainer.project_name=fedagent_fed",
            f"trainer.experiment_name=round{round_num}_persistent",
        ]
        if int(cfg.total_training_steps) > 0:
            cmd.append(f"trainer.total_training_steps={cfg.total_training_steps}")
        else:
            cmd.append("trainer.total_training_steps=null")
        cmd += [str(o) for o in (cfg.client_overrides or [])]
        inject_rollout_mode(cmd, cfg)
        if str(cfg.get("adv_estimator", "grpo")).lower() == "gae":
            cmd += ["algorithm.adv_estimator=gae"]
            if critic_model_path:
                cmd += [f"critic.model.path={critic_model_path}"]

        env = dict(env_base)
        env["FEDAGENT_PERSISTENT"] = "1"             # sitecustomize -> arm the reload patch on workers
        env["VERL_RAY_JOB_ID"] = f"{_RUN_TAG}-persist"  # disjoint weight-xfer socket (long-lived worker)
        env["FEDAGENT_PERSISTENT_PLAN"] = str(plan_path)
        if str(cfg.get("eval_mode", "inline")).lower() == "worker" and cfg.get("val_env_spec"):
            # eval_mode=worker: the worker evals each round's starting model on its HOT engine (no
            # second vLLM). It dumps round_<k>/eval/val_samples; the orchestrator reads them + evals
            # the FINAL model once after the worker stops (see run()'s worker-eval collection).
            env["FEDAGENT_WORKER_EVAL"] = str(cfg.val_env_spec)
            env["FEDAGENT_WORKER_EVAL_DIR"] = str(cfg.output_dir)
            env["FEDAGENT_WORKER_EVAL_URL"] = val_service_url(cfg)
            env["FEDAGENT_WORKER_EVAL_TEMP"] = str(cfg.val_temperature)
            # the worker evals the per-round GLOBAL model EVERY round (paper red line, server-aggregated);
            # only the round-0 BASE point is gated by val_before_train -- matching the orchestrator's
            # run_eval. test_freq is verl's WITHIN-job step cadence (client-end circles), NOT this global
            # eval, so it is NOT a cadence gate here. The FINAL round is evaled by the orchestrator after
            # the worker stops, so the worker only needs 0..T-1.
            env["FEDAGENT_WORKER_EVAL_VBT"] = "1" if cfg.val_before_train else "0"
            # client-end circles on the hot engine (after each client's fit); collected from the
            # per-client dumps below after the worker stops.
            env["FEDAGENT_WORKER_CLIENT_END_EVAL"] = "1" if cfg.get("client_end_eval", False) else "0"
        if cfg.env_kind in ("webshop", "alfworld"):
            # per-client routing: the worker rewrites this file with each client's service URL before
            # its fit(); the shared agent-loop workers read it (resolve_service_url) per episode. Drop
            # any single launch-env URL so the file is authoritative within the one shared process.
            env["FEDAGENT_SERVICE_URL_FILE"] = str(Path(cfg.output_dir) / "current_service_url")
            env.pop("WEBSHOP_SERVICE_URL", None)
            env.pop("ALFWORLD_SERVICE_URL", None)
        if cfg.get("fedprox_mu", 0) and cfg.fedprox_mu > 0:
            env["FEDPROX_MU"] = str(cfg.fedprox_mu)  # per-client anchor reset handled by reload_client_model

        if cross:
            # ONE long-lived process for the WHOLE run, driven by signal files in <out>/_xround.
            xdir = Path(cfg.output_dir) / "_xround"
            xdir.mkdir(parents=True, exist_ok=True)
            for stale in [*xdir.glob("go_*"), *xdir.glob("done_*"), xdir / "stop"]:
                stale.unlink(missing_ok=True)   # never inherit a prior run's signals
            env["FEDAGENT_CROSS_ROUND"] = "1"
            env["FEDAGENT_XROUND_DIR"] = str(xdir)
            env["FEDAGENT_XROUND_START_ROUND"] = str(round_num)
            proc = BgProc(cmd, env, log_path, tag="xround-persist")
            xstate["proc"], xstate["xdir"], xstate["log"] = proc, xdir, log_path
            log(f"cross-round: launched long-lived worker (start round {round_num}); awaiting it")
            _wait_signal(xdir / f"done_{round_num}", proc, f"round {round_num} done")
        else:
            rc = stream(cmd, env, log_path, tag=f"r{round_num}-persist")
            if rc != 0:
                raise RuntimeError(f"persistent round {round_num} FAILED (rc={rc}); see {log_path}")

    results = {}
    for c in selected:
        ckpt_root = round_dir / f"client_{c}" / "checkpoints"
        actor = latest_actor_dir(ckpt_root)
        if actor is None:
            raise RuntimeError(
                f"persistent round {round_num} client {c}: no checkpoint under {ckpt_root}; "
                f"see {log_path}")
        results[c] = (actor, critic_dir_for(actor))
        log(f"persistent round {round_num} client {c} OK -> {actor}"
            + (f" (+critic {results[c][1]})" if results[c][1] else ""))
    # per-client metrics from the shared persistent log (best-effort). Cross-round: ALL rounds' steps
    # stream to the ONE launch log (round_r>1 has no own log), and BgProc is still running -> flush it
    # first so the just-finished round's steps are on disk before we parse (else metrics.json == []).
    # Note: for cross-round the parse is cumulative (rounds 1..r) since they share the launch log; the
    # authoritative per-step data lives in that persistent_training.log.
    try:
        from fedagent.fed.metrics_logger import parse_training_log, summarize, write_metrics_json
        metrics_src = log_path
        if cross and xstate is not None and xstate.get("proc") is not None:
            xstate["proc"].flush()
            metrics_src = xstate.get("log", log_path)
        write_metrics_json(metrics_src, round_dir / "json_logs")
        log(f"persistent round {round_num} reward: {summarize(parse_training_log(metrics_src))}")
    except Exception as e:
        log(f"[warn] metrics parse r{round_num}-persist: {e}")
    return results


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
        "torchrun", "--standalone", f"--nproc_per_node={ws}", str(AGGREGATOR),
        "--phase", "aggregate",
        "--client-actor-dirs", ",".join(str(a) for a in client_dirs),
        "--output-actor-dir", str(agg),
        "--global-step", "0",
    ]
    if cfg.weights:
        cmd += ["--weights", cfg.weights]
    log_path = Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / f"aggregate_{kind}.log"
    # --standalone gives this torchrun its OWN rendezvous (a free port, not the default
    # localhost:29500), so concurrent FedAvg aggregations on one node -- parallel clients,
    # or two run_fed experiments sharing a node -- don't collide on 29500 (one would die
    # rc=1 mid-aggregate). Clear any inherited torch-distributed env so it cannot override
    # --standalone's auto-assigned port. Affects only the aggregator's comm port: the FedAvg
    # math, rollout, and eval are untouched. (PPO critic FedAvg routes through here too.)
    agg_env = dict(env_base)
    for _k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        agg_env.pop(_k, None)
    rc = stream(cmd, agg_env, log_path, tag=f"agg-{kind}-r{round_num}")
    if rc != 0:
        raise RuntimeError(f"FedAvg {kind} round {round_num} FAILED (rc={rc}); see {log_path}")
    if not list(agg.glob("model_world_size_*_rank_*.pt")):
        raise RuntimeError(f"FedAvg {kind} round {round_num}: no aggregated shards in {agg}")
    if not (agg / "huggingface").is_dir():
        raise RuntimeError(f"FedAvg {kind} round {round_num}: missing huggingface/ config in {agg}")
    log(f"FedAvg {kind} round {round_num} OK (ws={ws}) -> {agg}")
    return agg


def merge_to_hf(cfg, round_num: int, agg_dir: Path, env_base: dict,
                kind: str = "actor", out_hf: Optional[Path] = None) -> Path:
    """Merge aggregated FSDP shards -> a complete HF model dir for the next round's model.path
    (actor) or critic.model.path (critic). The merger auto-detects the architecture from the
    shard's huggingface/config.json (both serialize as ...ForCausalLM; the value model just
    carries an extra scalar value head), so no per-kind flag is needed. ``out_hf`` overrides the
    default aggregated/ path -- used by the client-end eval to merge a single client's actor to
    round_<r>/client_<c>/hf (which survives cleanup_round_checkpoints)."""
    sub = "hf" if kind == "actor" else f"{kind}_hf"
    hf_dir = Path(out_hf) if out_hf is not None else (
        Path(cfg.output_dir) / f"round_{round_num}" / "aggregated" / sub)
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(agg_dir),
        "--target_dir", str(hf_dir),
    ]
    log_path = hf_dir.parent / f"merge_{kind}.log"
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
    env_base.update(history_length_env(cfg))   # windowed=2 / concat=0 -> faithful per-mode prompt

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
            _reps = int(cfg.get(f"{cfg.env_kind}_replicas", 1) or 1)
            _band = int(cfg.total_clients) * _reps         # per-client band: base + c*reps + j
            # val band is [val_port, val_port+reps); the two ranges must be disjoint
            if _base < _vp + _reps and _vp < _base + _band:
                raise ValueError(
                    f"{cfg.env_kind}_val_port band [{_vp}, {_vp + _reps}) overlaps the per-client "
                    f"service band [{_base}, {_base + _band}) (replicas={_reps}); move "
                    f"{cfg.env_kind}_val_port or {cfg.env_kind}_base_port apart.")
        log(f"eval ON: unperturbed val every test_freq={cfg.test_freq} rounds "
            f"(val_before_train={cfg.val_before_train}, temp={cfg.val_temperature}) -> {cfg.val_env_spec}")
        vs = start_val_service(cfg, env_base)
        if vs:
            val_services.extend(vs)   # replica-aware: start_val_service returns a LIST of handles

    is_ppo = str(cfg.get("adv_estimator", "grpo")).lower() == "gae"
    if is_ppo:
        log("adv_estimator=gae -> PPO: federating the critic (value model) alongside the actor "
            "each round (round-1 critic = base model)")

    # eval/training GPU-sharing mode (docs §7.7). Resolve the GPU partition + how cross_round+eval coexist.
    eval_mode = str(cfg.get("eval_mode", "inline")).lower()
    eval_gpu_ids = None
    if do_eval and eval_mode == "worker" and not (cfg.get("persistent", False) or cfg.get("cross_round", False)):
        raise ValueError("eval_mode=worker needs persistent=true or cross_round=true (the worker that "
                         "runs eval on its hot engine); use eval_mode=inline for the subprocess path.")
    if do_eval and eval_mode == "worker":
        log("eval_mode=worker: the persistent worker evals each round's starting model on its HOT vLLM "
            "(verl _validate, no second engine -> no cold-start, no OOM); orchestrator finals model_T.")
    if do_eval and eval_mode == "parallel":
        n_train, n_eval = int(cfg.n_gpus_per_node), int(cfg.get("eval_gpus", cfg.n_gpus_per_node))
        eval_gpu_ids = ",".join(str(g) for g in range(n_train, n_train + n_eval))
        env_base["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in range(n_train))  # pin TRAINING
        log(f"eval_mode=parallel: train on GPU(s) [0,{n_train}), eval CONCURRENT on GPU(s) {eval_gpu_ids} "
            f"(needs a {n_train + n_eval}-GPU node). eval is read-only -> bit-equivalent to serial eval, "
            f"but off the critical path.")
    elif do_eval and eval_mode == "shared":
        # pin BOTH training and eval to the same GPUs so eval genuinely COEXISTS with the worker (not
        # silently lands on free cards); eval's reduced KV pool (eval_gpu_mem_util) fits the VRAM the
        # worker leaves. This is the no-spare-GPU (saturated) cross_round answer.
        env_base["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in range(int(cfg.n_gpus_per_node)))
        log(f"eval_mode=shared: eval coexists on the SAME GPU(s) [0,{cfg.n_gpus_per_node}) as training "
            f"at gpu_memory_utilization={cfg.eval_gpu_mem_util} (fits the VRAM the cross_round worker "
            f"leaves free; eval stays serial but pays no extra GPUs).")

    # cross_round + per-round eval on the SAME GPUs is incompatible: the idle worker holds its vLLM KV
    # cache (~gpu_memory_utilization) between rounds, so an inline eval's own full-util vLLM can't
    # allocate -> "Free memory < desired utilization" (GPU-confirmed). parallel (disjoint GPUs) and
    # shared (low-util coexist) AVOID this; only INLINE needs the fallback to per-round persistence
    # (worker exits between rounds -> eval gets free GPUs), keeping the eval curve at per-round speed.
    if cfg.get("cross_round", False) and do_eval and eval_mode == "inline":
        log("[warn] cross_round + inline eval: the long-lived worker holds GPU memory between rounds, so "
            "eval's vLLM would OOM. Falling back to per-round persistence (still skips per-client "
            "cold-start within a round). Use eval_mode=parallel (spare GPUs) or =shared (low-util) to "
            "keep cross_round speed; or run without per-round eval.")
        with open_dict(cfg):
            cfg.cross_round = False
            cfg.persistent = True
    use_persistent = cfg.get("persistent", False) or cfg.get("cross_round", False)
    if use_persistent:
        if cfg.env_kind in ("webshop", "alfworld"):
            # per-client routing (FEDAGENT_SERVICE_URL_FILE, see run_round_persistent +
            # _route_service + resolve_service_url) lets the ONE shared process send each client's
            # rollouts to its own service. Mechanism unit-tested + equivalent on tinyguess; a full
            # multi-service webshop/alfworld GPU run is the remaining end-to-end check.
            log(f"persistent + env_kind={cfg.env_kind}: routing each in-process client to its own "
                "service via FEDAGENT_SERVICE_URL_FILE (per-client URL rewritten before each fit).")
        if cfg.get("cross_round", False):
            log("cross_round=true -> lever #4 extended: ONE process spans ALL rounds (cold-start "
                "paid ONCE for the whole run); between rounds the worker idles while the SAME "
                "external FedAvg/merge runs, then resets to the merged model.")
        else:
            log("persistent=true -> lever #4: each round trains its clients in ONE process "
                "(init_workers once); per-client checkpoints feed the SAME FedAvg/merge.")
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
    xstate: dict = {}   # lever #4 cross_round: holds the long-lived worker's BgProc + control dir
    pending_eval: Optional[dict] = None   # eval_mode=parallel: the in-flight async eval handle

    def run_eval(model_path: str, rnum: int) -> None:
        """Dispatch one global-model eval by eval_mode and fold the result into val_history.
        inline/shared = blocking; parallel = launch async on the disjoint GPU subset and collect the
        PREVIOUS async eval first (so at most one eval overlaps the next round's training)."""
        nonlocal pending_eval
        if eval_mode == "worker":
            return   # the persistent worker evals on its hot engine; collected after it stops (below)
        label = "base" if rnum == 0 else "aggregated"
        if eval_mode == "parallel":
            if pending_eval is not None:
                prev = collect_eval(pending_eval); pending_eval = None
                if prev and prev[1]:
                    val_history.append({"round": prev[0],
                                        "model": "base" if prev[0] == 0 else "aggregated", **prev[1]})
            pending_eval = launch_eval_async(cfg, model_path, rnum, env_base, val_url, eval_gpu_ids)
        else:
            mu = float(cfg.eval_gpu_mem_util) if eval_mode == "shared" else None
            m = eval_global(cfg, model_path, rnum, env_base, val_url, mem_util=mu)
            if m:
                val_history.append({"round": rnum, "model": label, **m})

    try:
        history = []
        client_history: List[dict] = []    # paper per-client "circle" marks (client_end_eval)
        pending_prewarm: List[dict] = []   # lever #2: round r+1's env services, launched early in round r
        current_model = base_model
        # PPO round-1 value model starts from the base (random value head on the backbone),
        # mirroring the original's critic.model.path=<base>; thereafter the aggregated critic.
        current_critic = base_model if is_ppo else None

        # round-0 point: the base model on the unperturbed val set (paper val_before_train)
        if do_eval and cfg.val_before_train:
            run_eval(base_model, 0)

        for r in range(1, cfg.total_rounds + 1):
            selected = select_clients(r, cfg.total_clients, cfg.clients_per_round, cfg.base_seed,
                                      local_client_id=lid)
            banner(f"ROUND {r}/{cfg.total_rounds}  |  clients={selected}  |  "
                   f"model={'BASE' if r == 1 else 'round %d aggregated' % (r - 1)}")
            log(f"round {r} starting model: {current_model}"
                + (f"  |  critic: {current_critic}" if is_ppo else ""))

            # lazy per-round services: start ONLY this round's selected clients' env services,
            # train, then tear them down (services aren't needed for aggregation/merge/eval).
            # Lever #2: if the previous round prewarmed THIS round's services (launched early to
            # overlap their warmup with that round's FedAvg/merge/eval), adopt + health-check them
            # here; else start fresh. After training, prewarm the NEXT round's services.
            round_services: List[dict] = []
            client_actors, client_critics = [], []
            try:
                if pending_prewarm:
                    round_services, pending_prewarm = pending_prewarm, []
                    try:
                        _wait_services_healthy(
                            cfg, round_services,
                            "WebShop" if cfg.env_kind == "webshop" else "ALFWorld")
                        log(f"round {r}: adopted {len(round_services)} prewarmed service(s) "
                            f"(warmup overlapped the previous round's aggregation)")
                    except Exception as e:
                        log(f"[warn] round {r}: prewarmed services unhealthy ({e}); starting fresh")
                        stop_services(round_services)
                        round_services = []
                if not round_services:
                    if cfg.env_kind == "webshop":
                        round_services = start_webshop_services(cfg, env_base, client_ids=selected)
                    elif cfg.env_kind == "alfworld":
                        round_services = start_alfworld_services(cfg, env_base, client_ids=selected)
                if use_persistent:
                    # lever #4: ONE persistent process trains all of this round's clients (cross_round:
                    # the SAME process across all rounds -- xstate carries it between iterations).
                    results = run_round_persistent(cfg, r, selected, current_model, env_base,
                                                   critic_model_path=current_critic, xstate=xstate)
                    for c in selected:
                        actor, critic = results[c]
                        client_actors.append(actor)
                        if critic is not None:
                            client_critics.append(critic)
                else:
                    for c in selected:
                        actor, critic = run_client(cfg, r, c, current_model, env_base,
                                                   critic_model_path=current_critic)
                        client_actors.append(actor)
                        if critic is not None:
                            client_critics.append(critic)
                        if c != selected[-1] and cfg.wait_between_clients > 0:
                            time.sleep(cfg.wait_between_clients)
                # lever #2: launch next round's services now -> warmup overlaps FedAvg/merge/eval
                pending_prewarm = prewarm_next_round_services(cfg, env_base, r)
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

            # client-end eval (paper per-client "circle" marks, §7.4): score EACH client's post-training
            # model on the unperturbed val service. MUST run before cleanup (it reads the client shards
            # to merge client_<c>/hf, which then survives cleanup). Non-worker modes only -- worker mode
            # evals clients on its hot engine in-process (persistent_task_runner), since the GPU-holding
            # worker would OOM a separate eval vLLM. Read-only -> no equivalence risk.
            if do_eval and cfg.get("client_end_eval", False) and eval_mode != "worker":
                cmu = cfg.eval_gpu_mem_util if eval_mode == "shared" else None
                for c, actor in zip(selected, client_actors):
                    cm = eval_client(cfg, r, int(c), actor, env_base, val_url, mem_util=cmu)
                    if cm:
                        client_history.append({"round": r, "client": int(c), **cm})

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

            # score the aggregated GLOBAL model on the unperturbed val set EVERY round -- the paper's
            # per-round red line (server-aggregated), one point per round (val_before_train semantics).
            # test_freq is verl's WITHIN-job step cadence (client-end circle marks): with epochs_per_round
            # steps/round it only fires is_last_step, NOT this global eval -- so it does NOT gate here.
            # eval_mode=parallel: this LAUNCHES eval(model_r) on the eval GPUs and returns immediately,
            # so the next round's training (on the train GPUs) overlaps it.
            if do_eval:
                run_eval(current_model, r)
    finally:
        # drain a still-in-flight parallel eval (e.g. the final round's, which has no next round to
        # overlap) so its metrics land in val_history before we summarize.
        if pending_eval is not None:
            prev = collect_eval(pending_eval); pending_eval = None
            if prev and prev[1]:
                val_history.append({"round": prev[0],
                                    "model": "base" if prev[0] == 0 else "aggregated", **prev[1]})
        stop_persistent_cross_round(xstate)  # lever #4 cross_round: end the long-lived worker
        # eval_mode=worker: the worker dumped round_0..T-1 val_samples on its hot engine (no cold-start);
        # fold them in, then eval the FINAL model_T ONCE on the now-free GPUs (worker stopped above) --
        # the val service is still up (torn down below).
        if eval_mode == "worker" and do_eval:
            for k in range(0, int(cfg.total_rounds)):
                d = Path(cfg.output_dir) / (f"round_{k}" if k > 0 else "round_0") / "eval" / "val_samples"
                mk = summarize_val_dump(d)
                if mk:
                    val_history.append({"round": k, "model": "base" if k == 0 else "aggregated", **mk})
                    log(f"worker-eval round {k}: success={mk['success_rate']} reward={mk['reward_mean']}")
            mfin = eval_global(cfg, current_model, int(cfg.total_rounds), env_base, val_url)
            if mfin:
                val_history.append({"round": int(cfg.total_rounds), "model": "aggregated", **mfin})
            # client-end circles (worker mode): the hot-engine worker dumped each client's post-training
            # model eval to round_<r>/client_<c>/eval after its fit(); fold them into client_history just
            # like the orchestrator path does for the non-worker modes.
            if cfg.get("client_end_eval", False):
                for k in range(1, int(cfg.total_rounds) + 1):
                    for cd in sorted((Path(cfg.output_dir) / f"round_{k}").glob("client_*/eval/val_samples")):
                        try:
                            c = int(cd.parent.parent.name.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        cm = summarize_val_dump(cd)
                        if cm:
                            client_history.append({"round": k, "client": c, **cm})
                            log(f"client-end eval r{k} c{c}: success={cm['success_rate']} "
                                f"reward={cm['reward_mean']}")
        stop_services(pending_prewarm)   # lever #2: any un-adopted prewarmed services (loop end/error)
        stop_services(val_services)      # round services are torn down per-round; only val remains

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
        **({"client_curve": client_history} if (do_eval and client_history) else {}),
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
