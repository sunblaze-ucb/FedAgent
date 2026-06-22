"""ALFWorld remote env service — wraps a single-instance textworld env behind HTTP.

Runs in the ``verl-agent-alfworld`` conda env (has alfworld + textworld + gymnasium +
the torch/torchvision pins). The verl-0.8 trainer (incompatible env) drives ALFWorld
through the thin ``fedagent.envs.alfworld.AlfworldEnv`` HTTP client. We:

  - build the ``AlfredTWEnv`` interface ONCE (it walks ``$ALFWORLD_DATA`` collecting the
    solvable ``game.tw-pddl`` files — the slow part), then pre-warm a POOL of single
    textworld gym envs via ``AlfredTWEnv.init_env(batch_size=1)`` (each registers the
    games + ``textworld.gym.make``) so episodes don't pay registration startup;
  - serve episodes via borrow(``/create``) -> ``/reset(seed)`` -> ``/step(text)``* -> return(``/close``);
  - parse the model's action text SERVER-SIDE with the original ``alfworld_projection``
    (loaded in isolation), then call the textworld env -- mirroring verl-agent's
    AlfworldWorker + AlfWorldEnvironmentManager.

We pool the SINGLE-instance textworld env (``AlfredTWEnv.init_env(batch_size=1)``),
deliberately NOT the multiprocess/Ray ``AlfworldEnvs`` wrapper: the agent-loop is
per-row async, so one env == one episode, and Ray actors would only add overhead here.
The textworld batch env at ``batch_size=1`` returns length-1 batched results
(``obs`` is ``[str]``; ``infos`` is a dict-of-length-1-lists with ``won`` /
``admissible_commands`` / ``extra.gamefile``); we unbatch index 0 exactly like
``AlfworldWorker`` does.

Per-episode game selection: textworld's ``env.reset()`` takes no game argument — it
advances a shuffled iterator. So we call ``env.seed(seed)`` (deterministic reshuffle)
immediately before ``reset()`` so each ``seed`` value maps to a fixed game, the analog
of WebShop's per-seed goal selection.

Launch via ``service/run_service.sh`` (fedagent/envs/alfworld/service/). Reads ``$ALFWORLD_DATA`` (game data root)
and ``$ALF_CONFIG`` (config_tw.yaml path; defaults to verl-agent's bundled config).
Phase 4: federated heterogeneity (PARTITION_STRATEGY / CLIENT_ID / CLIENT_NUM / ...) is
read from the environment here and forwarded to ``AlfredTWEnv``, so one service instance
== one client's game shard (a distinct hidden transition kernel P_i).
"""
import asyncio
import importlib.util
import os
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
# fedagent/envs/alfworld/service/ -> the vendored ALFWorld engine is a sibling dir
# (../engine). It preserves the ``agent_system/environments/`` package anchor so the
# engine's absolute imports (``agent_system.environments.partition_strategy`` and
# ``agent_system.environments.env_package.alfworld.alfworld...``) still resolve. The
# vendored ``agent_system/environments/__init__.py`` is EMPTY: the original imported
# env_manager -> the old verl 0.3.x, which this overlay does not use (neutralized so no
# verl-agent dependency remains; AlfredTWEnv itself needs only textworld/alfworld).
_ENGINE = os.path.abspath(os.path.join(_HERE, "..", "engine"))
if _ENGINE not in sys.path:
    sys.path.append(_ENGINE)

_ALF_PKG = os.path.join(_ENGINE, "agent_system", "environments", "env_package", "alfworld")

# Load the original action parser in isolation (it only imports re/typing), avoiding the
# agent_system package __init__ (which would pull verl-0.3.1/torch) AND the alfworld
# package __init__ (which imports envs.py -> ray/torchvision). Mirrors webshop_service.
_PROJ = os.path.join(_ALF_PKG, "projection.py")
_spec = importlib.util.spec_from_file_location("alfworld_projection_mod", _PROJ)
_proj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proj)
alfworld_projection = _proj.alfworld_projection

POOL_SIZE = int(os.environ.get("ALFWORLD_POOL_SIZE", "4"))
# config_tw.yaml: data paths use $ALFWORLD_DATA, env type AlfredTWEnv, max 50 steps.
ALF_CONFIG = os.environ.get("ALF_CONFIG") or os.path.join(_ALF_PKG, "configs", "config_tw.yaml")
# train | eval_in_distribution | eval_out_of_distribution (game split to draw from).
TRAIN_EVAL = os.environ.get("ALFWORLD_TRAIN_EVAL", "train")
ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", "")

# --- federated heterogeneity (Phase 4): ONE client's game shard -------------------
# When CLIENT_NUM>1, the WHOLE pool is built from this client's disjoint slice of the
# game files (a distinct hidden transition kernel P_i — the env arm of the Input-Dynamics
# Asymmetry). One service instance == one client's environment. Bridged via env vars
# (PARTITION_STRATEGY/CLIENT_ID/CLIENT_NUM/MIN_GOALS_PER_CLIENT), mirroring verl-agent's
# AlfredTWEnv federated sharding (which shards only the TRAIN split; eval stays full).
PARTITION_STRATEGY = os.environ.get("PARTITION_STRATEGY", "uniform").strip().lower() or "uniform"
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_NUM = os.environ.get("CLIENT_NUM")
MIN_GOALS_PER_CLIENT = int(os.environ.get("MIN_GOALS_PER_CLIENT", "100"))
# Optional per-task-type filter for the 6-way eval breakdown: comma-separated AlfredTWEnv
# task-type IDs (1=Pick/pick_and_place, 2=Look/look_at_obj_in_light, 3=Clean, 4=Heat,
# 5=Cool, 6=Pick2/pick_two_obj). Empty/unset => all types (normal train/eval). Applied via
# the env's NATIVE task_types config filter (collect_game_files keeps only these types).
ALFWORLD_TASK_TYPES = os.environ.get("ALFWORLD_TASK_TYPES", "").strip()
# Task-het knobs forwarded to AlfredTWEnv -> partition_dataset (preference/coverage/hardness).
# Only the kwargs relevant to PARTITION_STRATEGY are passed (else the verbatim partition fns
# reject unexpected kwargs). uniform/env_disjoint take none.
OMEGA = os.environ.get("OMEGA", "")
SIZE_STD = os.environ.get("SIZE_STD", "")
SUCCESS_STD = os.environ.get("SUCCESS_STD", "")
TRAJECTORIES_FILE = os.environ.get("TRAJECTORIES_FILE", "")
_CLIENT_ID = int(CLIENT_ID) if CLIENT_ID not in (None, "") else None
_CLIENT_NUM = int(CLIENT_NUM) if CLIENT_NUM not in (None, "") else None


def _partition_kwargs() -> dict:
    """Strategy-specific het kwargs for AlfredTWEnv (forwarded to partition_dataset)."""
    s = PARTITION_STRATEGY
    if s == "preference" and OMEGA:
        return {"omega": float(OMEGA)}
    if s == "coverage" and SIZE_STD:
        return {"size_std": float(SIZE_STD)}
    if s == "hardness":
        kw = {}
        if SUCCESS_STD:
            kw["success_std"] = float(SUCCESS_STD)
        if TRAJECTORIES_FILE:
            kw["trajectories_file"] = TRAJECTORIES_FILE
        return kw
    return {}  # uniform / env_disjoint take no extra kwargs

_pool: asyncio.Queue = None
_sessions: dict = {}
_base_env = None  # the shared AlfredTWEnv (game-file index); init_env() spawns pooled envs
_num_games = 0
# textworld's PDDL grammar parser (tatsu) is a SHARED module-level singleton with mutable
# rule-stack state -> concurrent reset()/step() across the pooled envs corrupts it
# (IndexError: pop from empty list). Serialize all textworld env ops with one process-global
# lock. Env transitions are ms-fast vs seconds of LLM generation, so the pool's real
# concurrency benefit (overlapping LLM rollouts) is preserved.
_TW_LOCK = threading.Lock()


def _load_config():
    import yaml

    with open(ALF_CONFIG) as f:
        return yaml.safe_load(f)


def _build_base_env():
    """Build the AlfredTWEnv ONCE (walks $ALFWORLD_DATA collecting solvable games).

    Forwards federated partition kwargs so the game pool is this client's shard.
    """
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
        get_environment,
    )

    config = _load_config()
    if ALFWORLD_TASK_TYPES:
        # restrict to these task-type IDs (the 6-way eval breakdown); native config filter
        config["env"]["task_types"] = [int(x) for x in ALFWORLD_TASK_TYPES.split(",") if x.strip()]
        print(f"[alfworld-service] task_types filter -> {config['env']['task_types']}", flush=True)
    env_type = config["env"]["type"]  # AlfredTWEnv
    pkw = _partition_kwargs()
    if pkw:
        print(f"[alfworld-service] partition_kwargs -> {pkw}", flush=True)
    base = get_environment(env_type)(
        config,
        train_eval=TRAIN_EVAL,
        client_id=_CLIENT_ID,
        client_num=_CLIENT_NUM,
        partition_strategy=PARTITION_STRATEGY,
        min_games_per_client=MIN_GOALS_PER_CLIENT,
        **pkw,                      # preference(omega)/coverage(size_std)/hardness(success_std,trajectories_file)
    )
    return base


def _make_env():
    """One single-instance textworld gym env (batch_size=1) from the shared base."""
    return _base_env.init_env(batch_size=1)


def _unbatch_info(infos: dict) -> dict:
    # textworld batch env returns infos as a dict-of-lists (length == batch_size == 1).
    # Mirror AlfworldWorker: take index 0 of every key.
    return {k: v[0] for k, v in infos.items()}


def _admissible(info: dict):
    # exclude 'help' just like AlfWorldEnvironmentManager.build_text_obs
    cmds = info.get("admissible_commands", []) or []
    return [c for c in cmds if c != "help"]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pool, _base_env, _num_games
    _base_env = await asyncio.to_thread(_build_base_env)
    _num_games = int(getattr(_base_env, "num_games", 0))
    _pool = asyncio.Queue()
    envs = await asyncio.gather(*[asyncio.to_thread(_make_env) for _ in range(POOL_SIZE)])
    for e in envs:
        _pool.put_nowait(e)
    print(
        f"[alfworld-service] warmed {POOL_SIZE} envs (split={TRAIN_EVAL} num_games={_num_games} "
        f"partition={PARTITION_STRATEGY} client={_CLIENT_ID}/{_CLIENT_NUM})",
        flush=True,
    )
    yield
    while _pool is not None and not _pool.empty():
        try:
            _pool.get_nowait().close()
        except Exception:
            pass


app = FastAPI(lifespan=_lifespan)


class Sid(BaseModel):
    session_id: str


class ResetReq(BaseModel):
    session_id: str
    seed: int = 0


class StepReq(BaseModel):
    session_id: str
    text: str


@app.get("/health")
async def health():
    return {
        "ok": True,
        "free": _pool.qsize() if _pool else 0,
        "sessions": len(_sessions),
        "num_games": _num_games,
        "split": TRAIN_EVAL,
        "task_types": ALFWORLD_TASK_TYPES or "all",
        "partition": PARTITION_STRATEGY,
        "client_id": _CLIENT_ID,
        "client_num": _CLIENT_NUM,
        "alfworld_data": ALFWORLD_DATA or None,
    }


@app.post("/create")
async def create(r: Sid):
    env = await _pool.get()  # borrow (waits if the pool is exhausted)
    _sessions[r.session_id] = env
    return {"ok": True}


@app.post("/reset")
async def reset(r: ResetReq):
    env = _sessions.get(r.session_id)
    if env is None:
        raise HTTPException(404, "unknown session")

    def _do():
        # textworld reset() takes no game arg; seed(seed) deterministically reshuffles
        # the game iterator so each seed maps to a fixed game (per-seed game selection).
        # reset() loads the game -> parses its PDDL grammar via the shared tatsu parser,
        # so hold the global lock across the whole op (see _TW_LOCK).
        with _TW_LOCK:
            env.seed(int(r.seed))
            obs, infos = env.reset()
        info = _unbatch_info(infos)
        return obs[0], _admissible(info), info.get("extra.gamefile")

    obs, avail, gamefile = await asyncio.to_thread(_do)
    return {"obs": obs, "admissible_commands": avail, "gamefile": gamefile}


@app.post("/step")
async def step(r: StepReq):
    env = _sessions.get(r.session_id)
    if env is None:
        raise HTTPException(404, "unknown session")

    def _do():
        # alfworld_projection mutates the actions list; pass a fresh single-element list.
        # action_pools (admissible cmds) only drive the think/Chinese-char validity check.
        acts, valids = alfworld_projection([r.text], [[]])
        # step() executes PDDL actions on the loaded game (shared tatsu parser state) ->
        # serialize with the same global lock as reset() (see _TW_LOCK).
        with _TW_LOCK:
            obs, scores, dones, infos = env.step([acts[0]])
        info = _unbatch_info(infos)
        won = bool(info.get("won", False))
        # text-only reward == compute_reward(info) == 10.0 * won (see envs.compute_reward)
        reward = 10.0 * float(won)
        return obs[0], reward, bool(dones[0]), _admissible(info), won, int(valids[0])

    obs, reward, done, avail, won, valid = await asyncio.to_thread(_do)
    return {
        "obs": obs,
        "reward": reward,
        "done": done,
        "admissible_commands": avail,
        "success": won,
        "is_action_valid": valid,
    }


@app.post("/close")
async def close(r: Sid):
    env = _sessions.pop(r.session_id, None)
    if env is not None:
        _pool.put_nowait(env)  # return to the pool for the next episode
    return {"ok": True}
