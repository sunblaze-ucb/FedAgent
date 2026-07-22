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
    # INSERT AT THE FRONT, not append: the verl-agent-alfworld conda env carries an EDITABLE
    # verl-0.3.1 install whose .pth puts the ORIGINAL verl-agent repo on sys.path at startup --
    # `agent_system` is a namespace package, so submodule resolution follows path order and an
    # appended engine dir silently LOSES to the old repo (the service would run the un-vendored
    # engine; caught 2026-07-02 when the vendored manifest-cache patch never fired). Prepending
    # makes the vendored engine authoritative; any submodule absent from the vendored tree still
    # falls through to the old repo via the namespace merge.
    sys.path.insert(0, _ENGINE)

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


def _float_env(name: str, value: str, default: float) -> float:
    if value.strip().lower() in ("", "none", "null"):
        return default
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {value!r}") from e


def _partition_kwargs() -> dict:
    """Strategy-specific het kwargs for AlfredTWEnv (forwarded to partition_dataset)."""
    s = PARTITION_STRATEGY
    if s == "preference":
        return {"omega": _float_env("OMEGA", OMEGA, 0.5)}
    if s == "coverage":
        return {"size_std": _float_env("SIZE_STD", SIZE_STD, 1.0)}
    if s == "hardness":
        if not TRAJECTORIES_FILE.strip():
            raise ValueError(
                "ALFWorld hardness partition requires TRAJECTORIES_FILE "
                "(run_fed config key: trajectories_file)."
            )
        if not os.path.isfile(TRAJECTORIES_FILE):
            raise FileNotFoundError(f"ALFWorld hardness trajectories file not found: {TRAJECTORIES_FILE}")
        return {
            "success_std": _float_env("SUCCESS_STD", SUCCESS_STD, 1.0),
            "trajectories_file": TRAJECTORIES_FILE,
        }
    return {}  # uniform / env_disjoint take no extra kwargs


# The vendored engine (fedagent/envs/alfworld/engine, sys.path-imported above) dispatches on the
# same unified vocabulary as the configs: uniform/preference/coverage/hardness/env_disjoint
# (alfred_tw_env.py raises "Invalid partition strategy" for anything else), so the strategy name
# passes through untranslated.
def _engine_partition_strategy() -> str:
    return PARTITION_STRATEGY


_pool: asyncio.Queue = None
_sessions: dict = {}
_pending: dict = {}          # session_id -> creation-in-flight event (idempotent /create; see create())
_create_lock: asyncio.Lock = None   # guards _sessions/_pending bookkeeping (init in _lifespan)
_base_env = None  # the shared AlfredTWEnv (game-file index); init_env() spawns pooled envs
_num_games = 0


class _Session:
    """Per-episode server state: the borrowed env + idempotency bookkeeping for /step.

    ``/step`` mutates env state, so a retried request (lost response, or a socket reset
    mid-flight during the full-PPO storm) MUST apply the transition exactly once. The client
    carries a monotone ``step_id`` and increments it only after a success, so it only ever
    re-sends the CURRENT id -> a SINGLE-SLOT cache (last id + its response) is sufficient.
    ``lock`` serializes concurrent same-session requests so the env steps once and the retry
    returns the cached response. (This is the per-session async lock; it composes with the
    process-global _TW_LOCK that serializes the actual textworld transition.)
    """
    __slots__ = ("env", "lock", "next_step_id", "last_step_id", "last_resp")

    def __init__(self, env):
        self.env = env
        self.lock = asyncio.Lock()
        self.next_step_id = 0
        self.last_step_id = -1
        self.last_resp = None

    def reset_steps(self):
        self.next_step_id = 0
        self.last_step_id = -1
        self.last_resp = None
# textworld's PDDL grammar parser (tatsu) is a SHARED module-level singleton with mutable
# rule-stack state -> concurrent reset()/step() across the pooled envs corrupts it
# (IndexError: pop from empty list). Serialize all textworld env ops with one process-global
# lock. Env transitions are ms-fast vs seconds of LLM generation, so the pool's real
# concurrency benefit (overlapping LLM rollouts) is preserved.
_TW_LOCK = threading.Lock()

# Deterministic per-index game selection for hardness LABELLING (off by default -> normal
# train/eval keep the seeded-shuffle selection untouched). When ALFWORLD_SEED_IS_INDEX=1,
# /reset treats `seed` as a direct index into the env's registered game_files, so contiguous
# seeds 0..num_games-1 cover EVERY train game exactly once (a bijection). The default
# seed->game map (RandomState(seed).shuffle(gamefiles)[0]) is only ~pseudo-random, so
# coupon-collecting all N games would need ~N ln N episodes; index mode makes full-pool
# labelling exactly N episodes. See tools/gen_alfworld_hardness_trajectories.py.
SEED_IS_INDEX = bool(os.environ.get("ALFWORLD_SEED_IS_INDEX"))


def _tw_batch(env):
    """Reach the underlying TextworldBatchGymEnv (has .gamefiles + ._gamefiles_iterator)
    through any gym wrapper chain (AlfredDemangler / AlfredInfos)."""
    e = env
    for _ in range(16):
        if hasattr(e, "gamefiles") and hasattr(e, "_gamefiles_iterator"):
            return e
        nxt = getattr(e, "env", None) or getattr(e, "unwrapped", None)
        if nxt is None or nxt is e:
            break
        e = nxt
    raise RuntimeError("could not locate TextworldBatchGymEnv (.gamefiles) for index mode")


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
        partition_strategy=_engine_partition_strategy(),   # unified name, accepted by the vendored engine as-is
        min_games_per_client=MIN_GOALS_PER_CLIENT,
        **pkw,                      # preference(omega)/coverage(size_std)/hardness(success_std,trajectories_file)
    )
    return base


def _make_env():
    """One single-instance textworld gym env (batch_size=1) from the shared base.

    Serialized under _TW_LOCK: init_env -> textworld.gym.register_games bumps a PROCESS-GLOBAL
    `registry` dict with a non-atomic check-then-max-version (textworld/gym/utils.py), so
    POOL_SIZE concurrent _make_env threads can hit 'dict changed size during iteration' or race
    to the same env_id. Payload is homogeneous (all pool envs register the identical game list),
    so this only guards the one-time startup against a registry crash -- transitions still run
    concurrently (their _TW_LOCK is taken separately, per reset/step)."""
    with _TW_LOCK:
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
    global _pool, _base_env, _num_games, _create_lock
    _create_lock = asyncio.Lock()
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
    step_id: int = -1   # idempotency key; -1 = legacy caller (no replay cache)


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
    # Idempotent, exactly-once borrow. Under the full-PPO storm the client resends /create after a
    # mid-flight socket reset while the original is still parked in _pool.get(); a bare check-then-
    # borrow lets that retry borrow a SECOND env (the check sees the session absent because the first
    # create has not inserted yet), orphaning the first and slowly draining the pool to a permanent
    # hang. The FIRST caller owns the borrow; any overlapping caller waits on a per-session event.
    async with _create_lock:
        if r.session_id in _sessions:
            return {"ok": True}                       # already ready
        ev = _pending.get(r.session_id)
        first = ev is None
        if first:
            ev = asyncio.Event()
            _pending[r.session_id] = ev
    if not first:
        await ev.wait()                               # a concurrent create is borrowing; wait for it
        return {"ok": r.session_id in _sessions}      # ready iff that borrow succeeded
    try:
        env = await _pool.get()                       # borrow (waits if the pool is exhausted)
    except BaseException:
        async with _create_lock:
            _pending.pop(r.session_id, None)
        ev.set()                                      # wake waiters; session absent -> they see ok=False
        raise
    async with _create_lock:
        _sessions[r.session_id] = _Session(env)
        _pending.pop(r.session_id, None)
    ev.set()
    return {"ok": True}


@app.post("/reset")
async def reset(r: ResetReq):
    sess = _sessions.get(r.session_id)
    if sess is None:
        raise HTTPException(404, "unknown session")

    # Serialize under the SAME per-session lock /step uses: the client retries /reset, so a stale
    # retried /reset must not re-run env.seed()/env.reset() after the episode already advanced via
    # /step. (_TW_LOCK below serializes the textworld transition across sessions; sess.lock serializes
    # requests WITHIN this session.)
    async with sess.lock:
        env = sess.env

        def _do():
            # textworld reset() takes no game arg; seed(seed) deterministically reshuffles
            # the game iterator so each seed maps to a fixed game (per-seed game selection).
            # reset() loads the game -> parses its PDDL grammar via the shared tatsu parser,
            # so hold the global lock across the whole op (see _TW_LOCK).
            with _TW_LOCK:
                if SEED_IS_INDEX:
                    # Labelling: seed IS the game index. Point the iterator at exactly
                    # gamefiles[idx] (one-shot) so this reset loads that game. reset()
                    # calls next(iterator) batch_size(=1) times, so a 1-element iter suffices.
                    batch = _tw_batch(env)
                    gfs = list(batch.gamefiles)
                    batch._gamefiles_iterator = iter([gfs[int(r.seed) % len(gfs)]])
                else:
                    env.seed(int(r.seed))
                obs, infos = env.reset()
            info = _unbatch_info(infos)
            return obs[0], _admissible(info), info.get("extra.gamefile")

        obs, avail, gamefile = await asyncio.to_thread(_do)
        sess.reset_steps()   # new episode -> restart the /step idempotency counter
        return {"obs": obs, "admissible_commands": avail, "gamefile": gamefile}


@app.post("/step")
async def step(r: StepReq):
    sess = _sessions.get(r.session_id)
    if sess is None:
        raise HTTPException(404, "unknown session")

    # Serialize same-session requests so a retry that races the original applies the env
    # transition exactly once (see _Session). Different sessions hold different locks; the
    # actual textworld transition is additionally serialized process-wide by _TW_LOCK.
    async with sess.lock:
        if r.step_id >= 0:
            if r.step_id == sess.last_step_id:
                return sess.last_resp    # replay: env already stepped -> return cached response
            if r.step_id != sess.next_step_id:
                raise HTTPException(409, f"step_id {r.step_id} out of order (expected "
                                         f"{sess.next_step_id} or replay {sess.last_step_id})")
        env = sess.env

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
            return obs[0], reward, bool(dones[0]), _admissible(info), won, int(valids[0]), acts[0]

        obs, reward, done, avail, won, valid, action = await asyncio.to_thread(_do)
        resp = {
            "obs": obs,
            "reward": reward,
            "done": done,
            "admissible_commands": avail,
            "success": won,
            "is_action_valid": valid,
            "action": action,        # PROJECTED action (for windowed-mode history; matches legacy)
        }
        if r.step_id >= 0:           # commit single-slot replay cache (only the latest id is retried)
            sess.last_step_id = r.step_id
            sess.last_resp = resp
            sess.next_step_id = r.step_id + 1
        return resp


@app.post("/close")
async def close(r: Sid):
    async with _create_lock:
        sess = _sessions.pop(r.session_id, None)
    if sess is not None:
        # Drain under the session lock: wait for any in-flight /step or /reset to finish before
        # recycling, so an env still being driven in a worker thread is never handed to a 2nd session.
        async with sess.lock:
            _pool.put_nowait(sess.env)  # return to the pool for the next episode
    return {"ok": True}
