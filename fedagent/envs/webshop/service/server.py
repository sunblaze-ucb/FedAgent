"""WebShop remote env service — wraps the in-process WebAgentTextEnv behind HTTP.

Runs in the ``verl-agent-webshop`` conda env (has gym 0.24 / pyserini / Java / the
Lucene index). The verl-0.8 trainer (incompatible env) drives WebShop through the thin
``fedagent.envs.webshop.WebShopEnv`` HTTP client. We:

  - pre-warm a POOL of ``WebAgentTextEnv`` instances (``gym.make`` ~26s each) so episodes
    don't pay JVM+index startup;
  - serve episodes via borrow(``/create``) -> ``/reset(goal)`` -> ``/step(text)``* -> return(``/close``);
  - parse the model's action text SERVER-SIDE with the original ``webshop_projection``
    (loaded in isolation), then call the gym env -- mirroring verl-agent's WebshopWorker.

Launch via ``service/run_service.sh`` (fedagent/envs/webshop/service/). Phase 4: heterogeneity env_kwargs
(catalog_filter_asins / bm25_in_memory_config / ...) get read from the environment here
so the whole pool reflects one client's variant.
"""
import asyncio
import importlib.util
import os
import random
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
# fedagent/envs/webshop/service/ -> the vendored WebShop engine is a sibling dir
# (../engine). Holds web_agent_site + the shipped catalog data; self-contained, no
# verl-agent dependency.
_ENGINE = os.path.abspath(os.path.join(_HERE, "..", "engine"))
_WEBSHOP = os.path.join(_ENGINE, "webshop")  # web_agent_site + catalog data
if _WEBSHOP not in sys.path:
    sys.path.append(_WEBSHOP)

# Load the original action parser in isolation (it only imports re/typing).
_PROJ = os.path.join(_ENGINE, "projection.py")
_spec = importlib.util.spec_from_file_location("webshop_projection_mod", _PROJ)
_proj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proj)
webshop_projection = _proj.webshop_projection

POOL_SIZE = int(os.environ.get("WEBSHOP_POOL_SIZE", "4"))
NUM_GOALS = int(os.environ.get("WEBSHOP_NUM_GOALS", "6910"))  # confirmed pool size in standalone smoke
ENV_KWARGS = {"observation_mode": "text", "num_products": None}

# --- train/val goal split (paper: train = goals[VAL_SIZE:], val = goals[0:VAL_SIZE]) ---
# WEBSHOP_SPLIT=val makes this service the shared UNPERTURBED validation env: it ignores any
# client partition and draws from the held-out goals[0:VAL_SIZE] on the full catalog, so every
# arm is scored on the same fixed val set (a fair cross-arm comparison). WEBSHOP_SPLIT=train
# (default) draws from goals[VAL_SIZE:] only -- the het partitions already live in [VAL_SIZE:]
# (start_idx=500), and the uniform/centralized path is offset here so it never leaks val goals.
WEBSHOP_SPLIT = os.environ.get("WEBSHOP_SPLIT", "train").strip().lower()
VAL_SIZE = int(os.environ.get("WEBSHOP_VAL_SIZE", "500"))

# Goal-id logging (for the hardness-trajectories generator only): when FEDAGENT_LOG_GOAL_ID
# is set, /reset also returns this goal's TASK_ID so a labelling pass can record per-goal
# (task_id -> success). Off by default -> zero overhead on normal train/eval runs.
# CRITICAL: the task_id MUST use the exact formula hardness_partition keys on
# (f"{asin}_{md5(goal_options)}", or asin+instruction_text hash, else asin), computed from the
# env's REAL server.goals (which carry goal_options). A bare-asin id (the old behavior) would
# NOT match the partition's options-hash lookup -> every goal would fall to "low success".
LOG_GOAL_ID = bool(os.environ.get("FEDAGENT_LOG_GOAL_ID"))
_GOAL_TASKIDS = None  # runtime list: goal index -> task_id (filled in _lifespan when LOG_GOAL_ID)


def _goal_taskid(goal: dict) -> str:
    """task_id for one server.goals dict -- VERBATIM with hardness_partition (webshop_hardness.py)."""
    import hashlib
    asin = goal.get("asin")
    if asin is None:
        return None
    if goal.get("goal_options"):
        options_str = str(sorted(goal["goal_options"].items()))
        return f"{asin}_{abs(int(hashlib.md5(options_str.encode()).hexdigest(), 16))}"
    if "instruction_text" in goal:
        h = abs(int(hashlib.md5(goal["instruction_text"].encode()).hexdigest(), 16))
        return f"{asin}_{h}"
    return asin

# --- env-level heterogeneity (Phase 4): ONE client's Catalog-Split variant ---
# When PARTITION_STRATEGY=catalog_split, the WHOLE pool is built with this client's
# disjoint catalog (search/click restricted to CATALOG_ASINS) and every reset draws
# its goal from this client's slice (CLIENT_GOAL_IDXS). One service instance == one
# client's environment (a distinct hidden transition kernel P_i — the env arm of the
# Input-Dynamics Asymmetry). Bridged via env vars (CLIENT_ID/CLIENT_NUM/ENV_DIV/
# KEEP_RATIO/MIN_GOALS_PER_CLIENT/HOLDOUT_FILE), mirroring verl-agent's fed_env_manager.
PARTITION_STRATEGY = os.environ.get("PARTITION_STRATEGY", "").strip().lower()
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
CLIENT_NUM = int(os.environ.get("CLIENT_NUM", "1"))
CATALOG_ASINS = None
CLIENT_GOAL_IDXS = None
ENV_VARIANT_KWARGS: dict = {}  # transition-level variants (BM25/lookalike/rank) merged into gym.make
# TASK-level partitions (preference/coverage/hardness) are CONTENT-dependent: which goals a
# client gets depends on each goal's category/size/hardness. The original verl-agent partitions
# the env's ACTUAL `server.goals` (seed-42 shuffled) and maps back via goals.index(), so the
# served goal at index i carries the property the partition selected. We therefore DEFER these
# to _lifespan (after the first env exists) and compute CLIENT_GOAL_IDXS from env.server.goals.
# (catalog_split/task_disjoint use a contiguous index RANGE whose values are order-independent,
# and bm25/lookalike/rank use uniform goals -- both safe to compute at import time below.)
_DEFERRED_TASK_PARTITION = None  # set to the strategy name when its idxs are computed at runtime
# catalog_split = disjoint goal slice + disjoint catalog (ENV heterogeneity, hidden P_i).
# task_disjoint = the SAME disjoint goal slice but FULL catalog (TASK heterogeneity only,
#   observable in the goals). The two differ ONLY by the catalog filter -> a clean
#   ablation of the env effect with the task partition held fixed.
if PARTITION_STRATEGY in ("catalog_split", "task_disjoint"):
    from fedagent.hetero.webshop_catalog_split import catalog_split_for_client

    _catalog, CLIENT_GOAL_IDXS = catalog_split_for_client(
        CLIENT_ID, CLIENT_NUM,
        env_div=float(os.environ.get("ENV_DIV", "0.7")),
        keep_ratio=float(os.environ.get("KEEP_RATIO", "0.7")),
        min_goals_per_client=int(os.environ.get("MIN_GOALS_PER_CLIENT", "100")),
        holdout_file=os.environ.get("HOLDOUT_FILE") or None,
    )
    CATALOG_ASINS = _catalog if PARTITION_STRATEGY == "catalog_split" else None  # task_disjoint -> full catalog
    print(f"[webshop-service] {PARTITION_STRATEGY} client {CLIENT_ID}/{CLIENT_NUM}: "
          f"|catalog|={len(CATALOG_ASINS) if CATALOG_ASINS is not None else 'FULL'} "
          f"|goal_idxs|={len(CLIENT_GOAL_IDXS)}", flush=True)
elif PARTITION_STRATEGY in ("preference", "coverage", "hardness"):
    # TASK-level heterogeneity (Preference omega / Coverage xi / Hardness xi'), FULL catalog
    # (env unperturbed). DEFERRED to _lifespan: these select goals by CONTENT (category / size /
    # hardness), so the indices must address the env's REAL goal order. CLIENT_GOAL_IDXS is
    # computed in _compute_task_partition() from env.server.goals once the pool is warmed.
    _DEFERRED_TASK_PARTITION = PARTITION_STRATEGY
    CATALOG_ASINS = None  # task-level -> full catalog
    print(f"[webshop-service] {PARTITION_STRATEGY} client {CLIENT_ID}/{CLIENT_NUM}: "
          f"goal partition DEFERRED to runtime (computed from env.server.goals; FULL catalog)",
          flush=True)
elif PARTITION_STRATEGY in ("bm25_field_subset", "bm25_reweight", "lookalike", "rank_wrapper"):
    # ENV-level (transition-level) heterogeneity, paper Variants 2-5: perturb the
    # search/ranking dynamics (a distinct hidden P_i) via env_kwargs merged into
    # gym.make. The task/goal split stays UNIFORM (no goal_idxs) -- a clean
    # transition-kernel shift with the task distribution held fixed.
    from fedagent.hetero.webshop_env_variants import (
        bm25_variant_for_client,
        lookalike_injection_for_client,
        rank_wrapper_for_client,
    )

    # N = number of distinct env variants in the pool (paper sweeps N in {2,4,8}); clients are
    # deterministically assigned to one of N. VARIANT_N unset => each fn's paper default.
    _vn = os.environ.get("VARIANT_N", "").strip()
    _N = {"N": int(_vn)} if _vn else {}
    if PARTITION_STRATEGY == "bm25_field_subset":      # Variant 2 (Field-Subset Index)
        ENV_VARIANT_KWARGS = bm25_variant_for_client(CLIENT_ID, CLIENT_NUM, variant_pool="fields_only", **_N)
    elif PARTITION_STRATEGY == "bm25_reweight":        # Variant 3 (BM25 Reweighting)
        ENV_VARIANT_KWARGS = bm25_variant_for_client(CLIENT_ID, CLIENT_NUM, variant_pool=None, **_N)
    elif PARTITION_STRATEGY == "lookalike":            # Variant 4 (Lookalike Injection)
        ENV_VARIANT_KWARGS = lookalike_injection_for_client(CLIENT_ID, CLIENT_NUM, **_N)
    else:                                              # Variant 5 (Rank Wrapper)
        ENV_VARIANT_KWARGS = rank_wrapper_for_client(CLIENT_ID, CLIENT_NUM, **_N)
    CATALOG_ASINS = None  # variants perturb dynamics, not the catalog filter
    print(f"[webshop-service] {PARTITION_STRATEGY} client {CLIENT_ID}/{CLIENT_NUM}: "
          f"env_variant_keys={list(ENV_VARIANT_KWARGS)} (UNIFORM goals, FULL catalog)", flush=True)
elif PARTITION_STRATEGY:  # non-empty but unrecognized -> FAIL FAST (was: silent homogeneous run).
    # A typo (e.g. legacy 'bm25_reweighting' vs new 'bm25_reweight', or 'distractor_disjoint')
    # would otherwise fall through every branch -> CLIENT_GOAL_IDXS/CATALOG_ASINS stay None ->
    # /reset draws uniformly from the full pool/catalog, silently running a HOMOGENEOUS arm.
    raise ValueError(
        f"[webshop-service] unknown PARTITION_STRATEGY={PARTITION_STRATEGY!r}; expected one of "
        "catalog_split, task_disjoint, preference, coverage, hardness, bm25_field_subset, "
        "bm25_reweight, lookalike, rank_wrapper (or '' for the uniform/centralized baseline)."
    )

_pool: asyncio.Queue = None
_sessions: dict = {}


def _make_env(seed: int):
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the gym id)

    kw = dict(ENV_KWARGS, seed=seed)
    if CATALOG_ASINS is not None:
        kw["catalog_filter_asins"] = CATALOG_ASINS  # restrict search/click to this client's catalog
    if ENV_VARIANT_KWARGS:
        # DEEP-COPY per env: WebShop engine.load_products() mutates each product dict
        # in-place (e.g. pricing str -> parsed list at engine.py:579). ENV_VARIANT_KWARGS
        # is a single module global shared across all POOL_SIZE gym.make() calls, so for
        # `extra_products` (lookalike injection) the first env would convert the shared
        # dicts and the next env would re-process the already-converted `pricing` list ->
        # `'list' object has no attribute 'split'`. A per-env deepcopy isolates them.
        import copy
        kw.update(copy.deepcopy(ENV_VARIANT_KWARGS))  # transition-level variant (bm25/lookalike/rank)
    return gym.make("WebAgentTextEnv-v0", **kw)


def _avail(env) -> dict:
    try:
        return env.get_available_actions()
    except Exception:
        return {"has_search_bar": False, "clickables": []}


def _server_goals(env):
    """The env's REAL goal list (seed-42 shuffled dicts) -- the order the original partitions."""
    return env.unwrapped.server.goals


def _compute_task_partition(server_goals):
    """Compute CLIENT_GOAL_IDXS for a DEFERRED task-level strategy from the env's real goals.

    Mirrors verl-agent envs.py: partition env.server.goals (the seed-42 shuffled list) so the
    served goal at index i carries the category/size/hardness the partition selected. Only
    reached for preference/coverage/hardness (catalog_split/variants are computed at import).
    """
    global CLIENT_GOAL_IDXS
    strat = _DEFERRED_TASK_PARTITION
    min_goals = int(os.environ.get("MIN_GOALS_PER_CLIENT", "100"))
    if strat == "preference":
        from fedagent.hetero.webshop_task import preference_for_client
        CLIENT_GOAL_IDXS = preference_for_client(
            CLIENT_ID, CLIENT_NUM, omega=float(os.environ.get("OMEGA", "0.5")),
            min_goals_per_client=min_goals, env_goals=server_goals)
    elif strat == "coverage":
        from fedagent.hetero.webshop_coverage import coverage_for_client
        CLIENT_GOAL_IDXS = coverage_for_client(
            CLIENT_ID, CLIENT_NUM, size_std=float(os.environ.get("SIZE_STD", "1.0")),
            min_goals_per_client=min_goals, env_goals=server_goals)
    elif strat == "hardness":
        from fedagent.hetero.webshop_hardness import hardness_for_client
        CLIENT_GOAL_IDXS = hardness_for_client(
            CLIENT_ID, CLIENT_NUM, success_std=float(os.environ.get("SUCCESS_STD", "1.0")),
            trajectories_file=os.environ.get("TRAJECTORIES_FILE", ""),
            min_goals_per_client=min_goals, env_goals=server_goals)
    print(f"[webshop-service] {strat} client {CLIENT_ID}/{CLIENT_NUM}: runtime |goal_idxs|="
          f"{len(CLIENT_GOAL_IDXS) if CLIENT_GOAL_IDXS else 0} (from env.server.goals, FULL catalog)",
          flush=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pool, _GOAL_TASKIDS
    _pool = asyncio.Queue()
    envs = await asyncio.gather(*[asyncio.to_thread(_make_env, i) for i in range(POOL_SIZE)])
    for e in envs:
        _pool.put_nowait(e)
    # Compute runtime, goal-order-dependent state from the env's REAL goals (all pool envs share
    # the same seed-42 shuffled server.goals -- catalog filter doesn't perturb it, GPU-verified).
    server_goals = _server_goals(envs[0])
    if _DEFERRED_TASK_PARTITION is not None:
        await asyncio.to_thread(_compute_task_partition, server_goals)
    if LOG_GOAL_ID:
        _GOAL_TASKIDS = [_goal_taskid(g) for g in server_goals]
        print(f"[webshop-service] LOG_GOAL_ID: built {len(_GOAL_TASKIDS)} task_ids from env.server.goals",
              flush=True)
    print(f"[webshop-service] warmed {POOL_SIZE} envs (NUM_GOALS={NUM_GOALS})", flush=True)
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
        "split": WEBSHOP_SPLIT,
        "client_id": CLIENT_ID,
        "partition": PARTITION_STRATEGY or "none",
        "catalog_size": len(CATALOG_ASINS) if CATALOG_ASINS is not None else None,
        "goal_slice": len(CLIENT_GOAL_IDXS) if CLIENT_GOAL_IDXS else None,
        "env_variant_keys": list(ENV_VARIANT_KWARGS) or None,
    }


@app.post("/create")
async def create(r: Sid):
    if r.session_id in _sessions:
        return {"ok": True}    # idempotent: a retried /create (lost response) must NOT borrow a
                               # 2nd env -- that would orphan the 1st and slowly drain the pool.
    env = await _pool.get()  # borrow (waits if the pool is exhausted)
    _sessions[r.session_id] = env
    return {"ok": True}


@app.post("/reset")
async def reset(r: ResetReq):
    env = _sessions.get(r.session_id)
    if env is None:
        raise HTTPException(404, "unknown session")

    def _do():
        if WEBSHOP_SPLIT == "val":
            sess = int(r.seed) % VAL_SIZE                       # held-out val goals[0:VAL_SIZE] (deterministic coverage; eval sets no base seed)
        elif CLIENT_GOAL_IDXS:
            # Pick a goal from THIS client's shard via the full seed entropy (mirrors the original
            # RandomState(federated_seed).choice(goal_idxs)). NOT `seed % len`: the round-threaded
            # FEDAGENT_BASE_SEED enters as base*100000 in r.seed, and len|100000 for a ~100-goal
            # shard => `% len` annihilates the round term => the client would train on the SAME first
            # len goals every round. random.Random(seed) uses all bits, so goals vary per (round,row)
            # and cover the shard across rounds (matching the original's per-round re-draw).
            sess = CLIENT_GOAL_IDXS[random.Random(int(r.seed)).randrange(len(CLIENT_GOAL_IDXS))]
        else:
            sess = VAL_SIZE + int(r.seed) % max(1, NUM_GOALS - VAL_SIZE)  # uniform TRAIN pool, excludes val holdout
        res = env.reset(session=sess)
        obs = res[0] if isinstance(res, tuple) else res
        gid = None
        if LOG_GOAL_ID and _GOAL_TASKIDS is not None:
            gid = _GOAL_TASKIDS[sess] if 0 <= sess < len(_GOAL_TASKIDS) else None
        return obs, _avail(env), gid

    obs, avail, gid = await asyncio.to_thread(_do)
    return {"obs": obs, "available_actions": avail, "goal_id": gid}


@app.post("/step")
async def step(r: StepReq):
    env = _sessions.get(r.session_id)
    if env is None:
        raise HTTPException(404, "unknown session")

    def _do():
        acts, valids = webshop_projection([r.text])  # parse <action>..</action> server-side
        obs, reward, done, info = env.step(acts[0])
        info = info or {}
        # SPARSE training reward, exactly like verl-agent envs.py:32-40: the env's graded
        # score in [0,1] is kept as task_score (info), but the TRAINING reward is binary
        # {0,10} -- 10 iff the episode ends with a perfect match (done and score==1.0). The
        # overlay previously returned the dense score, which changes the GRPO/PPO objective
        # on every WebShop arm (partial-match trajectories get nonzero group-relative adv).
        dense = float(reward)
        won = bool(done and dense == 1.0)
        sparse = 10.0 if won else 0.0
        return obs, sparse, dense, bool(done), _avail(env), won, int(valids[0])

    obs, sparse, dense, done, avail, won, valid = await asyncio.to_thread(_do)
    return {
        "obs": obs,
        "reward": sparse,        # sparse {0,10} for training (matches original)
        "task_score": dense,     # dense graded score in [0,1] (diagnostic only)
        "done": done,
        "available_actions": avail,
        "success": won,
        "is_action_valid": valid,
    }


@app.post("/close")
async def close(r: Sid):
    env = _sessions.pop(r.session_id, None)
    if env is not None:
        _pool.put_nowait(env)  # return to the pool for the next episode
    return {"ok": True}
