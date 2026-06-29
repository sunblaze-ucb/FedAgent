# Extending FedAgent

FedAgent is a **thin overlay on stock verl 0.8**, and it is a *library first*: the
federated control loop, the environments, the heterogeneity constructors, and the
model aggregation are deliberately decoupled, so you can replace any one of them
without touching the others. This document is the reference for doing exactly that.

The maintained entry point never changes — every extension is reachable through

```bash
python -m fedagent.fed.run_fed --config <yaml>
```

The driver (`fedagent/fed/run_fed.py`) is **verl-agnostic**: a client is just a
subprocess (`python -m fedagent.main_ppo_fed`). It trains each selected client
**sequentially**, FedAvgs the resulting FSDP shards, merges them back to a HuggingFace
model, and re-enters the next round from that aggregated model. Because of this split,
each extension point is isolated to a small number of files:

| # | Extension point | Primary file(s) | Selected by |
|---|---|---|---|
| 1 | **New environment** | `fedagent/envs/<name>/` + `fedagent/envs/registry.py` + a `config/envs/<name>.yaml` | the env-spec row's `name:` |
| 2 | **New heterogeneity strategy** | `fedagent/hetero/<name>.py` + the service env-var bridge + `run_fed.py` | `partition_strategy` (YAML / env `PARTITION_STRATEGY`) |
| 3 | **New RL algorithm** | the verl trainer (`algorithm.adv_estimator`); FedAgent only carries the checkpoints | `adv_estimator` (YAML → Hydra) |
| 4 | **New aggregation rule** | `tools/verl08_migration/aggregate_fedavg_fsdp.py` (server FedAvg) · `sitecustomize.py` + `fedagent/fedprox.py` (client-side hook) | the aggregator CLI / `fedprox_mu` |

> **How the layers fit together.** The overlay owns the *orchestration* (the round
> loop, per-client env services, aggregation, eval) and the *agent rollout*
> (`fedagent/agent_loops/gym_text_agent_loop.py` drives one `BaseTextEnv` per dataset
> row). Stock verl owns the *RL update* (advantage estimation, the actor/critic FSDP
> workers, the optimizer). Extension points 1 and 2 live entirely in the overlay;
> point 3 lives in verl (the overlay only selects it and carries the extra
> checkpoint); point 4 straddles both — the server rule is an overlay tool, the
> client-side proximal term is a verl monkeypatch loaded at interpreter startup.

> **Before you start.** Read [`./architecture.md`](./architecture.md) for the round
> loop and [`./installation.md`](./installation.md) for the **three** conda envs (the
> trainer env `fedagent-verl08` plus the per-service `verl-agent-webshop` /
> `verl-agent-alfworld` envs). Environments whose Python dependencies conflict with
> verl 0.8 (WebShop's pyserini/Java, ALFWorld's TextWorld) run in their own conda env
> behind an HTTP service; in-process environments (TinyGuess) need none of this.

---

## 1. Add an environment

### Where

Every environment a FedAgent agent-loop drives implements one async contract,
`BaseTextEnv` (`fedagent/envs/base.py`), and is looked up by name from
`fedagent/envs/registry.py`. The agent-loop instantiates **one env instance per
dataset row** and `await`s its reset/step (the verl-0.8 agent-loop is per-row async,
not the old batched, synchronous `EnvironmentManager`).

### The contract

`BaseTextEnv` is the whole surface — four methods, three of them abstract. The
observation convention is a dict with at least `obs_str` (the text shown to the
model):

```python
# fedagent/envs/base.py
Obs = Dict[str, Any]

class BaseTextEnv(ABC):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        self.env_config: Dict[str, Any] = dict(env_config or {})

    @abstractmethod
    async def system_prompt(self) -> Obs:
        """The system message ({"obs_str": ...}) shown once at episode start."""

    @abstractmethod
    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        """Reset to a fresh episode, deterministically in `seed`. Returns (obs, info)."""

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        """Apply the model's decoded text action. Returns (obs, reward, done, info)."""

    async def close(self) -> None:
        """Release any resources held by this instance (override if needed)."""
        return None
```

Three invariants the existing envs honor, and you must too:

- **`info` carries `success` (bool).** The agent-loop records the episode outcome from
  `info["success"]`; it becomes FedAgent's headline metric `val/success_rate`. (See
  how `TinyGuessEnv.step` returns `{"success": self.solved, ...}` and how
  `WebShopEnv.step` maps the service's `success` field.)
- **`reset` is deterministic in `seed`.** Every dataset row is a distinct instance
  with a distinct seed; reproducibility (and the per-`(round, client)` re-draw) rides
  on `seed` alone. The federated driver threads
  `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id` into the dataset, which sets
  each row's seed.
- **`obs_str` is the only required observation key.** Image/multimodal envs may later
  add `multi_modal_data` without changing the contract.

`TinyGuessEnv` (`fedagent/envs/tiny_guess.py`, ~70 lines) is the cleanest reference —
fully in-process, dependency-free, parses the action with a regex, returns
higher/lower. Read it first.

### Steps

1. **Subclass `BaseTextEnv`.** For an in-process env, do the work directly in
   `reset`/`step` (mirror `TinyGuessEnv`). For an env whose dependencies conflict with
   verl 0.8, make the subclass a **thin HTTP client** (see §1b) and put the real env in
   a service.

2. **Register it.** Add the class to `ENV_REGISTRY` in `fedagent/envs/registry.py`:

   ```python
   # fedagent/envs/registry.py
   from fedagent.envs.myenv import MyEnv

   ENV_REGISTRY: Dict[str, Type[BaseTextEnv]] = {
       "TinyGuess": TinyGuessEnv,
       "WebShop": WebShopEnv,
       "ALFWorld": AlfworldEnv,
       "MyEnv": MyEnv,          # <- the name the agent-loop looks up
   }
   ```

   The dict key is the `env_name`; `make_env(env_name, env_config)` raises
   `KeyError` for anything not registered.

3. **Write an env-spec YAML** under `fedagent/config/envs/`. This is what
   `fedagent.data.agentic_dataset.AgenticDataset` (verl's `data.custom_cls`) reads to
   emit one row per env instance. The schema is a flat list of env blocks:

   ```yaml
   # fedagent/config/envs/myenv.yaml
   envs:
     - name: MyEnv          # MUST match the ENV_REGISTRY key
       n_envs: 8            # rows (= distinct instances/seeds) emitted for this spec
       max_turns: 15        # per-episode turn budget enforced by the agent-loop
       agent_name: gym_text # optional (default: gym_text)
       config:              # forwarded verbatim as `env_config` to the constructor
         timeout: 180.0
   ```

   The dataset emits `n_envs` rows, each with a distinct `seed`; GRPO grouping is then
   handled downstream by verl's `rollout.n` (each row repeated `n` times = one GRPO
   group per instance). Keep `n_envs == data.train_batch_size` so the batch holds one
   group per step (see the `webshop_15.yaml` header for why).

4. **Point the federated config at the spec.** In your `run_fed` YAML set
   `env_spec: config/envs/myenv.yaml` (resolved relative to the package). For an
   in-process env that is all — set `env_kind` to a value other than `webshop`/
   `alfworld` so the driver starts **no** services (`tinyguess` is the in-process
   sentinel). See the smoke recipe at the end.

### 1b. Add a per-client HTTP service (only if deps conflict)

WebShop and ALFWorld cannot import alongside verl 0.8 (pyserini/Java/gym 0.24;
TextWorld), so the real env runs in its own conda env behind a FastAPI service and the
`BaseTextEnv` subclass is a thin client. Mirror `fedagent/envs/webshop/service/` (the
fullest reference) or `fedagent/envs/alfworld/service/`.

**The client side** reads its service URL from a per-env environment variable that the
driver sets *per client*, then ferries text in and observations out:

```python
# fedagent/envs/myenv/myenv_env.py  (mirrors fedagent/envs/webshop/webshop_env.py)
class MyEnv(BaseTextEnv):
    def __init__(self, env_config=None):
        super().__init__(env_config)
        self.base_url = (
            os.environ.get("MYENV_SERVICE_URL")          # set per-client by run_fed (authoritative)
            or self.env_config.get("service_url")        # ad-hoc single-service fallback
            or "http://localhost:8080"
        ).rstrip("/")
        self.session_id = uuid4().hex
        self._client = None  # lazily-created httpx.AsyncClient

    async def reset(self, seed=0):
        c = self._c()
        await c.post("/create", json={"session_id": self.session_id})
        r = await c.post("/reset", json={"session_id": self.session_id, "seed": int(seed)})
        ...  # format the service obs into {"obs_str": ...}, return (obs, {})

    async def step(self, action_str):
        r = await self._c().post("/step", json={"session_id": self.session_id, "text": action_str})
        d = r.json()
        info = {"success": bool(d.get("success", False)), ...}
        return {"obs_str": ...}, float(d.get("reward", 0.0)), bool(d.get("done", False)), info

    async def close(self):
        ...  # POST /close, aclose the httpx client
```

**The service side** is a FastAPI app that pre-warms a **pool** of env instances (so
episodes don't pay startup cost) and serves the borrow → reset → step\* → return
lifecycle. The five endpoints the client expects are `/health`, `/create`, `/reset`,
`/step`, `/close`. Parse the model's action text **server-side** (where the env's
projection function lives). The pool size is read from an env var and **must be ≥ the
generation batch**:

| Endpoint | Role |
|---|---|
| `GET /health` | readiness probe; the driver polls it until the pool is warm (and echoes partition info) |
| `POST /create` | borrow a warm env from the pool for a session |
| `POST /reset` | reset that env to a goal/seed; return obs + admissible actions |
| `POST /step` | parse text server-side, step the env, return obs/reward/done/success |
| `POST /close` | return the env to the pool |

Add a `run_service.sh` next to `server.py` that activates the right conda env and
launches uvicorn. Then in your `run_fed.py` you need a launcher analogous to
`start_webshop_services` / `start_alfworld_services` (it sets `MYENV_PORT`,
`MYENV_POOL_SIZE`, the partition env-var bridge below, and waits on `/health`), and an
`env_kind: myenv` branch in `run()` that calls it. Set the per-client
`MYENV_SERVICE_URL` in `run_client` so client *c* talks to the service on
`base_port + c`. The shared **unperturbed** validation service (one full-env service on
a separate port, used to score the aggregated global model each `test_freq` rounds) is
wired the same way in `start_val_service` / `eval_global`.

See [`../envs/README.md`](../envs/README.md) for the full client/service contract and
[`../envs/webshop/service/README.md`](../envs/webshop/service/README.md) /
[`../envs/alfworld/service/README.md`](../envs/alfworld/service/README.md) for the two
worked services.

---

## 2. Add a heterogeneity strategy

### Where

Each construction is a self-contained, numpy-only module under `fedagent/hetero/`
exposing a public **`*_for_client(...)`** function. (The functions are copied
*verbatim* from the verl-agent-0.3.1 `partition_strategy.py` so per-client assignment
is bit-identical to the baseline — the science red line; only the thin
`*_for_client` public API is new.) The strategy is selected by `partition_strategy`,
threaded into each env service through an **environment-variable bridge**.

### The two axes

| Axis | Enters via | Returns | Example helpers |
|---|---|---|---|
| **Task-level** (preference / coverage / hardness) | the *prompt* (which goals a client draws) | a list of **goal indices** for this client | `preference_for_client`, `coverage_for_client`, `hardness_for_client` |
| **Env-level** (catalog_split / variants 2–5) | the *transition kernel* `P_i` (catalog / search dynamics) | a catalog + goal-idx pair, or an `env_kwargs` dict | `catalog_split_for_client`, `bm25_variant_for_client`, `lookalike_injection_for_client`, `rank_wrapper_for_client` |

Task-level strategies leave the env unperturbed (full catalog) and are FedAvg-robust;
env-level strategies perturb the hidden dynamics and are worst-case non-robust. See
[`./heterogeneity.md`](./heterogeneity.md) and [`../hetero/README.md`](../hetero/README.md)
for the taxonomy and the knob-naming caveats (e.g. `size_std`/`success_std` are the
Beta *concentration* ξ/ξ′, not standard deviations).

### The contract

The public functions share a fixed shape (keyword-only knobs after `client_id`,
`client_num`). A **task-level** function returns the client's goal indices:

```python
# fedagent/hetero/webshop_catalog_split.py (env-level — returns a (catalog, idxs) pair)
def catalog_split_for_client(
    client_id: int,
    client_num: int,
    *,
    env_div: float = 0.7,
    keep_ratio: float = 0.7,
    min_goals_per_client: int = 100,
    holdout_file: Optional[str] = None,
    base_seed: int = 42,
    data_dir: Optional[str] = None,
) -> Tuple[List[str], List[int]]:        # (catalog_asins, client_goal_idxs)
    ...
```

Two invariants every strategy honors:

- **Deterministic in `client_id`.** Each client process runs the function
  independently and must agree on the global allocation, so FedAvg sees the *same*
  per-client slice every round. The existing code hardcodes `base_seed=42` and seeds a
  per-client RNG as `np.random.RandomState(42 + client_id)` (or a shared
  `default_rng(42)` indexed by `client_id`). **Do not** seed from Python's builtin
  `hash()` of a string (it is salted per interpreter).
- **Guarantee the floor.** Top up to `min_goals_per_client` if your draw comes up
  short.

### Steps

1. **Write `fedagent/hetero/myhet.py`** exposing `myhet_for_client(client_id,
   client_num, *, <knobs>, min_goals_per_client=100, env_goals=...)`. Task-level
   functions that select by goal *content* (category/size/hardness) take the env's real
   goal list as `env_goals` and return absolute indices into it; env-level functions
   return a catalog/idx pair or an `env_kwargs` dict to merge into the env constructor.

2. **Wire the env-var bridge in the service `server.py`.** The driver passes knobs as
   environment variables; the service dispatches on `PARTITION_STRATEGY` and calls your
   function. The current WebShop bridge (`fedagent/envs/webshop/service/server.py`)
   reads:

   | Env var | Meaning |
   |---|---|
   | `PARTITION_STRATEGY` | the dispatch key (your new key) |
   | `CLIENT_ID` / `CLIENT_NUM` | this client's id and the cohort size |
   | `MIN_GOALS_PER_CLIENT` | the per-client floor |
   | `OMEGA` | preference Dirichlet spread (ω) |
   | `SIZE_STD` / `SUCCESS_STD` | coverage / hardness Beta concentration (ξ / ξ′) |
   | `ENV_DIV` / `KEEP_RATIO` | catalog-split strength / distractor density |
   | `VARIANT_N` | # of env-variant arms in the pool (bm25/lookalike/rank) |
   | `TRAJECTORIES_FILE` | hardness: the `task_id → success` labels file |

   Add an `elif` branch:

   ```python
   # fedagent/envs/webshop/service/server.py
   elif PARTITION_STRATEGY == "myhet":
       from fedagent.hetero.myhet import myhet_for_client
       # task-level (content-dependent) idxs are DEFERRED to _lifespan, where the warmed
       # env's real server.goals exist (see _compute_task_partition). Order-independent
       # strategies (a contiguous range / env_kwargs) can be computed here at import time.
       _DEFERRED_TASK_PARTITION = "myhet"   # then add a branch in _compute_task_partition()
   ```

   Note the **deferral subtlety**: a content-dependent task partition must address the
   env's *actual* seed-42-shuffled `server.goals`, which only exist after the pool is
   warmed — so those are computed in `_compute_task_partition()` (called from
   `_lifespan`), not at import. Catalog/variant strategies (order-independent ranges or
   `env_kwargs`) are computed at import time. The unperturbed **validation** service is
   always launched with `PARTITION_STRATEGY` cleared, so divergence is attributable to
   the perturbation alone.

3. **Forward the knobs from `run_fed.py`.** Add any new knob as a key in the `DEFAULTS`
   dict of `fedagent/fed/run_fed.py`, then export it as an env var inside the service
   launcher (`start_webshop_services` / `start_alfworld_services`), e.g.:

   ```python
   env.update({
       "PARTITION_STRATEGY": cfg.partition_strategy or "",
       "CLIENT_ID": str(c), "CLIENT_NUM": str(cfg.total_clients),
       "MYHET_KNOB": str(cfg.get("myhet_knob", <default>)),   # <- your new knob
       ...
   })
   ```

4. **Select it from config.** Heterogeneity is chosen entirely through the federated
   YAML (a **flat** schema = the `DEFAULTS` dict — there is no nested `federated:`
   block):

   ```yaml
   partition_strategy: "myhet"   # the dispatch key
   myhet_knob: 0.5               # your knob (also add it to DEFAULTS)
   min_goals_per_client: 100
   ```

---

## 3. Add an RL algorithm (beyond GRPO / PPO)

### Where

The RL update lives entirely in **stock verl**, selected by `algorithm.adv_estimator`.
The federated overlay never touches the loss; it only (a) selects the estimator and (b)
carries whatever checkpoints the client writes.

### How GRPO and PPO are selected today

The federated YAML key `adv_estimator` drives a single branch in `run_fed.py`'s
`run_client`:

```python
# fedagent/fed/run_fed.py
if str(cfg.get("adv_estimator", "grpo")).lower() == "gae":
    cmd += ["algorithm.adv_estimator=gae"]          # PPO: flips need_critic on
    if critic_model_path:
        cmd += [f"critic.model.path={critic_model_path}"]
```

- **GRPO** (default, `adv_estimator: grpo`): actor-only. The group size *G* comes from
  `rollout.n` (paper *G = 8* via `actor_rollout_ref.rollout.n=8` in `client_overrides`).
  The client writes **only** an actor checkpoint, and the driver FedAvgs/merges just the
  actor — the GRPO command is byte-identical to the verified baseline.
- **PPO** (`adv_estimator: gae`): adds a **critic** (value model). The driver detects
  the critic shard dir alongside the actor (`critic_dir_for`), FedAvgs **both**
  components, merges each to HF, and carries the federated value model forward
  (`critic.model.path` is set per round; round 1's critic = the base model — a random
  value head on the backbone). For this to work the PPO config **must** include
  `critic.checkpoint.save_contents=[model]` (and `...actor.checkpoint.save_contents=[model]`)
  in `client_overrides`, so the aggregator finds the value-model weights. See
  `fedagent/config/examples/webshop/scaled/ppo.yaml`.

### What federating a new algorithm entails

1. **Add the estimator in verl**, so `algorithm.adv_estimator: my_algo` resolves
   (verl ships PPO/GAE, GRPO, and others upstream; a new one is added alongside in
   verl's advantage-computation dispatch). FedAgent does not patch this.

2. **Select it from the federated YAML.** If your algorithm is actor-only and needs no
   extra checkpoint, add a one-line branch in `run_client` (mirror the `gae` branch) to
   pass `algorithm.adv_estimator=my_algo`; surface its hyperparameters through
   `client_overrides` (each entry is a literal Hydra override applied to the per-client
   subprocess).

3. **Mind the checkpoint shape — this is the only federation-level concern.** The
   aggregator (§4) operates on the **FSDP shard layout**
   (`checkpoints/global_step_<n>/<component>/model_world_size_*_rank_*.pt`), not on
   algorithm internals. An **actor-only** algorithm needs nothing extra. If your
   algorithm adds a trainable component beyond the actor (as PPO adds the critic), make
   sure that component lands under the same `global_step_<n>/<component>/` layout *with*
   `checkpoint.save_contents=[model]`, and FedAvg/merge it with the same machinery —
   `run_fed.fedavg(..., kind="<component>")` and `merge_to_hf(..., kind="<component>")`
   are already component-agnostic (they average and merge whatever shard dir they are
   given). The merger reads the architecture from the shard's `huggingface/config.json`,
   and **both** the actor and the value model serialize as `...ForCausalLM` (the value
   model just carries an extra scalar value head), so no per-component special-casing is
   needed.

---

## 4. Add an aggregation rule (beyond FedAvg / FedProx)

Aggregation has **two** seams, because FedProx is not a server rule — it is a
client-side proximal term, with server aggregation left as FedAvg.

### 4a. Server-side rule — `aggregate_fedavg_fsdp.py`

### Where

`tools/verl08_migration/aggregate_fedavg_fsdp.py` is the live server aggregator. The
driver shells out to it once per round (per component) via `torchrun`:

```python
# fedagent/fed/run_fed.py :: fedavg()
cmd = [
    "torchrun", f"--nproc_per_node={ws}", str(AGGREGATOR),
    "--phase", "aggregate",
    "--client-actor-dirs", ",".join(str(a) for a in client_dirs),
    "--output-actor-dir", str(agg),
    "--global-step", "0",
]                       # + ["--weights", "0.5,0.5"] when cfg.weights is set
```

### The shape (and why)

verl 0.8 FSDP1 saves per-rank shards as `torch` `ShardedTensor`, which **cannot** be
loaded single-process. So FedAvg runs under a **matched-world-size process group**
(`torchrun --nproc_per_node == the save-time world_size`): each rank loads *its own*
rank shard from every client, (weighted-)averages the **local** tensors in place, and
`torch.save`s the dict back. The output is byte-structurally identical to a verl
checkpoint (same `ShardedTensor` objects, only local values changed), so the next round
loads it with verl's own FSDP wrap unchanged.

```python
# the averaging core (aggregate_fedavg_fsdp.py)
sds = [torch.load(c / rank_file, weights_only=False) for c in clients]
base = sds[0]
for k in base:
    acc = _get_local(base[k])              # writable local shard (ShardedTensor/DTensor/plain)
    acc.mul_(weights[0])
    for w, other in zip(weights[1:], sds[1:]):
        acc.add_(_get_local(other[k]), alpha=w)
torch.save(base, out / rank_file)          # same objects, averaged local values
```

### The contract for a new rule

1. **Add a `--phase` (or a sibling averaging routine).** The CLI is
   `--phase {aggregate,verify}`, `--client-actor-dirs A,B`, `--output-actor-dir OUT`,
   `--weights`, `--global-step`. Reuse the **load → average-local-shard → save**
   skeleton above; most rules (trimmed mean, median, FedAvgM, per-client weighting by
   `|X_i|`) are a different reduction over the same per-rank `sds` list. Keep the output
   structure intact: write `model_world_size_<ws>_rank_<rank>.pt`, copy
   `fsdp_config.json` + `huggingface/`, and write `latest_checkpointed_iteration.txt` —
   that *is* the format the next round's clients load from.
2. **A weighting hook already exists.** If your rule is "FedAvg but weighted", you do
   not need new code in the aggregator: compute the weights and pass them through
   `cfg.weights` (`--weights w0,w1,...`; they must sum to 1). The driver forwards
   `cfg.weights` to every component.
3. **Validate with the built-in `verify` phase.** `--phase verify` re-loads the written
   shards and asserts the local values equal the (weighted) mean of the clients
   (FedAvg correctness) and that they round-trip as `ShardedTensor` (so verl will load
   them). Run it on a real round before trusting a new rule.

### 4b. Client-side rule — the `sitecustomize` FedProx hook

FedProx adds `mu * (w - w_t)` to the actor gradient before each optimizer step, where
`w_t` is the round-start global model. In FedAgent's subprocess-per-round design each
client-round is a **fresh process** that loads the aggregated model, so `w_t` is simply
the params at the first optimizer step — no external per-round reset is needed.

The hook is injected **without** a Ray `runtime_env` worker hook (which clobbered
verl's per-worker `CUDA_VISIBLE_DEVICES`). Instead, `sitecustomize.py` at the repo
root is auto-imported by CPython at interpreter startup in **every** process on
`PYTHONPATH` (the driver and its Ray workers, since `run_fed` sets
`PYTHONPATH=REPO_ROOT`), gated on `FEDPROX_MU`:

```python
# sitecustomize.py (repo root)
import importlib.util, os
mu = float(os.environ.get("FEDPROX_MU", "0") or "0")
if mu > 0:
    if importlib.util.find_spec("verl") is None:
        pass   # non-trainer env (e.g. a service conda env without verl) -> silent no-op
    else:
        from fedagent.fedprox import install_deferred_patch
        if not install_deferred_patch(mu):   # arms a meta-path hook; fail CLOSED
            raise RuntimeError("FedProx requested (FEDPROX_MU>0) but the patch could not be armed")
```

`install_deferred_patch` arms a `sys.meta_path` finder that monkeypatches
`FSDPEngine.optimizer_step` on verl's **first** import of
`verl/workers/engine/fsdp/transformer_impl.py` — which happens *after* the Ray worker has
its per-rank `CUDA_VISIBLE_DEVICES` set. (Importing `FSDPEngine` eagerly here at interpreter
startup would pull in torch/verl before device assignment and break per-rank GPU isolation
at multi-GPU, "Duplicate GPU detected".) The patch snapshots `w_t` on the first call, then
adds the proximal gradient per local shard (FSDP1 sharded view / FSDP2 DTensor — the
elementwise `grad.add_` is correct on each shard) before the original step. The driver
sets `FEDPROX_MU` per client when `fedprox_mu > 0`; plain FedAvg leaves `mu = 0` (a
no-op, and `fedagent` is never even imported). Eval always strips `FEDPROX_MU`.

To add **another** client-side rule on the same pattern: write the patch in a module
under `fedagent/`, add an env-var gate to `sitecustomize.py`, and have `run_fed.py` set
that env var per client. Keep the patch CUDA-free at import time (importing
`FSDPEngine` must not initialize CUDA — it runs *before* verl assigns devices).

---

## Smoke-test recipe

Validate any extension end-to-end with the in-process TinyGuess smoke (2 clients × 2
rounds, 1 step/client/round — closes the full federated loop in minutes, no services):

```bash
# inside the fedagent-verl08 conda env, on a GPU node
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml
```

This trains each client (`python -m fedagent.main_ppo_fed`), FedAvgs the two clients'
FSDP shards under a matched process group, merges back to HF, and re-enters round 2
from the aggregated model. A clean run ends with `FEDERATED LOOP CLOSED` and writes
`<output_dir>/federated_summary.json` (per-round provenance:
`started_from → aggregated_hf`). CLI flags override the YAML:

```bash
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml \
    --rounds 3 --clients 4 --n-gpus 2 --output-dir /tmp/my_smoke
```

What to check per extension point:

| You changed | The smoke proves | Then exercise |
|---|---|---|
| **A new in-process env** | point `env_spec` at your `config/envs/<env>.yaml` — it resolves, rolls out, and aggregates | full run on real data |
| **A service-backed env** | (TinyGuess can't test the service) | set `env_kind: <env>`, watch `/health` come up, confirm episodes step |
| **A heterogeneity strategy** | set `partition_strategy` + knobs; confirm the service `/health` echoes your partition and per-client slices differ | the het arm under [`./reproducing.md`](./reproducing.md) |
| **A new algorithm** | set `adv_estimator`; for a critic-bearing algo confirm both components FedAvg/merge | `examples/webshop/scaled/ppo.yaml` as the PPO template |
| **An aggregation rule** | run `--phase verify` on a real round's shards | a multi-round run, diffing against average-of-clients |

For the run-mode/GPU matrix and the federated-key reference see
[`./running.md`](./running.md) and [`./configuration.md`](./configuration.md); for the
round loop, [`./architecture.md`](./architecture.md); for per-component code detail,
[`../fed/README.md`](../fed/README.md), [`../envs/README.md`](../envs/README.md), and
[`../hetero/README.md`](../hetero/README.md).
