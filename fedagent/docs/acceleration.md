# FedAgent verl-0.8 — Performance Analysis & Acceleration Plan

> **Status:** Design + analysis doc. The migration is *functionally* closed (windowed GRPO+PPO and
> concat federated loops all verified green on GPU); this doc is about **speed**, not correctness.
> A Chinese translation lives in [acceleration_cn.md](acceleration_cn.md).
>
> **TL;DR:**
> 1. **Why it's slow:** a measured **76% (0.5B, warm cache, this session) → 88% (1.5B smoke)** of
>    wall-clock is the **per-(client×round) subprocess cold-start** (Ray + FSDP + vLLM + kernel compile;
>    2.5 min warm / 5–14 min cold), repeated ~140× for a paper run. Actual training steps are only
>    ~12–24%. See §2.6.
> 2. **verl's async runs, but only at the rollout layer** (episodes in a batch dispatched concurrently to
>    vLLM); the **pipeline layer (client / round / eval) is fully serial** — the largest unexploited
>    parallelism.
> 3. **Four levers, by ROI:** `#2 env prewarm` (safest, zero numerics) → `#4 persistent trainer/vLLM`
>    (the only single-node lever that kills the 88%, with equivalence risk) → multi-node `#3 parallel
>    clients` + `#1 eval∥train`.
> 4. **Iron law:** hard-parallelizing eval+train on the *same* GPUs is VRAM/kernel contention, not
>    speedup; real gains require either *eliminating* work (#4) or *resource isolation* (#1/#3, multi-node).
>
> **✅ Session result (#4 GPU-validated — see §3 Lever #4 / §7):** the persistent trainer is built
> (overlay, no verl fork) and integrated into `run_fed`. Per-round (`persistent: true`): a 2-round
> federated GRPO loop closes in **515 s vs 909 s subprocess = −43%**. Cross-round (`cross_round: true`,
> §7.2): **ONE cold-start for the whole run, 342 s = −62%**. Both keep the final aggregated model
> **numerically equivalent** to the subprocess path (full-loop max|Δ|=1.13e-5, bf16 noise). PPO critic
> reload (§7.1) and **per-client service routing — validated on the real WebShop env** (§7.3, 32/32
> episodes to distinct services) — also landed; cross-round + per-round eval OOMs and auto-falls-back
> to per-round (§7.4). It is the single-node lever that doesn't break reproducibility. Windowed-default
> blocker fixed too (§7.5).

---

## 0. Scope & references

- **➜ Companion docs.** *This* doc is the **analysis & plan**. Also:
  [acceleration_report.md](acceleration_report.md) — the **complete end-to-end walkthrough** (every lever &
  feature in depth, the investigations + corrections, all results); and
  [acceleration_results.md](acceleration_results.md) — the **results at a glance** (status table + numbers).
- **Repo (overlay):** `/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent`
- **verl 0.8 source (editable):** `/gpfs/projects/b1222/userdata/canyu/kangyu/others/verl/verl`
- **Bar:** scientific equivalence with the FedAgent paper (reproduce within 3-seed noise).
  Every acceleration here is judged first on *whether it perturbs the numerics*, then on speed.
- **Evidence:** all timings below are from real run logs on the cluster (cited inline), not estimates.

---

## 1. Current state — what is implemented

### 1.1 Migration architecture: thin overlay, no fork
FedAgent runs on **stock verl 0.8** via a thin `fedagent/` overlay (no verl fork). verl is an
editable install; the overlay adds the federated orchestrator, the env services, the faithful
windowed rollout, FedProx, and the paper config generator — nothing in verl is patched on disk
(the one runtime patch is a *scoped, tag-gated* monkeypatch applied from the overlay; see §1.3).

### 1.2 Federated loop: `fedagent/fed/run_fed.py` — subprocess per (client, round)
The orchestrator is verl-agnostic. For each round `r`:

```
model_r = base                 (r == 1)
        = round_{r-1}/aggregated/hf   (r > 1)   # merged FedAvg'd FSDP shards
for each selected client c (SEQUENTIAL):                 # run_fed.py:1046
    python -m fedagent.main_ppo_fed  model.path=model_r  default_local_dir=round_r/client_c ...
        -> FSDP actor (+critic for PPO) checkpoint shards
FedAvg:  torchrun aggregate_fedavg_fsdp.py  --client-actor-dirs c0,c1 ...   # run_fed.py:1074
merge :  python -m verl.model_merger  ->  round_r/aggregated/hf            # run_fed.py:1116
eval  :  (every round) a SEPARATE val-only subprocess (inline) | hot engine (worker)   # run_fed.py
```

Key properties (all confirmed in code):
- **One fresh OS subprocess per (client, round)** — clean isolation, guaranteed GPU release.
  Clients within a round are **sequential** (`for c in selected`, `run_fed.py:1046`).
- **Rounds are a hard barrier:** round `r+1` cannot start until `FedAvg(r)`+`merge(r)` produce `model_r`.
- **Eval is its own subprocess** (`eval_global`, `run_fed.py:717`, `trainer.val_only=true`) → it
  pays a *second* full cold-start, and it runs **after** the round, blocking nothing but blocked by it.
- **Env services** (WebShop/ALFWorld) are **per-client remote FastAPI services**, started **lazily**
  per round (only the round's selected clients), torn down before aggregation (`run_fed.py:1386`).
  The unperturbed **val service** starts once and stays up.
- Baselines (federated / centralized / local), FedProx (via `sitecustomize.py`), and the
  unperturbed eval curve (`val_before_train`, `test_freq`) are all wired.

### 1.3 Two rollout modes (the faithful-vs-stock axis)
| mode | what it is | samples | history | default | how |
|---|---|---|---|---|---|
| **windowed** | the paper's per-turn rollout | 1 per **turn** | `history_length=2` template | ✅ yes | `WindowedAgentLoopManager` |
| **concat** | stock verl GymTextAgentLoop | 1 per **episode** | full history | opt-in | stock `AgentLoopManager` |

`run_fed.inject_rollout_mode` (`run_fed.py:647`) injects the windowed manager into **both**
train and eval cmds unless an explicit `manager_class` override is present.

**The windowed fix** (`fedagent/agent_loops/windowed_manager.py`): windowed expands one episode into
N per-turn rows, which violates verl 0.8's "1 sample per input prompt" contract (silent truncation in
train, an AssertionError crash in eval). The overlay fixes this *without forking verl* via a tag-gated
scoped monkeypatch of `DataProto.slice`/`union` + a worker-side eval-collapse + `adjust_batch`-style
divisor padding (mirrors legacy `del batch; batch = gen_batch_output` + `adjust_batch`). It is
idempotent and isolated (verified 0 leak into concat runs).

### 1.4 What is verified green on GPU
- **windowed GRPO** federated loop closed (per-turn rows folded, FedAvg, eval per-episode).
- **windowed PPO** federated loop closed (actor **and** critic FedAvg'd each round).
- **concat** (via `rollout_mode=concat`) closed; monkeypatch isolation holds.

### 1.5 What async ALREADY exists (and is being used) — verl 0.8 agent-loop rollout
verl 0.8 replaced legacy verl-agent's batched-sync rollout with an **async agent loop**:

```
one training step:
  train_batch_size × rollout.n  episodes launched as async coroutines
  each episode (async def):  await env.reset(); await generate(); await env.step(); await generate(); ...
  vLLM dynamically batches the concurrent generate() requests across all episodes
```

This async **is running** — the WebShop PPO paper config launches `64×8 = 512` trajectories that all
`reset()` at once (the source of the env-service "connection storm" we hardened). Concretely:
- `GymTextAgentLoop.run()` / `WindowedGymTextAgentLoop.run_episode_windowed()` are `async def`;
  `server_manager.generate()`, `env.reset()`, `env.step()` are all `await`ed → episodes in a batch
  advance concurrently.
- WebShop/ALFWorld env clients use `httpx.AsyncClient`; services are FastAPI/uvicorn async handlers
  with an `asyncio.Queue` env pool → the I/O layer is non-blocking.
- vLLM batches the concurrent generation requests — the core payoff of async rollout.

**So: the rollout layer uses async. The federated pipeline layer does not.** §2 is about that gap.

---

## 2. Deep analysis — why there is no speedup yet

The headline: **the work that runs is mostly not training.** On a windowed PPO federated smoke,
total wall-clock was **31m33s** but the sum of `timing_s/step` across all training steps was only
**~223.5s** → **~88% of wall-clock is fixed overhead**, almost none of it the actual RL step.

There are three distinct reasons, in order of impact.

### 2.1 ⭐ Dominant: per-(client×round) subprocess **cold-start** (~88%)
Every `python -m fedagent.main_ppo_fed` subprocess rebuilds the **entire** stack before training
step 1. Measured from a real **1.5B / max_seq_len=8192** run log
(`<output_dir>/round_1/client_0/training.log`):

```
20:17:49.776  Ray local instance started           (worker.py:2003)
   │   ~2 min: worker-pool construction, FSDP actor/ref/critic init + weight load,
   │           CUDA-graph capture-size sweep, kernel prep
20:19:58.178  flashinfer autotuning  (39 ms — CACHED here; see note)   (autotuner.py:256)
20:20:02.017  vLLM "Initializing a V1 LLM engine"  seed=0, prefix_caching=True, dummy load
20:20:20      LLMServerManager ready: 4 server addresses                (ray_trainer log)
   →  first training step begins
20:20:20+     step:1   timing_s/step=37.94  (gen=30.9, update_actor=1.78, update_weights=2.14)
```

**Cold-start ≈ 2.5 min** for this config *with a warm kernel cache*. **Critical nuance:** the
flashinfer autotune above was **39 ms because the JIT/inductor cache was already warm**. On a **cold
node / first run**, flashinfer JIT + `torch.compile`/inductor compilation is **not** cached and the
same phase balloons to the **5–14 min** we have observed. Either way it is **one-time setup**:

| cold-start phase | warm cost | persistent process can skip? | evidence |
|---|---|---|---|
| Python import + Hydra | ~1–2 s | ✅ skip | — |
| Ray init | ~2–5 s | ✅ skip (cluster stays up) | log L5 |
| Worker pool + FSDP actor/ref/critic init + weight load | ~60–90 s | ✅ skip; hot-swap weights instead | log L920–921 |
| CUDA-graph capture + flashinfer/torch.compile kernels | ~30 s warm / **5–14 min cold** | ✅ **skip entirely** (kernels stay compiled) | log L863–886 |
| vLLM engine init + KV-cache alloc | ~18 s | ✅ skip; `update_weights` (~0.5–2 s) instead | log L882–978 |
| Dataloader + agent-loop manager | ~3 s | ✅ cheapen (rebuild dataset only) | — |

This setup is paid **once per (client × round)** *plus* **once per eval**. A paper run is
~`clients_per_round × rounds (+ evals)` ≈ **140 subprocesses**. At 2.5–14 min each, cold-start alone
is **hours**, dwarfing the ~38 s/step of actual training. **This is the 88%.** A process that stays
alive and hot-swaps weights pays it **once**.

### 2.2 Intra-step slowdowns (the ~12% that does run)
Two faithfulness/env costs make even the running 12% slower than legacy:

1. **Windowed breaks vLLM prefix cache.** Windowing shifts the context window each turn → every turn
   is a prefix-cache *miss*; concat reuses the growing prefix → hits. Measured **windowed gen ~43 s vs
   concat gen ~30 s** (same node, 1.43×). This is the *price of faithfulness*, accepted deliberately —
   not a bug. (Step-level: gen is ~81% of the 38 s step, so this directly moves wall-clock.)
2. **`_TW_LOCK` serializes ALFWorld env stepping.** The ALFWorld service holds a **process-global
   `threading.Lock`** around every textworld `reset`/`step` because the tatsu PDDL parser is a
   process-global mutable singleton. Measured **86 ms/step → 160 serialized steps = 13.7 s** of pure
   lock-serialized env time per windowed client-step, versus legacy's **parallel per-env Ray actors
   (~0.9 s)**. **WebShop has no such lock** (so this is ALFWorld-only). The async rollout's env-side
   concurrency is real but throttled to single-file by this lock.

> Net of §2.2: even with infinite orchestration parallelism, windowed is ~1.47× slower *per step*
> than concat, and ALFWorld adds a serialized env tax. These are understood and bounded.
>
> **⚠️ 2026-07-01 correction (§9):** "bounded" held only at the small batch measured here (160
> steps → 13.7 s). At paper-scale batches (~3200 env steps/optimizer-step) the same 86 ms/step
> constant becomes **73 % of a 4-GPU ALFWorld step** — now FIXED by env-service replica sharding
> (`alfworld_replicas`): gen 219→52 s, step 298→128 s, end-to-end −31 %. See §9.

### 2.3 The unexploited parallelism — the whole pipeline layer is serial
Nothing in `run_fed.py` overlaps:

| serial today | independent? | could overlap? | blocker |
|---|---|---|---|
| **clients within a round** (`for c in selected`, L854) | ✅ yes — each reads `model_{r-1}`, writes its own ckpt; FedAvg is order-free | ✅ yes | GPU contention (single node) → needs more GPUs/nodes |
| **rounds** | ❌ no — `train(r+1)` needs `FedAvg(r)` | ❌ no (hard barrier) | data dependency |
| **eval(r)** (after merge, L896) | ✅ yes — reads `model_r` only | ✅ **yes, with `train(r+1)`** | both want GPU |
| **next round's env warmup** | ✅ yes — services are CPU, clients are deterministic | ✅ yes (overlap with FedAvg/merge/eval) | nothing — pure scheduling |

### 2.4 The dependency graph (what *can* overlap)
```
                        ┌────────────► eval(r)            (reads model_r; pure measurement)
merge(r) ──► model_r ───┤
                        └────────────► train_client_c(r+1) ∀c  (reads model_r)
within round r:  train_client_0(r), train_client_1(r), ...  are siblings reading model_{r-1}
                 FedAvg(r) is the JOIN (needs ALL clients of round r)
```
Therefore the legal overlaps are exactly: **(a) clients within a round**, **(b) eval(r) ∥ train(r+1)**,
**(c) env-warmup(r+1) ∥ post-train(r)**. Cross-round training cannot overlap.

### 2.5 The iron law (the crux)
On a **single shared GPU set**, hard-parallelizing eval+train (or two clients) is **not speedup — it
is VRAM + kernel contention.** Real benefit requires either (i) **eliminating** the work (#4: stop
re-paying cold-start) or (ii) **resource isolation** (#1/#3: give each concurrent job its own
GPUs/node). Any "async pipeline" that ignores this just timeshares one set of kernels.

### 2.6 Empirical confirmation (this session, GPU — qgpu3022/qgpu3013, 4×H100)
Fresh runs that validate the analysis above.

**Baseline (TinyGuess, 0.5B, 4-GPU, concat, warm caches, 3 steps/client, federated 2 clients × 2 rounds, rc=0):**

| per (client,round) subprocess | mean |
|---|---|
| total wall | **125 s** |
| step-compute (3 steps) | 31 s (≈10 s/step) |
| **fixed overhead (cold-start + teardown)** | **94 s = ~76%** |

- Even in the *cheapest* case (tiny model, warm kernel + fs cache, 3 steps), **~76% of every subprocess is not training.** Scale to 1.5B + 8192-ctx + ~140 subprocesses and this is the **88%** measured on the windowed PPO smoke. Import alone (process→Ray) was **27 s warm vs 86 s on a cold GPFS read**.
- → **#4 (persistent process) is THE lever**: it pays this ~94 s *once*, not ~140×.

**Windowed-default crash — confirmed release blocker.** `rollout_mode=windowed` (the new default) + stock `agent.yaml` (registers only `gym_text`) → `AttributeError: 'GymTextAgentLoop' object has no attribute 'run_episode_windowed'` ([windowed_manager.py](../agent_loops/windowed_manager.py)#L152). Windowed was only ever green with the *explicit* `tools/verl08_migration/poc/windowed` agent config; any config on the stock `agent.yaml` crashes. → the **paused windowed config migration** (register `gym_text_windowed`) is a prerequisite for windowed-as-default.

**#2's benefit is config-specific (often ≈0).** WebShop *homogeneous* services (`partition_strategy=""`) become healthy in **seconds** — they build no catalog, so there is nothing for prewarm to overlap. #2 pays off only for **expensive-warmup** arms (catalog_split with large catalogs, ALFWorld game collection), and even then the per-subprocess cold-start (#4) dwarfs env warmup. **Conclusion: #2 stays a minor, opt-in lever** — correct and ready (CPU-validated), but not where the wall-clock is.

**Subprocess-path fragility (more #4 motivation).** Across 4 sequential cold-starts on one node we saw vLLM `/dev/shm` `KeyError`s and a `DataLoader worker killed (SIGKILL)` during teardown (survived only via the 5 s inter-client wait; legacy polled `nvidia-smi` until GPU-free). The 4-GPU rollout also tripped `data size must be divisible by force_group_size × micro_batch` — the same divisibility class the windowed fix handles, here on a hand-made config. A persistent process sidesteps this repeated alloc/teardown churn entirely.

---

## 3. Acceleration levers

### Lever #2 — pre-warm next round's env services *(cheapest, safest, single-node win)*
Overlap round `r+1`'s env-pool warmup (minutes for WebShop/ALFWorld) with round `r`'s FedAvg/merge/eval.
- **Why safe:** pure scheduling, **zero numerical impact**. CPU-only services overlap GPU aggregation —
  no resource contention.
- **Feasible:** `select_clients(r+1, …)` is deterministic (seeded `base_seed + round - 1`,
  `run_fed.py:799-811`) → next round's clients are known before round `r` ends. Ports are
  client-indexed (`base_port + client_id`) → no cross-round collision **except** a client selected in
  both consecutive rounds (handle: skip prewarm for the overlap, or reuse the live service).
- **Patch shape** (`run_fed.py`): add `prewarm_next_round_services(cfg, env_base, r)` (calls the existing
  `start_*_services` with `select_clients(r+1)`); call it **after** the client loop, **before**
  `fedavg`; stash handles in `prewarmed_next_services[r+1]` and adopt them at the next iteration's top
  instead of starting fresh; tear down on the final round / on failure. Preserves the
  "≤ clients_per_round alive" invariant (at most ~2× briefly during the overlap).
- **Win:** saves the env-warmup minutes that currently block each round's start. **But §2.6 measured WebShop homog warmup at ~seconds → benefit ≈ 0 there; only material for expensive-warmup arms** (catalog_split large catalogs, ALFWorld). Bounded and config-specific.

### Lever #4 — persistent trainer / vLLM across clients *(biggest single-node lever; equivalence risk)*
Keep **one** `RayPPOTrainer` alive across all (client, round) calls; between clients, hot-swap the
actor (and critic) to the round's aggregated model via `update_weights` (the same call verl already
runs every step, ~0.5–2 s), repoint the dataloader at the next client's env/seed, **rebuild the
optimizer/scheduler**, and run another E epochs. **This is the only lever that attacks the 88%** — it
pays cold-start *once* instead of ~140 times.

> **✅ PROTOTYPE BUILT + GPU-VALIDATED (this session) — see §7.** Overlay-only (no verl fork):
> `fedagent/fed/persistent_{patch,task_runner,main}.py` + a `sitecustomize` gate. The per-client
> reset is `reload_client_model` → re-point `engine.model_config.local_path` + `engine.initialize()`
> (verified to rebuild module+optimizer+scheduler in one call) + `del engine._fedprox_w_t`. A/B on a
> 2-client/1-round TinyGuess smoke: **207 s vs 327 s (−37%)** and per-client checkpoints **equivalent
> to the subprocess path (max\|Δ\|≈1e-6, bf16 noise)**. The reset reproduces fresh-Adam / fresh-LR /
> dropped-FedProx-anchor; the vLLM-RNG divergence risk did not surface at checkpoint level here.
> Since confirmed at the **full-loop** level (2-round final aggregated model max\|Δ\|=1.13e-5,
> §7.1) and for **PPO** (critic reload validated, §7.1); remaining: confirm at larger step counts.

- **Feasibility:** **Yes, with caveats.** verl already re-enters "load-weights → push-to-rollout →
  train" every step; the worker groups / vLLM / FSDP are designed to be long-lived. Nothing in the
  architecture forbids driving N clients through one trainer. The caveats are **all about numerical
  equivalence**, not mechanism.
- **Cleanest seam:** factor the body of `RayPPOTrainer.fit()` into a re-enterable `_fit_one_client()`,
  driven by an outer loop. The natural cut is **`ray_trainer.py:1383-1410`** — everything *above*
  (`init_workers()`) is one-time; from `self.global_steps = 0` (L1383) down is already per-`fit()` and
  becomes per-client. The overlay driver seam is **`run_fed.py:1046`** (the `for c in selected`
  loop becomes `trainer.train_client(c, model_r, seed_c)` instead of `subprocess.Popen`).
- **Sharp detail:** the FedAvg aggregator writes **only** `model_world_size_*` shards and **strips the
  optimizer/extra shards** (`aggregate_fedavg_fsdp.py:75-76`). So you **cannot** reuse
  `load_checkpoint` as-is (it asserts the optim shard exists). Load with `load_contents=["model"]`
  (or push merged-HF weights into `engine.module` in-memory), then `update_weights()` to vLLM.

#### Per-client reset checklist (the equivalence-audit spec)
The subprocess design's defining property: **every client starts from fresh Adam moments, a fresh
cosine schedule at step 0, and a FedProx anchor = the just-loaded aggregated model.** A persistent
process must reproduce each of these by hand:

| # | state | persistent path must… | cite |
|---|---|---|---|
| 1 | actor weights | load aggregated FSDP shards (model-only) → `update_weights()` | aggregator:75; ray_trainer:1387 |
| 2 | **Adam m/v** | **rebuild optimizer** (don't keep prev client's) — *biggest trap* | transformer_impl:451 |
| 3 | **LR scheduler** | **rebuild scheduler** at step 0; recompute `total_training_steps` if `len(dataloader)` differs | ray_trainer:438 |
| 4 | `global_steps` | reset to 0 per client (already L1383) | ray_trainer:1383 |
| 5 | torch/numpy/py RNG | re-seed per client (else the stream *continues*) | transformer_impl:135 |
| 6 | **env seed** | rebuild `AgenticDataset` so new `FEDAGENT_BASE_SEED` takes effect (read only at `__init__`) | agentic_dataset:55 |
| 7 | dataloader iterator + sampler RNG | rebuild `train_dataloader`/sampler per client | ray_trainer:374 |
| 8 | vLLM GPU weights | covered by #1's `update_weights` | vllm_rollout/utils:288 |
| 9 | vLLM KV/prefix cache | flush (auto **iff** swap routes through `ServerAdapter.update_weights`) | vllm_rollout:194 |
| 10 | **vLLM sampler RNG** | **no clean reset API** — inject per-request `seed` into `SamplingParams` | vllm_async_server:505 |
| 11 | **FedProx anchor `w_t`** | `del engine._fedprox_w_t` after swap (else anchors to prev client's model) | fedprox:35 |
| 12 | ref-policy weights | re-point ref to the swapped aggregated weights | engine_workers:452 |
| 13 | critic (PPO) | same as #1–#3 for `critic_wg` (GRPO: skip) | ray_trainer:1008 |
| 14 | GPU mem fragmentation | `aggressive_empty_cache(force_sync=True)` per swap (operational, not numerical) | engine_workers:738 |

#### Top-3 equivalence risks (ranked)
1. **Optimizer / LR / FedProx-anchor carryover** (#2,#3,#11) — silently changes every client-after-the-first's
   optimization trajectory; invisible in logs. The aggregator *enforces* the fresh-Adam invariant by
   stripping optim shards; persistence must replicate it. **Mandatory rebuild.**
2. **vLLM sampler RNG continuity** (#10) — engine is seeded once at construction; no API re-seeds it,
   so the sampling stream *continues* across clients. Drives which trajectories each client sees →
   rollouts diverge from the subprocess path even at identical weights/env-seed. Only partially
   fixable (per-request seed). May force "equivalence" to be defined **statistically**, not bit-exact.
3. **Stale dataset/env seed + KV cache** (#6,#7,#9) — reusing the dataset object pins every client to
   client-0's goals (a *correctness* bug); bypassing `update_weights` leaves stale prefix blocks.
   Straightforward to fix, silent if missed.

> **Prior art:** both legacy verl-agent 0.3.1 and the current overlay are subprocess-per-(client,round)
> — neither ever kept a trainer alive (legacy even polled `nvidia-smi` until GPU-free between clients).
> So #4 is a **genuinely new capability** with no reference to port: it must solve *in-process* what
> legacy solved by process death. → must be A/B-validated against the subprocess path before adoption.

### Lever #3 — parallel clients within a round *(single-node wins for small models — GPU-validated; multi-node for large)*
`for c in selected` → concurrent, each client on its own GPU subset/node. **Numerically identical**
to sequential (FedAvg is order-free; the per-client env seed is `base_seed + round*100 + client`
(`run_fed.py:877`), *client-indexed, not order-dependent*). Splitting 4 GPUs → 2+2 changes FSDP
`world_size` → shard layout (aggregator reads `world_size_of` dynamically; same global batch ⇒ same numerics).

**GPU-validated on ONE node (2 client × 2 GPU, 1.5B, paper settings) — and it's ~35% faster, not a wash.**
Two independent verl/Ray/vLLM jobs **coexist** on the 4-GPU node: engines load on disjoint pairs
(6519 MiB ×4), no Ray-port / GPU / `/dev/shm` collision — isolation is just per-job `CUDA_VISIBLE_DEVICES`
+ `RAY_TMPDIR`. (One *non-obvious* shared-`/tmp` exception — the FSDP→vLLM weight-transfer socket — was
**race-lucky** at 2 jobs and only deadlocked once a 3rd concurrent job joined; **second robustness bug**
below, and the reason #3's own coexistence is only safe *after* that fix.) Timing:

| arm | wall-clock |
|---|---|
| `t1` — 1 client, 4 GPU | 558s |
| `t1` — 1 client, 2 GPU | 725s |
| **#3 — 2 client × 2 GPU, concurrent** | **727s** |
| sequential — 2 client × 4 GPU | 2×558 = **1116s** |

The win is **sub-linear FSDP scaling at small model size**: 4 GPUs are only `725/558 = 1.30×` faster than
2 for 1.5B (FSDP all-gather/reduce-scatter overhead + the env-latency-bound WebShop rollout + fixed
cold-start all weigh more when per-GPU compute is small). So 4→2+2 split + both clients concurrent beats
sequential-at-"full"-4-GPU. **This inverts the earlier "single-node #3 = contention" claim** — for small
models it's a real win. *Caveat:* for a large model where 4-GPU scaling ≈2×, single-node #3 ties/loses →
that's the regime that genuinely needs **≥2 nodes (one client per node)**.

**Robustness bug found + fixed (FedAvg rendezvous port).** Concurrency exposed a real bug: the FedAvg
step (`torchrun --nproc_per_node=ws aggregate_fedavg_fsdp.py`) used torchrun's **default c10d rendezvous
`localhost:29500`**, so two clients aggregating at the same time **collide on 29500** → one dies `rc=1`
mid-aggregate (`CUDA_VISIBLE_DEVICES`/`RAY_TMPDIR` don't isolate a TCP port; this also makes *any* two
`run_fed`s sharing a node unsafe at aggregation). Fix: `torchrun --standalone` (auto free port) + clear
inherited `MASTER_*`/`RANK`/`WORLD_SIZE` on the aggregator env (`run_fed.py fedavg()`). Touches **only**
the aggregator's comm port — FedAvg math, rollout, eval unchanged; PPO-critic FedAvg routes through the
same path. GPU-validated: the exact concurrent A+B that failed now closes **both** (`rc=0`, no `EADDRINUSE`).
*Debugging note:* the surface symptom was `DataLoader worker killed (SIGKILL)` — a **red herring** (benign
`__del__` teardown noise in *both* runs; dmesg showed no OOM-killer, node RAM 966 G / `/dev/shm` 504 G /
cgroup unlimited all free). The real failure was the 29500 collision one step later.

**Second robustness bug found + fixed (FSDP→vLLM weight-transfer socket — same family).** Pushing to a
**THIRD** concurrent verl job (2 train + 1 eval, §7.7) exposed a deeper collision. verl's FSDP→vLLM weight
sync (`bucketed_weight_transfer.py`) ships the rollout engine its new weights over a **ZMQ IPC socket on
the shared `/tmp`**: `ipc:///tmp/rl-colocate-zmq-<job_id>-replica-<r>-rank-<lr>.sock`, namespaced by the
**Ray job id** *precisely* to keep concurrent jobs disjoint. But FedAgent runs each client/eval as its
**own isolated Ray cluster** (`RAY_TMPDIR`), and every fresh cluster assigns the **same first job id
`01000000`** (verified — two clusters, identical id) → all jobs compute the **same** socket path → the
senders' `os.remove` + re-`bind` race cross-wires them and the weight `send` **deadlocks** (GPU-confirmed:
both trainers hung **44 min at 0 % util**, `do_epoll_wait` inside `update_weights`; env services healthy
with **zero sessions** — rollout never started). The 2-client case had been **race-lucky**; 3 jobs deadlock
reliably — so this threatened **#3 itself**, not just eval-parallel. **Fix (overlay-first):** `run_fed`
exports a **unique `VERL_RAY_JOB_ID` per launched verl subprocess** (per-process tag + role/client/round),
and a **2-line verl patch** makes the sender (`vllm_rollout.py`) and the receiver (`vllm_async_server.py`)
**honor that override** instead of the colliding job id; stock single-cluster runs (override unset) are
byte-for-byte unchanged. GPU-validated: the exact 3-job layout that deadlocked now closes all three
(`rc=0`; watcher saw weight-sync pass). **Same lesson as the FedAvg-port bug — verl's per-job isolation
assumes ONE shared Ray cluster; FedAgent's cluster-per-client model breaks that assumption** — both fixes
give the shared-host resource (rendezvous port / `/tmp` socket) a per-job-unique name.

### Lever #1 — eval(r) ∥ train(r+1) *(needs extra GPU; bounded)*
Both read `model_r` (§2.4) → independent. Run eval on a spare allocation while round `r+1` trains.
Eval runs **every round** (the per-round red line, §7.4), and each `inline` eval is a full cold-start
subprocess — so cheapening/overlapping it matters *every* round, not occasionally. **Compounds with
#4:** with a persistent trainer, eval becomes `update_weights(model_r) + val pass` (seconds, not a
cold-start) — this is exactly **`eval_mode=worker`** (§7.4), which needs no extra GPU; `parallel`
overlaps it onto spare GPUs. Pure measurement ⇒ **zero numerical risk.**

### ROI ranking by hardware
| hardware | do first | the big lever | skip (contention) |
|---|---|---|---|
| **single 4-GPU node** (default) | **#2** (free, zero-risk) | **#4** (only thing that touches the 88%); **#3 for *small* models** (2×2 = −35% on 1.5B, GPU-validated §Lever #3) | #1 (needs spare GPU); #3 on *large* models (4-GPU scaling ≈2× ⇒ wash) |
| **≥2 nodes / 8 GPUs** | #2 + **#3** (huge, bit-equivalent) | #1 free on spare alloc; #4 still ultimate GPU-hour win | — |

---

## 4. Scientific-equivalence summary (the project bar)
| lever | numerical impact | verdict |
|---|---|---|
| #2 env prewarm | none (pure scheduling) | ✅ safe |
| #1 eval ∥ train | none (measurement) | ✅ safe |
| #3 parallel clients | none (FedAvg order-free, client-indexed seed) | ✅ safe; single-node 2×2 GPU-validated (−35% on 1.5B) + **two** concurrency bugs fixed: FedAvg rendezvous-port (`--standalone`) and FSDP→vLLM weight-transfer socket (`VERL_RAY_JOB_ID`) — §Lever #3 / §7.7 |
| #4 persistent trainer | **none measured**: smoke max\|Δ\|≈1e-6, full-loop max\|Δ\|=1.13e-5; high IF resets missed | ✅ validated (§7) incl. full-loop + PPO critic reload; confirm at larger step counts |

---

## 5. Experimental plan + GPU validation
1. **Baseline timing (ground the 88%):** instrument `run_fed.py` with per-phase wall-clock
   (`service_start`, `run_client` total + its in-subprocess cold-start, `fedavg`, `merge`, `eval`).
   Run a small federated smoke; emit a phase breakdown table. *(This is the empirical anchor for §2.1.)*
2. **#2 A/B:** same config with/without prewarm on WebShop/ALFWorld; compare round-start latency.
3. **#4 spike + equivalence A/B:** prototype `_fit_one_client()` driving 2 clients in one process;
   compare per-step metrics (loss, reward, advantages) against the subprocess path **bit-for-bit
   where possible, statistically otherwise** (the vLLM RNG caveat, risk #2).
4. **#3 (if multi-node available):** run 2 clients concurrently on 2 allocations; confirm the
   aggregated model matches the sequential run within fp noise.

## 6. Recommended roadmap (phased)
- **Phase 0 — measure.** Instrument + baseline-time a smoke (turns the 88% from "summary number" into
  a fresh per-phase table). *Low risk, high information.*
- **Phase 1 — #2 prewarm.** Implement + A/B. Zero numerical risk, immediate single-node win.
- **Phase 2 — #4 prototype. ✅ DONE + GPU-validated (this session, §7).** Overlay-only: per-round −43%
  (§7.1), **cross-round −62% with ONE cold-start for the whole run (§7.2)**, PPO critic reload (§7.1)
  — all at full-loop max\|Δ\|=1.13e-5 EQUIVALENT on TinyGuess; **per-client service routing validated
  on the real WebShop env** (§7.3, 32/32 episodes to distinct services). The cross-round + per-round
  eval OOM is GPU-confirmed and **auto-handled by fall-back to per-round persistence** (§7.4).
  **Remaining:** a paper-scale (1.5B) persistent run + a larger-step equivalence A/B.
- **Phase 3 — multi-node #3 / #1.** When ≥2 allocations are available, parallelize clients and float
  eval onto a spare node.

---

## 7. Session results — GPU-validated #4 prototype

**Built (overlay-only, no verl fork):**
- [persistent_patch.py](../fed/persistent_patch.py) — `reload_client_model` (`ONE_TO_ALL`), attached to
  the worker class via a deferred import hook (mirrors FedProx) so it lands on every Ray FSDP worker;
  gated by `FEDAGENT_PERSISTENT=1` in [sitecustomize.py](../../sitecustomize.py).
- [persistent_task_runner.py](../fed/persistent_task_runner.py) — `PersistentFedTaskRunner(TaskRunner)`:
  `init_workers()` once, then `fit()`-per-client with `_reset_for_client` between.
- [persistent_main.py](../fed/persistent_main.py) — Hydra entry (`run_ppo(config, task_runner_class=…)`).
- [compare_fsdp_checkpoints.py](../../tools/verl08_migration/compare_fsdp_checkpoints.py) — tensor diff.

**Per-client reset (verified mechanism):** `reload_client_model` re-points
`engine.model_config.local_path` → `engine.initialize()` (transformer_impl.py:183→543: `_build_module`
reads `local_path` → new weights; `_build_optimizer`/`_build_lr_scheduler` → fresh Adam + fresh
schedule) → `del engine._fedprox_w_t` (re-anchor FedProx). Dataloader rebuilt per seed; driver RNG reseeded.

**A/B (TinyGuess, 0.5B, 4-GPU, concat, GRPO, 2 clients × 1 round, 2 steps, matched seeds 142/143):**

| metric | persistent | subprocess |
|---|---|---|
| wall-clock | **207 s** (1 cold-start) | 327 s (2 cold-starts + FedAvg + merge) |
| **saving** | — | **120 s = 37%** |
| client-0 ckpt max\|Δ\| | 4.0e-6 → **EQUIVALENT** | (reference) |
| client-1 ckpt max\|Δ\| | 7.6e-6 → **EQUIVALENT** | (reference) |

The persistent trainer is **numerically faithful (1e-6 = bf16 noise) AND 37% faster** on the smallest
case. The saving = `(clients_per_round − 1)` cold-starts/round here; with `cross_round` (§7.2) the
process spans the whole run and captures **all ~140 cold-starts** (one paid total).

### 7.1 Integrated into `run_fed` (`persistent: true`) — full federated loop
A whole round's clients now train in ONE process via `run_round_persistent` ([run_fed.py](../fed/run_fed.py)):
it writes a plan JSON, launches `persistent_main` once, and scans the per-client checkpoints → the
**same** FedAvg/merge runs downstream (byte-identical). A **2-round TinyGuess GRPO** A/B:

| | persistent (`persistent: true`) | subprocess |
|---|---|---|
| 2-round federated loop | **closed rc=0, 515 s** | closed rc=0, **909 s** |
| **saving** | — | **394 s = −43%** (~2 cold-starts avoided) |
| final aggregated model (round_2) | **max\|Δ\|=1.13e-5, mean 1.7e-7 → EQUIVALENT** (atol 1e-4) | (reference) |

The round-2 final aggregated actor is **bit-equivalent** across the whole loop (worst tensor
`layers.15.self_attn.o_proj.weight`, 1.13e-5 — bf16 round-trip noise), so the speedup is free:
the persistent path's hot-swap reset reproduces the subprocess path's fresh-Adam / fresh-LR /
fresh-dataloader exactly, not just per-round but compounded across rounds through FedAvg.

Per-round saving = `(clients_per_round − 1)` cold-starts. **PPO critic reload — GPU-validated.**
A 2-client `adv_estimator=gae` persistent smoke closes rc=0 with the value model FedAvg'd each
round: client 1's per-client reset rebuilds the critic engine (`reload_critic_model` on
`TrainingWorker`) to produce its `.../global_step_2/critic`, which then FedAvg's with client 0's
→ aggregated `actor` **and** `critic` both written, loop closed. (This first required a `critic:`
block in [fedagent_ppo.yaml](../config/fedagent_ppo.yaml) — verl's gae path needs the value
model's micro-batch set; inert under GRPO since the critic is disabled there.) WebShop/ALFWorld
in persistent mode are unblocked by per-client service routing (§7.3).

### 7.2 Cross-round persistence (`cross_round: true`) — GPU-validated
The per-round path still pays one cold-start *per round*. `cross_round: true` keeps **ONE process
alive across the entire run**: after a round's clients train + save, the worker writes a `done_<r>`
signal and **idles** (holding its GPUs) while the orchestrator runs the **same external
FedAvg/merge** (byte-identical → equivalence preserved); the orchestrator then publishes
`plan_round_<r+1>.json` + touches `go_<r+1>`, and the worker resets to the merged model and trains
the next round — all in the same process. The aggregator inits a *separate* NCCL world and uses
~1 GB/rank, so it coexists with the paused worker on the same H100s.

A **2-round TinyGuess GRPO** A/B (same seeds as §7.1, directly comparable):

| | cross-round | per-round persistent | subprocess |
|---|---|---|---|
| cold-starts (whole run) | **1** | 2 | 4 |
| wall (2-round loop) | **342 s** | 515 s | 909 s |
| **vs subprocess** | **−62%** | −43% | (reference) |
| final aggregated model (round_2) | **max\|Δ\|=1.13e-5 → EQUIVALENT** | 1.13e-5 → EQUIVALENT | (reference) |

Exactly **one** `Started a local Ray instance` for the whole run; worker exits rc=0 on `stop`. The
final model is bit-equivalent (worst tensor `layers.11.mlp.gate_proj.weight`, 1.13e-5) — the
reset-equivalence argument holds *across the FedAvg boundary*, so spanning all rounds in one process
costs nothing in fidelity. Implementation: [run_fed.py](../fed/run_fed.py) `BgProc` + `_wait_signal`
+ `stop_persistent_cross_round`; [persistent_task_runner.py](../fed/persistent_task_runner.py)
`_wait_next_round` (the worker-side outer loop).

### 7.3 Per-client service routing (WebShop/ALFWorld) — GPU-validated on real WebShop
In persistent mode all clients share ONE process, so process-env routing (`WEBSHOP_SERVICE_URL`)
can't give each client its own service. Fix: a **file channel**. The driver rewrites
`$FEDAGENT_SERVICE_URL_FILE` with the current client's URL (`base_port + client_id`) before each
`fit()` (`_route_service`); the shared agent-loop workers — which build the env per episode in a
*separate* process — read that file (`resolve_service_url`, priority: file > env-var > config >
default). Topology-agnostic (the file is on the shared FS), so it sidesteps having to reach into the
async agent-loop worker processes. Unit-tested (file beats env-var; per-client switch takes effect
with no restart; missing-file fallback; tinyguess no-op), **and validated end-to-end on the real
`verl-agent-webshop` service**: a 2-client persistent smoke (ports 8100/8101) closed rc=0 with
`route client 0 → :8100`, `route client 1 → :8101`, and — decisively — **each client's own service
served exactly its 32 episodes** (distinct env seeds 11 vs 12). Broken routing would send all 64 to
one service (or to the default `:8080`); 32/32 to distinct services is the discriminating proof.

### 7.4 Eval/training GPU sharing (`eval_mode`: inline / parallel / shared / worker)
**Eval cadence (what the curve is).** The paper's "server-aggregated" red line is **one eval per
round** of the round's aggregated global model on the shared unperturbed val set, **every round**
(`val_before_train` adds the base model as the round-0 point). It is **not** gated by `test_freq` —
that knob is verl's *within-job* step cadence (with `epochs_per_round` steps/round it only fires
`is_last_step`, the per-client "client-end" marks). One eval of the round's aggregate equals the
**expectation** of the paper's per-client `val_before_train`(step-0) **average**: every client of the
round starts from the *same* aggregate, so a single shared eval is the same curve at a fraction of the
rollout cost. Gate: `run_eval` every round ([run_fed.py](../fed/run_fed.py)); `worker` evals the
round's *starting* model at `i==0`, the others eval the merged model after the round.

**Why eval is read-only (zero-equivalence-risk).** Eval loads the merged `model_r`, scores it on the
val set, and writes nothing back to training (no RNG/data/weights). So `eval(model_r)` run async /
parallel / deferred / on-the-hot-engine yields a **bit-identical training trajectory** — unlike lever
#4 (which had to reproduce reset state). The only constraint is GPU memory.

**The resource physics.** vLLM *pre-reserves* `gpu_memory_utilization × total` per GPU as its KV
cache, **independent of model size**. Two vLLM engines on the *same* GPU collide: 0.6 + 0.6 > 1.0 →
the `cross_round` OOM (`Free 28.9 < desired 47.5 GiB`). So "eval ∥ train" needs either **disjoint
GPUs**, **a shrunk second engine**, or **the one hot engine** — hence four modes (`eval_mode`):

| mode | mechanism | when | cost |
|---|---|---|---|
| **inline** (default) | blocking eval after merge, on `n_gpus_per_node` GPUs | training **saturates** the node → eval gets the whole node in its window | eval cold-start, on the critical path; `cross_round`+inline OOMs → auto-fallback to per-round |
| **parallel** | eval on a **disjoint GPU subset** (`CUDA_VISIBLE_DEVICES`), concurrent with the next round's training; non-blocking launch + deferred collect | training uses **< node** GPUs (e.g. 2 of 4) → free async | needs `n_gpus_per_node + eval_gpus ≤ node`; off the critical path |
| **shared** | a **second** eval vLLM coexists on the worker's GPUs at reduced `eval_gpu_mem_util=0.3` | single node, `cross_round`, no spare GPUs | eval serial + a cold-start per round (small KV pool); keeps `cross_round` |
| **worker** | the cross-round worker evals the round's starting model on its **own hot vLLM** (verl `_validate()`) — no second engine | single node, `cross_round`, no spare GPUs (the paper's 4-GPU case) | **no OOM, no eval cold-start**; eval serial but cheap; needs the FSDP→vLLM weight sync (below) |

**inline auto-fallback.** `cross_round` + `eval_mode=inline` would OOM (worker holds the GPUs), so
`run()` falls back to per-round persistence. `parallel`/`shared`/`worker` keep `cross_round` speed by
isolating, shrinking, or reusing the engine.

**GPU-validated comparison** (2-client × 2-round WebShop, eval every round, 0.5B; cross-round base
= 342s no-eval, per-round-persistent base = 909s; training on 2 of the node's 4 GPUs):

| eval_mode | process base | wall-clock | val_curve r0/r1/r2 | eval OOM? | needs spare GPUs? |
|---|---|---|---|---|---|
| **inline** | per-round persistent | 1018s | −0.6 / −0.6 / −0.6 | n/a (serial) | no |
| **parallel** | cross-round | **690s** | −0.6 / −0.6 / −0.6 *(identical)* | none (disjoint GPUs) | **yes (≥2)** |
| **shared** | cross-round | 874s | −0.6 / −0.6 / −0.6 *(identical)* | none (0.3 util) | no |
| **worker** | cross-round | **703s** | −0.6 / −0.6 / −0.6 *(identical)* | none (one engine) | no |

All four `val_curve`s are byte-identical — eval is read-only, so `eval_mode` changes only *where/when*
eval runs, never the result. `rc=0` for all four (`shared`/`worker` emit benign `__del__` DataLoader
teardown noise on shutdown, run still closes clean).

**Saturated 4-GPU paper case → `worker`.** With `n_gpus_per_node=4` there are no spare GPUs, so
`parallel` is N/A *as 4-train* (it can still run as **2 train + 2 eval**, below) and `shared` pays a
per-round eval cold-start. `worker` reuses the **hot** rollout engine (no second vLLM → no OOM, no
cold-start), so per-round eval is cheap and `cross_round` speed is kept. Validated end-to-end (**703s**
vs shared **874s** on the 2-round smoke; refined at 1.5B/n=500 below, where `shared` is in fact *slowest*).

**1.5B paper-settings, 4-card comparison (GPU-validated).** The 0.5B table above floors at −0.6; re-run
at **paper settings** (1.5B, G=8, `webshop_15` 15-turn, response 512, **n=500 val**, 100-client uniform
partition 2/round, seed 42, 2 rounds), with every mode using the **full 4-card node** — three at 4 training
GPUs, `parallel` at **2 train + 2 eval**. (`n_gpus_per_node` is not an algorithm parameter — FSDP across
2 vs 4 GPUs is the same math — so all four stay paper-algorithm-consistent.)

| eval_mode | GPU layout | wall-clock | rc |
|---|---|---|---|
| **parallel** | 2 train + 2 eval | **2493s** | 0 |
| **worker** | 4 train (hot-engine eval) | 2637s | 0 |
| **inline** | 4 train (blocking eval after merge) | 3090s | 0 |
| **shared** | 4 train + 2nd eval engine @ 0.3 util | **3316s** | 0 |

All four run clean — **no OOM** even for `shared`'s coexisting 2nd engine and `parallel`'s split.
Ranking `parallel < worker < inline < shared` — two things the 0.5B floor hid:
- **`shared` flips to slowest at a large val set.** At 0.5B/n=8 `shared` (874s) beat `inline` (1018s);
  at 1.5B/**n=500** `shared` (3316s) is *slowest*, past `inline` (3090s). Cause: `shared`'s reduced-KV
  (0.3-util) eval engine caps batch concurrency, so a 500-episode eval is throttled — a penalty that
  **scales with val-set size** and was invisible at n=8. So `shared` is the wrong pick when val is large.
- **`parallel` wins by hiding the expensive eval.** Its full-util eval engine on the disjoint 2 cards
  overlaps the next round's training (n=500 eval off the critical path); `worker` is a close 2nd (full-util
  hot engine, serial but cheap); `inline` pays per-round cold-start **and** blocks on the eval.

The val numbers vary across modes by **eval sampling** (temp=0.4, n=500 but only 3–25 successes) and
eval-path, not training: cross-mode **weight equivalence** was confirmed directly (worker vs inline 1.5B
aggregates, max|Δ| 3.8e-6 / 7.6e-6), so `eval_mode` still never changes the trajectory.

**What `worker` needed** (verl-lifecycle fixes, [persistent_task_runner.py](../fed/persistent_task_runner.py)).
verl's `_validate()` is built to run *inside* `fit()`'s engine lifecycle; driving it from the persistent
loop means reproducing that lifecycle:
1. **`global_steps`** — `_validate()` reads it for the step label; only `fit()` sets it → seed `=0` when absent.
2. **FSDP→vLLM weight sync** *(the real CUDA-crash root cause)* — verl inits the rollout vLLM with
   **dummy** weights and leaves it **asleep** after `init_workers` (`ray_trainer.py:972`); real weights are
   synced by `checkpoint_manager.update_weights` per rollout. The worker-eval runs *before* the round's
   `fit()`, so without the sync vLLM holds dummy weights → **CUDA illegal-memory-access / EngineDeadError**.
   Fix: `update_weights` (sync+wake) before `_validate`, `sleep_replicas` after — mirroring `fit()`'s
   `ray_trainer.py:1387`. (`enforce_eager` only *moved* the symptom — not the fix.)
3. **dump executor** — each `fit()` shuts down verl's dump `ThreadPoolExecutor` at its end; the next round's
   worker-eval would submit to a dead executor (`cannot schedule new futures`). Fix: re-init if shut down
   before `_validate`, exactly as `fit()` does at `ray_trainer.py:1369`.
4. **`val_batch_size`** — honor `config.data.val_batch_size` (stock verl) instead of `len(val)`, so a full
   WebShop/ALFWorld val set isn't fired in one batch (the env-service storm).

**Per-client "client-end" circles (`client_end_eval`, default off).** The red line scores the round's
*aggregate*; the paper also plots each client's **post-training** model as a circle (one per client per
round). Enabling `client_end_eval: true` adds `clients_per_round` evals/round on the **unperturbed val
set** and emits a `client_curve` (one entry per `(round, client)`) in `federated_summary.json`,
alongside `val_curve`. Two paths, by `eval_mode`:
- **orchestrator** (inline/parallel/shared): `eval_client` merges the client's trained actor to
  `round_<r>/client_<c>/hf`, then scores it through the normal `eval_global` path against the
  **unperturbed val service** — sidestepping the within-job routing problem (the env can't tell a train
  rollout from a val rollout to swap service URLs mid-job, so the client's *own* job can't self-eval on
  the clean set). Must run **before** `cleanup_round_checkpoints` (it reads the client shards; the merged
  `hf` survives).
- **worker** (hot engine): after each client's `fit()`, `_worker_validate(r, client_id=c)` scores the
  just-trained model on the **hot** rollout engine — no merge, no second service.

GPU-validated on the 2×2 WebShop smoke for **both** paths: `client_curve` = 4 circles
(r1c0, r1c1, r2c0, r2c1, all `−0.6`), matching the 3-point red line `val_curve` (r0 base, r1/r2
aggregate). Off by default — the red line is the headline curve; circles are opt-in diagnostics.

### 7.5 Windowed-default release blocker — FIXED
The new `rollout_mode=windowed` default crashed on the stock `agent.yaml`
(`AttributeError: GymTextAgentLoop has no run_episode_windowed`, §2.6). Fixed **without** editing the
7 env specs or regenerating the 176 paper configs:
- [agent.yaml](../config/agent.yaml) registers `gym_text_windowed` → `WindowedGymTextAgentLoop`.
- [windowed_manager.py](../agent_loops/windowed_manager.py) `_run_agent_loop` **auto-maps**
  `agent_name=gym_text → gym_text_windowed`, so one shared env spec drives both modes.
- `FEDAGENT_HISTORY_LENGTH` (set by `run_fed` per rollout_mode: **windowed=2 / concat=0**, authoritative
  over the spec; read by `alfworld_env`/`webshop_env`) makes the same spec faithful in either mode —
  windowed gets the paper's 2-history template, concat gets `history_length=0` (the GymTextAgentLoop owns
  history). **✅ GPU-validated:** the windowed-default config that previously crashed now runs the entire
  federated loop end-to-end (4 steps, FedAvg + merge, `FEDERATED LOOP CLOSED` rc=0, no `AttributeError`).

### 7.6 Remaining (ranked)
1. **Larger-step / real-env equivalence A/B** — equivalence is verified at 2 steps on TinyGuess
   (max\|Δ\|=1.13e-5); confirm at more steps and on a real-env persistent A/B.
2. **Paper configs:** regenerated to the windowed `response_length=512` budget (176 files, §8.1); a
   real paper-scale (1.5B) persistent run is the final integration check.
3. **vLLM sampler-RNG / /dev/shm teardown** at long horizons (benign teardown noise — `DataLoader
   worker killed` / `resource_tracker KeyError` at exit, rc still 0; watch + add
   `aggressive_empty_cache` / `SamplingParams.seed` if it ever bites). *(To quiet the `DataLoader
   worker killed` line specifically: drop `data.dataloader_num_workers` 8 → 0/2 in the smoke/accel
   profile — it's a teardown-only `__del__` SIGKILL, rc unaffected.)*

### 7.7 "2 train on 1 GPU + 2 GPU eval" layout — correctness OK, **not** the fast path
**Question (a hardware-allocation idea):** on the 4-GPU node, is **2 clients each on 1 GPU (parallel) +
eval on the spare 2 GPUs** — i.e. **#3 ⊗ #1** — faster than keeping 2 GPUs/client? It *hides* eval on
dedicated cards, which #3-at-full-4-GPU cannot. GPU-tested at 1.5B, paper settings, n=64 WebShop val.

**What it proves (all GPU-confirmed):**
- **3-job 4-GPU coexistence works** — 2 trainers (1 GPU each) + 1 eval (2 GPU) load on disjoint cards
  (peak A/B 26 GB, eval 40 GB), no OOM — **but only after** the weight-transfer-socket fix (§Lever #3,
  *second* bug; this layout is exactly what exposed the 3-job deadlock).
- **Eval is genuinely hidden:** the standalone val pass (faithful — drives `run_fed._build_eval` → the
  same verl val-only code path the loop's eval uses) finished in **407 s** while training ran **995 s**,
  so eval held the spare cards &lt;½ the round and cost **zero** extra wall-clock.

**But it is not the fast path** — 1-GPU training dominates:

| layout (4-GPU node, M=2, per-round) | wall-clock |
|---|---|
| 4-GPU solo client (no client-parallelism) | 558 s |
| **#3 — 2 client × 2 GPU concurrent** (train only) | **727 s** |
| #3 + **`eval_mode=worker`** (hot-engine eval, no eval cold-start) | **≈ 845 s** |
| #3 + serial (cold-start) eval | 727 + 407 ≈ 1134 s |
| **this layout — 2 client × 1 GPU + 2-GPU eval (hidden)** | **995 s** |

1-GPU training measured **~180–226 s/optimizer-step**, **995 s** for the round — vs **725 s** at 2 GPU
(**1.37× slower, not 2×**: the WebShop rollout is **env-latency-bound**, so halving GPUs barely moves the
rollout; only the FSDP fwd/bwd + vLLM generation scale, and those are a minority of the wall-clock).
Hiding eval saves ~407 s, but the 1-GPU penalty (995 − 727 = **+268 s/round**) is paid **every round**,
and the right baseline isn't *serial* eval — it's **`eval_mode=worker`**, which already makes eval nearly
free (hot engine, no second cold-start). So **#3 (2 GPU/client) + worker eval (≈ 845 s) beats
this layout's 995 s**: don't starve training to 1 GPU just to free eval cards. **Verdict: correctness-OK
and a clean demonstration that eval hides on dedicated GPUs — but #3 + worker eval is the recommended
single-node fast path; the 1-GPU split is not a default.** The investigation's lasting value is the
**weight-transfer fix** above, which hardens #3 and eval-parallel for *every* concurrent-job layout.

> **⚠️ 2026-07-01 correction (§9.1):** the "env-latency-bound" explanation and the "1.37×" figure
> were **wall-ratio inferences** on a 3-step round diluted by ~390 s of fixed overhead. WebShop's
> first `timing_s` decomposition shows the opposite: WebShop is **GPU-compute-bound** (74 % of a
> 1-GPU step) and the **per-step** 1-GPU penalty is **2.41×** — the verdict above (don't starve
> training to 1 GPU) gets *stronger*. It is ALFWorld whose rollout is env-bound (§9).

---

## 8. Implementation reference (this session's changes)

Everything below is **overlay-only** — with **one deliberate exception**: a **2-line patch to
`others/verl`** (`vllm_rollout.py` + `vllm_async_server.py`) so the FSDP→vLLM weight-transfer socket
honors a `VERL_RAY_JOB_ID` override (§7.7 / §Lever #3). verl itself stays **pristine upstream**; the two
lines are captured as
[`tools/verl08_migration/patches/verl_weight_transfer_jobid.patch`](../../tools/verl08_migration/patches/verl_weight_transfer_jobid.patch)
(applied at env setup, base commit `7aed6b2`). That is a genuine **upstream-assumption fix** (verl's
per-job isolation assumes one shared Ray cluster) and is PR-worthy; everything else remains no-fork.

### 8.1 Files added / changed

**New (lever #4 — persistent trainer):**

| file | purpose | key symbols |
|---|---|---|
| [persistent_patch.py](../fed/persistent_patch.py) | attach per-client reset methods to verl worker classes via a deferred import hook | `reload_client_model` (ActorRolloutRefWorker), `reload_critic_model` (TrainingWorker), `install_deferred_persistent_patch()` |
| [persistent_task_runner.py](../fed/persistent_task_runner.py) | `PersistentFedTaskRunner(TaskRunner)`: init_workers once, fit-per-client; **cross-round outer loop**; **per-client service routing**; **`eval_mode=worker`** hot-engine eval + **client-end circles** (§7.4) | `run()`, `_reset_for_client(spec)`, `_wait_next_round(xdir,r)`, `_route_service(spec)`, `_worker_validate(r, client_id=None)` (update_weights+global_steps+dump-executor+sleep_replicas; `client_id` → client-end circle), `_should_worker_eval(r)` (every-round gate) |
| [persistent_main.py](../fed/persistent_main.py) | Hydra entry → `run_ppo(config, task_runner_class=ray.remote(...)(PersistentFedTaskRunner))` | `main()` |
| [compare_fsdp_checkpoints.py](../../tools/verl08_migration/compare_fsdp_checkpoints.py) | tensor-by-tensor FSDP-checkpoint equivalence diff | `compare_dir(a,b,atol)` |

**Changed:**

| file | change |
|---|---|
| [sitecustomize.py](../../sitecustomize.py) | gated `FEDAGENT_PERSISTENT=1` → `install_deferred_persistent_patch()` (every Ray worker gets the reset methods) |
| [run_fed.py](../fed/run_fed.py) | **#4 per-round:** `persistent` flag + `run_round_persistent()` + run-loop branch. **#4 cross-round:** `cross_round` flag + `BgProc` (line-buffered log) + `_wait_signal` + `stop_persistent_cross_round` + signal-file handshake. **routing:** `client_service_url()` + plan `service_url` + `FEDAGENT_SERVICE_URL_FILE`. **eval modes (§7.4):** `eval_mode` inline/parallel/shared/worker — `_build_eval`/`eval_global`/`launch_eval_async`/`collect_eval`; per-round eval **every round** (`if do_eval:`, not `r%test_freq`); `cross_round`+inline → auto-fallback to per-round. **client-end circles (§7.4):** `client_end_eval` flag + `eval_client()` (merge client actor → `client_<c>/hf` + eval on val service, before cleanup) + `merge_to_hf(out_hf=)`/`_build_eval(client_id=)`; emits `client_curve` in summary. **metrics:** flush BgProc + parse the launch log (cross-round) so `metrics.json` isn't `[]`. **#2:** `prewarm_next_round_services()`. **windowed:** `history_length_env()`. **#3 (concurrent aggregation, §Lever #3):** `fedavg()` uses `torchrun --standalone` + clears `MASTER_*`/`RANK`/`WORLD_SIZE` so two clients/experiments aggregating on one node don't collide on the default rendezvous port 29500. **#3/#1 (concurrent weight transfer, §7.7):** module-level `_RUN_TAG` (uuid) + `env["VERL_RAY_JOB_ID"]` set on every verl launch (train/eval/persistent, `_RUN_TAG`+role/client/round) so isolated Ray clusters (all `job_id=01000000`) don't collide on the `/tmp` FSDP→vLLM weight-transfer ZMQ socket — pairs with the 2-line verl honor-override patch. **logging:** `stream()` now `lf.flush()`es each line so an inner log isn't 0-byte during a run or hang |
| [fedagent_ppo.yaml](../config/fedagent_ppo.yaml) | added `critic:` block (PPO/gae value-model micro-batch; inert under GRPO) |
| `fedagent/config/paper/*.yaml` (176) | **regenerated** to the windowed `response_length=512` budget (was `6144`/`8192`); via `tools/verl08_migration/gen_paper_configs.py --out fedagent/config/paper` |
| [base.py](../envs/base.py) | `resolve_service_url(env_var, cfg, default)` — file-channel routing helper (file > env-var > config > default) |
| [agent.yaml](../config/agent.yaml) | registered `gym_text_windowed → WindowedGymTextAgentLoop` |
| [windowed_manager.py](../agent_loops/windowed_manager.py) | `_run_agent_loop` auto-maps `agent_name → {name}_windowed` |
| [alfworld_env.py](../envs/alfworld/alfworld_env.py), [webshop_env.py](../envs/webshop/webshop_env.py) | `history_length` reads `FEDAGENT_HISTORY_LENGTH`; service URL via `resolve_service_url` (per-client file routing) |

### 8.2 Persistent trainer — end-to-end walkthrough

**(1) The multi-process patch problem & fix.** `reload_client_model` must exist on the worker class
*inside every Ray FSDP-worker process*, not just the driver. Importing `verl.workers.engine_workers`
eagerly at interpreter start pulls the FSDP engine in before Ray sets per-rank `CUDA_VISIBLE_DEVICES`
→ "Duplicate GPU detected". So `install_deferred_persistent_patch()` arms a one-shot `MetaPathFinder`
that wraps `engine_workers`'s `exec_module` and attaches the methods the moment verl imports it (after
device assignment) — the exact FedProx deferral pattern. `sitecustomize.py` calls it in every process
with `FEDAGENT_PERSISTENT=1` + verl on PYTHONPATH. *Validated:* `[persistent] … attached` printed in the
driver, the TaskRunner actor, and all 4 FSDP workers.

**(2) The plan.** `run_round_persistent` writes `round_<r>/persistent_plan.json` =
`[{client, model_path, critic_path, seed, out_dir, exp}, …]` for the round's selected clients (all share
the round's `model_path`/`critic_path`; `seed = base_seed + round*100 + client`, **identical to**
`run_client`). It launches ONE `persistent_main` with `FEDAGENT_PERSISTENT=1` +
`FEDAGENT_PERSISTENT_PLAN=<path>`, then scans `round_<r>/client_<c>/checkpoints/global_step_*/actor` and
returns `{client: (actor_dir, critic_dir)}` so the **existing** `fedavg()`/`merge_to_hf()` run unchanged.

**(3) Per-client reset** (`_reset_for_client` — reproduces what a fresh subprocess gets for free):

| reset | how | verl anchor |
|---|---|---|
| actor weights+optimizer+scheduler | `reload_client_model` → re-point `engine.model_config.local_path` + `engine.initialize()` (rebuilds module(new weights)+optimizer(zero Adam)+scheduler in ONE call) | transformer_impl.py:183→543 |
| ref policy | same `_reset_engine(self.ref.engine)` (forward_only → weights only) | engine_workers.py:537 |
| critic (PPO) | `reload_critic_model` (TrainingWorker.engine rebuild) | engine_workers.py:165 |
| FedProx anchor | `del eng._fedprox_w_t` after reset (survives `initialize()`) | fedprox.py:37 |
| dataset/env seed | `os.environ[FEDAGENT_BASE_SEED]=seed` then `_create_dataloader(None…)` (driver-side) | ray_trainer.py:374 |
| global_steps | stock `fit()` sets `=0` then `+=1` (free) | ray_trainer.py:1383 |
| vLLM weights + KV/prefix | stock `fit()` `update_weights()` after the reload (full `load_weights` + cache flush) | ray_trainer.py:1387 |
| driver RNG | `random/np/torch.manual_seed(seed)` + `torch.cuda.empty_cache()` | — |

Then `trainer.fit()` runs E epochs and saves the per-client checkpoint.

### 8.3 Windowed release-blocker — mechanism

Three coordinated pieces let the **same** env spec (`agent_name: gym_text`, no `history_length`) drive
both rollout modes, so **no env-spec edit and no 176-config regeneration**:
1. **Registry** — `agent.yaml` now has `gym_text_windowed → WindowedGymTextAgentLoop` (the subclass with
   `run_episode_windowed`).
2. **Auto-map** — `windowed_manager._run_agent_loop` resolves `f"{agent_name}_windowed"` in
   `_agent_loop_registry` (fallback `agent_name`). The concat path (stock manager) keeps using `gym_text`.
3. **History** — `FEDAGENT_HISTORY_LENGTH` (run_fed: windowed=`windowed_history_length` default 2, concat=0;
   authoritative over the spec) read by `alfworld_env`/`webshop_env`. Windowed → the paper's 2-history
   per-turn template; concat → `0` so the `GymTextAgentLoop` owns the growing chat.

### 8.4 Reproduce

```bash
# lever #4 A/B (full 2-round federated GRPO, TinyGuess, 4 GPU): persistent vs subprocess
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/dev/persist_full.yaml   # persistent: true
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/dev/subproc_full.yaml   # persistent: false
python tools/verl08_migration/compare_fsdp_checkpoints.py \
  --a .../subproc_full_out/round_2/aggregated/checkpoints/global_step_0/actor \
  --b .../persist_full_out/round_2/aggregated/checkpoints/global_step_0/actor --atol 1e-4

# windowed-default no-crash check (rollout_mode defaults to windowed):
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/dev/tinyguess_windowed_check.yaml

# enable lever #4: `persistent: true` (per-round) or `cross_round: true` (one process, whole run).
```

---

> **Status note (this session):** the windowed-default release blocker (§7.5) and lever #4 — per-round
> persistence (§7.1), **cross-round persistence (§7.2)**, PPO critic reload (§7.1), **per-client
> service routing (§7.3)** — are **done + GPU-validated** (cross-round on TinyGuess: ONE cold-start/run,
> −62%, max\|Δ\|=1.13e-5 EQUIVALENT; routing on the **real `verl-agent-webshop`** env: 32/32 episodes to
> distinct per-client services), overlay-only, **uncommitted**.
>
> **The five review gaps — resolved this session:** (1) **real-env routing** ✅ GPU-validated (§7.3);
> (2) **cross-round + `val_env_spec`** ✅ — GPU-confirmed that the GPU-holding worker OOMs eval's vLLM,
> now **auto-falls-back to per-round persistence** (§7.4), keeping the eval curve; (3) **paper configs**
> ✅ regenerated to the windowed `response_length=512` (176 files, was `6144`/`8192`, §8.1); (4)
> **metrics drop** ✅ fixed — was `BgProc` buffering the log (line-buffer + flush-before-parse + cross-round
> launch-log path; parser itself was fine); (5) **teardown noise** — `DataLoader worker killed` /
> `resource_tracker KeyError` at exit is **benign** (rc=0), documented in §7.6. The windowed
> *history-length* migration is superseded by auto-map + `FEDAGENT_HISTORY_LENGTH`; the
> *response-length* regen in (3) is the separate, now-done part. **Remaining:** a paper-scale (1.5B)
> persistent run + a larger-step equivalence A/B (§7.6).

---

## 9. Tier-1 — env-service replica sharding (2026-07-01, GPU-validated end-to-end)

**The finding that motivated it.** §2.2 measured ALFWorld's `_TW_LOCK` at **86 ms/step** on a
160-step batch (13.7 s) and filed it as "understood and bounded." At **paper-scale batches that
qualitative call inverts**: a windowed ALFWorld training step is 64 episodes × ~50 turns ≈ **3200
env steps**, and 86 ms × 3200 ≈ 275 s — the measured `timing_s/gen` (219–228 s, **flat across
1/2/4 GPU**) is almost exactly the lock-serialized env time. The lock's design comment assumed
"env transitions are ms-fast vs *seconds* of LLM generation" — but windowed responses average
**~100 tokens/turn** (measured), so the LLM hides under the lock, not the reverse. **gen ≈ the
lock**, 73 % of a 4-GPU step.

**The fix (implemented, both envs).** Run **K identical service processes per client** over the
SAME shard — K processes = K independent parsers/locks; sessions spread round-robin client-side:
- [envs/base.py](../envs/base.py) `resolve_service_url` accepts a **comma-separated replica list**
  from any routing source (URL-file / process-env / spec); each env instance binds one replica
  round-robin (PID-offset cursor, per-worker balance ±1) and stays sticky for its episode.
- [fed/run_fed.py](../fed/run_fed.py): `alfworld_replicas` / `webshop_replicas` (default 1 =
  byte-identical legacy). Train **and val** services replicate; ports `base + c*K + j`; the
  configured pool is split ~evenly (+2 slack); the val/client port-collision guard is band-aware.
- **Equivalence:** every replica gets the client's identical shard env-vars → identical game/goal
  distribution, iid sampling; which replica an episode lands on changes scheduling only (same class
  as the existing pool-borrow order). update_actor/old_log_prob/ref were bit-untouched by design and
  measured unchanged.

**Validation chain (all GPU-measured, 1.5B, response 4096, batch 8×8):**

| level | experiment | result |
|---|---|---|
| mechanism | same-node K-sweep (1×H100, pool 64) | gen **217.5 (K1) → 65.8 (K4) → 61.8 (K8)** |
| control | K=1 + pool 64 vs pool 8 baseline | gen 217.5 ≈ 228 → **pool size irrelevant; the lock is the whole story** |
| component | `alf_scale_g4_r8` (4×H100, K=8) | gen **219.3 → 51.7 (−76 %)**, step **298.4 → 127.6 (−57 %, 2.34×)**; update_actor 43.3→43.7 (untouched ✓) |
| component | `alf_scale_g1_r8` (+ node-1 K4/K8) | step **534.5 → 350–359 (−33 %)** on BOTH nodes; K=4 suffices on an 8-core node |
| end-to-end | `alf_em_worker_r8` — same config as the 3509 s baseline + `alfworld_replicas: 8` | **2412 s (−31 %)**; training steps 359→**127 s mean (−65 %)**; rc=0, val healthy |

Residual gen ≈ 52–66 s = the new floor (episode critical path: ~50 turns × (LLM ~0.2–0.3 s + env
86 ms/K + HTTP)); more replicas stop paying once per-replica serial load < that path (K=4–8).

**Strategic consequence — the 1-GPU story dies post-fix.** With the env floor removed, GPU compute
dominates and ALFWorld's per-step 1-GPU penalty grows **1.79× → 2.81×** (358.8/127.6). The ALFWorld
production recipe is `cross_round + eval_mode=worker + alfworld_replicas=8` on 4 GPUs (or 2×2 #3).

### 9.1 WebShop decomposition — the OPPOSITE bottleneck (first measurement; corrects §7.7's inference)

`ws_scale_g{1,g1b,g4}` (1 step, eval off, paper settings, pool 16) — WebShop had never had a
`timing_s` decomposition; §7.7 *inferred* "env-latency-bound" from wall ratios. The decomposition
says otherwise:

| WebShop | gen | old_log_prob | ref | update_actor | GPU-compute Σ | step |
|---|---|---|---|---|---|---|
| 1×H100 (same node) | 54.6 (24 %) | 26.6 | 39.1 | 100.1 | **165.8 (74 %)** | 225.2 |
| 4×H100 | 44.1 (47 %) | 10.2 | 8.8 | 27.9 | 46.9 (50 %) | 93.4 |

- **WebShop is GPU-compute-bound** (74 % at 1 GPU) — the mirror image of ALFWorld's 73 % env.
- gen is flat-ish (54.6→44.1) but *small*; GPU compute scales 3.54×.
- **Correction:** the earlier "1-GPU only 1.37× slower" was **fixed-overhead dilution** of a 3-step
  wall (995 ≈ 3×202 + ~390 overhead reconciles exactly). Per-step the WebShop 1-GPU penalty is
  **2.41×** (225.2/93.4). §7.7's layout verdict (don't starve training to 1 GPU) gets *stronger*.
- Node effect ruled out: 1-GPU probe on the 8-core node = 202.1 s vs 225.2 s on the 64-core node (±10 %).

**WebShop levers measured** (4-GPU): pool 16→64 alone **hurts** (gen 44.1→50.1 — more concurrent
sessions in ONE process amplifies GIL contention; the "wave-throttle" hypothesis is refuted);
pool 64 + `webshop_replicas: 4` → gen **35.7**, step **82.2 (−12 %)** — GIL sharding is real but
modest. WebShop's big lever remains GPU compute (#3 / more GPUs); replicas are a free garnish.

### 9.2 Reproduce

```bash
# ALFWorld replica probes (4-GPU K=8, 1-GPU K=8, control K=1/pool64):
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/alfworld/alf_scale_g4_r8.yaml
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/alfworld/alf_scale_g1_r8.yaml
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/alfworld/alf_scale_g1_r1n1.yaml
# end-to-end A/B vs the 3509s worker baseline:
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/alfworld/alf_em_worker_r8.yaml
# WebShop decomposition + levers:
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/webshop/ws_scale_g4.yaml       # + g1, g1b
python -m fedagent.fed.run_fed --config tools/verl08_migration/accel/webshop/ws_scale_g4_p64r4.yaml # + g4_p64
```
