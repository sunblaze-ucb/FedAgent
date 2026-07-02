# Agent-RL engine design — rollout, async, acceleration, infra

> **What this document is.** The design deep-dive of FedAgent's **agent-RL subsystem**: how a
> multi-turn episode becomes training data (rollout design), how the system stays busy
> (the async model and where its bounds are), how it got fast (the acceleration architecture as a
> stack of measured levers), and what runs it (services, isolation, orchestration, ops infra).
> Companions: [architecture.md](./architecture.md) (system overlay, code map, round loop),
> [acceleration.md](./acceleration.md) (the lever analyses; §9 = replica sharding),
> [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md) (the dated
> deep-validation report), [migration.md](./migration.md) (fidelity vs the verl-agent 0.3.1 fork).
>
> **Fidelity bar for every design choice below:** SCIENTIFIC EQUIVALENCE with the paper — same
> algorithm (GRPO, G=8), same per-turn prompting, reproduce conclusions within 3-seed noise. Speed
> is only taken where it provably does not touch the science.

---

## 1. The problem shape

FedAgent trains **LLM agents** (Qwen2.5-1.5B class) with RL on **multi-turn text environments**
(WebShop 15-turn, ALFWorld ≤50-turn), inside a **federated** wrapper: N clients × T rounds, each
round = local GRPO/PPO training per client → FedAvg of FSDP checkpoints → next round starts from
the aggregate. Three properties dominate the design:

1. **Episodes are long and env-mediated** — a rollout is a conversation with an external
   environment process, not a single generate() call. Latency lives on both sides.
2. **The original science is per-turn (“windowed”)** — the paper's agent sees a *sliding window*
   prompt each turn, not the growing full history. Faithfulness forces a custom rollout mode.
3. **Federation multiplies everything** — ~140 (client × round) training jobs + evals per paper
   run. Any fixed cost is paid ~140× unless the infra amortizes it.

## 2. Design overview — three planes

```
┌─ ORCHESTRATION plane (fedagent/fed/run_fed.py) ─────────────────────────────────────────┐
│ round loop · client subprocess/persistent lifecycle · FedAvg(torchrun) · merge · eval    │
│ service lifecycle (per-client + val, K replicas) · ports/isolation · metrics/summary     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
          │ launches (subprocess or in-process persistent)            │ starts/stops (HTTP health)
┌─ TRAINER plane (STOCK verl 0.8, per client-round) ──────────┐  ┌─ ENV-SERVICE plane ─────────┐
│ Ray cluster (isolated) · RayPPOTrainer · FSDP actor/ref     │  │ FastAPI/uvicorn per client   │
│ vLLM server-mode rollout engines (+CUDA graphs)             │  │ (× K replicas) · env pool    │
│ AgentLoopManager → WindowedAgentLoopWorkers (async)         │  │ sticky sessions · own conda  │
│   └─ GymTextAgentLoop coroutine per trajectory ──── HTTP ───┼──► env (per-worker textworld /  │
│ FSDP→vLLM weight sync (ZMQ bucketed transfer)               │  │ webshop gym) · partition vars│
└──────────────────────────────────────────────────────────────┘  └──────────────────────────────┘
```

- **Trainer plane is stock verl 0.8** — no fork. FedAgent contributes agent loops, env clients,
  and lifecycle drivers registered through verl's own extension points (one 2-line verl patch
  exists, for the weight-transfer socket namespace; see §8.3).
- **Envs live behind HTTP** (`BaseTextEnv` → thin async client → remote service). This is the
  central boundary: it isolates dependency hells (WebShop's gym-0.24/pyserini/Java, ALFWorld's
  textworld stack live in their *own* conda envs), gives federation a natural per-client unit
  (one service = one client's hidden transition kernel), and makes every env-side fix
  trainer-invisible.
- **Orchestration owns every lifecycle** — subprocesses, services, aggregation, eval — and is the
  only place that knows about federation.

## 3. Rollout design

### 3.1 The per-row async contract

verl 0.8's agent-loop runs **one coroutine per dataset row**. FedAgent's
[`GymTextAgentLoop`](../agent_loops/gym_text_agent_loop.py) implements the episode loop:

```
system_prompt → reset(seed) → [ generate(turn prompt) → env.step(action) → build next prompt ]* → done
```

- `env.reset/step` are `await`-able (`BaseTextEnv`, [envs/base.py](../envs/base.py)); the env
  instance is constructed **per episode** and stays bound to one service session (sticky).
- Seeding is client-indexed (`base_seed + round*100 + client`), so client *order* never matters —
  a precondition for client-parallelism (#3).
- Rewards/success come back through `step()`'s info; the loop records the trajectory and returns
  token ids + masks to verl.

### 3.2 Windowed vs concat — the faithfulness axis

| mode | prompt each turn | samples/episode | fidelity | cost |
|---|---|---|---|---|
| **windowed** (default; the paper) | task + last-`history_length` (obs, action) pairs + current obs | **one per turn** (`run_episode_windowed`) | faithful to verl-agent 0.3.1 | every turn is a vLLM **prefix-cache miss** (~1.43× gen vs concat) |
| concat (stock verl) | full growing history | 1 | stock behavior, opt-in | prefix-cache hits |

Mechanics worth knowing:
- The **windowed manager** ([windowed_manager.py](../agent_loops/windowed_manager.py)) replaces the
  stock manager when `rollout_mode=windowed`; it **auto-maps** `agent_name=gym_text →
  gym_text_windowed` so ONE env spec drives both modes; `FEDAGENT_HISTORY_LENGTH` (set by run_fed:
  windowed=2, concat=0) overrides the spec so the same file is faithful in either mode.
- **Batch math:** `train_batch_size=8 × rollout.n=8` = 64 episodes; windowed slicing turns them
  into **~3200 per-turn rows** on ALFWorld (measured: `adopted 3184 per-turn rows`) — each row
  ~440-token prompt + ~100-token response. The optimizer step trains on the *windows*, GRPO groups
  by episode (G=8 per goal).
- **Consequence:** per-turn responses are tiny (~100 tokens) → LLM time per turn is ~0.2–0.3 s →
  the *env* side, not generation throughput, is the natural bottleneck candidate (§5, §6).

## 4. The async model — three layers, and where the bounds are

**Layer 1 — trajectory coroutines (unbounded).** All 64 (up to 512 at paper batch) episodes run as
independent asyncio tasks; there is deliberately **no client-side semaphore**. While trajectory A
awaits `env.step()`, B..N generate; while A generates, B..N step. Env latency and LLM latency hide
under each other *across* trajectories.

**Layer 2 — vLLM dynamic batching.** All concurrent `generate()` calls funnel into the vLLM
server-mode engines (one per GPU group), which batch them continuously (CUDA graphs pre-captured,
prefix caching on). Generation throughput is effectively never the limiter at 1.5B.

**Layer 3 — env-service async handlers.** Each service is FastAPI with `async def` endpoints;
blocking env work runs in `asyncio.to_thread`. Concurrency is bounded here — by design and by
accident:
- **by design:** the env pool (`asyncio.Queue`, `*_pool_size`) caps live sessions; `/create`
  *blocks* until an env frees (intentional — see §5).
- **by accident (fixed 2026-07-01):** ALFWorld's `_TW_LOCK` serialized ALL env stepping in a
  process (tatsu PDDL parser is a process-global singleton); WebShop's pure-Python `env.step`
  contends on the **GIL**. One service process = one file line. **This was the hidden bound**: at
  paper batches it made ALFWorld's rollout 73 % of the training step, flat across GPU counts. The
  fix is replica sharding (§6.3) — K processes = K locks/GILs.

**The resulting latency model** (what actually gates `timing_s/gen`):

```
gen ≈ max( slowest episode critical path,  Σ env-steps / (K × per-process service rate),  LLM )
      └ ~50 turns × (LLM 0.2-0.3s + env 86ms + HTTP)   └ the lock/GIL term the replicas divide
```

Post-sharding the first term dominates (~52–66 s on ALFWorld); pre-sharding the second did (219 s+).

## 5. The HTTP boundary — contract details that keep 512 concurrent episodes correct

The env client ([webshop_env.py](../envs/webshop/webshop_env.py) /
[alfworld_env.py](../envs/alfworld/alfworld_env.py)) is deceptively small; each design point exists
because a failure mode demanded it:

| mechanism | why |
|---|---|
| **per-trajectory `httpx.AsyncClient`**, closed in `finally` | isolation per episode; no shared connection-pool state across trajectories |
| **retry w/ backoff+jitter on transport errors** (`/create`,`/reset`,`/step`) | at full batch, hundreds of episodes hit one service near-simultaneously; sockets reset mid-response (`httpx.ReadError`) — retries spread the stampede |
| **idempotency key `step_id` on `/step`** | `/step` mutates env state; a blind retry would double-apply. The server applies each id exactly once and replays the cached response — retry-safe without giving up exactly-once semantics. HTTP 4xx/5xx are *not* retried (a real desync must surface loudly). |
| **blocking `/create`** (read-timeout disabled, connect/write bounded) | borrowing a pooled env legitimately waits for a free env — that wait scales with batch/pool, not with a fixed timeout; a hard timeout here killed whole rollouts. No timeout → no re-send → no duplicate borrow. |
| **sticky sessions** (session_id → env, borrowed at `/create`, returned at `/close`) | episode state lives server-side in that env instance |
| **replica routing** (`_pick_replica`) | a service URL may be a comma list; each episode binds one replica round-robin (PID-offset cursor, per-worker balance ±1) — sticky thereafter. One implementation point covers both envs × all three routing sources. |

Routing priority (`resolve_service_url`): `FEDAGENT_SERVICE_URL_FILE` (persistent mode's per-client
re-pointing; a file because one process's `os.environ` can't vary per client) → process env var
(subprocess mode) → spec → default.

## 6. Env-service infra

### 6.1 Service anatomy (both envs, same shape)

FastAPI + uvicorn (one process), lifespan warm-up builds the env pool in parallel threads;
endpoints `/health /create /reset /step /close`; per-session `asyncio.Lock` (serializes retries of
the *same* session, not different sessions); per-request thread offload. The **partition env-var
bridge** (`CLIENT_ID/CLIENT_NUM/PARTITION_STRATEGY/OMEGA/...`) makes the service build exactly its
client's data shard at boot — heterogeneity is injected *here*, invisibly to the trainer.

### 6.2 Per-client services + the shared val service

One service per selected client per round (Design A: service == the client's hidden transition
kernel), lazily started per round, health-gated (`/health` polls with generous timeouts — ALFWorld
walks an 8810-game collection at boot, ~3 min). One **UNPERTURBED val service** (held-out split,
no partition) serves the every-round global eval; client and val port bands are guard-checked.

### 6.3 Replica sharding (`alfworld_replicas` / `webshop_replicas`, 2026-07-01)

The serialization bound of §4-layer-3, removed without touching the service code: run **K
identical processes per client** over the same shard (ports `base + c*K + j`; pool split ~evenly
+2 slack; val replicated with the same K). Same distribution → science-safe; K=1 is byte-identical
legacy. Validated mechanism→control→component→end-to-end: ALFWorld 4-GPU step **298→127.6 s**,
full run **3509→2412 s**; WebShop (GPU-bound) −12 %. Full data:
[the dated report](./acceleration_tier1_report_2026-07-01.md).

## 7. Trainer plane — stock verl 0.8, and the two lifecycle seams

- **Training:** stock `RayPPOTrainer`, FSDP actor (+ref; +critic for PPO/GAE), GRPO advantage over
  G=8 episode groups; `old_log_prob`/`ref` recomputed under FSDP (exactness over speed —
  deliberately not taken from vLLM logprobs).
- **Rollout engines:** vLLM in server mode (`vLLMHttpServer` per GPU group), dummy-load at init,
  then **FSDP→vLLM weight sync** per rollout via verl's bucketed **ZMQ IPC transfer**; engines
  sleep between rollouts and wake on sync.
- **Seam 1 — cluster-per-job vs verl's one-cluster assumption (§8.3):** verl namespaces shared
  host resources (the weight-transfer `/tmp` socket) by Ray job id, assuming one shared cluster.
  FedAgent runs each client/eval as its *own* Ray cluster, and isolated clusters all mint the same
  first job id → identical socket path → cross-wired weight sends (a 44-min silent deadlock).
  Fixed by a per-launch unique `VERL_RAY_JOB_ID` (+ the 2-line verl honor patch). The FedAvg
  `torchrun` rendezvous port was the same bug class (fixed with `--standalone`). **Design rule
  learned: every shared-host resource must carry a per-job-unique name.**
- **Seam 2 — driving `_validate()` outside `fit()` (worker eval):** verl's validation assumes
  `fit()`'s engine lifecycle. The persistent worker eval reproduces it: seed `global_steps`, run
  `update_weights` (sync+wake) *before* validating — vLLM otherwise still holds dummy weights
  (the root cause of a CUDA illegal-access class of crashes) — re-init the dump executor, honor
  `val_batch_size`, sleep engines after.

## 8. Federated orchestration & process lifecycles

### 8.1 Three lifecycle modes (lever #4)

| mode | processes | cold-start paid | when |
|---|---|---|---|
| subprocess (baseline) | one `main_ppo_fed` per (client, round) | ~140× per paper run (was **76–88 %** of wall) | maximal isolation; debugging |
| `persistent: true` | one process per round, in-process per-client reset | once per round | — |
| `cross_round: true` | **one process for the whole run** | **once** | production default |

The persistent runner re-points clients via the URL-file (§5), resets data/seeds in-process, and
keeps FedAvg/merge **external and byte-identical** — equivalence `max|Δ|≈1e-5` through compounded
rounds is the checked invariant.

### 8.2 Aggregation pipeline

`save FSDP shards (save_contents=[model]) → torchrun --standalone matched-PG FedAvg over shards →
verl.model_merger → HF dir → next round / eval load`. Exact averaging, order-free; PPO federates
the critic through the same path. (Known next optimization: direct shard-load to skip the HF merge
— see §9 "next".)

### 8.3 Isolation model (what makes same-node concurrency safe)

Per launched job: `CUDA_VISIBLE_DEVICES` (disjoint GPUs) + `RAY_TMPDIR` (own cluster) +
`VERL_RAY_JOB_ID` (own socket namespace) + port bands (client `base + c*K + j`, val `val + j`,
guard-checked) + `--standalone` FedAvg. Under this set, 2–3 concurrent verl jobs coexist cleanly
(GPU-validated; it is how #3 client-parallel and eval∥train run).

### 8.4 Eval design

Eval is **read-only** (loads merged model_r, writes nothing back) → zero equivalence risk by
construction, so it is freely movable: `inline` (blocking) / `parallel` (disjoint GPUs) / `shared`
(second engine at 0.3 util) / `worker` (the persistent trainer's hot engine — no second engine, no
cold-start). Cadence: the paper's red line = one eval of the round aggregate per round;
`client_end_eval` adds per-client circles. Mode choice is pure wall-clock (measured rankings differ
per env — WebShop parallel-first, ALFWorld worker-first).

## 9. The acceleration architecture — a stack of measured levers

Every lever attacks one term of `round ≈ cold-start + rollout + train-compute + eval`, and each
exposed the next bottleneck (full genealogy + data in the
[dated report](./acceleration_tier1_report_2026-07-01.md) §7):

| lever | term | mechanism | headline |
|---|---|---|---|
| #4 persistent / cross_round | cold-start | one process, hot-swap weights | −43 % / −62 % |
| eval modes (worker/parallel) | eval | hot-engine or disjoint-GPU eval | eval ≈ free off the critical path |
| #3 client-parallel | train-compute | sub-linear FSDP → 2×2 beats 4-serial | −35 % (WebShop) |
| concurrency fixes | (enabler) | per-job-unique names for shared resources | 3-job coexistence |
| **replica sharding** | rollout (env) | K service processes = K locks/GILs | ALFWorld step −57 %, run −31 % |

**Decision rule for a new environment** (the transferable method): run a 1-step `timing_s` probe at
two GPU counts. **Flat gen → env-bound → `*_replicas`. Scaling gen → GPU-bound → more GPUs /
#3.** Current recipes: ALFWorld 4×H100 = `cross_round + eval_mode=worker + alfworld_replicas: 8`;
1×H100 = `alfworld_replicas: 4`; WebShop = GPUs first, replicas optional (−12 %).

**Next (identified, not built):** inter-round plumbing (direct shard-load / in-process FedAvg /
service manifest cache — the −31 % vs −65 % gap), #3 × replicas composition (~−18 % predicted,
needs a parallel-round launcher), multi-node #3.

## 10. Ops infra (SLURM) — running and watching long jobs

Patterns that survived hard lessons (details in EXPERIMENTS.md / memory):

- **Durable launch:** run drivers in the *foreground of a long-lived `srun --overlap` step*. A
  `setsid nohup` orphan does NOT survive — Slurm's cgroup cleanup kills it when the launching step
  exits (setsid escapes the session, not the cgroup).
- **Self-queuing barrier files on GPFS:** each driver appends `[stage] rc=… wall=…` lines +
  `=== DONE ===` to a barrier file; the next driver spin-waits on it. Monitors read GPFS from the
  login node (no srun needed for reads).
- **Liveness = log mtime staleness + GPU util** — never `pgrep -f <pattern>` (it self-matches
  sibling watchers; caused a 53-min blind spot once).
- **Health noise triage:** `DataLoader worker killed` / `Engine core died` at teardown are benign
  `__del__` noise (rc unaffected); a silent 0 %-util + stale-log state is a real death.
- **Ports:** keep client/val/replica bands documented per config; fresh bands per experiment avoid
  stale binds.

## 11. Design principles (distilled)

1. **Overlay, not fork.** Stock verl 0.8; extension points only; one 2-line exception, carried as
   a patch file, honored-by-env-var, byte-identical when unset.
2. **The env lives behind HTTP.** Dependency isolation, per-client federation unit, and an
   acceleration surface (replicas) the trainer never sees.
3. **Faithfulness first, then speed.** Windowed mode pays 1.43× for the paper's per-turn prompt;
   speed comes from lifecycle/scheduling/services — layers that provably don't touch sampling.
4. **Eval is read-only — exploit it.** Anything read-only may be moved anywhere (hot engine,
   spare GPUs, deferred) at zero risk.
5. **Per-job-unique names for every shared host resource.** Ray job ids, rendezvous ports, tmp
   sockets — the recurring bug class of cluster-per-client designs.
6. **Measure → decompose → fix → verify at four levels.** Every lever earned its place via
   mechanism / control / component / end-to-end evidence with pre-registered predictions — and the
   two wrong predictions are documented alongside the hits.

## See also

- [architecture.md](./architecture.md) — system code map, round loop, one-subprocess anatomy
- [acceleration.md](./acceleration.md) · [acceleration_results.md](./acceleration_results.md) —
  lever analyses + numbers
- [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md) — the
  replica-sharding deep-validation report
- [acceleration_cross_env.md](./acceleration_cross_env.md) — WebShop vs ALFWorld synthesis
- [migration.md](./migration.md) — what changed vs the verl-agent 0.3.1 fork and why it's faithful
- 中文版: [agent_rl_design_cn.md](./agent_rl_design_cn.md)
