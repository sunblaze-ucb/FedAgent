# Deep-validation report (2026-07-01) — env-service replica sharding & the mirror-image bottlenecks

> **What this document is.** The self-contained report of the 2026-07-01 deep-validation campaign:
> what was measured, **what was built and why**, the complete experimental data, and **how the new
> improvement relates to every prior acceleration lever**. Companions:
> [acceleration.md](./acceleration.md) (the running analysis; §9 is this campaign's summary),
> [acceleration_cross_env.md](./acceleration_cross_env.md) (WebShop-vs-ALFWorld synthesis),
> [alfworld_testing.md](./alfworld_testing.md) (the ALFWorld strategy doc; §7),
> [acceleration_results.md](./acceleration_results.md) (numbers at a glance).
>
> **TL;DR.** The two benchmark environments have **mirror-image bottlenecks**: ALFWorld spends
> **73 %** of a 4-GPU training step inside one lock-serialized env-service process; WebShop spends
> **74 %** of a 1-GPU step in GPU compute. We built **env-service replica sharding**
> (`alfworld_replicas` / `webshop_replicas`) and validated it at four levels — mechanism, control,
> component, end-to-end: ALFWorld 4-GPU step **298 → 127.6 s (−57 %)**, full federated run
> **3509 → 2412 s (−31 %)**; WebShop (as predicted from its decomposition) gains only −12 %.
> Two prior published conclusions were corrected along the way.

**Constants for every number below** (unless stated): Qwen2.5-1.5B-Instruct, GRPO G=8, windowed
rollout, `train_batch_size=8 × rollout.n=8` = 64 episodes/step, seed 42, eval off for probes.
ALFWorld: `response_length=4096`, ≤50 turns, ~3200 env-steps/optimizer-step. WebShop: `webshop_15`
(15 turns), `response_length=512`, 960 env-steps/step. Hardware: 4×H100 (qgpu3021, 64 cores) and
1×H100 (qgpu3010, 8 cores). Probes are 1 optimizer step, single-run (±5–10 % noise).

---

## 1. Where we started — the prior improvement stack

Every earlier lever attacked one term of the per-round wall-clock equation
(`round ≈ cold-start + rollout + train-compute + eval`):

| lever (when) | term attacked | headline result |
|---|---|---|
| **#4 persistent / cross_round** (June) | cold-start (was **76–88 %** of wall) | −43 % per-round, −62 % cross-round; equivalence `max\|Δ\|≈1e-5` |
| **eval modes** inline/parallel/shared/worker (June) | eval placement | WebShop n=500: parallel 2493 < worker 2637 < inline 3090 < shared 3316 |
| **#3 client-parallel** (June) | train-compute via sub-linear FSDP | 2×2 GPU = 727 s vs sequential 1116 s (−35 %); exposed + fixed two concurrency bugs (FedAvg :29500, ZMQ `/tmp` socket `VERL_RAY_JOB_ID`) |
| **ALFWorld campaign** (2026-06-30) | measurement | eval-mode ranking flips (worker 3509 fastest); GPU-scaling probe: **gen flat 228→225→219 s across 1/2/4 GPU** → rollout is env-bound |

The 06-30 campaign ended with a measurement, not a fix: ALFWorld's rollout time does not respond to
GPUs. **This report is the fix — and the proof.**

## 2. The diagnosis — why gen was flat (and why nobody had fixed it)

Three facts, each independently verified on 07-01, close the causal chain:

1. **Code audit.** The ALFWorld service holds a process-global `threading.Lock` (`_TW_LOCK`,
   `envs/alfworld/service/server.py:180`) around **every** textworld `reset`/`step` (:315, :350),
   because the tatsu PDDL parser is a process-global mutable singleton (concurrent use corrupts its
   rule stack). The service runs as **one** uvicorn worker; the env pool is an in-process
   `asyncio.Queue` — so `alfworld_pool_size` cannot bypass the lock, and the agent-loop's unbounded
   async concurrency all funnels into one file line.
2. **Arithmetic reconciliation.** acceleration.md §2.2 had already measured the lock at
   **86 ms/step** — on a 160-step batch (13.7 s), and filed it as "understood and bounded". At paper
   scale a windowed step is ~**3200** env-steps (measured: `adopted 3184 per-turn rows`), and
   86 ms × 3200 ≈ **275 s** — bracketing the measured gen of 219–228 s. The doc's *constant* was
   right; its *qualitative verdict* didn't survive the batch-size scaling.
3. **The design assumption inverted.** The lock's own comment says *"env transitions are ms-fast vs
   **seconds** of LLM generation, so the pool's real concurrency benefit is preserved."* In windowed
   mode the mean response is **~100 tokens/turn** (measured 99.6): the LLM turn is ~0.2–0.3 s, not
   seconds. The LLM hides under the lock — the reverse of the assumption — and the serialized env
   becomes the floor: **73 % of a 4-GPU step**.

Why it hadn't been fixed: at the batch sizes of the June smokes the tax was seconds; only the
06-30 paper-scale probe made it dominant, and only the flat-gen signature identified *where* it sat.

## 3. What was built — replica sharding, and the design decisions

**Change (committed `e593dd2`):** run **K identical service processes per client** over the *same*
data shard; each episode binds to one replica.

| decision | choice | why (alternatives rejected) |
|---|---|---|
| Where to break the lock | **K separate service processes** | (a) patching textworld/tatsu for per-env parsers = deep upstream surgery, regression risk; (b) a multiprocessing pool *inside* the service = IPC layer + fork-safety work. (c) K processes needs **zero service-code change** — the parser is per-process, so K processes = K independent locks for free. |
| Client-side routing | comma-separated URL list handled in **one** place, `envs/base.py::resolve_service_url` | covers BOTH envs × all three routing sources (persistent URL-file / process-env / spec) with one edit; each env instance picks a replica at construction → naturally **sticky** for its whole episode. |
| Balance policy | **round-robin** (PID-offset cursor), not hashing | bounded imbalance (±1 per agent-loop worker) lets the per-replica pool be `ceil(pool/K)+2` without `/create` starvation; hashing gives √n-scale skew and would need much larger slack. |
| Pool semantics | `*_pool_size` stays **total per client**, split ~evenly (+2 slack) across replicas | preserves the documented invariant "pool ≥ gen_batch"; configs stay comparable. |
| Val service | replicated with the **same K** | eval walks the same lock (48–140 games × 50 turns); the win applies to the every-round eval too. |
| Back-compat | `K=1` reduces **exactly** to the legacy single service (same port `base+c`, same log names) | the default path is byte-identical; all June configs unaffected. |
| Ports | `base + c*K + j`; val `val_port + j`; collision guard made band-aware | keeps the per-client port bands disjoint at any K. |

**Equivalence argument (science-safety).** Every replica receives the client's identical shard
env-vars (`CLIENT_ID`/`CLIENT_NUM`/partition knobs) → identical game/goal distribution; episodes
sample iid from the same set, so *which* replica serves an episode is a scheduling detail of the
same class as the existing pool-borrow order. The trainer plane is untouched — confirmed by
measurement (update_actor 43.3 → 43.7 s, old_log_prob/ref within noise).

## 4. Campaign design

Two allocations run in parallel with self-queuing drivers (each stage starts when the previous
barrier file says DONE; the durable foreground-`srun` pattern from 06-30):

- **1×H100 / 8 cores (2.5 h walltime left)** — code-independent probes first: WebShop 1-GPU
  decomposition, then ALFWorld K=4 / K=8, then the K=1+pool-64 **control**.
- **4×H100** — WebShop 4-GPU + 1-GPU-pinned (same-node pair), then ALFWorld K=8 component probes,
  then the **end-to-end A/B**, then the WebShop lever pair.

Every experiment carried a **falsifiable numeric prediction written down before launch** (§8).
The replica code path was first exercised by unit smoke (routing balance, URL construction, K=1
reduction), then by 11 GPU runs covering every path (K=1 legacy webshop+alfworld / K>1 train / K>1
train+val+cross_round+worker / K>1 webshop). All 13 runs of the day: rc=0.

## 5. Complete experimental data

### 5.1 ALFWorld — baselines (2026-06-30, for reference)

| GPUs | gen | old_log_prob | ref | update_actor | **step** | wall (1 step, incl. fixed) |
|---|---|---|---|---|---|---|
| 1 | 228.3 | 52.2 | 107.9 | 140.0 | **534.5** | 1050 s |
| 2 | 225.3 | 31.9 | 33.0 | 92.2 | **386.9** | 865 s |
| 4 | 219.3 | 18.5 | 14.1 | 43.3 | **298.4** | 778 s |

gen flat (−4 % over 4× GPUs); update_actor ~linear (3.2×); pool_size was 8.

### 5.2 ALFWorld — mechanism sweep + control (1×H100, qgpu3010, pool 64 total)

| config | K | gen | update_actor | **step** | wall |
|---|---|---|---|---|---|
| `alf_scale_g1_r1n1` (**control**) | 1 | **217.5** | 139.5 | 511.7 | 1056 s |
| `alf_scale_g1_r4n1` | 4 | **65.8** | 138.8 | 358.1 | 714 s |
| `alf_scale_g1_r8n1` | 8 | **61.8** | 137.3 | 350.2 | 702 s |

- Control (K=1, pool 8→64): gen 217.5 ≈ 228 baseline → **pool size is irrelevant; the lock is the
  whole story.** (Kills the "it was just a bigger pool" explanation.)
- Same node, same pool, single variable K: **217.5 → 65.8 → 61.8**.
- K=4 ≈ K=8 → the residual ~60 s is the new floor (episode critical path: ~50 turns ×
  (LLM ~0.2–0.3 s + env 86 ms/K + HTTP)); an 8-core node needs only **K=4**.

### 5.3 ALFWorld — component probes at K=8 (qgpu3021)

| config | GPUs | gen | old_log_prob | ref | update_actor | **step** | vs baseline |
|---|---|---|---|---|---|---|---|
| `alf_scale_g4_r8` | 4 | **51.7** | 15.2 | 14.1 | 43.7 | **127.6** | **−57 %** (2.34×) |
| `alf_scale_g1_r8` | 1 | 62.7 | 40.7 | 108.1 | 141.0 | **358.8** | −33 % |

- gen −76 % at 4 GPU; **update_actor 43.3→43.7 = the GPU plane untouched** (the no-side-effect check).
- 1-GPU step agrees across nodes (358.8 vs 358.1/350.2) → node-independent.
- **Post-fix consequence:** with the env floor gone, GPU compute dominates → the ALFWorld 1-vs-4-GPU
  per-step penalty grows **1.79× → 2.81×**. The "1 GPU per client" layout is now dead on both envs.

### 5.4 ALFWorld — end-to-end A/B (the config from the 06-30 eval-mode sweep, ONLY + `alfworld_replicas: 8`)

2 clients × 2 rounds, eval every round (48-game val), `cross_round + eval_mode=worker`, train **and
val** services sharded:

| | baseline `alf_em_worker` | `alf_em_worker_r8` | Δ |
|---|---|---|---|
| total wall | 3509 s | **2412 s** | **−31 %** |
| training steps (4 client-rounds) | 408 / 320 / 338 / 370 (mean 359) | 147 / 113 / 121 / 126 (mean **127**) | **−65 %** |
| rc / val | 0 / healthy | 0 / healthy (r2 success 0.083) | ✓ |

The −31 % vs −65 % gap locates the **next bottleneck**: with steps at 508 s of 2412, the run is now
dominated by trainer cold-start, aggregation/merge, service loads and eval plumbing (the Tier-2
inter-round candidates).

### 5.5 WebShop — first-ever timing decomposition (corrects an inference)

| config | node | GPUs | gen | old_log_prob | ref | update_actor | GPU-Σ | **step** | wall |
|---|---|---|---|---|---|---|---|---|---|
| `ws_scale_g1` | 8-core | 1 | 50.6 | 23.8 | 35.0 | 87.9 | 146.7 (73 %) | 202.1 | 619 s |
| `ws_scale_g1b` | 64-core | 1 | 54.6 | 26.6 | 39.1 | 100.1 | **165.8 (74 %)** | **225.2** | 543 s |
| `ws_scale_g4` | 64-core | 4 | 44.1 | 10.2 | 8.8 | 27.9 | 46.9 (50 %) | **93.4** | 481 s |

- **WebShop is GPU-compute-bound** — the mirror image of ALFWorld. gen is flat-ish (54.6→44.1) but
  *small*; GPU compute scales 3.54×.
- **Correction:** the published "1-GPU only 1.37× slower" was a 3-step wall diluted by ~390 s fixed
  overhead — reconciliation: 995 ≈ 3 × 202 + 390 ✓. The per-step 1-vs-4 penalty is **2.41×**.
- Node effect ruled out: ±10 % between the 8-core and 64-core nodes.

### 5.6 WebShop — lever pair (4×H100)

| config | pool | K | gen | update_actor | **step** |
|---|---|---|---|---|---|
| `ws_scale_g4` (baseline) | 16 | 1 | 44.1 | 27.9 | 93.4 |
| `ws_scale_g4_p64` | 64 | 1 | **50.1** ⚠ | 27.9 | 100.5 |
| `ws_scale_g4_p64r4` | 64 | 4 | **35.7** | 26.0 | **82.2 (−12 %)** |

- Pool 16→64 alone **hurts** (+14 % gen): 64 concurrent sessions in ONE process amplify GIL
  contention — the "wave-throttle" hypothesis is **refuted**.
- Replicas give WebShop a real but modest −12 % (GIL sharding). WebShop's lever remains GPU count.

## 6. Analysis — why the numbers are what they are

**The one-line physics.** A training step is `gen + GPU-compute`. gen is gated by whatever the env
service can serialize; GPU-compute is gated by FSDP scaling. The two environments sit at opposite
ends: ALFWorld = 3200 lock-serialized env-steps (gen ≈ the lock, 73 %), WebShop = 960 cheap env
steps against a heavy 4096/512-token FSDP+logprob pipeline (GPU 74 %). Hence the same intervention
(replicas) is worth **−57 %** on one and **−12 %** on the other — and *measuring the split first*
(a 1-step `timing_s` probe at two GPU counts) is the transferable decision rule for any new env.

**Why the residual floor is ~60 s.** After sharding, per-replica serialized load (219/K ≈ 27 s at
K=8) drops below the episode critical path (~50 turns × ~0.3 s ≈ 15–20 s plus tail/HTTP): the
longest single episode, not the aggregate env throughput, now bounds gen. More replicas cannot help;
shorter episodes or fewer turns would (science changes — out of scope).

**Why end-to-end −31 % ≠ step −65 %.** Fixed costs (one trainer cold-start, K service loads,
aggregation + HF merge, eval engine sync) don't shrink with the lock. At paper scale (70 rounds ×
3-epoch rounds) the step term dominates more, so the end-to-end gain grows toward the step gain —
but that extrapolation is *not yet measured* (open item).

**Why the penalty GROWS post-fix (1.79× → 2.81×).** Removing a GPU-count-invariant term (env) from
the numerator and denominator of the 1-vs-4 ratio leaves the strongly-scaling GPU-compute term
exposed. Counter-intuitive but arithmetically inevitable — and strategically important: env fixes
make *more* GPUs per client more valuable, not less.

## 7. How this composes with the prior stack

The genealogy — each lever exposed the next bottleneck:

```
#4 persistent/cross_round   killed cold-start (76–88 % of wall)      → exposed rollout & eval
eval modes (worker/parallel) took eval off the critical path          → exposed the training step
#3 client-parallel          exploited sub-linear FSDP (2×2 −35 %)     → needed the concurrency fixes
:29500 + VERL_RAY_JOB_ID    made ANY same-node concurrency safe       → #3/eval∥train reliable
06-30 measurement           found ALFWorld's step is 73 % env         → THIS round:
Tier-1 replica sharding     killed the env serialization (−57 % step) → next: GPU compute + inter-round plumbing
```

Composition status:
- **Stacks with #4 + worker-eval:** the end-to-end A/B *is* `cross_round + worker + replicas` — the
  three compose cleanly (that run is the production recipe).
- **Stacks with #3 (predicted, not yet run):** post-fix 2-GPU step ≈ 55 + 157 ≈ 210 s → two parallel
  2-GPU clients ≈ 210 s vs serial 2×127.6 = 255 s → a further ~−18 %; needs a parallel-round
  launcher in `run_fed` (the June #3 evidence came from two separate processes).
- **Orthogonal to the ZMQ fix:** replicas add processes on the *env* plane only; the weight-transfer
  socket namespace is per-verl-job and unaffected (worker_r8 ran clean under it).
- **Corrections forced on prior docs:** §2.2 "bounded" (batch-scaling), §7.7 "WebShop
  env-latency-bound / 1.37×" (overhead dilution) — both now carry dated correction notes in place.

## 8. Prediction scorecard

| prediction (written before launch) | measured | verdict |
|---|---|---|
| ALFWorld gen 219 → 40–70 s (K=8) | 51.7 | ✅ |
| ALFWorld step 298 → 120–150 s | 127.6 | ✅ |
| ALFWorld 1-GPU step → ~350 s | 350–359 (both nodes) | ✅ |
| control: K=1+pool64 gen stays ~220 s | 217.5 | ✅ |
| end-to-end < 2500 s (vs 3509) | 2412 | ✅ |
| small node: fewer replicas suffice | K=4 ≈ K=8 | ✅ |
| WebShop gen dominant & flat | flat-ish ✓ but only 24–47 % — **not dominant** | ⚠ half-wrong (the earlier back-solved "t_env ≈ 455 s/round" was fixed overhead, not env) |
| WebShop pool 16→64 cuts gen | gen +14 % (GIL amplification) | ❌ refuted |

The two misses are as informative as the hits: they killed the WebShop-replica plan *before* it
consumed GPU-hours, and they are recorded in the docs rather than silently dropped.

## 9. Production recipes

| hardware × env | recipe | expected |
|---|---|---|
| 4×H100 × ALFWorld | `cross_round: true` + `eval_mode: worker` + `alfworld_replicas: 8` + `alfworld_pool_size: 64` | step 2.34×; end-to-end ≥ −31 % |
| 1×H100 (8-core) × ALFWorld | same, `alfworld_replicas: 4` | step −33 % |
| 4×H100 × WebShop | June fast path (#3 2×2 or 4-GPU+worker) + optional `webshop_replicas: 4` (pool ≥ batch) | replicas add −12 % |
| 1×H100 × WebShop | no service-layer magic exists; per-step 2.41× vs 4-GPU | avoid if possible |
| any NEW env | 1-step `timing_s` probe at 2 GPU counts: **flat gen → `*_replicas`; scaling gen → GPUs** | the decision rule |

## 10. Limitations & next steps

- **1-step, single-run probes** (±5–10 %); multi-round steady-state walls not yet measured — the
  end-to-end gain should *grow* toward −65 % at paper scale, unverified.
- **#3 × replicas composition** unimplemented (needs a parallel-round launcher; predicted ~−18 %).
- **#4's isolated contribution on ALFWorld** never re-isolated (it is inside both A/B arms).
- **Replica startup cost**: K parallel game-collection walks (~3–5 min); a manifest cache would
  remove it.
- **Next bottleneck** (from the −31 % vs −65 % gap): inter-round plumbing — direct shard-load into
  the persistent trainer (skip the HF merge), in-process FedAvg, service-load caching.

## 11. Provenance

- **Code:** commit `e593dd2` — `fedagent/envs/base.py` (`_pick_replica` routing),
  `fedagent/fed/run_fed.py` (`alfworld_replicas`/`webshop_replicas`, train+val replication, pool
  split, band-aware port guard). K=1 byte-identical; unit smoke + 11 GPU runs cover all paths.
- **Configs:** `tools/verl08_migration/accel/alfworld/alf_scale_g{4,1}_r8.yaml`,
  `alf_scale_g1_r{1,4,8}n1.yaml`, `alf_em_worker_r8.yaml`;
  `accel/webshop/ws_scale_g{1,g1b,g4}.yaml`, `ws_scale_g4_p64{,r4}.yaml` (each README maps them).
- **Runs (gitignored):** `runs/ws_scale/`, `runs/alf_scale/`, `runs/alf_em/worker_r8*` — barriers
  carry per-run rc/wall/timing extracts.
- 中文版: [acceleration_tier1_report_2026-07-01_cn.md](./acceleration_tier1_report_2026-07-01_cn.md)
