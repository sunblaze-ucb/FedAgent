# Acceleration across environments — WebShop vs ALFWorld

**What this doc answers in one line:** *which acceleration choices transfer from WebShop to ALFWorld,
which flip, and the single principle that predicts it.*

This is the self-contained cross-environment synthesis. The per-environment detail lives in
[`acceleration.md`](./acceleration.md) (WebShop levers + analysis),
[`acceleration_results.md`](./acceleration_results.md) (WebShop numbers), and
[`alfworld_testing.md`](./alfworld_testing.md) (ALFWorld strategy + §6 results). Both are
Qwen2.5-1.5B-Instruct, 4×H100, GRPO (G=8), paper settings.

> ### ⚠️ Read this first — what is and isn't comparable
> **Absolute wall-clock seconds are NOT comparable across the two environments.** They differ in
> val size (WebShop eval-mode sweep n=500 vs ALFWorld n=48), episode length (15 vs 50 turns), and
> per-step env weight. **Compare the *rankings*, the *relative %* penalties, and the *mechanisms* —
> never "ALFWorld 3509s vs WebShop 2493s".** Where a number's *metric* matters (per-step vs full-run
> wall), it is labelled inline.

---

## 1. At a glance

| Axis | WebShop (15-turn) | ALFWorld (50-turn) | Cross-env verdict |
|---|---|---|---|
| **Eval-mode — fastest** | `parallel` (2493s, n=500) | **`worker`** (3509s, n=48) | **FLIPS** — worker overtakes parallel |
| **Eval-mode — slowest** | `shared` (3316s, n=500) | **`inline`** (4738s, n=48) | **FLIPS** — inline worst, not shared |
| **Eval-mode — structure** | parallel < worker < inline < shared | **worker < parallel < shared < inline** | decoupled-beats-coupled holds; order within shifts |
| **Step bottleneck** (measured decomposition) | **GPU compute 74 %** of a 1-GPU step (gen only 24 %) | **env (gen) 73 %** of a 4-GPU step | **OPPOSITE** — mirror images |
| **1-GPU penalty (per-step)** | **2.41×** (225.2/93.4) | 1.79× pre-fix → **2.81× post-replica-fix** | heavy on both; grows on ALFWorld once env is fixed |
| **GPU↔rollout coupling** | gen flat-ish but *small* (54.6→44.1) | **gen FLAT across 1/2/4 GPU (env-bound, `_TW_LOCK`)** | ALFWorld-specific floor |
| **Replica sharding** (`*_replicas`) | step **−12 %** (93.4→82.2, GIL relief) | step **−57 %** (298→127.6); end-to-end **−31 %** (3509→2412 s) | **THE ALFWorld lever; garnish on WebShop** |
| **2-job concurrency (ZMQ fix)** | PASS (3-job) | PASS (2-job, both rc=0) | **TRANSFERS** |
| **Persistent trainer (#4)** | −43%/round, −62% cross-round | in the 3509s/2412s baselines (cross_round on) | transfers |

**Three sentences:** The two environments have **mirror-image bottlenecks** — ALFWorld spends 73 % of
a step inside its lock-serialized env service (fixed by `alfworld_replicas`: step −57 %, end-to-end
−31 %), while WebShop spends 74 % of a 1-GPU step in GPU compute (its lever is GPUs; replicas give
only −12 %). Decoupling eval from the training critical path wins on **both** envs, with the mode
ranking flipping on ALFWorld (worker's cold-start amortization). The concurrency fix is
environment-agnostic and holds on both.

---

## 2. Eval-mode ranking — the big flip

Same 4-mode sweep (inline / parallel / shared / worker), each = eval running at a different place
relative to training. Full wall-clock of a 2-client × 2-round run, eval every round:

| Rank | WebShop (n=500) | ALFWorld (n=48) |
|---|---|---|
| 1 (fastest) | parallel 2493s | **worker 3509s** |
| 2 | worker 2637s | parallel 3620s |
| 3 | inline 3090s | shared 4560s |
| 4 (slowest) | shared 3316s | **inline 4738s** |

**What stays the same:** the two **eval-decoupled** modes (`worker`, `parallel`) beat the two
**eval-coupled** modes (`shared`, `inline`). Whether eval sits on the 4-GPU training critical path is the
dominant factor in both envs.

**What flips, and why:**
- **`worker` overtakes `parallel`.** ALFWorld's eval engine cold-start (vLLM init + CUDA-graph capture +
  loading the 8810-game service) is *expensive*. `worker` pays it **once** (persistent cross-round) and
  keeps all 4 GPUs for training; `parallel` hides eval but trains on only 2 GPUs (+30%/step). When eval
  is heavy, amortizing the cold-start beats hiding it.
- **`inline` becomes worst (not `shared`).** `inline` re-spins that expensive eval engine **every round**
  on the critical path. On WebShop the eval was light enough that inline's re-spin was cheap and
  `shared`'s 0.3-util KV throttle was the worst sin; on ALFWorld the heavy per-round re-spin dominates,
  so `inline` sinks below even throttled-`shared`.

> **Comparability caveat.** WebShop's "shared slowest" was specifically a **large-val (n=500)** effect;
> ALFWorld ran n=48. So the shared↔inline ordering is partly val-size, not pure env. The robust,
> val-size-independent claim is the **mechanism**: ALFWorld's heavy *per-eval cold-start* is what makes
> `inline` the loser and rewards `worker`'s amortization.

---

## 3. GPU scaling — both decompositions now measured (2026-07-01)

| per-step | WebShop gen | WebShop GPU-Σ | WebShop step | ALFWorld gen | ALFWorld GPU-Σ | ALFWorld step |
|---|---|---|---|---|---|---|
| 1 GPU | 54.6 (24 %) | **165.8 (74 %)** | 225.2 | 228.3 (43 %) | 300.2 | 534.5 |
| 4 GPU | 44.1 (47 %) | 46.9 (50 %) | 93.4 | **219.3 (73 %)** | 75.9 | 298.4 |
| **1-vs-4 penalty** | | | **2.41×** | | | 1.79× (pre-fix) |

- **Mirror-image bottlenecks:** WebShop = GPU-compute-bound (gen small and flat-ish); ALFWorld =
  env-bound (gen large and FLAT — the `_TW_LOCK` floor, since fixed by replicas → post-fix step
  127.6 s and the penalty grows to **2.81×**).
- **Correction to the earlier wall-based numbers:** the old "WebShop +37 % / 1.37×" figures were
  3-step walls diluted by ~390 s fixed overhead (995 ≈ 3×202 + overhead reconciles exactly);
  per-step the WebShop 1-GPU penalty is **2.41×**. ALFWorld's "+21 % wall" was the same artifact.

**The new mechanism (ALFWorld only, measured):** split each step into rollout vs training —

| GPUs | gen (rollout) | update_actor (training) |
|---|---|---|
| 1 | 228.3s | 140.0s |
| 2 | 225.3s | 92.2s |
| 4 | 219.3s | 43.3s |
| scaling | **FLAT (−4%)** | **~linear (3.2×)** |

`gen` is **flat across GPU count** → ALFWorld rollout is **env-latency-bound**: the `_TW_LOCK`-serialized
TextWorld service gates generation, not GPU compute. Only `update_actor` scales.
**Practical lever (validated 2026-07-01):** NOT `pool_size` (K=1 control: pool 8→64 left gen at 217.5 s
— the lock serializes regardless) but **service replicas** (`alfworld_replicas: K` = K processes = K
locks): gen 217.5 → 65.8 (K4) → 61.8 (K8); 4-GPU step 298→**127.6 s**; end-to-end **3509→2412 s (−31 %)**.
WebShop's split (measured, §3 table) is the mirror image — GPU-bound; `webshop_replicas: 4` yields only
−12 % (GIL relief), its lever is GPU compute.

---

## 4. Concurrency / the ZMQ fix — environment-agnostic

The FSDP→vLLM weight-transfer deadlock (every isolated Ray cluster picks the same first job id
`01000000` → identical `/tmp` ZMQ socket → 44-min hang) and its fix (`VERL_RAY_JOB_ID` per verl
subprocess + a 2-line verl honor-override patch) live entirely in the **env-agnostic verl/Ray plane**.

| | WebShop | ALFWorld |
|---|---|---|
| Test | 3 concurrent jobs (client-parallel + eval∥train) | 2 concurrent training jobs, GPUs {0,1}+{2,3} |
| Result | PASS (rc=0) after fix | **PASS** (both rc=0; A 392s, B 473s) |

ALFWorld is the *stronger* stress test — its slow service cold-start widens the socket race window — and
the fix holds. This is the expected outcome: nothing about the bug or the fix touches the env service.

---

## 5. The principle (why all of the above follows)

ALFWorld differs from WebShop along three axes — **longer episodes (50 vs 15 turns)**, **heavier
per-step env (TextWorld + a process-global `_TW_LOCK`)**, **larger/heavier eval**. Each one shifts where
the wall-clock goes:

```
            WebShop  ────────────────►  ALFWorld
 cost moves FROM:  GPU compute    TO:  eval-engine cold-start  +  env-latency (rollout)
```

That single shift predicts every result above:
- **eval-cold-start grows** → the mode that *amortizes* it (`worker`) wins and the mode that *repeats* it
  (`inline`) loses → **eval-mode ranking flips**.
- **rollout becomes env-latency-bound** → adding GPUs stops helping generation (`gen` flat) → the lever
  for rollout becomes **env-service replicas** (K processes = K locks; pool size alone does nothing),
  and the per-GPU *training* penalty is unchanged (it was never about rollout).
- **the trainer plane is untouched** → the concurrency fix transfers verbatim.

**Decision rule for a new environment:** estimate (a) eval-engine cold-start cost and (b) how
env-latency-bound the rollout is (run a 1-step `timing_s` probe at two GPU counts: flat gen = env-bound).
High (a) → prefer `worker`/`parallel`, avoid `inline`. High (b) → set `*_replicas` until per-replica
serial load < the episode critical path (K≈4–8), THEN scale GPUs — post-fix the GPU-count penalty
*grows* (ALFWorld 1.79×→2.81×), so don't starve clients to 1 GPU on either env.

---

## 6. Settled vs open

**Settled (measured both envs):** eval-mode ranking + the decouple-eval principle; ZMQ concurrency fix
env-agnostic; **both step decompositions** (2026-07-01: WebShop GPU-bound 74 % vs ALFWorld env-bound
73 % — mirror images); **replica sharding** (`*_replicas`) validated end-to-end on ALFWorld
(mechanism K-sweep 217→66→62 s + pool control + component −57 % + end-to-end −31 %) and measured
modest on WebShop (−12 %); per-step 1-GPU penalties (WS 2.41×, ALF 2.81× post-fix — the 1-GPU-client
layout is dead on both).

**Open / not yet isolated:**
- **Persistent-trainer (#4) isolated A/B on ALFWorld** — it is *inside* the 3509/2412 baselines
  (cross_round on), but its solo contribution wasn't re-isolated.
- **#3 client-parallel × replicas composition** — 2×2-GPU parallel clients each with sharded services
  (predicted ~−18 % over serial-4-GPU post-fix); needs a parallel-round launcher in run_fed.
- **Full-val ALFWorld numbers** — these used n=48; the in-loop `valid_seen` is 140 and the offline set is
  274 (`tools/verl08_migration/eval_alfworld_by_tasktype.py`).
- **Multi-step steady-state walls** — probes were 1 step; a multi-round run confirms wall penalties
  converge to the per-step figures.

---

## Provenance & see also
- **WebShop numbers:** [`acceleration_results.md`](./acceleration_results.md),
  [`acceleration.md`](./acceleration.md) §7.4 (eval modes) / §7.7 (layouts) / §Lever #3.
- **ALFWorld numbers:** [`alfworld_testing.md`](./alfworld_testing.md) §6 (predictions resolved +
  scorecard); [`EXPERIMENTS.md`](../EXPERIMENTS.md) "ALFWorld acceleration economics (2026-06-30)".
- **Configs:** `tools/verl08_migration/accel/webshop/`, `…/accel/alfworld/`,
  `…/accel/client_parallel/` (each has a README).
- **The fix:** `tools/verl08_migration/patches/` (`VERL_RAY_JOB_ID` honor-override).
- Chinese version: [`acceleration_cross_env_cn.md`](./acceleration_cross_env_cn.md).
