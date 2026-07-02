# ALFWorld testing strategy — why ALFWorld is tested *this* way

> **The one-line thesis.** ALFWorld does **not** need the full WebShop sweep re-run. Correctness is
> already covered because every fix lives in the **env-agnostic** verl/Ray plane (§1). What *is* worth
> GPU-hours is the **acceleration economics** that ALFWorld changes — longer episodes (50 turn vs 15),
> heavier per-step env (TextWorld + a process-global lock), a larger val rollout — because two of the
> WebShop conclusions may **flip** on ALFWorld: *(a)* whether the 1-GPU penalty is small enough that
> "1 GPU/client + hidden eval" becomes competitive, and *(b)* whether eval is now too big to hide.
>
> Companions: [acceleration.md](./acceleration.md) (the lever design + cold-start dissection),
> [acceleration_results.md](./acceleration_results.md) (the WebShop numbers this doc reasons from),
> [architecture.md](./architecture.md) (the overlay), [running.md](./running.md) (how to launch).
>
> **Conventions.** WebShop baselines are 4×H100, Qwen2.5-1.5B-Instruct, paper settings. "1-GPU penalty"
> numbers (995/727/407 s) were measured at WebShop **n=64** val; the eval-mode sweep at **n=500**.
> Every cited number is anchored to a section in the acceleration docs.

---

## 0. Verdict

| question | answer | where it's settled |
|---|---|---|
| Do the concurrency/transport fixes need re-validating on ALFWorld? | **No** — they're in the env-agnostic plane; byte-identical regardless of env service | §1 |
| Does the persistent trainer / eval_mode / #3 code need re-validating? | **No** — verl-lifecycle code; the only env-specific seam (service routing) is already wired + validated for both envs | §1 |
| What *does* change on ALFWorld? | The **wall-clock decomposition** — each lever's relative value, because episode length / per-step weight / val size differ | §2–§3 |
| What's worth testing, then? | **Tier-1** (concurrency smoke + fast-path number) and **Tier-2** (the two economics flips) | §4 |
| What's safe to skip? | The full 1-GPU layout sweep and the 4-card eval sweep, re-run verbatim | §4 |

---

## 1. The architecture boundary — why the fixes are env-agnostic

FedAgent's env is an **independent HTTP service** behind the `BaseTextEnv` contract (see
[architecture.md](./architecture.md) → *Remote env services*). Inside the training process the
agent-loop (`GymTextAgentLoop`) reaches it only over HTTP. **The env lives past a process boundary.**
That splits the whole stack into two planes:

```
┌─ verl / Ray plane  (ENV-AGNOSTIC) ────────────────────────────────────────┐
│  FSDP actor ──ZMQ /tmp socket──► vLLM engine     ← weight-transfer fix here │  (Bug #2: VERL_RAY_JOB_ID)
│  torchrun aggregate_fedavg_fsdp                  ← FedAvg rendezvous fix here│  (Bug #1: --standalone)
│  run_fed orchestration / persistent / eval_mode / #3   ← all this CODE here │
│  metrics_logger stdout parse + flush                   ← here               │
└────────────────────────────────────────────────────────────────────────────┘
                    │  HTTP  (the ONLY env interface)
┌─ env service plane  (ENV-SPECIFIC) ───────────────────────────────────────┐
│  WebShop service   /   ALFWorld service          ← ONLY this plane swaps    │
└────────────────────────────────────────────────────────────────────────────┘
```

Walking the fixes one by one:

| fix / capability | which plane | why it transfers to ALFWorld unchanged |
|---|---|---|
| **ZMQ weight-transfer socket** (Bug #2) | verl/Ray | The socket is FSDP-actor → vLLM-engine, *inside* the trainer. The collision (every isolated Ray cluster picks the same first job id `01000000` → same `/tmp` socket path → deadlock) and the `VERL_RAY_JOB_ID` fix are byte-identical no matter which env service is on the other end of HTTP. |
| **FedAvg rendezvous port** (Bug #1) | verl/Ray | The aggregator (`torchrun --standalone`) operates on **FSDP checkpoint shards** — it never touches an env. |
| **persistent / cross_round / eval_mode / #3** | verl/Ray | This is `run_fed` orchestration + verl lifecycle. The *only* env-specific seam is **service routing** (`WEBSHOP_SERVICE_URL` vs `ALFWORLD_SERVICE_URL`) — and both routes are implemented **and** GPU-validated (per-client routing, acceleration.md §7.3). |
| **stream / stdout flush** | verl/Ray | `metrics_logger.py` parses the client's `training.log` stdout; nothing env-aware. |

**Conclusion: correctness already covers ALFWorld.** Re-running the whole WebShop sweep would mostly
re-confirm things the env boundary guarantees are unchanged. The interesting question is not *whether
it works* but *how fast* — and that is where ALFWorld genuinely differs.

---

## 2. Wall-clock decomposition — what the acceleration logic depends on

Per-round wall-clock ≈ **cold-start + rollout + train-compute**, plus an eval term. Each lever attacks
exactly one of these. The lever's value therefore depends on **how big that term is** — and the term
sizes are set by the env. That is the root reason ALFWorld differs.

| lever | term it attacks | WebShop 1.5B measured | source |
|---|---|---|---|
| **#4 persistent** | **cold-start** (paid once instead of ~140×) | ramp ≈ 2.5 min warm / 5–14 min cold; **~76–88%** of paper-run wall-clock; **−43%** per-round, **−62%** cross-round | accel §2.1; results §2 |
| **#3 / GPU-scaling** | **train-compute + vLLM generation** (FSDP/TP parallel) | t1(4)=**558** → t1(2)=**725** → t1(1)=**995**; 4-GPU only **1.30×** over 2-GPU (sub-linear) | results §4; accel §7.7 |
| **eval-hiding** (#1 / worker) | the **eval** term | eval **407 s** (n=64) hidden under **995 s** train; n=500 sweep: parallel 2493 / worker 2637 / inline 3090 / shared 3316 | accel §7.7; results §3 |

The relative value of each lever is just the relative size of the term it attacks. Change the term
sizes — which the env does — and the rankings move.

---

## 3. How ALFWorld changes each term (falsifiable predictions)

ALFWorld vs WebShop: **longer episodes** (50 turn vs 15), **heavier per-step env** (TextWorld
simulation vs WebShop retrieval), and — decisively — a **process-global `_TW_LOCK`** that serializes
*every* textworld `reset`/`step` (the tatsu PDDL parser is a process-global mutable singleton;
~86 ms/step serialized, no env-layer parallelism — WebShop has no such lock; accel §2.2). Pushing each
term through:

### 3.1 Cold-start share ↓ → lever #4's *relative* benefit shrinks
The rollout term grows (50 turn × heavy env), cold-start is ~unchanged → cold-start's **share** of
wall-clock falls. So "kill the cold-start" (#4) buys a **smaller** relative win on ALFWorld than the
WebShop **−43% / −62%**. #4 is still worth keeping on (it's free and never hurts), just less dominant.

### 3.2 1-GPU penalty ↓ → the "1 GPU/client" layout conclusion may loosen, or flip ⭐
This is the most interesting prediction. GPU count drives **train-compute + vLLM generation
throughput**; it does **not** touch env latency (HTTP to a CPU env service is GPU-independent — and
ALFWorld's env stepping is *already serialized* by `_TW_LOCK` regardless of GPU count). ALFWorld's
rollout is far more **env-latency-bound**, so a larger fraction of its wall-clock is GPU-independent
waiting → cutting GPUs 4→2→1 bites a **smaller** slice.

- WebShop 1-GPU penalty = **995 / 727 = +37%** (+268 s/round) — enough that the doc rules
  "2 client × 1 GPU + 2-GPU hidden eval" **not** the fast path (accel §7.7; report §9.1).
- **ALFWorld prediction: smaller penalty (perhaps +15–25%)** → the "2 clients each on 1 GPU + 2 GPUs
  hiding eval" layout may become **competitive** on ALFWorld. This is precisely the flip **Tier-2's
  t1(1) vs t1(2)** measures.

### 3.3 eval cost ↑ → hiding eval matters more, but it may not *fit*
ALFWorld's val is a big rollout, and `_TW_LOCK` makes it un-parallelizable at the env layer. Three
consequences:

- **Hiding eval saves more** (eval is dearer) → eval-hiding's value **rises**.
- **But "hidden" requires eval < a training round.** A 50-turn × `_TW_LOCK`-serialized × heavy-step
  val can **exceed one training round** → eval **overflows** the training window → the eval-hiding
  lever may **fail** on ALFWorld. *This must be measured.*
- **`shared` almost certainly loses.** "shared slowest at large val" (the WebShop n=500 finding,
  results §3) compounds: shared's 0.3-util reduced-KV eval engine throttles an already env-serialized
  ALFWorld val → near-certain confirmation that `shared` is wrong for ALFWorld.
- **`worker`'s lead over `inline` shrinks.** worker's edge is skipping the eval cold-start; when the
  eval *rollout itself* is huge, the saved cold-start is a smaller fraction → the eval-mode ranking may
  re-order.

> **Magnitude correction vs the napkin estimate.** The in-loop ALFWorld val is **140 games**
> (`valid_seen`, 50-turn — `config/envs/alfworld_val.yaml`), **not** 274. The full **274-trial** set =
> `valid_seen`(140) + `valid_unseen`(134) and is scored **offline** by
> `tools/verl08_migration/eval_alfworld_by_tasktype.py` on the final model, not in the loop. So by raw
> game count ALFWorld in-loop (140) is *smaller* than WebShop (n=500). The reason eval still balloons is
> **not** game count — it's **episode length (50 vs 15) × heavier per-step × `_TW_LOCK` serialization**.
> That mechanism is also why §3.2 holds: env stepping is serialized regardless of GPU count.

---

## 4. Test design — one prediction, one test

Each tier maps directly onto a prediction above. Run the smallest set that resolves the flips.

### Tier 1 — concurrency smoke + ALFWorld fast-path number (highest value, one job, three answers)
**Layout:** 2 ALFWorld clients × 2 GPU concurrent + `persistent` + `eval_mode=worker`.

Why highest value: the ALFWorld service cold-start is **slow** (it must load the game collection),
which is exactly where the **`/tmp` socket race window is widest** → the **strongest** stress test of
the ZMQ weight-transfer fix (Bug #2). One job delivers: ✅ ALFWorld fast-path number, ✅ 4-GPU
two-job coexistence on ALFWorld, ✅ the hardest case for the concurrency fixes — confirming they are
env-agnostic under real ALFWorld load.

### Tier 2 — the ALFWorld-specific economics (the two flips)
1. **t1(1) vs t1(2)** — does the 1-GPU penalty actually shrink/flip (§3.2)? Measures whether
   "1 GPU/client + hidden eval" becomes competitive.
2. **eval-mode mini-sweep on the 140-game val** — does eval *hide*, and which mode wins (§3.3)?
   Expect `shared` to lose; watch whether `worker` still leads `inline`, and whether any mode keeps
   eval under one training round.

This is the only genuinely **ALFWorld-specific** cadence conclusion — it cannot be inferred from
WebShop.

### Skip (conclusions transfer; not worth GPU-hours)
- The **full 1-GPU layout sweep** re-run verbatim.
- The **4-card eval sweep** re-run verbatim.

Both transfer from WebShop; only the *flip points* in Tier-2 are new.

---

## 5. Mechanism & pitfalls (wire these up before testing)

| item | detail |
|---|---|
| **Separate conda env** | The ALFWorld service runs in its own conda env (`verl-agent-alfworld`), a **different interpreter** from the py3.12 trainer (`fedagent-verl08`). Bring the service up first (`envs/alfworld/service/run_service.sh`). |
| **Slower cold-start** | The service loads the game collection on boot → give `/health` a **generous** wait. Starting several services concurrently raises **host-CPU contention** — stagger or widen the timeout. |
| **Val split** | In-loop val = `alfworld_val_split: eval_in_distribution` = **`valid_seen` (140)** → the round→success curve. Full **274** (`valid_seen`+`valid_unseen`) + the per-task-type breakdown come from `tools/verl08_migration/eval_alfworld_by_tasktype.py` on the **final** model. `alfworld_task_types` (`""`=all 6; else `1=Pick..6=Pick2`) selects the breakdown subset. |
| **Pool size** | `alfworld_pool_size ≥ gen_batch` — the TextWorld env pool must cover one rollout batch (`fed/run_fed.py` DEFAULTS / [`fed/README.md`](../fed/README.md)). |
| **`_TW_LOCK`** | A process-global `threading.Lock` around every textworld `reset`/`step` (tatsu PDDL parser is a process-global mutable singleton) serializes ALFWorld env stepping (~86 ms/step; ~13.7 s/windowed client-step vs legacy's ~0.9 s parallel Ray actors). Bounded and understood — and the **reason** ALFWorld is env-latency-bound (§3.2) and its eval is hard to hide (§3.3). WebShop has no such lock. |
| **Context budget** | 50-turn ALFWorld episodes pair with a widened window (paper sets `rollout.max_model_len=16384`, `response_length=8192`); the agent loop hard-guards overflow. Confirm no truncation before `done` on verbose rooms (`config/envs/alfworld.yaml` GPU-VERIFY note). |

---

## 6. Results — predictions resolved (2026-06-30, 1.5B, 4×H100, qgpu3021)

All three tiers ran. **Setup:** in-loop val reduced to **n=48** (`alf_em` configs →
[`alfworld_val_48.yaml`](../../tools/verl08_migration/accel/alfworld/alfworld_val_48.yaml), 48 of the
140 `valid_seen`) to keep a 4-mode sweep tractable; training = the sweep's minimal
`epochs=1, total_training_steps=1` per round. Wall-clock unless noted. Configs:
[`tools/verl08_migration/accel/alfworld/`](../../tools/verl08_migration/accel/alfworld/).

### 6.1 Tier-1 — concurrency: **PASS** ✅
Two independent ALFWorld training jobs (GPUs {0,1}+{2,3}), each its own Ray cluster + ALFWorld service,
both doing FSDP→vLLM weight sync on the shared `/tmp` socket — the exact path that deadlocked pre-fix.
Both `rc=0` in 16 min (A 392s, B 473s; B's +22% = host CPU/RAM contention between the two 8810-game
services, not a GPU/correctness effect). **The `VERL_RAY_JOB_ID` fix holds under ALFWorld's heavier
2-service load** → §1's env-agnostic claim confirmed under real load. (`alf_conc_{A,B}.yaml`.)

### 6.2 Tier-2 scaling — resolves §3.2 (the 1-GPU-penalty prediction)
`timing_s/step` at 1/2/4 GPU (`alf_scale_g{1,2,4}.yaml`, eval off, 1 step):

| GPUs | step | gen (rollout) | update_actor |
|---|---|---|---|
| 1 | 534.5s | 228.3s | 140.0s |
| 2 | 386.9s | 225.3s | 92.2s |
| 4 | 298.4s | 219.3s | 43.3s |
| scaling | — | **FLAT (−4%)** | **~linear (3.2×)** |

- **Mechanism CONFIRMED ✅:** `gen` is flat across GPU count → ALFWorld rollout is **env-latency-bound**
  (the `_TW_LOCK`-serialized, `pool_size`-throttled env service gates it, not GPU compute). §3.2's
  premise is directly measured.
- **Magnitude (+15–25%) too optimistic at the per-step level ✗:** the **per-step** 1-GPU penalty is
  **+38%** (534.5/386.9) ≈ WebShop's +37% — it did **not** shrink. The flat env-bound gen *dampens* it
  (pure-compute would be ~+90%), but `update_actor` still scales, so per-step matches WebShop.
- **Wall-clock 1-GPU penalty is +21%** (1050/865s) — but that is a **single-step-probe artifact**: ~490s
  of fixed overhead (service load + Ray/vLLM init + teardown) doesn't scale and dilutes one step. Over a
  real multi-step run the fixed cost amortizes and the wall penalty climbs toward the +38% per-step
  figure. **So §3.2's "1-GPU becomes competitive" loosening does NOT hold at steady state** — the layout
  conclusion stands as on WebShop. (The env-bound lever for ALFWorld rollout is `pool_size`, not GPUs.)

### 6.3 Tier-2 eval-mode mini-sweep — resolves §3.3
2 client × 2 round, eval every round, 48-game val (`alf_em_{inline,parallel,shared,worker}.yaml`):

| mode | wall | vs fastest |
|---|---|---|
| **worker** | **3509s** | — |
| parallel | 3620s | +3% |
| shared | 4560s | +30% |
| inline | 4738s | +35% |

- **Ranking `worker < parallel ≪ shared < inline`.** Eval-**decoupled** {worker, parallel} beat
  eval-**coupled** {shared, inline} by **~25–30%** → decoupling eval from the training critical path is
  what matters on ALFWorld's heavy eval; *how* you decouple (persistent worker vs concurrent GPUs) is a toss-up.
- **§3.3 "shared loses / slowest" → WRONG ✗.** shared (4560) **beat** inline (4738). On ALFWorld,
  inline's per-round cold-start of the *heavy* eval engine costs more than shared's 0.3-util throttle, so
  **inline is slowest**, not shared. (The WebShop "shared slowest at large val" finding did not transfer.)
- **§3.3 "worker's lead over inline shrinks" → did NOT shrink ✗.** worker still beats inline by **26%** —
  the cold-start it amortizes is a *bigger* prize when eval is heavy. worker also edged parallel
  (cross-round 4-GPU-training + amortized cold-start > parallel's hidden-but-2-GPU-trained eval).

### 6.4 Prediction scorecard

| § | prediction | verdict |
|---|---|---|
| §3.2 mechanism | rollout env-bound → gen GPU-insensitive | ✅ confirmed (gen flat 228→219) |
| §3.2 magnitude | 1-GPU penalty shrinks to +15–25% | ✗ per-step +38% ≈ WebShop; only the 1-step *wall* is +21% (fixed-overhead dilution) |
| §3.3 shared | shared loses / slowest | ✗ shared (4560) beat inline (4738); **inline** slowest |
| §3.3 worker | worker's lead over inline shrinks | ✗ worker keeps a 26% lead |
| §4 Tier-1 | ZMQ fix env-agnostic under ALFWorld load | ✅ PASS, both rc=0 |

**Net:** the *mechanisms* the doc reasoned from (env-bound rollout; decoupling eval matters) are
**confirmed**; two *magnitude/ordering* predictions **flipped** — the per-step 1-GPU penalty does not
shrink, and **inline** (not shared) is the eval-mode loser. ALFWorld single-node fast path = **`worker`
or `parallel`** (~25–30% over inline). Run logs: gitignored `runs/alf_em`, `runs/alf_scale`, `runs/alf_conc`.

---

## 7. Tier-1 fix — env-service replica sharding kills the env-bound floor (2026-07-01)

§6.2's env-bound `gen` (flat 219–228 s across GPU counts) is `_TW_LOCK` serialization: the tatsu
PDDL parser is process-global, one service process = one lock = single-file env stepping
(86 ms × ~3200 steps/optimizer-step ≈ the whole gen). **Fix:** `alfworld_replicas: K` — K identical
service processes per client over the *same* game shard, sessions spread round-robin client-side
(comma-URL list in `resolve_service_url`; run_fed replicates train **and val** services). Same
episode distribution → science-safe. Validation chain (1.5B, batch 8×8, GPU-measured):

| level | result |
|---|---|
| mechanism (same-node K-sweep, pool 64) | gen **217.5 (K1) → 65.8 (K4) → 61.8 (K8)** |
| control (K=1, pool 8→64) | gen 217.5 ≈ 228 → **pool irrelevant; the lock is the whole story** |
| 4×H100 component (K=8) | gen **219→51.7 (−76 %)**, step **298→127.6 (−57 %)**; update_actor untouched |
| 1×H100 component (both nodes) | step **534→350–359 (−33 %)**; **K=4 suffices on an 8-core node** |
| **end-to-end** (§6.3 worker config + K=8) | **3509 → 2412 s (−31 %)**; train steps −65 %; val healthy |

Residual gen ≈ 52–66 s = episode critical path (new floor). **Post-fix consequence:** GPU compute
now dominates → the per-step 1-GPU penalty grows **1.79× → 2.81×**, so the 1-GPU-per-client idea is
dead for good. **ALFWorld production recipe: `cross_round + eval_mode=worker + alfworld_replicas: 8`
(4×H100) / `alfworld_replicas: 4` (1×H100, −33 %).** Details + WebShop contrast (GPU-bound, replicas
only −12 %): [acceleration.md](./acceleration.md) §9.

---

## In one sentence

ALFWorld should **not** be "tested all over again": correctness rides on the env-agnostic fixes (§1),
already covered. What's worth GPU-hours is the **acceleration economics that the longer/heavier/larger
ALFWorld rollout changes** — above all the two points that may **flip**: *is the 1-GPU penalty small
enough to loosen the layout conclusion (§3.2)*, and *is eval now too big to hide (§3.3)*. Tier-1
confirms the fixes are env-agnostic under real ALFWorld load **and** banks the fast-path number; Tier-2
resolves the two flips. **Both ran (§6):** fixes hold (concurrency PASS), fast path = `worker` (3509s) —
and both predictions *flipped*: the per-step 1-GPU penalty did **not** shrink (+38% ≈ WebShop), and
**inline** (not shared) is the eval-mode loser.

## See also
- [acceleration_cross_env.md](./acceleration_cross_env.md) — **WebShop vs ALFWorld side by side** (the §6 results distilled into one master table + the transfer/flip principle)
- [acceleration.md](./acceleration.md) — lever design, cold-start dissection, equivalence audits
- [acceleration_results.md](./acceleration_results.md) — the WebShop numbers this doc reasons from
- [acceleration_report.md](./acceleration_report.md) — the complete acceleration walkthrough
- [architecture.md](./architecture.md) — the overlay, the two planes, remote env services
- [running.md](./running.md) — launching `run_fed.py` (eval modes, GPUs, ALFWorld services)
