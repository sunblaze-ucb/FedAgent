# FedAgent verl-0.8 Acceleration — Complete Engineering Report

> **The definitive, end-to-end account** of the acceleration + validation workstream: the problem, the
> design space, every lever and feature *in depth* (mechanism → equivalence argument → measured result),
> the investigations (including the false trails and corrections), and how to use it all.
>
> **The three acceleration docs, by purpose:**
> - [acceleration.md](acceleration.md) — the original **analysis & plan** (cold-start dissection, the iron law, lever design, equivalence-risk audits).
> - [acceleration_results.md](acceleration_results.md) — the **quick results** reference (status table + numbers at a glance).
> - **this doc** — the **complete walkthrough**: everything, in order, with the reasoning and the empirical findings.
>
> **Conventions.** GPU on 4×H100 (qgpu30xx). Bar = reproduce the paper within 3-seed noise. **EQUIVALENT** =
> FSDP-checkpoint `max|Δ| ≤ 1e-4` (bf16 noise floor) vs the stock-verl subprocess baseline. Model =
> Qwen2.5-1.5B-Instruct (smokes: 0.5B / TinyGuess). Thin `fedagent/` overlay on **stock verl 0.8, no fork.**

---

## 1. The problem and the bar

FedAgent is federated agent-RL: each round, a few **clients** each train a local policy (GRPO/PPO, multi-turn
rollouts against a remote env service), then the server **FedAvg**s their weights into a new global model.
The original code was a fork of verl-agent 0.3.1; the migration re-implements it as a **thin overlay on stock
verl 0.8** (no fork). The non-negotiable **bar is scientific equivalence**: every acceleration must be judged
*first* on whether it perturbs the numerics (checkpoint `max|Δ|`), *then* on speed. A faster run that changes
the trained weights beyond the bf16 floor is a failure, not a speedup.

The paper scale: **100 clients, 2/round, 70 rounds, 3 epochs/round, GRPO G=8** (PPO variant too), 4-GPU FSDP,
Qwen2.5-1.5B-Instruct, WebShop + ALFWorld, plus heterogeneity arms — **176 configs**.

## 2. Why there was no speedup at first — the cold-start thesis

GPU-measured, **~76% (0.5B, warm cache) to ~88% (1.5B smoke) of wall-clock is per-(client×round) subprocess
cold-start**: Ray init + FSDP shard load + vLLM engine init + kernel compile, paid **~140×** for a paper run
(70 rounds × 2 clients). verl's async agent-loop rollout *is* used intra-client, but the **federated pipeline**
(clients → rounds → eval) was fully serial subprocesses. So the real lever isn't squeezing the ~12% of compute —
it's **not re-paying the cold-start**.

**The iron law (the crux).** On a single saturated node, "run two things at once" doesn't help — they just
contend for the same GPUs/VRAM/PCIe. Real gains require either **(i) eliminating work** (stop re-paying
cold-start) or **(ii) resource isolation** (give each concurrent job its own GPUs/node). This framing drives
the whole design: #4 eliminates; #1/#3 isolate. (One important exception to the iron law surfaced later — see §9.)

## 3. The design space — four levers

| lever | idea | regime | numerical risk |
|---|---|---|---|
| **#4 persistent trainer** | one process across clients/rounds; in-process reset | single node — **the** lever | the only one with risk (must reproduce reset state) |
| **#1 eval ∥ train** | overlap eval(model_r) with train(round r+1) on spare GPUs | needs ≥1 spare GPU | none (eval is read-only) |
| **#3 client-parallel** | clients of a round trained concurrently | multi-node ideal; single-node wins for *small* models (§9) | none (FedAvg order-free) |
| **#2 env prewarm** | overlap next round's env-service warmup | minor / opt-in | none (pure scheduling) |

ROI by hardware: **single 4-GPU node** → do #2 (free) + #4 (the 88%) + #3 for small models; **≥2 nodes** → add
#3 (huge, bit-equivalent) + #1 free on the spare alloc.

---

## 4. Lever #4 — the persistent trainer (the big lever)

**Mechanism.** Instead of a fresh subprocess per (client, round), one Ray/FSDP/vLLM process is kept alive and
**reset in-process** between clients. The reset re-points the engine at the next client's model + clears the
FedProx anchor, rebuilding module (new weights) + optimizer (fresh Adam) + LR scheduler in one
`engine.initialize()`. Two scopes:
- **`persistent: true`** (per-round): one process per round, reused across that round's clients (cold-start once/round).
- **`cross_round: true`** (whole run): **one process for the entire run** — cold-start paid **once**. A signal-file
  handshake in `<out>/_xround/` lets the worker idle (holding GPUs) while the orchestrator runs the *same*
  external FedAvg/merge, then resume on the merged model. The aggregator inits a separate ~1 GB/rank NCCL world
  so it coexists with the paused worker.

FedAvg/merge stay **external and byte-identical** to the subprocess path — that's what keeps equivalence.

**Results (TinyGuess GRPO, 2-round, matched seeds):**

| arm | wall | Δ | final aggregate `max|Δ|` |
|---|---|---|---|
| subprocess (baseline) | 909s | — | — |
| `persistent` (per-round) | 515s | **−43%** | `1.13e-5` → EQUIVALENT |
| `cross_round` (whole run) | 342s | **−62%** | `1.13e-5` → EQUIVALENT |

Equivalence holds **compounded across rounds** through the FedAvg boundary (worst tensor
`layers.15.self_attn.o_proj.weight`, mean 1.7e-7). **PPO critic reload is GPU-validated**: the gae path rebuilds
the critic engine per client and FedAvg's actor **and** critic (required adding a `critic:` block to
`fedagent_ppo.yaml` — a pre-existing gap hitting *both* subprocess and persistent PPO; inert under GRPO).

**Per-client reset = the equivalence-critical surface.** The top risks are optimizer/LR/FedProx-anchor carryover
(must be fully reset, not inherited from the previous client) and vLLM sampler RNG state. The per-client reset
checklist + ranked risks live in `acceleration.md` §Lever #4.

**Per-client service routing (real WebShop/ALFWorld).** A persistent worker can't spawn a fresh env per client,
so routing uses a **file channel** `FEDAGENT_SERVICE_URL_FILE`: the driver rewrites it with the current client's
URL (`base_port + c`) before each `fit()`; the shared agent-loop workers read it via `resolve_service_url`
(priority file > env-var > config > default). GPU-validated: 2-client smoke served exactly its 32 episodes per
distinct service (seeds 11 vs 12), proving the routing isn't leaking to a default.

---

## 5. Eval/train GPU sharing — the four `eval_mode`s

Eval is **read-only**: it loads the merged `model_r`, scores it, writes nothing back (no RNG/data/weights). So
`eval(model_r)` run inline / async / on-the-hot-engine yields a **bit-identical training trajectory** — `eval_mode`
changes only *where/when* eval runs. The only constraint is GPU memory: vLLM pre-reserves
`gpu_memory_utilization × VRAM` per GPU independent of model size, so two engines on one GPU collide (0.6+0.6>1.0).
Hence four modes:

| mode | mechanism | when |
|---|---|---|
| **inline** (default) | blocking eval after merge, on all node GPUs | training saturates the node |
| **parallel** (= #1) | eval on a **disjoint** GPU subset, concurrent with next round's train; async launch + deferred collect | training uses < node GPUs |
| **shared** | a **2nd** eval vLLM at reduced `eval_gpu_mem_util=0.3` on the worker's GPUs | single node, no spare GPUs |
| **worker** | the cross-round worker evals on its **own hot vLLM** (verl `_validate()`) — no 2nd engine | single node, saturated (the paper case) |

**0.5B, 2-client × 2-round WebShop, eval every round** (val floored at −0.6 → byte-identical across modes,
confirming read-only):

| eval_mode | base | wall | val r0/r1/r2 |
|---|---|---|---|
| inline | per-round persistent | 1018s | −0.6 / −0.6 / −0.6 |
| **parallel** | cross-round | **690s** | identical |
| shared | cross-round | 874s | identical |
| **worker** | cross-round | **703s** | identical |

**1.5B, PAPER settings (G=8, webshop_15 15-turn, response 512, n=500 val), 4-card, 2 rounds** — all `rc=0`, **no OOM**:

| eval_mode | GPU layout | wall |
|---|---|---|
| **parallel** | 2 train + 2 eval | **2493s** |
| **worker** | 4 train (hot eval) | 2637s |
| inline | 4 train (blocking) | 3090s |
| shared | 4 train + 2nd engine @0.3 | **3316s** |

**Finding the 0.5B floor hid: `shared` flips to *slowest* at a large val set.** At 0.5B/n=8 shared (874s) beat
inline (1018s); at 1.5B/**n=500** shared (3316s) is slowest, because its reduced-KV (0.3-util) eval engine caps
batch concurrency — the n=500 eval is throttled, a penalty that *scales with val-set size*. So `shared` is the
wrong pick when the val set is large; `parallel` wins (full-util eval overlapped off the critical path), `worker`
a close second.

**`worker` needed four verl-lifecycle fixes** (it drives `_validate()` *outside* `fit()`'s lifecycle):
1. **`global_steps`** — `_validate()` reads it; only `fit()` sets it → seed `=0` when absent.
2. **FSDP→vLLM weight sync = the real CUDA-crash root cause** — verl inits the rollout vLLM with **dummy** weights,
   asleep after `init_workers`; real weights are synced by `checkpoint_manager.update_weights` per rollout. A
   worker-eval *before* `fit()` would hit dummy weights → CUDA illegal-memory-access / EngineDeadError. Fix:
   `update_weights` (sync+wake) before `_validate`, `sleep_replicas` after (mirroring `fit()`). (`enforce_eager`
   only *moved* the symptom — not a fix.)
3. **dump executor** — `fit()` shuts down verl's dump ThreadPoolExecutor at its end; the next round's worker-eval
   would submit to a dead executor → re-init if shut down.
4. **`val_batch_size`** — honor `config.data.val_batch_size`, not `len(val)`, so a full WebShop/ALFWorld val set
   isn't fired in one batch (the env-service storm).

---

## 6. Eval cadence semantics (a correction worth stating)

The paper's "server-aggregated" red line is **one eval of the round's aggregate, every round** — *not* gated by
`test_freq`. `test_freq` is verl's **within-job** step cadence (with `epochs_per_round` steps/round it only fires
`is_last_step`). A single shared eval of the round's aggregate equals the **expectation** of the paper's per-client
`val_before_train`(step-0) **average** (every client of the round starts from the *same* aggregate) → same curve,
a fraction of the rollout cost. Code: the global-eval gate is `if do_eval:` (every round), not `r % test_freq`;
`do_eval = bool(val_env_spec)` (so `val_env_spec: ""` turns eval fully off — used to isolate training time in §9).

---

## 7. Client-end eval — the per-client "circle" marks

The red line scores the round's *aggregate*; the paper also plots **each client's post-training model** as a
circle (one per client per round). `client_end_eval: true` (default off) adds `clients_per_round` evals/round on
the **unperturbed val set** and emits a `client_curve` alongside `val_curve`.

**The within-job routing problem (why this isn't trivial).** During a client's training job, the env-service URL
is routed to *that client's* (perturbed) service, and the agent-loop workers can't tell a train rollout from a
val rollout to swap URLs mid-job. So a client's *own* job can't self-eval on the clean set. Two paths solve it
**outside** the per-client routing:
- **orchestrator** (inline/parallel/shared): `eval_client` merges the client's trained actor → `round_<r>/client_<c>/hf`,
  then scores it via the normal `eval_global` path against the **unperturbed val service** (must run *before*
  `cleanup_round_checkpoints` — reads the client shards; the merged `hf` survives).
- **worker**: `_worker_validate(r, client_id=c)` scores the just-trained model on the **hot** engine after each `fit()`.

Both GPU-validated: `client_curve` = 4 circles (r1c0/r1c1/r2c0/r2c1), matching the 3-point red line.

---

## 8. Equivalence validation — methodology, results, and the eval-noise nuance

**Methodology.** Matched-arm A/Bs that differ **only** in the accel mechanism (eval off, cleanup off to preserve
shards), compared tensor-by-tensor (`tools/verl08_migration/compare_fsdp_checkpoints.py`, atol 1e-4).

| A/B | actor `max|Δ|` | verdict | note |
|---|---|---|---|
| WebShop **GRPO** (subprocess vs cross-round) | **9.8e-6** | EQUIVALENT | |
| **PPO** (subprocess vs cross-round) | **1.16e-5** | EQUIVALENT | backbone ~1e-4; critic **value-head** `score.weight` 5.92e-2 |
| 1.5B cross-**mode** (worker vs inline aggregates) | 3.8e-6 / 7.6e-6 | EQUIVALENT | |

**The PPO value-head 5.92e-2** is *not* a divergence: the critic's value head (shape (1,896)) is a random init
that isn't reproduced across arms; the backbone matches ~1e-4, and the **actor** matches 1.16e-5 because advantage
normalization washes the value-head offset out of the policy gradient. Harmless for the trained policy.

**The eval-noise nuance (1.5B).** At 0.5B the val_curves were *byte-identical* across modes — but that was the
**failure floor** (model always fails → −0.6 regardless of sampling). At 1.5B the floor is gone and val uses
`temperature=0.4` sampling, so val *numbers* differ slightly across modes (e.g. base eval 1.21 vs 1.24 on 8
episodes; or 3 vs 17 successes/500) — **eval-sampling noise, not a training divergence.** Proof: the **weights**
are equivalent (worker vs inline 1.5B aggregates `3.8e-6 / 7.6e-6`) even though the val numbers differ. So the
equivalence claim lives at the **checkpoint level**; eval numbers are read-only and carry sampling noise.

---

## 9. The #3 client-parallel investigation (the detective story)

**The test.** Run two clients **concurrently** on one 4-GPU node, each on a disjoint pair (A=0,1 / B=2,3), 1.5B,
paper settings, eval off — and ask: (a) do two verl/Ray/vLLM jobs coexist? (b) is 2×2-parallel faster than
2×4-sequential?

**(a) Coexistence: yes.** Both engines loaded on disjoint pairs (all 4 cards 6519 MiB), no Ray-port / GPU /
`/dev/shm` collision. Isolation needed only per-job `CUDA_VISIBLE_DEVICES` + `RAY_TMPDIR` (separate temp dirs →
each Ray picks its own free ports). This refuted the prior worry that two verl jobs can't share a node.

**(b) The crash, and the forensics.** One run (A) failed `rc=1`; the surface symptom was
`DataLoader worker killed by signal: Killed`. The default suspect is OOM — but **all three memory sources were
free**: node RAM 966 G / 1 TB, `/dev/shm` 504 G (48 M used), cgroup `memory.max` = unlimited. **dmesg showed no
OOM-killer.** So the SIGKILL wasn't the kernel. The full log told the real story:
- A's training *succeeded*: `step 1/2/3`, `Training Progress 100%`, `[Rank 0/1] Saved model`, `[fed] client 0
  round 1 OK` with rewards. The `DataLoader killed` was `Exception ignored in: ...__del__` — **benign teardown
  noise during GC**, printed by *both* A and B.
- A failed at the **next** step: `FedAvg actor round 1 FAILED`. The FedAvg is `torchrun --nproc_per_node=ws
  aggregate_fedavg_fsdp.py`, which uses torchrun's **default c10d rendezvous `localhost:29500`**. A and B finished
  training within seconds and launched their FedAvg `torchrun`s concurrently → **both grabbed 29500 → collision →
  one died.** (`CUDA_VISIBLE_DEVICES`/`RAY_TMPDIR` don't isolate a TCP port.)

So the `DataLoader SIGKILL` was a **red herring**; the real bug was the FedAvg rendezvous-port collision — which
also makes *any* two `run_fed`s aggregating on one node unsafe.

**The fix** (`run_fed.py fedavg()`): `torchrun --standalone` (auto free port) + clear inherited
`MASTER_*`/`RANK`/`WORLD_SIZE` on the aggregator env. Touches **only** the aggregator's comm port — FedAvg math,
rollout, eval unchanged; PPO-critic FedAvg routes through the same path. **Verified**: the exact concurrent A+B
that failed now both `rc=0`, no `EADDRINUSE`.

**The speed verdict (and a second correction).**

| arm | wall |
|---|---|
| `t1` — 1 client, 4 GPU | 558s |
| `t1` — 1 client, 2 GPU | 725s |
| **#3 — 2 client × 2 GPU, concurrent** | **727s** |
| sequential — 2 client × 4 GPU | 2×558 = **1116s** |

**#3 is ~35% faster — not the "wash" I'd predicted.** My earlier reasoning ("compute conserved → splitting halves
each client → same wall") assumed *linear* 4-GPU scaling. Empirically 4 GPUs are only `725/558 = 1.30×` faster
than 2 for 1.5B — **sub-linear**: FSDP all-gather/reduce-scatter overhead + the env-latency-bound WebShop rollout
(15 turns, fixed per-turn service latency) + fixed cold-start all weigh more when per-GPU compute is small. So
splitting 4→2+2 and running both clients concurrently beats sequential-at-full-4-GPU. This is the **exception to
the iron law**: for small models, single-node #3 *is* a real win. **Caveat:** for a large model where 4-GPU
scaling ≈2×, single-node #3 ties/loses → that's the regime that genuinely needs **≥2 nodes (one client per node)**;
the orchestrator's external FedAvg already supports it, it needs a parallel multi-node launcher (not yet built).

---

## 10. Paper-config validation (wiring + feasibility)

The **real** `uniform/Qwen2.5-1.5B/main/grpo/webshop` config (100 clients, 2/round, G=8, webshop_15 15-turn,
response 512 / prompt 4096, n=500 val, val temp 0.4), capped at 2 rounds in `worker` mode: **rc=0, loop closed.**
This is the first run on the *actual* paper config (not a smoke), and it exercised + passed every new surface:
100-client partition, per-client routing, **G=8 memory** at 4 GPUs, full 3-epoch rounds, the n=500 eval.

**Timing decomposition** (via artifact mtimes): cold-start + base eval 707s; one G=8 training round (2 clients ×
3 epochs) **496s**; one n=500 eval **630s**. Note the eval is *more expensive than the training round* — so eval
cadence is the main time lever at scale.

**70-round feasibility:** ≈ `70×475 + 71×630` ≈ **22h** (eval every round) or `70×475 + 15×630` ≈ **12h**
(`test_freq=5`) — **fits one node** (a 4×H100 allocation with ~1.5-day walltime). Val moved the **right** way even
in 2 rounds (success base `0.022 → 0.034`, n=500) — a positive omen for a full run.

The full 70-round curves (× 3 seeds × WebShop/ALFWorld × GRPO/PPO × heterogeneity arms) are a **multi-node,
multi-day campaign**, not yet run.

---

## 11. Levers #1 and #2 (brief)

- **#1 eval ∥ train** is exactly `eval_mode=parallel` (§5) — eval(r) on the disjoint GPUs overlaps train(r+1).
  GPU-validated; zero numerical risk (read-only). Compounds with #4: a persistent trainer turns eval into
  `update_weights + val pass` (seconds), which is `eval_mode=worker`.
- **#2 env prewarm** (`prewarm_next_round_services`, default off): splits service launch from health-wait and
  adopts next round's services at the round top. CPU-validated, but **benefit ≈0 for homogeneous WebShop**
  (services warm in seconds — nothing to overlap). Material only for expensive-warmup arms (catalog_split large
  catalog, ALFWorld game collection), and even then the per-subprocess cold-start (#4) dwarfs it. Stays minor/opt-in.

---

## 12. How to run — config reference

| flag | effect |
|---|---|
| `persistent: true` | #4 per-round persistent base (cold-start once/round) |
| `cross_round: true` | #4 cross-round (one process for the whole run) |
| `eval_mode:` `inline`/`parallel`/`shared`/`worker` | eval/train GPU sharing (§5) |
| `eval_gpus: N` | `parallel`: GPUs given to eval (train gets `n_gpus_per_node`; sum ≤ node) |
| `eval_gpu_mem_util: 0.3` | `shared`: the 2nd eval engine's KV pool |
| `client_end_eval: true` | per-client circles → `client_curve` (§7) |
| `val_env_spec: ""` | eval OFF (isolate training; §6) |
| `test_freq: N` | verl within-job step cadence (NOT the global red-line gate) |

**Recipes.**
- **Single 4-GPU node (paper default):** `cross_round: true` + `eval_mode: worker` — the saturated-node answer
  (no 2nd engine, no OOM, no eval cold-start).
- **Spare GPUs:** `eval_mode: parallel` — fastest (eval overlapped off the critical path).
- **Small model, ≥2 clients/round, one node:** #3 client-parallel (run `run_fed` per client pinned to a disjoint
  GPU subset; the `--standalone` FedAvg fix makes concurrent aggregation safe) — ~35% win.

---

## 13. Open items & roadmap

- **Full 70-round reproduction** — wiring validated; full curves not run (≈12–22h/config; the 3-seed × env × algo ×
  heterogeneity matrix is a multi-node, multi-day campaign).
- **Multi-node #3** — not implemented. Single-node 2×2 validated (wins for *small* models); large models need a
  one-client-per-node parallel launcher (external FedAvg already supports it).
- **#2 env prewarm** — implemented + CPU-validated, but its non-zero benefit (expensive-warmup arms) not
  GPU-demonstrated.
- **vLLM sampler RNG at long horizons** — watch reset-equivalence + `/dev/shm` teardown over the paper's 70 rounds
  (smokes pass at 2 rounds).

---

## 14. Appendix — the complete numbers + file/symbol map

**All GPU-validated numbers, one place:**

| measurement | value | config |
|---|---|---|
| cold-start fraction of wall | 76% (0.5B) → 88% (1.5B) | GPU-measured |
| #4 per-round | 515s (−43% vs 909s) | TinyGuess GRPO 2-round |
| #4 cross-round | 342s (−62%) | TinyGuess GRPO 2-round |
| #4 / equiv-A/B GRPO | `max|Δ|` 1.13e-5 / 9.8e-6 | TinyGuess / WebShop |
| equiv-A/B PPO | actor 1.16e-5 (value-head 5.92e-2 harmless) | WebShop PPO |
| eval modes 0.5B | inline 1018 / parallel 690 / shared 874 / worker 703 | 2c×2r WebShop, eval/round |
| eval modes 1.5B paper | parallel 2493 / worker 2637 / inline 3090 / shared 3316 | 4-card, G=8, n=500, 2r |
| cross-mode weight equiv 1.5B | 3.8e-6 / 7.6e-6 | worker vs inline aggregates |
| #3 scaling | t1(4)=558, t1(2)=725 (1.30×) | 1 client, eval off, paper |
| #3 parallel vs sequential | 727s vs 1116s (−35%) | 2 client, 1.5B |
| paper unit costs | 475s/train-round, 630s/n=500-eval | 1.5B worker, G=8 |
| paper 70-round estimate | 12h (test_freq=5) / 22h (every-round) | one node |

**Key files** (overlay): `fed/run_fed.py` (orchestrator: persistent/cross_round, routing, eval modes,
client-end eval, the `fedavg() --standalone` fix), `fed/persistent_{patch,task_runner,main}.py` (the persistent
worker + `_worker_validate`), `envs/base.py` (`resolve_service_url`), `tools/verl08_migration/{aggregate_fedavg_fsdp,
compare_fsdp_checkpoints}.py`. Per-symbol detail: `acceleration.md` §8.
