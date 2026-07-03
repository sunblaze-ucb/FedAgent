# FedAgent acceleration — the definitive record (2026-07-03)

> **This document is self-contained.** It answers four questions without requiring any other
> doc: *what is the final acceleration recipe → how much faster is it → why does it work →
> and how was it found, step by step (including everything that failed)*. The companion docs
> it distills are linked in §6 for readers who want raw logs and per-experiment detail.
>
> Constants everywhere below: Qwen2.5-1.5B policy, GRPO with G=8 samples/prompt, the paper's
> windowed rollout, 4×H100 on one node (qgpu3021), stock verl 0.8 + the thin `fedagent/`
> overlay (no fork). "The paper config" = `uniform/1.5B/main/grpo/<env>`: 100 clients,
> 2 sampled per round, 3 optimizer steps per client-round, 70 rounds, WebShop & ALFWorld.

---

## 0. Thirty seconds of background (so the rest makes sense)

FedAgent trains agents **federatedly**: in each *round*, a few clients are sampled, each
fine-tunes its own copy of the model on its own environment slice, and the server averages
the resulting weights (**FedAvg**). One round of the ORIGINAL implementation looks like this:

```
round r:
  1. start env services        one fleet of environment servers per client (WebShop pool /
                               ALFWorld games), fresh processes every round
  2. for each client (serially):
       spawn a TRAINING SUBPROCESS:  build Ray cluster + FSDP model engines + vLLM engine,
                                     load weights                        ← "cold start"
       rollout + 3 PPO/GRPO steps    (multi-turn episodes against the client's services)
       save FSDP checkpoint shards, tear everything down
  3. FedAvg the client checkpoints  → aggregated FSDP shards
  4. model_merger                   → convert shards to a HuggingFace folder  ("HF export")
  5. evaluation                     spawn ANOTHER subprocess (another cold start) to score
                                    the aggregated model on validation games
  6. stop env services
final: one more cold subprocess scores the last model ("final eval")
```

Two structural facts drive everything below:

- **The GPU work per round is small** (2 clients × 3 steps of a 1.5B model), so anything
  that is *not* the training math — process boots, engine builds, environment walks,
  format conversions, cold evaluations — is a large *fraction* of the wall clock.
- **The science bar is equivalence, not "roughly the same":** every optimization must leave
  the training *byte-stream* identical (same episodes, same batches, same seeds), verified
  by comparing final checkpoints at `max|Δ| ≤ 1e-4`. Anything that changes the algorithm is
  out (or explicitly quarantined as an opt-in, §4.8).

---

## 1. The final recipe

Everything below ships as **individually optional config knobs, all default-OFF** — with no
knobs set, the code path is byte-identical to the original. The adopted stacks:

```yaml
# ---------- ALFWorld paper runs ----------
use_persistent_trainer: true      # one trainer process per RUN, not per client  (§4.1)
persistent_scope: cross_round     #   … kept alive across rounds too             (§4.1)
eval_mode: worker                 # evals run on that trainer's hot engine       (§4.2)
alfworld_replicas: 8              # shard each client's env service ×8           (§4.3)
alfworld_manifest_cache: true     # cache the 8810-game disk walk                (§4.6)
service_scope: run                # keep env-service fleets warm across rounds   (§4.6)
final_eval_mode: worker           # score the FINAL model on the hot engine too  (§4.6)
hf_export: final                  # skip per-round shard→HF conversion           (§4.6)

# ---------- WebShop paper runs ----------
use_persistent_trainer: true
persistent_scope: cross_round
eval_mode: worker
service_scope: run
final_eval_mode: worker
hf_export: final
client_overrides:
  - +actor_rollout_ref.model.use_fused_kernels=True   # fused log-prob/entropy CUDA kernels (§4.4)
# note: NO webshop replicas — measured as a wash at the real config (§4.5)
```

**Deliberately NOT in the recipe** (each tried and measured, §4):
`parallel_clients` lanes (wash: −2 %), WebShop replicas (wash at paper config),
`use_dynamic_bsz` (made things *slower*: +8–11 %), more asynchrony inside a round
(saturated: every remaining barrier is a data dependency), and verl's
`one_step_off_policy` (works, −10 % realized — but changes the algorithm; kept as an
explicitly off-policy ADDITIONAL OPTION, §4.8).

## 2. How much faster

Three reference points, all on the REAL paper configs (70 rounds, eval every 5 rounds):

| stack | ALFWorld 70-round | WebShop 70-round | what it is |
|---|---|---|---|
| A. original subprocess stack | ≈ 41 h (reconstructed¹) | ≈ 32 h (reconstructed¹) | the §0 diagram as-is |
| B. wiring stack (persistent + worker eval + ALF replicas) | **29.3 h** (measured blocks²) | **20.4 h** (measured blocks²) | the stack after §4.1–4.3 |
| C. **final recipe** | **16.7 h** | **9.4 h** | B + §4.4–4.6 knobs |

- **C vs B: −43 % (ALFWorld), −54 % (WebShop)** — measured 2-round paper-config runs,
  projected with the formula below.
- **C vs A: ≈ ×2.5 (ALFWorld), ≈ ×3.5 (WebShop)** — the full campaign.

¹ A = B plus two measured costs B had already removed: a ~310 s engine cold-start per
client fit (measured directly: the probe's subprocess client spends ~310 s from launch to
its first optimizer step) ×2 clients ×70 rounds ≈ +12 h, plus cold subprocess evals.
² Projection formula, every term a measured block from the same-day 2-round runs:
`T(70) ≈ one_time + 70 × steady_round + 14 × eval + final_eval`, e.g. ALFWorld combo
= 791 + 70×762 + 14×389 + 389 s. The 2-round runs themselves: ALFWorld 3719 → 3202 s,
WebShop 2802 → 2309 s (smaller deltas than the projections because one-time costs dominate
a 2-round run; the *steady-state round* is the number that scales: **762 vs 1125 s** and
**402 vs 905 s**).

Where the C-vs-B win comes from (per steady round, ALFWorld):

| removed cost | mechanism | evidence |
|---|---|---|
| ~250 s/round service re-warm | fleets stay alive (`service_scope: run`) | A/B arm: −16 % |
| ~146 s/wave game-directory walk | manifest cache | A/B arm: −18 %, 24/24 HIT logs |
| ~40–60 s/round shard→HF merge | `hf_export: final` + direct shard reload | A/B arm equivalence 8.8e-6 |
| 578 → 389 s final eval | hot-engine eval-only plan | worker log: "scored on the hot engine; no fit" |

## 3. Why this is the shape of the answer

Profiling (§4.4) kept pointing at the same conclusion: **at this model size the pipeline was
paying fixed costs over and over, not compute**. The four bottleneck families, in the order
they were discovered and removed:

1. **Cold starts (the big one).** Every client fit and every eval built a Ray cluster, FSDP
   engines, a vLLM engine, and loaded weights from disk — ~76–88 % of small-run wall time,
   ~310 s each at 1.5B. *Fix: one persistent trainer per run; clients become plan files fed
   to it; evals run on its already-hot engine.* (§4.1–4.2)
2. **Environment plumbing.** ALFWorld's env service serializes all agents through one
   TextWorld lock (86 ms × thousands of steps), walks 8810 game directories on GPFS at every
   boot (146 s × 24 services × every round), and fleets were torn down/rebuilt each round.
   *Fix: replica sharding ×8; manifest cache; run-scoped fleets.* (§4.3, §4.6)
3. **Handoff overhead.** Each round converted FSDP shards → HF folder just so the next round
   could load it, and the final model was scored by yet another cold process. *Fix: hand the
   shards over directly (the worker's checkpoint manager loads them model-only), export HF
   once at the end; score the final model on the hot engine.* (§4.6)
4. **The GPU itself (mostly already fine).** The only surviving intra-step lever is fused
   log-prob/entropy kernels on WebShop (−6.5 % step). Batch-shape tricks backfired;
   parallelizing clients across GPU halves is a wash because the fit is GPU-bound — 2 clients
   × 2 GPUs ≈ sequential 4-GPU. Deeper asynchrony would break the on-policy equivalence bar.
   (§4.4–4.5, §4.7–4.8)

A cross-cutting reason the wins are *safe*: none of the adopted knobs touch what the model
sees. Same processes or same inputs → same episode streams → same batches; the checkpoint
comparisons (§5) confirm it to below measurement noise.

## 4. The journey — every station, including the dead ends

### 4.1 "Why is nothing faster?" → the persistent trainer

The migration to verl 0.8 was correct but not fast. Instrumentation showed 76–88 % of a
small federated run was engine cold-start. Rebuilding the trainer as ONE long-lived process
(clients arrive as plan files; between clients it resets weights/optimizer from the base
model — preserving the "fresh optimizer per round" semantics of the original) gave, on the
dev rig: subprocess **909 s → persistent 515 s (−43 %) → cross-round persistent 342 s
(−62 %)**, with full-loop checkpoint equivalence `max|Δ| = 1.13e-5` — equivalence holding
*compounded across rounds through FedAvg*, not just per client. PPO/critic (GAE) gets the
same treatment (critic engine rebuild + critic FedAvg), GPU-validated.

### 4.2 Eval without cold starts → `eval_mode: worker`

Four eval modes were built and GPU-validated (inline / parallel / shared / worker). The
adopted one — `worker` — scores round models on the persistent trainer's own hot engine
(cross-mode weight-equivalence checks 3.8e-6/7.6e-6). One real crash was root-caused on the
way: evaluating inside the worker while vLLM still held stale weights → the FSDP→vLLM weight
sync is the fix, not a workaround. Windowed rollout (the paper's fresh-prompt-per-turn mode)
was made the default; eval cadence semantics (per-round vs within-job) were pinned down.

### 4.3 The ALFWorld floor → replica sharding (`alfworld_replicas`)

ALFWorld rollouts stayed slow even with a hot engine. Diagnosis: every parallel episode
funnels through ONE TextWorld interpreter lock in the env service — 86 ms per env-step,
~3200 steps per batch, fully serialized. Sharding each client's service into 8 replicas
(games partitioned, agents routed) cut the ALFWorld step **298 → 127.6 s (−57 %)** and the
end-to-end run **−31 %**. A WebShop probe suggested −12 %; §4.5 killed that at paper scale.
Operational rule discovered here and later caught live by a guard: with 100 clients × K
replicas the service port band is `[base, base+100K)` — validation ports must sit beyond it.

### 4.4 The frontier study — profile, refute, shortlist

With the big rocks gone, a phase-by-phase decomposition of real runs identified ~800–1000 s
of *addressable plumbing* per short run (service warm, re-warm, merges, cold final eval) —
the Tier-2 shortlist. Just as important, it **refuted** the tempting GPU-side ideas with
probes instead of adopting them on vibes:

- `use_dynamic_bsz` (token-balanced micro-batching): **slower on both envs** (ALFWorld step
  127.6 → 141.7 s, +11 %; WebShop 82.2 → 88.9 s, +8 %). The GPU term was already FLOP-bound,
  and shape churn defeats kernel/CUDA-graph reuse.
- fused log-prob/entropy kernels: WebShop **−6.5 %** step (bandwidth-bound terms shrink:
  old-log-prob −30 %, ref −22 %), ALFWorld wash (+2 %) — adopted for WebShop only, later
  equivalence-verified (1.116e-5).
- more asynchrony: the trajectory level is already saturated (the bound is the episode
  critical path; every remaining barrier is a data dependency). The only remaining async
  form — training on the previous step's rollouts — is *off-policy* by construction: §4.8.

### 4.5 Paper-config reality checks (the survivorship filter)

Probes are necessary but lie in both directions, so everything was re-anchored on the REAL
paper configs capped at 2 rounds:

- **First-ever full ALFWorld paper-config run**: 3719 s (r1 1766 + r2 1375 + cold final
  eval 578), validation success 0.043 → 0.114 — the 70-round projection ≈ 29 h fits a
  2-day allocation. WebShop wiring: 2802 s (490/764/905/643).
- **Cold probes are ~3× pessimistic about steady state**: hot-engine steady steps run
  ~50 s (gen ~10 s) vs 82 s (gen 36 s) in cold single-step probes — cross-step prefix
  caching plus a warm engine. Probe arithmetic systematically understates long-run wins.
- **WebShop replicas: the −12 % probe did NOT survive** the real config (wash) — dropped
  from the recipe.
- A stale-docs correction mattered here: the real paper ALFWorld geometry is windowed
  prompt 2048 / response 512 / max_model_len 2560 (not the retired 16384/8192 concat
  design), so all probe numbers were at paper geometry after all.

### 4.6 Tier-2: everything else, as default-off knobs

The four plumbing fixes from §4.4 plus lanes and one_step_off became six independent knobs
(§1 recipe box; composition gates enforced at startup). Each got a matched A/B pair — arms
identical except the knob, seed 42 — comparing the round-2 aggregated actor checkpoint:

| knob | wall effect (A/B rig) | equivalence max\|Δ\| |
|---|---|---|
| same-config replicate (control) | −6.5 % "for free" | **9.293e-5 = the noise floor** |
| `alfworld_manifest_cache` | **−18 %** (kills 146 s/wave walks ×3 waves) | 9.199e-5 ✓ at floor |
| `service_scope: run` | **−16 %** (no round re-warm) | 9.090e-5 ✓ at floor |
| `final_eval_mode: worker` | **−11 %** (no cold final eval) | 9.241e-5 ✓ at floor |
| all four Tier-2 together | **−24 %** (sub-additive: overlaps) | 8.752e-5 ✓ at floor |
| `hf_export: final` (trainer-plane rig) | −37 % on that rig | **8.825e-6** ✓ |
| `parallel_clients: 2` (trainer-plane rig) | −32 % on that rig (⚠ see §4.7) | **1.144e-5** ✓ (via HF export diff) |

The control row is a finding in itself: two *identical* runs differ by 9.293e-5 (GPU
nondeterminism), so the knob arms — all at or below that floor — introduce **no divergence
distinguishable from noise**. That control exists thanks to a caught bug: the first cache
arm silently ran with the cache inert, because the conda env's editable verl-0.3.1 `.pth`
had been shadowing the vendored ALFWorld engine on `sys.path` all along (namespace-package
resolution race; fixed with `sys.path.insert(0, ...)`) — an independent latent
provenance bug this suite flushed out.

### 4.7 The optimal combos — and lanes dying at paper scale

The winning knob set was then measured exactly like the wiring baselines (real configs,
2 rounds): **ALFWorld 3202 s vs 3719; WebShop 2309 s vs 2802**, steady rounds **762 vs
1125 s** and **402 vs 905 s**, hot final evals 389 vs 578 / 326 vs 643 s — the §2 numbers.
Lanes (2 clients concurrently on 2+2 GPUs), despite −32 % on the tiny rig and an earlier
−35 % small-model probe, came out a **wash on BOTH envs** (−2 %): the 1.5B fit is GPU-bound,
so two 2-GPU fits ≈ one 4-GPU fit run twice, and the 2-GPU hot final eval gives back the
env-overlap savings. Same survivorship lesson as WebShop replicas: **small-rig concurrency
wins do not transfer to GPU-bound paper configs.**

### 4.8 `one_step_off` — the additional option that stays optional

verl 0.8's experimental `one_step_off_policy` generates batch t+1 on dedicated GPUs while
batch t trains (step wall → `max(gen, train)`). Wiring it under FedAgent took five layers of
fixes (a hydra primary-config rule → config split; a batch-divisibility constraint → 2+2 GPU
split; a disaggregation assert → `hybrid_engine: False`; upstream's resource-block copy →
mirrored in our entry; and per-token rollout log-probs → both agent loops now emit
`response_logprobs`, which also unlocks verl's rollout-correction family). The probe then
ran clean: steps **116 → 65 → 72 s** — generation truly vanishes from steps 2–3
(`gen ≈ 0`; the 71/64 s `generate_async` hides fully under training) vs the serial 93.4 s
step: **−23…−31 % steady**. But FedAgent's 3-step client-rounds re-pay the 116 s pipeline
prime every round: **253 vs 280 s ≈ −10 % realized**. That, plus being off-policy by one
step (the update batch was sampled from the previous weights — a *scientific* change, not an
engineering one), keeps it out of the recipe: available behind `one_step_off: true`,
subprocess path only, sign-off required.

## 5. The measurement discipline (why these numbers can be trusted)

1. **Default-off knobs**: with nothing set, the byte path is the legacy one — adopting a
   lever is a config decision, never a code migration.
2. **Matched A/B + checkpoint compare**: every knob measured against an arm differing only
   in that knob; equivalence = final aggregated actor `max|Δ| ≤ 1e-4` (FSDP shard diff, or
   HF-export diff when shard layouts differ).
3. **Noise floor first**: the same-config replicate (9.293e-5) calibrates what "equivalent"
   can even mean on this hardware; knob deltas are judged against it, not against zero.
4. **Paper-config survivorship**: no probe win enters the recipe until it survives the real
   config — this filter killed WebShop replicas (−12 % → 0) and lanes (−35 % → −2 %).
5. **Steady-state over cold probes**: long-run claims come from steady-round blocks of real
   2-round runs (cold probes proved ~3× pessimistic), projected with an explicit formula.
6. **Val curves are not the gate** at these scales: 140-game evals at near-chance success
   rates are sampling-noise dominated (equivalent weights scored 0.114 vs 0.021 on different
   runs); the weight comparison is the scientific gate.

## 6. Provenance — where each chapter lives

| chapter | doc |
|---|---|
| persistent trainer + eval modes + the lever stack | [acceleration_report.md](./acceleration_report.md) · numbers: [acceleration_results.md](./acceleration_results.md) · plan: [acceleration.md](./acceleration.md) |
| ALFWorld vs WebShop transfer principles | [acceleration_cross_env.md](./acceleration_cross_env.md) |
| replica sharding deep-validation | [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md) |
| frontier study (decomposition + refutations) | [acceleration_frontier_2026-07-02.md](./acceleration_frontier_2026-07-02.md) |
| Tier-2 knobs, noise floor, combos, one_step_off | [acceleration_tier2_2026-07-02.md](./acceleration_tier2_2026-07-02.md) |
| running log (PM entries) | `fedagent/EXPERIMENTS.md` |

Implementation lives in `fedagent/fed/run_fed.py` (knobs, orchestration),
`fedagent/fed/persistent_task_runner.py` + `persistent_patch.py` (the persistent trainer),
`fedagent/agent_loops/` (concat + windowed loops, log-prob plumbing),
`fedagent/envs/*/service/` (replicated env services, manifest cache); A/B configs and
drivers under `tools/verl08_migration/accel/`; raw run logs under gitignored `runs/`.

中文版: [acceleration_final_2026-07-03_cn.md](./acceleration_final_2026-07-03_cn.md)
