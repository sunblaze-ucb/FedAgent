# Acceleration frontier study (2026-07-02) — what's left, is more async worth it, what verl 0.8 still offers

> **The three questions this doc answers**, asked after Tier-1 replica sharding landed
> ([the 07-01 report](./acceleration_tier1_report_2026-07-01.md)):
> 1. **Can FedAgent still be accelerated?** Yes — a quantified ~1.6–2× remains, but it moved: the
>    frontier is now **inter-round plumbing** (~800–1000 s addressable of a 2412 s run), not the
>    training step. The within-step config surface is **measured to exhaustion** this round —
>    including two refuted candidates.
> 2. **Is (more) async useful?** Trajectory-level async is already the backbone and is *saturated*;
>    the remaining safe async ≈ 0 (every barrier left is a data dependency). The next async tier
>    (phase-level overlap / one-step-off) **exists in verl 0.8 and is production-ready — but it is
>    off-policy**, i.e. a science change requiring explicit sign-off.
> 3. **What verl 0.8 characteristics remain useful?** Audited exhaustively (§5): the safe list is
>    nearly harvested; the two untried config levers were probed this round — **`use_dynamic_bsz`
>    REFUTED (+8–11 % slower); `use_fused_kernels` = WS −6.5 % garnish (equivalence-verified §8), ALF wash** — leaving offload tuning and
>    the sign-off tier (one-step-off, rollout-logprob reuse) as the only unexplored verl features.
>
> Companions: [acceleration.md](./acceleration.md) §9, [the 07-01 tier-1 report](./acceleration_tier1_report_2026-07-01.md),
> [agent_rl_design.md](./agent_rl_design.md) (async model §4), [acceleration_cross_env.md](./acceleration_cross_env.md).
> Constants: 1.5B, GRPO G=8, windowed, batch 8×8; 4×H100 qgpu3021; probes = 1 step, single-run (±5–10 %).

---

## 1. Method — three evidence sources, then single-variable probes

1. **Phase decomposition** of the current best end-to-end run (`worker_r8`, 2412 s) from log
   timestamps — where the non-step ~1900 s actually goes.
2. **Effective-config extraction** from run-log config dumps — what performance flags are *actually
   in force* today (not what defaults claim).
3. **verl 0.8 tree audit** (`others/verl`) — every performance characteristic, its default, its
   maturity, and whether it touches the training algorithm.

Each resulting hypothesis was then probed with a **single-variable A/B against a measured baseline,
with the prediction written down first** — the same discipline as the 07-01 campaign, and this round
it mattered: the biggest prediction was wrong (§4).

## 2. Q1 — where the remaining time is

### 2.1 The end-to-end budget (worker_r8 = 2412 s, phase-decomposed)

| phase | measured | fix | addressable |
|---|---|---|---|
| 4 training steps | 508 s | (post-Tier-1; see §2.2) | — |
| service warm, round 1 (24 replica processes walk the 8810-game dir) | ~270 s | **manifest cache** (walk once, load a file) | → seconds |
| **service RE-warm, round 2** — the same 16 replicas restarted identically | ~200–250 s | **persist services across rounds** (uniform shards are round-independent) | ~all |
| **final eval runs as a COLD subprocess** (the hot worker is torn down before the last aggregate is scored) | ~330 s | one more hot-engine `_validate` before teardown | ~250 s |
| 2 × FedAvg + HF merge | ~260 s | direct shard-load (skip `model_merger` on the training path) | ~150 s |
| trainer cold-start + r0/r1 hot evals + teardown | remainder | partial (teardown trim) | some |

**~800–1000 s of 2412 s is addressable plumbing** — none of it touches training math. Combined
with §2.3, the worker-probe-scale run projects **2412 → ~1300–1500 s** (cumulative vs the 06-30
baseline: 3509 → ~1400 ≈ **2.5×**).

### 2.2 The within-step budget is now closed (this round's probes)

Post-Tier-1 step = gen (episode critical path) + GPU compute:

| 4×H100 per step | gen | old_log_prob | ref | update_actor | step |
|---|---|---|---|---|---|
| ALFWorld `g4_r8` (baseline) | 51.7 | 15.2 | 14.1 | 43.7 | **127.6** |
| + `use_dynamic_bsz` | 49.5 | 21.2 | 18.7 | 49.2 | **141.7 (+11 %) ❌** |
| + `use_fused_kernels` (triton) | 44.0¹ | 16.6 | 13.9 | 52.2 | **129.7 (≈ wash)** |
| WebShop `g4_p64r4` (baseline) | 35.7 | 10.8 | 10.6 | 26.0 | **82.2** |
| + `use_dynamic_bsz` | 33.8 | 10.8 | 10.6 | 31.2 | **88.9 (+8 %) ❌** |
| + `use_fused_kernels` (triton) | 31.3¹ | **7.6 (−30 %)** | **8.3 (−22 %)** | 27.0 | **76.9 (−6.5 %)** |

¹ gen varies ±15 % run-to-run (44–52 s ALF); fused kernels don't touch the rollout path — read the
gen deltas as noise, the olp/ref deltas as the (confirmed-applied) fused effect. Numerical
equivalence verified same-day (§8): full-loop off/on A/B → final-aggregate actor
max|Δ| = **1.116e-5** ≤ the 1e-4 bar.

- **gen floor** (52–66 s ALF / ~34 s WS) = the longest episode's critical path (~50 turns ×
  (LLM 0.2–0.3 s + env 86 ms/K + HTTP)); more replicas measured useless past K=4–8.
- **GPU-compute floor**: the dyn-bsz refutation (§4) shows the 73 s (ALF) / 47 s (WS) GPU term is
  already FLOP/comm-bound at this model size — not scheduling-bound. No config knob moves it;
  only more GPUs per client (#3 composition) or model/algorithm changes would.

### 2.3 The remaining lever list, ranked

| lever | attacks | est. gain | cost | science |
|---|---|---|---|---|
| service persistence across rounds + manifest cache | plumbing | ~400–500 s/run | small run_fed change + service warm-cache | safe |
| final-eval on hot engine | plumbing | ~250 s/run | small persistent-runner change | safe (eval read-only) |
| direct shard-load (skip HF merge in-loop) | plumbing | ~150 s/run (scales with rounds) | moderate | safe (exact load path) |
| #3 × replicas (parallel-round launcher in run_fed) | train segment | ~−18 % of steps | moderate | safe (order-free FedAvg, client-indexed seeds) |
| multi-node #3 | rounds ∥ clients | ~linear in nodes | launcher + alloc | safe |
| one-step-off (verl `experimental/one_step_off_policy`) | gen∥train overlap | step → max(gen, GPU) ≈ −35 % | GPU split + config | **off-policy — sign-off required** |

## 3. Q2 — the async verdict

Async in this system has three tiers; the audit + measurements settle each:

1. **Trajectory-level (harvested, saturated).** 64–512 episode coroutines hide env and LLM latency
   under each other; vLLM dynamic-batches everything; `agent.num_workers=8` is already the verl
   default and the bound is the env, not workers. This tier is the *reason* gen equals the episode
   critical path and nothing else — there is nothing left to schedule away.
2. **Pipeline-level (safe remainder ≈ 0).** Every remaining synchronization is a **data
   dependency**: FedAvg needs all clients of the round (federation semantics); round r+1 needs
   model_r. Eval was the one movable phase and is already off the critical path (worker/parallel
   modes). Service warm-up overlap (lever #2) becomes moot once services persist (§2.1).
3. **Phase-level (exists, but costs science).** verl 0.8 ships
   `verl/experimental/one_step_off_policy` — generation of batch t+1 on separate GPUs while batch t
   trains; production-ready per upstream (their DAPO-32B case: −40 %). Applied here it would take
   the step from `gen + GPU` to `max(gen, GPU)` ≈ 76 s (ALF). **But it makes GRPO one-step
   off-policy** — outside the paper-reproduction bar. `fully_async_policy` is the same trade,
   younger. Verdict: *within the on-policy bar, async is done; the next async tier is an algorithm
   decision, not an engineering one.*

## 4. The dyn-bsz refutation (worth recording — it closes a hypothesis class)

**Hypothesis** (from the effective-config audit): with rmpad ON but `use_dynamic_bsz` OFF, ALFWorld
trains its ~3200 windowed rows in ~200 micro-batches of 4 rows (~2.2 k tokens) versus a 16 k
allowance — "the GPU term is scheduling-bound; token-packing will cut it ~2×."

**Result:** slower on BOTH envs, consistently across all three GPU components
(olp +40 %, ref +32 %, update_actor +13 % on ALFWorld; step +11 % / +8 %).

**Why the hypothesis was wrong:** the reconciliation cuts the other way — 43.7 s / 200
micro-batches ≈ 218 ms per micro-batch of 2.2 k tokens, which is already FLOP-dominated for a 1.5B
forward+backward with gradient checkpointing. Packing to 16 k saves no FLOPs and adds
Karmarkar-Karp balancing, concat copies, and **shape churn that defeats torch.compile/CUDA-graph
reuse**. Lesson: *"underfed micro-batches" must be checked against per-batch milliseconds, not
against the token allowance.* The refutation closes the whole "config-level GPU-compute lever"
class — the GPU term is real work.

## 5. Q3 — the verl 0.8 characteristics audit (three tiers)

**✅ Safe & useful — status after this round**

| feature | key | status here |
|---|---|---|
| remove padding (rmpad) | `model.use_remove_padding` | already ON — long harvested |
| dynamic token-packed micro-batches | `actor.use_dynamic_bsz` (+ref/rollout) | **probed, REFUTED** (§4) |
| fused logprob/entropy kernels | `model.use_fused_kernels` + `model.fused_kernel_options.impl_backend=triton` | **probed**: WS **−6.5 %** (olp −30 %, ref −22 % — the bandwidth mechanism is real), ALF **wash** (+2 %). **Equivalence PASSED** (§8: max|Δ|=1.116e-5 ≤ 1e-4) → **adoptable on WebShop**; skip on ALFWorld. |
| offload tuning | `param/optimizer/grad_offload` | candidate for the 1-GPU ref blow-up (108 s); untested, low priority post-“no 1-GPU clients” |
| seqlen balancing | `balance_batch` | already ON |
| prefix caching / chunked prefill / CUDA graphs / sleep / dummy-load / `free_cache_engine` | rollout.* | all already ON |
| weight-sync bucket | `update_weights_bucket_megabytes=2048` | sync is 0.3–0.9 s — not a bottleneck |

**⚠️ Exists, but changes the science (sign-off tier)**
- `experimental/one_step_off_policy` (§3) — the biggest known remaining step-level win (−35 %),
  off-policy by one step.
- `rollout.calculate_log_probs` + `actor.use_rollout_log_probs` (+ verl's importance-sampling
  correction helper) — skips the olp recompute (~15 s), but swaps the numeric path vLLM↔FSDP.
- `over_sample_rate` — aborts straggler episodes: **biases sampling toward short episodes**; do not use.
- LoRA / QAT-FP8 / MTP-speculative — change training math, quantize sampling, or target long
  responses (~100-token turns make speculation pointless).

**➖ Not applicable here:** fsdp2 migration (marginal at 1.5B), `multi_turn.*` tool-calling configs
(different rollout shape), disaggregation/layered-summon (multi-node large-model features).

## 6. Prediction scorecard (this round)

| prediction | measured | verdict |
|---|---|---|
| dyn-bsz: ALF step 127.6 → 85–105 s | 141.7 (+11 %) | ❌ **refuted** |
| dyn-bsz: WS step 82.2 → 60–75 s | 88.9 (+8 %) | ❌ **refuted** |
| fused-kernels: olp+ref shrink (bandwidth mechanism) | WS olp **−30 %**, ref **−22 %** → step −6.5 %; ALF wash (+2 %) | ⚠ mechanism confirmed, magnitude = garnish |
| fused-kernels: numerically equivalent (expected ~1e-5) | full-loop A/B final-aggregate max|Δ| = **1.116e-5** | ✅ equivalence bar passed (§8) |
| plumbing ≈ 1/3 of end-to-end is addressable | phase table §2.1 (~800–1000 s of 2412) | ✅ quantified |
| safe async remainder ≈ 0 | §3 dependency analysis | ✅ (analytical) |

Two misses in a row for "the GPU term is cheap to move" — the step is at its config-level floor.
The 07-01 report's scorecard discipline is why this is a *finding*, not an embarrassment: each
refutation closed a hypothesis class before it consumed a production run.

## 7. Where this leaves the roadmap

```
done      : cold-start (#4) · eval placement (modes) · env serialization (replicas) · concurrency (fixes)
this round: within-step config surface CLOSED (rmpad on; dyn-bsz REFUTED; fused = WS garnish EQUIVALENT/ALF wash)
next      : Tier-2 plumbing (~800-1000 s/run, all safe)  →  #3×replicas launcher (−18 % steps)
          →  multi-node #3 (campaign throughput)
sign-off  : one-step-off (−35 % step, off-policy)  ·  rollout-logprob reuse (−15 s, numeric path)
floor     : gen = episode critical path (52-66 s) + GPU compute = real FLOPs (73 s ALF / 47 s WS)
```

## 8. Same-day addendum (2026-07-02 PM) — equivalence + paper-geometry corrections

Four follow-ups landed the same afternoon, triggered by the "was this *fully* tested?" challenge:

1. **fused-kernels equivalence — PASSED.** Full-loop off/on A/B on the established equivalence rig
   (TinyGuess, 2 clients × 2 rounds × 2 steps, GRPO, seed 42, subprocess arms differing **only** in
   the two fused overrides; `accel/dev/fused_ab_{off,on}.yaml`): final-aggregate actor
   **max|Δ| = 1.116e-5** (mean 2.2e-7) ≤ the 1e-4 bar — the same order as the historical
   persistent-trainer (1.13e-5) and PPO (1.16e-5) A/Bs. The WS −6.5 % is now **adoptable**, not
   just measured.
2. **Paper-geometry correction (a stale-doc trap).** The ALFWorld probes were *not* "trimmed from
   paper 8192": the real paper geometry is the **bounded windowed template** — prompt 2048 /
   response cap **512** / max_model_len **2560** (`gen_paper_configs.py:148`). The 16384/8192
   figures in `config/envs/alfworld.yaml` and `reproducing.md` described the *retired* concat
   design (both fixed today). Measured windowed reality: response mean ~100 tok, 512-cap clip
   ratio 0.13 %. Consequence: the −57 %/−49 % ALFWorld results were effectively **at paper
   geometry all along** (probe caps were merely looser).
3. **1×H100 at true paper caps** (`accel/alfworld/alf_scale_g1_paper_r1.yaml`): step **514 s**
   (gen 212 / olp 52 / ref 105 / update_actor 139) vs 534 s at the old 4096/6144 caps — cap choice
   is immaterial, exactly as the env-bound model predicts. (The K=4 repeat arm was reclaimed with
   the allocation before finishing.)
4. **Port-band guard — first live catch at paper scale.** The first *real* paper-config launches
   (100 clients × K replicas) tripped the Tier-1 collision guard in 21 s: the per-client service
   band is `[base, base + total_clients×K)` — 800 ports for ALFWorld K=8 — and a `*_val_port`
   placed inside it is rejected before any GPU work. Deployment rule: **place `*_val_port` after
   `base + total_clients×K`** (fixed: ALF 43100, WS 10700). The ALFWorld wiring run itself landed
   same-day (`accel/alfworld/paper_alf_wiring_r8.yaml` — the first-ever run of the real ALFWorld
   paper config, carrying the adopted stack): **rc=0, 3719 s total** — round 1 (24-replica warm +
   base eval + 2 clients × 3 steps + eval + FedAvg/merge) **1766 s**, round 2 **1375 s**, cold
   final eval **578 s**; val success 0.0429 → **0.1143** (n=140); **70-round projection ≈ 27 h**
   (test_freq=5) ≈ 2.2× WebShop. The WebShop r4 re-run (`accel/webshop/paper_ws_wiring_r4.yaml`)
   hit the walltime and is queued for the next window.

## 9. Provenance

- Probes: `accel/alfworld/alf_scale_g4_r8dyn.yaml`, `alf_scale_g4_r8fused.yaml`;
  `accel/webshop/ws_scale_g4_dyn.yaml`, `ws_scale_g4_fused.yaml` (single-variable vs the 07-01
  baselines `g4_r8` / `g4_p64r4`). Logs: gitignored `runs/{alf,ws}_{g4dyn,fused}.log`.
- Phase decomposition: `runs/alf_em/worker_r8.log` timestamps.
- fused equivalence A/B (§8): `runs/fused_ab/{off,on}.log`; compare =
  `compare_fsdp_checkpoints.py --a <off actor> --b <on actor>` on `round_2/aggregated` (flags required).
- Effective-config extraction: `runs/alf_scale/g4_r8.log` config dump.
- Config-key trap (cost 3 failed launches): the fused backend is `model.fused_kernel_options.impl_backend`
  (a dict field, plain override — no `+`); `fused_kernels_backend` is the *function argument* in
  `apply_monkey_patch`, not a config key, and this verl's `HFModelConfig` rejects it as a kwarg.
- verl audit: `others/verl` @ the pinned 0.8 tree (`verl/workers/config/*.py`,
  `verl/experimental/one_step_off_policy/`, `verl/utils/seqlen_balancing.py`).
- 中文版: [acceleration_frontier_2026-07-02_cn.md](./acceleration_frontier_2026-07-02_cn.md)
