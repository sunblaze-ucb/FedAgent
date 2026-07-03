# Tier-2 optional acceleration knobs (2026-07-02) — implementation, validation, and the optimal combos

> **What this is.** Every remaining acceleration lever from the [frontier study](./acceleration_frontier_2026-07-02.md)
> — the four Tier-2 plumbing fixes, the #3 parallel-round lanes, fused-kernels adoption, and verl's
> experimental `one_step_off_policy` — implemented as **individually optional config knobs, all
> default-OFF** (default = byte-identical legacy behavior). Each knob is equivalence-gated
> (checkpoint `max|Δ| ≤ 1e-4` vs its off arm) and phase-timed; the **optimal combination** per
> environment is then measured on the **real paper config capped at 2 rounds** (the same wiring
> methodology as the 707/496/630 WebShop baseline and the 3719 s ALFWorld first run).
>
> Constants: Qwen2.5-1.5B, GRPO G=8, windowed (paper geometry), 4×H100 (qgpu3021).
> Companions: [acceleration_frontier_2026-07-02.md](./acceleration_frontier_2026-07-02.md) (the
> diagnosis these knobs answer) · [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md)
> (replica sharding) · [acceleration_report.md](./acceleration_report.md) (the lever stack).

---

## 1. The knob reference

| knob (run_fed key) | default | what it does | mechanism | science safety |
|---|---|---|---|---|
| `alfworld_manifest_cache` | `false` | caches the 8810-game manifest walk | the service's `collect_game_files` walk (os.walk + 2 JSON reads/game on GPFS) is identical for every service process; the first completed walk persists the **PRE-shuffle** list (atomic write, `(data_path, task_types)` key self-validation); shuffle/sharding/caps still run **natively** on identical input | byte-identical env streams; stale/foreign cache degrades to the full walk, never wrong data |
| `alfworld_manifest_dir` | `<repo>/runs/alfworld_manifest` | where manifests live | one file per split; persists across runs (the walk is paid ~once per cluster) | — |
| `service_scope: run` | `round` | keeps per-client env-service fleets **warm across rounds** | registry + LRU (cap `service_cache_clients`, default 4); reused fleets are health-checked and auto-restarted on failure; uniform-shard services are round-independent | same processes, same shards → byte-identical; eviction = plain teardown |
| `final_eval_mode: worker` | `subprocess` | scores model_T on the **hot engine** before worker teardown | an **eval-only plan** (round T+1, `eval_only` flag) reuses the worker's round-start eval path: reset to the final aggregate → `_worker_validate(T)` → skip fit | eval is read-only (cross-mode weight equivalence 3.8e-6/7.6e-6); falls back to the subprocess eval on any failure |
| `hf_export: final` | `every_round` | skips the per-round `model_merger` HF round trip | FedAvg still writes aggregated FSDP shards; the worker rebuilds each client engine from the BASE model (fresh optimizer/scheduler, as always) then **loads the aggregated shards model-only** via the engine's own FSDPCheckpointManager; HF is produced only for the final model | weights == the merge path's tensors (equivalence-gated); requires persistent/cross_round + (eval off or `eval_mode=worker`) |
| `parallel_clients: 2` | `1` | #3 lanes: the round's clients train **concurrently** on disjoint GPU halves | one long-lived worker per lane (per-lane `CUDA_VISIBLE_DEVICES` / `RAY_TMPDIR` / weight-transfer socket / service-URL file — the #3 concurrency fixes); lane 0 carries the worker-eval + final-eval duty | FedAvg is order-independent; per-client seeds derive from (round, client); **FSDP world size changes** (2 vs 4) → numeric-path change, equivalence-gated; small-model win only (1.5B: −35 % measured on WebShop) |
| `one_step_off` | `false` | **ADDITIONAL OPTION**: verl experimental `one_step_off_policy` | generation runs one step ahead on a dedicated GPU split (`rollout.n_gpus_per_node`); step wall → `max(gen, train)` | **OFF-POLICY by one step — outside the paper-reproduction bar.** Subprocess path only; requires explicit scientific sign-off; §5 |

Composition gates (enforced at startup): `hf_export=final` ⇒ persistent + (eval off ∨ worker);
`final_eval_mode=worker` ⇒ cross_round + worker (else warn + fallback); `parallel_clients>1` ⇒
cross_round + (eval off ∨ worker) + `P | n_gpus`; `one_step_off` ⇒ subprocess path only, no lanes.

## 2. Per-knob validation (equivalence + phase timing)

Equivalence rig: matched arms differing **only** in the knob, seed 42, cleanup off; compare the
round-2 aggregated actor (`compare_fsdp_checkpoints.py`, bar ≤ 1e-4). ALFWorld arms: 2c × 2r ×
1 step, r8, paper caps, 48-game val (worker mode). TinyGuess arms (for the trainer-plane knobs):
the established full-loop rig on the cross_round path.

**The reproducibility floor first.** The suite includes a same-config replicate (the pre-fix
cache arm ran with the cache inert — see §6 sys.path note — making it a pure base repeat):
**max|Δ| = 9.293e-5** between two identical-config runs. That is the GPU-nondeterminism floor
on this rig, sitting just under the 1e-4 bar. Every knob arm lands **at or below that floor** —
knob-induced divergence is indistinguishable from run-to-run noise.

| knob (ALFWorld arm, base wall 2185 s) | equivalence (max\|Δ\| vs base) | wall | timing effect |
|---|---|---|---|
| — same-config replicate (noise floor) | 9.293e-5 | 2043 s | −6.5 % = run-to-run noise band |
| `alfworld_manifest_cache` (warm manifest) | **9.199e-5 ✓** | 1784 s | **−18 %** — the 146 s/wave walk (tqdm 8810 games) skipped for all 24 services; ×3 waves at `scope=round` (r1+r2+val) ≈ the −401 s |
| `service_scope: run` | **9.090e-5 ✓** | 1843 s | **−16 %** — r2 fleet re-warm eliminated |
| `final_eval_mode: worker` | **9.241e-5 ✓** | 1941 s | **−11 %** — cold final-eval subprocess replaced by hot-engine scoring (eval-only plan verified: "scored round 2 model on the hot engine; no fit") |
| all four Tier-2 together | **8.752e-5 ✓** | 1669 s | **−24 %** — sub-additive: cache and scope overlap on the r2 wave |
| `hf_export: final` (TinyGuess rig) | **8.825e-6 ✓** (10× under the bar) | 271 s vs 427 s | shard-direct-load ≡ HF-merge path; wall −37 % on the tiny rig where mergers dominate (mechanism at scale verified in the `all` arm: round-1 HF deferred, worker shard-load) |
| `parallel_clients: 2` (TinyGuess rig) | **1.144e-5 ✓** (FSDP ws 2 vs 4; compared via the round-2 **HF exports** — `compare_hf_models.py`, since ws2-vs-ws4 shard sets can't be diffed directly) | 292 s vs 427 s | −32 % on the tiny rig |
| fused (WebShop) | **1.116e-5 ✓** (verified same-day, frontier §8) | — | −6.5 % step |

Note the two rigs' floors differ: the ALFWorld arms carry ~9e-5 of pure run-to-run noise
(hot-engine rollout at scale), while the TinyGuess trainer-plane arms reproduce to ~1e-5 —
which is why the lanes/hf_export deltas are 10× smaller despite being real numeric-path changes.

Manifest-cache evidence: first fleet writes (8× WROTE train + 8× eval, atomic last-writer-wins
on identical content), thereafter 24/24 HIT; `runs/alfworld_manifest/manifest_train.json`
(3553 games) + `manifest_eval_in_distribution.json` (140). Worth noting: the walk visits 8810
directories to keep 3553 — the cache also spares GPFS the 24-way concurrent stat storm.

## 3. The optimal combos — measured on the REAL paper config (2 rounds)

Per the project bar, the final stacks are measured exactly like the wiring baselines: the real
`uniform/1.5B/main/grpo/<env>` config, 70→2 rounds, same seeds/ports discipline.

**ALFWorld** (baseline: wiring_r8 = **3719 s** = r1 1766 + r2 1375 (incl. ~250 s redundant
re-warm) + cold final eval 578):

| stack | config | total | warm+boot+base eval | r1 fit+eval | r2 fit+agg+HF | final eval | Δ vs wiring_r8 |
|---|---|---|---|---|---|---|---|
| combo A = r8 + Tier-2 ×4 | `paper_alf_combo.yaml` | **3202 s** | 791 | 1252 | **762** | 389 (hot) | **−517 s (−13.9 %)** |
| combo B = A + lanes | `paper_alf_combo_lanes.yaml` | 3136 s | 856 | 1150 | 660 | 458 (hot, 2-GPU) | −583 s (−15.7 %) — **lanes ≈ wash on ALFWorld**: −102 s/round from env/gen overlap, given back via the slower 2-GPU final eval (+69) and dual worker boot (+65); the 1.5B fit is GPU-bound, so 2 clients × 2 GPUs ≈ sequential 4-GPU |

The 2-round total understates the win: one-time costs (service+worker cold start, base eval)
dominate a 2-round run. The number that scales is the **steady-state round: 762 s** vs the
wiring's 1125 s (1375 − the 250 s re-warm that `service_scope: run` now eliminates) = **−32 %
per round**; the hot final eval takes 389 s vs 578 s cold (−33 %). Mechanism evidence at paper
scale: 32 train-service manifest HITs + 8 val HITs (no tqdm walk anywhere), round-2 starting
model handed over as the FSDP shard dir (no per-round HF merge), r2 fleets reused warm.
Val trace (n=140): 0.0286 → 0.0214 → 0.0214 — at near-chance success rates the 140-game eval
is sampling-noise dominated (the wiring run scored 0.0429 → 0.1143 with equivalent weights);
the scientific gate is the checkpoint compare (§2), not the small-n val curve.

**WebShop** (baseline: same-day r4 wiring re-run = **2802 s**, phases 490/764/905/643):

| stack | config | total | warm+boot+base eval | r1 fit+eval | r2 fit+agg+HF | final eval | Δ vs r4 wiring |
|---|---|---|---|---|---|---|---|
| combo A = r4 + fused + Tier-2 (scope/feval/hf_export) | `paper_ws_combo.yaml` | **2309 s** | 769 | 802 | **402** | 326 (hot) | **−493 s (−17.6 %)** |
| combo B = A + lanes | `paper_ws_combo_lanes.yaml` | 2255 s | 761 | 782 | 362 | 337 (hot, 2-GPU) | −547 s (−19.5 %) — **lanes ≈ wash on WebShop too** (−20/−40 s per round, +11 s eval); the small-model probe's −35 % does not survive the real config, same fate as replicas |

Same shape as ALFWorld: the steady-state round collapses to **402 s** (vs the wiring's 905 s
r2 block) and the hot final eval to 326 s (vs 643 cold); what remains of the 2-round total is
dominated by the one-time warm+boot+base-eval block (769 s), which 70 rounds amortize away.
Val trace 0.012 → 0.010 → 0.020 (wiring: 0.018 → 0.012 → 0.030) — same small-n noise caveat.

**Recommended recipes** (final):

```yaml
# ALFWorld paper runs (= paper_alf_combo.yaml's knobs)
use_persistent_trainer: true
persistent_scope: cross_round
eval_mode: worker
alfworld_manifest_cache: true
service_scope: run
final_eval_mode: worker
hf_export: final
# NOT adopted: parallel_clients (wash: −2 % measured), replicas beyond r8 (Tier-1 verdict)

# WebShop paper runs (= paper_ws_combo.yaml's knobs)
use_persistent_trainer: true
persistent_scope: cross_round
eval_mode: worker
service_scope: run
final_eval_mode: worker
hf_export: final
client_overrides: "+actor_rollout_ref.model.use_fused_kernels=True ..."   # fused: −6.5 % step, 1.116e-5
# NOT adopted: parallel_clients (wash: −2.3 %), replicas (wash at paper config, Tier-1)
```

Excluded from both recipes: `one_step_off` (off-policy — §5), lanes (measured wash on both
envs: the 1.5B fit is GPU-bound, so 2 concurrent 2-GPU clients ≈ sequential 4-GPU, and the
2-GPU hot final eval gives back most of the overlap win), extra replicas (WebShop wash;
ALFWorld r8 is already in the baseline).

## 4. 70-round projections

Formula (test_freq = 5): `T(70) ≈ one_time + 70 × steady_round + 14 × eval_due + final_eval`,
all terms measured in the same-day 2-round paper-config runs (steady_round = the r2 block —
fit + FedAvg (+ merge in the wiring's case, + re-warm in the wiring's `scope: round` world);
eval_due = the per-eval wall; one_time = warm + worker boot + base eval).

| env | wiring stack | optimal combo | Δ |
|---|---|---|---|
| ALFWorld | 791 + 70×1375 + 14×578 + 578 ≈ **29.3 h** | 791 + 70×762 + 14×389 + 389 ≈ **16.7 h** | **−43 %** |
| WebShop | 490 + 70×905 + 14×643 + 643 ≈ **20.4 h** | 769 + 70×402 + 14×326 + 326 ≈ **9.4 h** | **−54 %** |

(The earlier "ALFWorld ≈ 27 h" estimate used the same wiring numbers with a coarser eval
model — consistent. The combo's `service_scope: run` term is conservative here: with 2/100
clients per round over 70 rounds, repeat draws land in the LRU warm cache and shave the
fleet start further; the projection prices every round as a fresh-fleet round.)

## 5. `one_step_off` — the additional option (off-policy; sign-off tier)

verl 0.8 ships `experimental/one_step_off_policy`: rollout workers live on a dedicated GPU split
and generate batch t+1 while batch t trains — step wall `gen + train → max(gen, train)`.
Wired here as `one_step_off: true` (subprocess client path; entry `fedagent.main_one_step_off`,
config `fedagent_one_step_off.yaml` layering upstream's deltas on `fedagent_ppo_body`; GPU split
via `client_overrides`, e.g. `trainer.n_gpus_per_node=3 rollout.n_gpus_per_node=1`).

Wiring this surfaced a hydra constraint worth recording: `hydra.searchpath` may only be
overridden from the PRIMARY config, so a second entry point cannot simply list `fedagent_ppo`
in its defaults. `fedagent_ppo.yaml` is therefore split into `fedagent_ppo_body.yaml` (all
FedAgent leaves, no hydra block) + a thin primary that adds only the searchpath — entry
points layer the body and declare their own searchpath. Composition-tested: the paper path's
resolved leaves are unchanged.

**Why it is NOT in any recommended combo:**
1. **It changes the algorithm** — the update batch was sampled from the previous step's weights
   (one-step off-policy). Learning curves are not expected to match the paper; adoption is a
   *scientific* decision, not an engineering one.
2. **FedAgent's round structure truncates the win**: with 3 optimizer steps per client-round, the
   pipeline drains at every round boundary, so the steady-state −35 % is never fully realized.

**Wiring record — five layers deep, all fixed in-repo.** Standing the experimental trainer
up under FedAgent surfaced, in order: (1) the hydra searchpath rule (→ the
`fedagent_ppo_body` split above); (2) upstream `validate_config` requires
train-world-size | real_train_batch_size (64) → GPU split **2+2**, not 3+1; (3)
`OneStepOffRayTrainer` asserts the disaggregated layout → `hybrid_engine: False` (+
`load_format: safetensors`, `layered_summon: True`), which upstream only sets in its shell
examples; (4) upstream copies the top-level `rollout:` resource block into
`actor_rollout_ref.rollout` in its own `main()`, not in the task runner → mirrored in
`fedagent.main_one_step_off`; (5) `bypass_mode=True` requires per-token
`rollout_log_probs` in the batch — upstream's base runner threads them from
`AgentLoopOutput.response_logprobs` (`agent_loop.py:944`), so **both FedAgent loops now
emit them** (mirroring `tool_agent_loop`: accumulate the server's `log_probs` per generated
segment; concat mode pads `0.0` on obs tokens; `None` whenever the server wasn't asked for
log-probs, i.e. `calculate_log_probs=false` → the legacy path is byte-identical). This
plumbing also unlocks verl's rollout-correction family on the main path.

**Probe result (WebShop paper geometry, 1 client × 1 round × 3 steps, 2 train + 2 gen
GPUs; total 683 s):** step walls **116.2 → 64.9 → 71.7 s**. Step 1 pays the blocking
pipeline-prime generation (`timing_s/gen` 44.4 s); steps 2–3 show `gen ≈ 0` — the next
batch's generation (`generate_async` 71.2 / 64.3 s) runs entirely hidden under training,
i.e. the step wall is `max(gen, train)` exactly as advertised, and at 2+2 the two sides
are nearly balanced. Against the serial 4-GPU reference (`ws_scale_g4`, 93.4 s/step) the
steady-state step is **−23…−31 %** — but a FedAgent client-round is only 3 steps, so the
round pays the prime every time: 116+65+72 = 253 s vs 3×93.4 = 280 s ≈ **−10 % realized**.
The "round structure truncates the win" argument above is now a measured fact, which —
together with the off-policy bar — keeps this an unadopted ADDITIONAL OPTION.

## 6. Provenance & incidental findings

**The sys.path shadowing fix (independent latent bug, caught by this suite).** The pre-fix
manifest-cache arm ran with the cache silently inert, which is how we found it: the
`verl-agent-alfworld` conda env carries an *editable* verl-0.3.1 install whose `.pth` files
put the ORIGINAL verl-agent repo on `sys.path` at interpreter startup; `agent_system` is a
namespace package, so the vendored engine path the service *appended* always LOST the
resolution race — ALFWorld services had been importing the un-vendored engine all along
(identical copies masked it until the cache edit landed only in the vendored copy). Fix:
`fedagent/envs/alfworld/service/server.py` now `sys.path.insert(0, _ENGINE)`. WebShop is not
affected (`web_agent_site` exists only in the vendored engine). Consequence recorded above:
the void arm became the same-config replicate that measures the reproducibility floor.

**Cold-probe pessimism (steady-state correction).** Same-day WS wiring showed hot-engine
steady-state steps at ~50 s (gen ~10 s) vs ~82 s (gen ~36 s) in cold single-step probes —
cross-step prefix caching plus a warm engine make steady state ≈ 3× cheaper than probe
arithmetic suggests. Probe-derived per-step numbers in the frontier doc are systematically
pessimistic for long runs; the combo tables above (real 2-round runs) supersede them.

- Implementation: `fedagent/fed/run_fed.py` (knobs, lanes, hot final eval, shard-load gating,
  service registry), `fedagent/fed/persistent_task_runner.py` (eval-only plans, shard-aware
  reset), `fedagent/fed/persistent_patch.py` (`reload_client_model(..., shard_dir)`),
  `fedagent/envs/alfworld/engine/.../alfred_tw_env.py` (manifest cache, surgical),
  `fedagent/envs/alfworld/service/server.py` (sys.path fix), `fedagent/main_one_step_off.py`,
  `fedagent/config/fedagent_one_step_off.yaml` + the `fedagent_ppo_body.yaml` split (§5).
- A/B configs: `accel/dev/t2_*.yaml`, `accel/alfworld/alf_t2_*.yaml`; combos:
  `accel/alfworld/paper_alf_combo*.yaml`, `accel/webshop/paper_ws_combo*.yaml`;
  probe: `accel/dev/oso_probe.yaml`. Drivers: `accel/run_t2_stack.sh` (chained after the WS
  r4 wiring), `accel/run_oso_probe.sh` (post-fix re-run). Tools: `compare_fsdp_checkpoints.py`,
  `compare_hf_models.py` (NEW — world-size-independent HF diff, used for the lanes arm).
  Logs under gitignored `runs/t2_*`, `runs/paper_*/combo*`, `runs/oso/`.
- 中文版: [acceleration_tier2_2026-07-02_cn.md](./acceleration_tier2_2026-07-02_cn.md)
