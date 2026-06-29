# FedAgent verl-0.8 Migration — Complete Engineering Report

> **The definitive, end-to-end account** of migrating FedAgent from the vendored **verl-agent 0.3.1 fork**
> to **stock verl 0.8** as a thin overlay: the route decision, the environment build, the hard
> compatibility problems and how each was solved, the two rollout modes, the fidelity record, and the
> verification journey.
>
> **The migration docs, by purpose:**
> - [architecture.md](architecture.md) — **how the overlay is built** (extension points, the round loop, data flow).
> - [migration.md](migration.md) — the **fidelity record** (what changed, the science-critical alignments, verification status).
> - **this doc** — the **complete walkthrough**: strategy + the hard technical problems *in depth* + the journey.
> - [acceleration_report.md](acceleration_report.md) — the separate **acceleration & validation** workstream (built *on* this migration).
>
> **Conventions.** Branch `migrate/verl-0.8.0`. Bar = **scientific equivalence** (reproduce the paper's
> conclusions within seed noise, *not* bit-identical curves). GPU = 4×H100 via `srun --overlap`. verl 0.8
> source (editable): `others/verl`. Reference for the overlay pattern: VAGEN-Lite (`others/VAGEN`).

---

## 1. The mission and the bar

FedAgent is **federated RL for LLM agents**: each round, a few clients each train a local policy (GRPO/PPO,
multi-turn rollouts against WebShop/ALFWorld), then the server **FedAvg**s their weights. The original code
**forked verl-agent 0.3.1** and wove federation into the trainer. This migration re-implements it as a **thin
overlay on stock verl 0.8 — no fork** — so the framework tracks upstream without fork maintenance.

The **bar is scientific equivalence**: reproduce the paper's *conclusions* (the input-dynamics asymmetry, the
heterogeneity sweep orderings, the baseline relationships) within 3-seed noise. This permits using verl 0.8's
native rollout instead of bit-reproducing the fork's, but it forbids any change to the **science red lines**:
task-vs-env heterogeneity independently sweepable; deterministic per-client-id assignment
(`RandomState(base_seed+client_id)`, `base_seed=42`); validation always on the **unperturbed** env; uniform
FedAvg; FedProx anchored to the round-start weights; the budget-matched `N=100 / M=2 / T=70 / E=3` protocol.

## 2. The route decision — fork → overlay

The fork (0.3.1) patched `verl/trainer/.../ray_trainer_fed.py` for **one** reason: to inject
`traj_collector.multi_turn_loop` (multi-turn rollout) + a GiGPO estimator into the trainer. **verl 0.8 has a
native `AgentLoopManager`** (async multi-turn rollout as a first-class seam) — so that reason is **gone**. The
chosen route ("**Route B**", VAGEN-style): a separate `fedagent/` package that **imports verl 0.8 as a library**
and plugs into its public extension points, driving **stock `RayPPOTrainer`** unchanged.

| extension point | what FedAgent plugs in |
|---|---|
| `data.custom_cls` | `AgenticDataset` — emits one env-spec row per env instance (not static text) |
| agent-loop registry (`agent.yaml`) | `GymTextAgentLoop` — multi-turn rollout on verl's async seam |
| Hydra `searchpath` | `fedagent_ppo.yaml` layered on verl's stock `ppo_trainer` |
| interpreter startup (`sitecustomize.py`) | FedProx proximal term, gated on `FEDPROX_MU` |
| process boundary (HTTP) | the WebShop / ALFWorld remote env services |

Two key codebases survive **verbatim** (verl-agnostic → zero migration risk): the heterogeneity constructions
(`partition_strategy.py`, ~3.7k LOC — the science crown jewel) and the WebShop/ALFWorld **engines** (the MDP).
The 0.3.1 `core/` control plane (~2.8k LOC of server + script-builder) is **not** carried over — it assumes a
`config['verl']` schema + `model_world_size_1` single-rank checkpoints; the lean `fed/run_fed.py` replaces it.

## 3. The environment (`fedagent-verl08`) — the dependency saga

A new conda env (**py3.12**): verl 0.8 + **vllm 0.11.0** + sglang 0.5.2 + flashinfer 0.3.1, FSDP-only
(`USE_MEGATRON=0`). The build surfaced five traps worth recording (all fixed; scripts in `tools/verl08_migration/`):

1. **flash-attn is MANDATORY.** verl 0.8's `ray_trainer` calls `unpad_input → flash_attn.bert_padding`
   **unconditionally** — `sdpa` does *not* avoid it (an early belief that was wrong). Prebuilt wheels need
   GLIBC_2.32 (node has 2.28) → **build from source**: FA 2.7.4.post1, `FLASH_ATTN_CUDA_ARCHS=90`, conda gcc-11 +
   cuda-12.1 nvcc, `--no-deps` (`build_fa.sh`).
2. **Never `pip install --force-reinstall` without `--no-deps`** — it cascades into bare `torch`, pulls
   torch 2.12+cu130 from PyPI, breaks CUDA on the 12.8 driver. Correct torch = **2.8.0+cu128** (the vllm 0.11 pin).
3. **CUDA-13-era `nvidia-*-cu13` pip packages clobber the cu12 `.so`s** (shared `nvidia/<lib>/lib/` namespace,
   last-installed wins) → torch loads NCCL 2.29.7 on a 12.8 driver → `ncclUnhandledCudaError` at FSDP
   param-broadcast. Fix: uninstall the cu13 orphans + `--force-reinstall` the torch trio (archived at
   `_scratch/archived_diagnostics/_fix_nvidia_stack.sh`).
4. **sglang pulled numpy 2.4** (breaks vllm's numba, needs ≤2.2) → pin `numpy==2.2.6`.
5. verl `copy_to_local` rejects a model path with a trailing `/`.

## 4. Architecture in brief

Two planes (full detail: [architecture.md](architecture.md)). **Control plane** — `fed/run_fed.py`, the
federated round loop; it **never imports verl** (a client is just a subprocess). **In-framework hooks** —
`envs/`, `agent_loops/`, `data/`, `fedprox.py`, run *inside* the verl client process via the extension points.

The round loop (subprocess-per-(client,round) is the *original* path; the persistent/cross-round accelerations
are [acceleration_report.md](acceleration_report.md) §4):

```
ROUND r:  for each selected client c (sequential):
            python -m fedagent.main_ppo_fed   model.path=model_r   FEDAGENT_BASE_SEED=base_seed+r*100+c
            → round_r/client_c/.../actor   (FSDP shards, ws = n_gpus)
          FedAvg:  torchrun aggregate_fedavg_fsdp.py  → round_r/aggregated/.../actor
          merge:   verl.model_merger  → round_r/aggregated/hf   → model_{r+1}
```

`model_1 = base`; `model_r = round_{r-1}/aggregated/hf`. PPO (`gae`) federates the **critic** the same way.

## 5. The hard problems solved

### 5.1 Checkpoint compatibility — the FSDP-shard saga

verl 0.8 saves FSDP checkpoints as per-rank `model_world_size_{WS}_rank_{R}.pt` + a **new** `fsdp_config.json`
(records FSDP version + world_size). The migration's hardest early question was *how to FedAvg these shards*.

- **A false alarm, then the truth.** A synthetic spike suggested FSDP1 saves **ShardedTensor** that **can't** be
  `torch.load`ed single-process ("world size at save 2, at load 1"). On **real** verl-0.8 training checkpoints,
  the params are **DTensor** and load single-process **fine** — the ShardedTensor error was an artifact of the
  spike's own synthetic save path, not verl's.
- **FedAvg = write-back, under a matched process group.** The validated/safe method: run aggregation as
  `torchrun --nproc_per_node=ws aggregate_fedavg_fsdp.py` — each rank loads **its own** rank shard from every
  client, averages the local values **in place** (`_get_local` handles DTensor/ShardedTensor/plain), and
  `torch.save`s back — **byte-structurally identical** to a verl save, so the next round loads it unchanged.
  *Do not* load verl shards into a freshly-wrapped model to re-save: verl's transformer auto-wrap shards params
  differently than a whole-model wrap → type-mismatch under SHARDED_STATE_DICT. **Validated FedAvg-exact**
  (`max|resumed − mean| = 0.0`).
- **Re-entry = Option C (model_merger → HF → `model.path`).** Round r+1 == round 1 with `model.path` swapped to
  the merged HF dir (fresh optimizer — exactly the original's "load-aggregated, fresh-optimizer"). `model_merger`
  reads HF config from `<local_dir>/huggingface` and writes a complete HF dir, no patching. **Caveat:**
  `model_merger` casts to **bf16**, so each round boundary truncates the fp32-aggregated weights — within the
  equivalence bar, but switch to Option B (`resume_from_path`, model-only load) if Phase-8 shows drift.

### 5.2 The async agent-loop seam

verl 0.8 generation is **async-only**, via `experimental/agent_loop/`, dispatched **per row** (each `AgentLoop`
sees **one** dataset row). The fork's batched-synchronous `multi_turn_loop` has no equivalent → envs must become
**per-instance async** (`reset/step/system_prompt/close`). The overlay's `GymTextAgentLoop` (`@register("gym_text")`,
an `AgentLoopBase` subclass) drives one `BaseTextEnv` per row on the native seam
(`reset → generate → parse action → env.step → …`), returning one concat `AgentLoopOutput` whose `response_mask`
is 1 on agent tokens, 0 on observation tokens (so PPO/GRPO trains only on actions). **Phase 0(b) proved the
seam**: a custom AgentLoop on **stock** `main_ppo` ran a full GRPO loop and emitted canonical 0.8 FSDP layout —
mutually consistent with what the FedAvg step (5.1) consumes.

### 5.3 Remote env services (dependency isolation)

WebShop (Java/pyserini/gym 0.24), ALFWorld (TextWorld/Fast-Downward), and the trainer (verl 0.8) have **mutually
conflicting deps**. So each env runs as its **own HTTP service in its own conda env**, one **per client**; the
trainer (`fedagent-verl08`) talks to them over HTTP. The services `sys.path`-inject the **vendored engines** — the
*same code the original ran* — so the MDP is unchanged. `run_fed` starts only the round's selected clients' services
(≤ `clients_per_round` alive at once), waits `/health`, tears them down per round before aggregation; one shared
**unperturbed val** service stays up for the whole run.

Two robustness findings: (i) the **`/step` storm** — retries under load replayed steps out of order → made services
**idempotent per `step_id`** (per-session `asyncio.Lock` + single-slot replay cache + 409 on out-of-order). (ii) the
**ALFWorld throughput bottleneck** is a process-global `_TW_LOCK`: the tatsu PDDL parser is a mutable singleton, so
all textworld ops serialize (measured 86 ms/step → ~13.7 s/client of serialized env-stepping; the fork parallelized
it across Ray-actor processes, one env each). The fix (engineering only, ALFWorld-only) is N worker processes, one
textworld env each — not yet applied.

### 5.4 Heterogeneity injection

The partition constructions are copied **verbatim** (numpy-only bodies; the file itself imports matplotlib so only
the functions are lifted), keyed by `RandomState(base_seed + client_id)` → a client's shard is bit-identical to the
0.3.1 baseline. Two levels: **env-level** (catalog) is **service-side** — `run_fed` passes
`PARTITION_STRATEGY/CLIENT_ID/ENV_DIV/…` as env vars, the service builds *that client's* catalog from the real
shuffled `server.goals`; **task-level** (goal distribution) is the `AgenticDataset._partition_specs` seam. **Critical
fix:** `WEBSHOP_SERVICE_URL` (env) must be **authoritative** over the spec's `service_url` — otherwise both clients
hit `:8080` and the heterogeneity **collapses**.

### 5.5 FedProx without a fork

FedProx is a one-method patch of `FSDPEngine.optimizer_step` (snapshot `w_t` on the first step, add `mu·(w − w_t)`
thereafter), applied via the repo-root **`sitecustomize.py`** (auto-imported at interpreter startup in every process
— client + Ray workers — gated on `FEDPROX_MU`). It is deliberately **not** a Ray `runtime_env` worker hook: that
**clobbered verl's per-worker `CUDA_VISIBLE_DEVICES`** assignment → "Duplicate GPU detected". `mu=0` → plain FedAvg.

## 6. The two rollout modes — the faithful-vs-stock axis

verl 0.8's native rollout is **concat** (one sample per episode, full verbatim history); the paper trains
**windowed** (per-turn samples, `history_length=2` legacy template). Both are kept, switched by one flag
`rollout_mode: windowed (default) | concat`, GPU-validated.

**Why windowed is hard on stock verl.** Per-turn expansion makes one prompt yield *many* training rows, but stock
verl 0.8 **hard-enforces 1 training sample per input prompt** (`fit()` slices the gen output to
`num_sampled_prompts`; `_validate()` unions test batches 1:1). A row-expanding manager is **silently truncated** in
train (corrupt: episode-uid ↔ turn-row misalign) and **crashes eval** (`AssertionError: 4 vs 76`). The fix (in
`windowed_manager.py`, **no verl fork**):
- scoped monkeypatch of `DataProto.slice` (tagged expanded batch → don't truncate) + `DataProto.union` (tagged
  `other` of different length → adopt it, pad to the mini-batch divisor, drop the tag), guarded by `len != len` so
  matched-size unions and **all** eval are untouched;
- **mini-batch divisibility** — `make_iterator` hard-asserts `batch % mini_batch == 0`; `use_dynamic_bsz` does *not*
  help (it only affects the micro split). `_compute_size_divisor = lcm(ppo_mini·n, world_size, [micro·ws], [critic
  terms if gae])` + pad the dynamic per-turn batch to it (mirrors the fork's `adjust_batch`);
- **two cascade fixes** because `union` returns `other` (discarding what stock would merge from `self`): re-merge
  `meta_info` (carries `temperature`, else `KeyError`) and force-add the per-turn `non_tensor` (carries `uid`, else
  `KeyError` at advantage); eval **collapses** each episode to 1 row (last turn + broadcast return) to stay 1:1.

**Faithfulness corrections** (two earlier beliefs were wrong): (1) the paper's GRPO did **not** per-trajectory
dedup — the fork's `seen_pairs` dedup is gated on a flag whose default disables it, so mean/std is over **all**
per-turn samples = **stock verl 0.8 grpo**. So no custom estimator is needed; `grpo_traj` is opt-in only. (2) the
**invalid-action penalty is per-turn** (−0.1 at turns whose own action was invalid), broadcast on top of the base
episode return — not a uniform per-episode subtraction.

**The A/B (answers "会有加速吗"): windowed is ~1.47× *slower*, not a speedup.** Same 16 episodes/step:
windowed 58.5 s/step vs concat 39.8 s; the gap is the **vLLM prefix-cache break** (windowing shifts the window each
turn → cache miss; gen 43.0 vs 30.1 s) + per-turn expansion (160 vs 16 training rows → `update_actor` 5.8×).
**Windowed is chosen for faithfulness + long-episode feasibility** (ALFWorld 50 turns: concat context blows up and
truncates at the response cap; windowed stays bounded) — **not** speed.

## 7. Fidelity record (condensed)

Full record: [migration.md](migration.md). The science-critical alignments verified during migration:

- **Algorithm** — GRPO **G=8** (`rollout.n=8`); stock verl multiplies `ppo_mini_batch_size` by `rollout.n`
  internally, so the original's "1 update/rollout-batch" is `ppo_mini_batch_size=8` (GRPO) / 64 (PPO), **not** 64×8.
- **Trajectories/step = `train_batch_size × rollout.n`** for PPO *and* GRPO (unconditional in the fork source) →
  original ran GRPO 8×8=64, PPO 64×8=512; **`rollout.n` must stay 8 for PPO** (dropping to 1 is unfaithful).
- **Sparse reward** `{0,10}` + per-turn `0.1×n_invalid` penalty (moved to the agent-loop; same total/episode).
- **Round-threaded data seed** `base_seed + round*100 + client` (a plain modulo collapsed the round term → every
  client saw the same goals every round).
- **Full E epochs/round** (`total_training_steps: 0` → no smoke step-cap leaks into paper runs); val on the shared
  **unperturbed** service, temperature 0.4, held-out split.
- **Config-generator fixes** (`gen_paper_configs.py`): WebShop `search_return_n` 200 (env-het) / **50** (elsewhere,
  matching the original baselines); ALFWorld `max_turns=50` + a 16384-token context for 50-turn transcripts.

## 8. Verification status

| path | status |
|---|---|
| TinyGuess (in-process) | ✅ GPU end-to-end |
| **WebShop GRPO federated** | ✅ GPU full 2-round loop (train → FedAvg → merge → round 2 → eval) |
| **WebShop PPO (gae critic federation)** | ✅ GPU (windowed smoke: actor **and** critic FedAvg + merge) |
| **windowed GRPO + PPO** train+eval+loop | ✅ GPU-green (per-turn rows trained, eval per-episode, no 4-vs-76 crash) |
| **concat** rollout | ✅ GPU-green (1 sample/episode; isolation confirmed — no windowed monkeypatch loaded) |
| ALFWorld (service + `max_turns=50`) | code-audited; GPU-VERIFY pending (OOM/truncation at 50 turns) |
| acceleration overlay (#4 / eval modes / #3) | ✅ GPU + equivalence — see [acceleration_report.md](acceleration_report.md) |

The migration also unlocked the **acceleration & validation** workstream (persistent trainer, eval modes,
client-parallel, the equivalence A/Bs) — built on top of this overlay and documented separately.

## 9. The config matrix

The paper configs (`fedagent/config/paper/`) mirror the original tree 1:1 — `uniform/<model>/<setting>/<algo>/`,
`env_heterogeneity/`, `task_heterogeneity/`, `decentralized/`, **176 configs** (model sizes
1.5B/3B/7B + Llama-3.2-3B; GRPO + PPO; WebShop + ALFWorld; the heterogeneity arms). One intentional deviation:
centralized/local baselines use `T=70 × E=3` (= 210 epochs) rather than 1 round × 210, because the runner draws goal
variety from **rounds** (the round-threaded seed) — same total epochs, correct coverage. Full matrix → run commands:
[reproducing.md](reproducing.md).

## 10. Gotchas & operational notes

- **Compute-node `/tmp` is invisible from the login node** — out-of-band `ls`/`cat` on a run's checkpoints must go
  through `srun --overlap --jobid=<JID>`. Put scripts on GPFS (`_scratch/`), not the login-local scratchpad.
- **Don't `pkill -f <pattern>`** when `<pattern>` appears in the wrapper's own command line — it self-matches and
  kills the srun step (instant exit, zero output). `run_fed` manages service lifecycle anyway.
- The benign `RuntimeError: DataLoader worker ... killed by signal: Killed` at interpreter teardown (atexit, after
  `fit()` saved) is **noise** (exit 0) — see the §9 of [acceleration_report.md](acceleration_report.md) for the case
  where it was mistaken for a crash cause (it wasn't).
- Config rename in 0.8: `checkpoint.contents` → `checkpoint.save_contents` / `.load_contents`.

## 11. Open items

- **ALFWorld 50-turn GPU-verify** — confirm no OOM / prompt truncation at the 16384-token / 50-turn budget.
- **ALFWorld service parallelism** — replace the process-global `_TW_LOCK` with N worker processes (one textworld
  env each) → ~22% faster ALFWorld rollout (engineering only, no science impact).
- **Full paper reproduction** — wiring validated (see acceleration_report §10); the 3-seed × model × env × algo ×
  heterogeneity matrix is a multi-node, multi-day campaign.
- **byte-exact windowed-obs vs legacy** — the one remaining faithfulness audit.

---

## Appendix — phase timeline

| phase | milestone | status |
|---|---|---|
| 0(a) | checkpoint round-trip + FedAvg (matched-PG, write-back) | ✅ exact |
| 0(b) | custom async AgentLoop on stock `main_ppo` (the seam proof) | ✅ |
| 1 | `fedagent/` package (entry, config, base env, agent-loop, dataset) | ✅ GPU (TinyGuess) |
| 2 | WebShop as a remote service | ✅ GPU smoke |
| 3 | ALFWorld + the two rollout modes (concat / windowed) | ✅ GPU (windowed + concat green) |
| 4 | heterogeneity (env-level service-side + task-level dataset seam) | ✅ (env-level GPU; first federated science run) |
| 5 | FedProx (`sitecustomize`) + `json_logs/metrics.json` logger | ✅ |
| 6 | FedAvg aggregation core (`aggregate_fedavg_fsdp.py`) | ✅ validated on real verl checkpoints |
| 6/7 | the federated loop closed (`run_fed.py`, model_merger re-entry) | ✅ GPU |
| 8 | scientific-equivalence validation | ✅ (GRPO/PPO A/Bs — acceleration_report §8) |
| (accel) | persistent trainer, eval modes, client-parallel | ✅ — [acceleration_report.md](acceleration_report.md) |
