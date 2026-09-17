# GPU recipes: 1 / 2 / 4 GPUs, and the accelerated paper matrix

One page: **which GPU count to use for what, what actually gets faster in each environment,
and the ready-made accelerated paper configs.** The mechanics of the GPU knobs live in
[running.md](./running.md#hardware-recipe); the measured record behind every number here is
[acceleration.md](./acceleration.md).

`n_gpus_per_node` (CLI `--n-gpus`) is the **FSDP world size on one node**, and the FedAvg
aggregator runs `torchrun --nproc_per_node=<ws>` against the saved shard layout, so training
and aggregation share the value. Pick a count per run and **keep it fixed for that run**
(resume included). The runner is single-node; there is no `nnodes`.

---

## Which GPU count (the one-table answer)

| `--n-gpus` | Use it for | WebShop | ALFWorld |
|---|---|---|---|
| `1` | debug, **and the single-H100 paper recipe** (`paper_accelerated_1gpu/`, [below](#the-single-h100-tree-configpaper_accelerated_1gpu)) | at 1.5B a full cell fits (peak 50 GiB GRPO / 64 GiB PPO) at about **×2** the 4-GPU per-round time: 17 min/round GRPO, 21 min PPO (the 4×H100 PPO twin: 10.6 min) | same: 40 min/round GRPO, 55 min PPO on one card (on ONE GPU ALFWorld is compute-bound, not env-bound: `update_actor` + `ref` are 55–72 % of the step) |
| `2` | the smoke default (`DEFAULTS`) | fine for smokes; ~half the paper recipe's compute | fine, same reasoning as 1 GPU |
| `4` | **the paper recipe**: every `paper/` and `paper_accelerated/` config pins it | ✅ the sweet spot: FSDP world size **is** WebShop's lever | ✅ paper-validated; the wins come from `alfworld_replicas`, not the count |

**The rule behind the table** (acceleration.md's "30-second rule"): run one training step at
two GPU counts and read the **gen** term of `timing_s`. Gen **scales** with GPUs ⇒ GPU-bound
(WebShop) ⇒ add GPUs. Gen **flat** ⇒ env-bound (ALFWorld) ⇒ add `<env>_replicas: K`, more
GPUs mostly idle against the env-service lock.

Per-count best practice:

- **1 GPU**: debugging, partition/wiring checks (`--rounds 2`), and since 2026-09-10 a
  **measured paper-scale option at 1.5B**: `paper_accelerated_1gpu/` (below) trades about ×2
  per-round wall-clock for ×4 fewer GPUs, so a cell costs roughly half the GPU-hours and four cells
  share one node. If you combine 1 GPU with `cross_round: true` (e.g. a 0.5B budget run on a 24 GB card),
  you need the 2026-08-18 reload hard-release fix ([bugfixes.md](./bugfixes.md)): ws=1
  degrades FSDP to NO_SHARD, and on older checkouts each client reload strands ~one fp32
  model copy (~1.33 GiB at 0.5B) → OOM at a headroom-dependent round.
- **2 GPUs**: smokes. On a 4-GPU node this also leaves 2 GPUs free for a second *small* run;
  give it its own `--output-dir` + `--port-base` ([running.md](./running.md#concurrent-runs-on-one-node)).
- **4 GPUs**: all paper runs, GRPO and PPO. Memory at the shipped settings: 1.5B fits
  comfortably (GRPO `gpu_memory_utilization=0.6`, PPO `0.5` + optimizer offload already in the
  configs); for larger backbones or tighter cards use the offload table in
  [running.md](./running.md#cpu-offload-and-gpu-memory-via-client_overrides).
- **>4 GPUs / multi-node**: not wired ([running.md](./running.md#honest-scope)).

---

## The accelerated paper matrix: `config/paper_accelerated/`

Every one of the 176 `config/paper/**` cells has an **accelerated twin at the same relative
path** under [`../config/paper_accelerated/`](../config/paper_accelerated/): same partition,
same seeds, same federation protocol, same eval cadence; only the fixed costs (engine
cold-starts, service restarts, cold evals) are removed. Every knob in the stack is
**A/B-equivalent**: final aggregated models match the legacy path within the measured
**9.3e-5** GPU-nondeterminism floor ([acceleration.md](./acceleration.md#why-its-safe-the-equivalence-bar)).

```bash
# any paper cell, accelerated, just swap paper/ -> paper_accelerated/:
python -m fedagent.fed.run_fed --config \
  fedagent/config/paper_accelerated/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

What a twin adds on top of its `paper/` original:

| knob | effect | envs |
|---|---|---|
| `cross_round: true` | ONE trainer+vLLM process for the whole run, the dominant win | both |
| `eval_mode: worker` + `final_eval_mode: worker` | per-round and final eval on the hot engine (no cold eval engines) | both |
| `service_scope: run` | per-client env-service fleets stay warm across rounds | both |
| `alfworld_replicas: 8` (+ pool 8→64) | shards the TextWorld process lock, env-step −57 % | ALFWorld |
| `alfworld_manifest_cache: true` | skips the 8810-game directory walk on warm boots (−18 %) | ALFWorld |
| fused log-prob/entropy kernels (triton) | −6.5 % on the GPU-bound step | WebShop, **Qwen2.5-1.5B twins only** (the backbone the A/B ran on; add the two `client_overrides` lines by hand to try another) |

Measured on the real 1.5B paper configs (4 GPUs): steady round **905 → 402 s** (WebShop) and
**1125 → 762 s** (ALFWorld); full 70-round budget **≈ ×3.5 / ×2.5** less wall-clock
([acceleration.md](./acceleration.md)).

Deliberate choices baked into the twins:

- **`hf_export: every_round`, not `final`.** Round-level resume scans per-round HF exports
  ([running.md](./running.md#resume)), so the twins keep them; a 10–17 h run that can hit a
  walltime limit should be resumable. If a run fits comfortably inside one allocation, flip to
  `hf_export: final` for the recipe's last saving (it skips the per-round FedAvg-merge-to-HF
  pass).
- **No `webshop_replicas`.** Measured a wash at paper scale (WebShop is GPU-bound); its
  absence is intentional, not an omission.
- **Disjoint port bands, all outside the kernel ephemeral range.** Twins use WebShop `22528+` /
  ALFWorld `28672+` (originals: `10000+` / `16384+`), so a cell and its twin can share a host.
  Every band sits below 32768 — a band inside the ephemeral range (32768–60999) can be squatted
  *mid-run* by any process binding port 0, which killed a round-13 client before the
  [2026-08-19 fix](./bugfixes.md). ALFWorld twins need wide bands (`replicas=8` ⇒ the 100-client
  band is 800 ports), so their 80 configs cycle 4 1024-port blocks, and the 128-block trees cycle
  48 — configs one cycle apart share a band **by design**: `run_fed` preflights the block at
  startup and relocates it (into the reserved `[61000, 65536)` pool) if it is occupied, so
  co-hosted configs deconflict themselves. `service_port_autoshift: false` opts out.

Regenerate the whole tree (it is generated, never hand-edited):

```bash
python tools/gen_paper_configs.py --accel
python tools/gen_paper_configs.py --accel --n-gpus 1   # -> config/paper_accelerated_1gpu/ (below)
```

### The single-H100 tree: `config/paper_accelerated_1gpu/`

`--n-gpus 1` emits the same 194 accelerated cells with `n_gpus_per_node: 1` (FSDP world size 1,
NO_SHARD). The science is untouched (prompts, `rollout.n`, minibatch, lr, KL, seeds, eval cadence);
only per-GPU memory placement changes: vLLM `gpu_memory_utilization` 0.5 (GRPO) / 0.4 (PPO)
because actor (+ critic) and engine share one 80 GB card, GRPO gains
`actor.fsdp_config.optimizer_offload=true` (PPO already had it), and every config carries a
`port_band_base` cycling over four disjoint 3300-port bands (5000/8400/11800/15200) so **four
single-GPU cells can share one 4-GPU node**. To co-host them, launch each driver with
`CUDA_VISIBLE_DEVICES=<k>` (its physical card) and its own `--output-dir`: `run_fed` maps its lane
pins through the driver's `CUDA_VISIBLE_DEVICES` (2026-09-10; before that the pin was the literal
`"0"`, so every co-hosted cell landed on physical GPU 0). Do not rely on `srun --overlap
--gres=gpu:1` steps for the split: overlapping steps are all handed the same GPU.

#### Measured: one H100 per cell (2026-09-10 → 09-13)

Qwen2.5-1.5B uniform cells, seed 42, four cells concurrently on one 4×H100 node, engine-default
Lucene backend, `ref_anchor: round` (the default of the time), three SLURM allocations with automatic
resume. A round = 2 clients × 3 epochs + FedAvg/merge + three 64-episode evals (the aggregate and
the two client-end circles); 25–33 % of it lies outside the training steps.

| cell (1 × H100) | median min/round | of which training (6 steps) | wall-clock | GPU·h | last-10 rounds |
|---|---|---|---|---|---|
| WebShop GRPO | **17.4** | 12.6 | 19.9 h / 70 rounds | 19.9 | task 0.802 / success 0.664 (best 0.901 @ r65) |
| WebShop PPO | **21.1** | 15.5 | 24.5 h / 70 rounds | 24.5 | task 0.794 / success 0.681 (best 0.828 @ r68) |
| ALFWorld GRPO | **39.8** | 25.9 | 44.7 h / 70 rounds | 44.7 | success 0.534 (r70 0.594) |
| ALFWorld PPO | **54.9** | 36.3 | 46.5 h / 54 rounds (~60 h projected) | 46.5 | success 0.336 (best 0.594 @ r39, regresses after r40; one seed) |

**How much slower than four GPUs: about ×2, not ×4.** The backend-matched 4×H100 twin of the
WebShop PPO cell (same accelerated config, cluster, backend and seed; 2026-09-09/10) ran at a
**median 10.6 min/round** (14–16 min in rounds 2–9, 8.5–10 min in rounds 61–70), ≈13 h and ≈50 GPU·h
for 70 rounds — so one H100 is **×2.0 slower per round for ×4 fewer GPUs: half the GPU·h per cell**,
and four cells finish together on one node instead of queueing. The training-step-only comparison
says the same: WebShop GRPO 12.6 min/round vs the 4-GPU steady round of 402 s (×1.9), ALFWorld GRPO
25.9 min vs 762 s (×2.0). Per-GPU memory: PPO peaks at 64 GiB of 80 (the checkpoint/reload phase;
actor update 45–51 GiB, vLLM ~36 GiB while awake — it sleeps during training, which is what makes
one card fit), GRPO at 50 GiB; four cells kept host RAM under 0.5 TB.

Caveats before quoting it: (1) **endpoints, not mid-run curves** — the single-GPU WebShop PPO cell
has a deep trough in rounds 19–34 (task 0.12–0.20, 84 % zero-score episodes while train reward keeps
rising) that recovers to the 4-GPU endpoint (0.794 / 0.681 vs 0.804 / 0.659); (2) one seed per cell;
(3) the `ref_anchor: base` A/B arm run on the first freed GPU (2026-09-12/13) held entropy 1.07–1.24
all run with no trough and finished at 0.780 / 0.609 — the evidence behind the 2026-09-16 default
flip ([revision.md](./revision.md)); (4) on one GPU an ALFWorld cell is compute-bound (gen 15–16 %,
`update_actor` + `ref` 55–72 % of the step), so "ALFWorld is env-bound" is a 4-GPU statement.

Co-hosting recipe, one driver per card (each config already carries its own port band):

```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python -m fedagent.fed.run_fed \
    --config fedagent/config/paper_accelerated_1gpu/uniform/Qwen2.5-1.5B-Instruct/main/grpo/<cell_$k>.yaml \
    --output-dir outputs/1gpu/cell_$k > logs/cell_$k.log 2>&1 &
done
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv   # four drivers, four distinct UUIDs
```

A cell that outlives its allocation resumes on the next one from `round_k/aggregated/{hf,critic_hf}`
([running.md § Resume](./running.md#resume)); a directory from before 2026-09-16 needs
`ref_anchor: round` in its config to continue under the objective it started with.

---

## What *not* to reach for

Measured dead ends ([acceleration.md](./acceleration.md#why-each-lever-works), "Measured and
rejected"): `parallel_clients` lanes on one node at 1.5B (wash on top of this stack, it stays
the multi-node-style lever), `use_dynamic_bsz` (slower on both envs), `one_step_off`
(**off-policy**: never for paper numbers).

With the stack on, the biggest remaining cost is **eval cadence**: one n=500 WebShop eval
(~630 s) outweighs a steady training round (402 s). Decide `client_end_eval` (the paper
figures' per-client circles; every paper config ships `true`) *before* launching.

---

## See also

- [acceleration.md](./acceleration.md): the final recipe, why each lever works, the
  equivalence bar.
- [running.md](./running.md): the driver, CLI flags, offload table, resume, SLURM pattern.
- [reproducing.md](./reproducing.md): which config backs which paper number; compute budget.
- [installation.md](./installation.md): the three conda envs and per-env data.
