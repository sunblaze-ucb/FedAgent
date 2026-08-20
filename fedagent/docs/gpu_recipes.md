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
| `1` | wiring checks / debug | smokes only: WebShop is **GPU-bound**, so the step time is ~the whole compute term on one card; [running.md](./running.md#hardware-recipe) recommends a small backbone + small config here | **legitimate budget option**: the paper geometry is GPU-verified no-OOM on 1×H100, and the bottleneck is the env service, not the GPU |
| `2` | the smoke default (`DEFAULTS`) | fine for smokes; ~half the paper recipe's compute | fine, same reasoning as 1 GPU |
| `4` | **the paper recipe**: every `paper/` and `paper_accelerated/` config pins it | ✅ the sweet spot: FSDP world size **is** WebShop's lever | ✅ paper-validated; the wins come from `alfworld_replicas`, not the count |

**The rule behind the table** (acceleration.md's "30-second rule"): run one training step at
two GPU counts and read the **gen** term of `timing_s`. Gen **scales** with GPUs ⇒ GPU-bound
(WebShop) ⇒ add GPUs. Gen **flat** ⇒ env-bound (ALFWorld) ⇒ add `<env>_replicas: K`, more
GPUs mostly idle against the env-service lock.

Per-count best practice:

- **1 GPU**: debugging, partition/wiring checks (`--rounds 2`), and ALFWorld-on-a-budget.
  Expect paper-scale WebShop to be painfully slow; that is the bottleneck class, not a bug.
  If you combine 1 GPU with `cross_round: true` (e.g. a 0.5B budget run on a 24 GB card),
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
```

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
