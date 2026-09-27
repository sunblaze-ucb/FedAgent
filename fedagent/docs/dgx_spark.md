# Running FedAgent on NVIDIA DGX Spark (GB10)

A community-validated recipe for running the FedAgent trainer on one **DGX Spark**
(GB10 Grace-Blackwell: SM121 GPU, aarch64 CPU, CUDA 13, **~120 GB of memory shared by CPU
and GPU**). The paper recipe targets x86 + H100 (see [gpu_recipes.md](./gpu_recipes.md)); this page
lists only what differs on GB10.

Status (2026-09-25): the TinyGuess federated loop (2 clients × 2 rounds, FedAvg, windowed
rollout, `--n-gpus 1`) completes end to end with non-zero reward at Qwen2.5-1.5B-Instruct (88 GiB peak)
and Qwen2.5-3B-Instruct (105 GiB peak) with the §2.4 settings.
WebShop/ALFWorld service envs have **not** been validated on aarch64 yet.

## 1. The stack: verl 0.8 on vLLM 0.20

No vLLM release within verl 0.8's `vllm<=0.12` pin runs on GB10. vLLM 0.20 does, and
verl 0.8's rollout/weight-sync glue works against it unchanged. So the recipe keeps the
`vllm/vllm-openai:v0.20.0` image's stack and installs verl 0.8 `--no-deps` on top of it:

| Component | Paper (H100) | DGX Spark |
|---|---|---|
| base | conda, Python 3.12 | `vllm/vllm-openai:v0.20.0` (Python 3.12.13) |
| torch | 2.8.0+cu128 | 2.11.0+cu130 (from the image) |
| vLLM | 0.11.0 | 0.20.0 (from the image; outside verl 0.8's pin) |
| transformers | 4.x | 5.6.2 (from the image; needs the `return_dict=False` fix, §3) |
| verl | 0.8.0 (`7aed6b2`) + the 2-line patch | the same, installed `--no-deps` |
| flash-attn | 2.7.4.post1, `FLASH_ATTN_CUDA_ARCHS=90` | **2.8.3**, `FLASH_ATTN_CUDA_ARCHS=120` (SM121 runs sm_120 cubins) |
| numpy | 2.2.6 | 2.2.6 (verl's `numpy<2` metadata is stale; `pip check` warns, harmless) |

[`tools/setup/Dockerfile.dgx-spark`](../../tools/setup/Dockerfile.dgx-spark) builds all of this
(10–15 min, mostly flash-attn). The key detail is a pip constraints file built from the image's own
`pip freeze`, so no later install can replace the GB10-working torch/vLLM:

```bash
docker build -f tools/setup/Dockerfile.dgx-spark -t fedagent:dgx-spark .
docker run --rm -it --gpus all --ipc=host --network host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v $PWD/outputs:/outputs \
  fedagent:dgx-spark python -m fedagent.fed.run_fed \
    --config <config.yaml> --n-gpus 1 --output-dir /outputs/<run>
```

flash-attn numerics on GB10 (causal, B=2, L=512, H=8, D=64, compared with SDPA): max |Δ| 9.8e-4 (bf16),
2.4e-4 (fp16), forward and backward.

Image last validated 2026-09-25 on an earlier snapshot of this branch (fixes 1–3 as first committed, no
`unified_memory.py`): unit suite 171 passed, 7 skipped (ALFWorld game data, spaCy), and the command above
with `fedagent/config/examples/tinyguess_2cl_2rd.yaml` exited 0 in 546 s. Rebuild before relying on it at
a later commit.

## 2. Memory: one pool for the trainer, vLLM and the host

On an H100, `rollout.gpu_memory_utilization=0.5` takes 40 GB of **dedicated** HBM. Ray, the dataloaders,
the CPU-offloaded ref model, and checkpoint save/merge all live in separate host RAM. On GB10 all of it
comes out of **one 119.6 GiB pool** (`free -m`: 122502 MiB), and the same 0.5 takes ~60 GB of it. With
the default body config a 1.5B windowed run fills the node, and Ray's memory monitor (threshold 0.95)
kills workers during `update_weights` (`ray.exceptions.OutOfMemoryError`, plus `NVRM … NV_ERR_NO_MEMORY`
in dmesg). verl's `perf/cpu_memory_used_gb` is `psutil.virtual_memory().used`, a system-wide number that
on GB10 includes GPU allocations; it is not a host-only figure.

All numbers below: one GB10, TinyGuess windowed, 1 step per client-round, page cache dropped first,
~4.6 GiB taken by an unrelated resident process. Peaks are system `used` from a 1 s `free -m` sample,
converted to GiB.

### 2.1 Rollout share (Qwen2.5-1.5B, 2 clients × 2 rounds, stock verl placement)

| rollout `gpu_memory_utilization` | exit | wall | peak used | Ray OOM kills |
|---|---|---|---|---|
| 0.50 (body default) | ✗ | — | ~114 GiB | fatal in `update_weights` |
| 0.30 | 0 | 14.1–14.3 min | 106.9–115.6 GiB (3 runs) | 0 |
| 0.25 | 0 | 13.9 min | 114.8 GiB | 0 |
| 0.20 | 0 | 13.5 min | 113.2 GiB | 0 |

Below 0.5 the rollout share stops being what fills memory. The rest of this section is where it goes.
(An earlier revision of this page listed `ref.fsdp_config.param_offload=false` as a lever. **It is a no-op
in verl 0.8**: the FSDP engine forces every forward_only engine to `CPUOffload(offload_params=True)`
whatever that key says (`transformer_impl.py`, `_build_fsdp_module`). The difference we attributed to it
was run-to-run noise: three 0.30 runs spread 106.9–115.6 GiB. §3.4 is the real fix.)

### 2.2 Where it goes

A 1 s profile (system `used`, per-process RSS, per-pid GPU memory, second-stamped log) of the training
worker (`ray::WorkerDict`), GPU MiB from `nvidia-smi` in GiB:

| verl phase | round 1 | round 2, before §3.3 | round 2, §3.3 |
|---|---|---|---|
| `init_model` | 13.5 | 19.1 | 13.5 |
| `compute_ref_log_prob` | 18.4 | 24.2 | 18.4 |
| `update_actor` (fp32 params + grads + Adam) | 37.0 | 42.8 | 37.0 |
| `save_checkpoint` | 37.5 | 43.4 | 37.5 |
| `update_weights` (vLLM wakes) | 45.9 | 51.8 | 45.9 |
| **system peak (whole run)** | | **115.6** | **104.6** |

Three things fill memory:
1. **Round 2 held one extra fp32 copy of the ref model (+5.8 GiB).** A real leak, fixed in §3.3. The copy is
   GPU allocator memory, so the leak is expected on other single-GPU (ws=1) systems too, but it was measured
   on GB10 only.
2. **The host-offloaded ref.** verl keeps the ref on the host even when told not to (§2.1). On GB10 that
   host copy is the same memory: the trainer's RSS grows ~13 GB at `init_model` and another ~16 GB at the
   first `compute_ref_log_prob` (FSDP's pinned staging, never returned; `malloc_trim(0)` frees nothing).
   Fixed in §3.4.
3. **The allocator cache at the weight copy.** `update_weights` wakes vLLM's weights first and empties the
   trainer's cache last. Measured at 1.5B at that moment: 17.3 GiB allocated (params 5.75 + Adam 11.5),
   36.7 GiB reserved, so ~19 GiB of freed backward/optimizer blocks sit in the trainer while vLLM
   re-allocates next to them. At 3B it is ~45 GiB and the node OOMs exactly there. Fixed in §3.4.

### 2.3 With the §3.4 fixes

Qwen2.5-1.5B, 2 clients × 2 rounds, rollout 0.3, same config, A/B on `unified_memory` (ref placement only;
the pre-sync release landed after this pair):

| | stock (`unified_memory: off`) | ref on GPU |
|---|---|---|
| system peak | 102.4 GiB | **88.3 GiB** |
| trainer host RSS at the peak (1-client run) | 26.3 GB | 3.9 GB |
| `timing_s/ref` | 15.2–16.0 s | **4.8–5.2 s** |
| `timing_s/save_checkpoint` | 9–11 s (one 22 s) | 25–26 s |
| wall | 745 s | **671 s** |

`save_checkpoint` writes ~17 GiB (fp32 params + Adam) per save through the page cache. Its time swings
9–26 s under both placements (the first large save in a fresh process is the slow one), but in this pair
the ref-on-GPU saves were consistently slow; the step is net ~3 s slower while the run is 10% faster. We
have not pinned down the mechanism.

Qwen2.5-3B-Instruct, 2 clients × 2 rounds, rollout 0.3:

| config | worker GPU at `update_weights` | system peak | result |
|---|---|---|---|
| §3.4(a) ref on GPU only | 96.5 GiB | 115.9 GiB | Ray OOM kill in `update_weights`, round 1 client 0 |
| + §3.4(b) pre-sync release | 78.6 GiB | 112.5 GiB | killed in `update_actor` (trainer 80.6 GiB) |
| + `ref.fsdp_config.model_dtype=bf16`, `actor.ppo_micro_batch_size_per_gpu=2` | 55.8–72.0 GiB | **105.2 GiB** | **exit 0, 22.4 min, all 4 client-rounds OK** |

3B step: `ref` 4.3–4.6 s, `update_actor` 67–69 s, `save_checkpoint` 59–81 s (~37 GiB per save),
`update_weights` 3.6–3.8 s, step 170–186 s.

**`ref.fsdp_config.model_dtype=bf16` is exact.** verl's engine default is `model_dtype: fp32`, so the ref is
stored in fp32 and cast to bf16 by `MixedPrecision(param_dtype=bf16)` on every forward. Storing it in bf16
gives the same bf16 weights (the HF checkpoint is bf16; an fp32 aggregate rounds the same way either path)
and the same fp32 buffers. Checked on Qwen2.5-1.5B, 581 tokens: max |Δ log p| = 0.0, bit-identical. It halves
the ref (−5.8 GiB at 3B) on any hardware.

### 2.4 Recommendation

```yaml
unified_memory: auto          # default; §3.4
client_overrides:
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.3
  - actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
  # 3B: - actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
```
plus the §3.3 fix. 1.5B then peaks at 88 GiB, 3B at 105 GiB of 119.6.

Also on unified memory:
- **Drop the page cache before a run** (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`). GB10
  counts page cache as used, and vLLM's startup free-memory check sees that.
- **CPU offload does not create capacity.** "CPU" and "GPU" are the same physical memory, so
  `*_offload=true` only adds copies.
- **`DataLoader worker (pid …) is killed by signal: Killed`** at the end of a client's last step is a
  teardown artifact (Ray tears down the TaskRunner after the checkpoint is written); the client-round is
  still reported OK and aggregated.

## 3. Code fixes found on this platform

All four are on the `dgx-spark` branch. 1–3 are platform-independent; 4 is gated to integrated GPUs:

1. **`gym_text_agent_loop._tokenize_chat`: `return_dict=False`.** transformers ≥5 makes
   `apply_chat_template(tokenize=True)` return a `BatchEncoding`, so `AgentLoopOutput.prompt_ids` fails
   pydantic validation on the first rollout. verl's own wrapper already pins this. No-op on 4.x.
2. **TinyGuess under the default windowed rollout never showed the rules.** `WindowedGymTextAgentLoop`
   sends a single user message = `obs_str` with no system turn. TinyGuess's rules lived only in
   `system_prompt()`, so the policy saw only `Make your first guess as <answer>N</answer>.` and the
   shipped smoke scored **0.0 on every rollout on any hardware**. TinyGuess now builds rules + the last
   N guesses + feedback when `FEDAGENT_HISTORY_LENGTH>0`, as WebShop/ALFWorld do. Concat mode is
   byte-identical. With the fix, 25/192 windowed samples score 1.0 at 1.5B (0.5B mostly fails to bisect
   even in concat mode; use 1.5B for a smoke that shows reward).

3. **`ref_anchor`: the ref was built twice on round ≥ 2, and the first copy was never freed.** verl's
   `init_model` builds the ref from a deepcopy of the actor's model config, which from round 2 on points at
   the round's aggregate. `ref_anchor` then dropped the ref and rebuilt it at the base. On ws=1
   (NO_SHARD), FSDP1's flat params outlive the dropped module, the same leak
   `persistent_patch._hard_release_fsdp_storages` already handles for the actor:
   `memory_allocated` was 11777 MiB before the drop, 11777 MiB after drop + gc + `empty_cache`, and
   17666 MiB after the rebuild. The fix sets the ref's own `local_path` to the base just before verl
   constructs the ref's `TrainingWorker`, so it is built once. Measured: round-2 worker GPU equals round 1,
   system peak drops from 115.6 to 104.6 GiB, and wall time from 844 to 746 s. The old rebuild remains as
   a logged fallback, and it still strands a copy. `tests/test_ref_anchor.py` runs verl's real `init_model`
   offline (Ray, the process group and the engines faked) and fails if the ref is loaded from anything but
   the base, or loaded twice.

4. **`fedagent/unified_memory.py`: two verl 0.8 placements that are wrong on unified memory** (§2.2, items 2–3).
   (a) verl forces the forward_only ref to `CPUOffload(offload_params=True)`; on an integrated GPU the ref
   now stays on the device (the patch rebinds `torch.distributed.fsdp.CPUOffload` for the duration of that
   one `_build_fsdp_module` call; actor/critic untouched; FSDP2 left stock and logged).
   (b) `ActorRolloutRefWorker.update_weights` now runs verl's own `aggressive_empty_cache` once more at its
   start, before vLLM's weights are resumed. Knob: `unified_memory: auto|on|off` (run_fed →
   `FEDAGENT_UNIFIED_MEMORY`, armed via sitecustomize). `auto` (default) applies both iff
   `torch.cuda.get_device_properties().is_integrated` (GB10: 1), so discrete-GPU runs are byte-identical to
   stock. Weights, dtypes and forwards are unchanged; only placement and timing. It is not fail-closed: if
   a verl upgrade moves either seam, the run logs a warning and keeps stock placement. Measured in §2.3;
   tests in `tests/test_unified_memory.py` (CPU-only, plus one fresh-interpreter check against the real
   verl classes). Worth fixing upstream in verl: honour `ref.fsdp_config.param_offload`, and empty the
   cache before `resume(tags=["weights"])`.

## 4. Environment variables

Set in the image. Listed here in case you build your own env:

| Variable | Why |
|---|---|
| `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1` | vLLM's usage thread calls `cpuinfo`, which raises `JSONDecodeError` on aarch64 (noise only) |
| `VLLM_USE_DEEP_GEMM=0`, `VLLM_SKIP_DEEP_GEMM_WARMUP=1` | DeepGEMM has no SM121 kernels |
| `CUDA_HOME=/usr/local/cuda` | flash-attn / JIT builds |

## 5. Not yet validated

- WebShop (JDK/pyserini) and ALFWorld (fast-downward) service envs on aarch64.
- Multi-GB10 (`--n-gpus 2` across two Sparks): the runner is single-node, so this is out of scope.
- Paper-length runs (70 rounds). Only smoke-length runs have been measured.
- Models above 3B. 3B fits with 14 GiB to spare; 7B (fp32 params + grads + Adam ≈ 112 GiB) does not fit
  a single GB10 without optimizer offload/sharding.
- Round-2 `init_model` is still slower than round 1 after §3.3: 64 / 55 s per client (101 / 97 s before)
  against 22–29 s, from 1 s process samples. Not diagnosed; candidates to profile are the aggregate's
  model/config load, checkpoint reads, and rollout init.
