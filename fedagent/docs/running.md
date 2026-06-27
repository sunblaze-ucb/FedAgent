# Running FedAgent

How to run the federated loop with [`fed/run_fed.py`](../fed/README.md) — the spine of the
**thin overlay on stock verl 0.8**. There is no trainer fork: each client is a plain
subprocess (`python -m fedagent.main_ppo_fed`); the driver orchestrates the rounds. For the
config-key reference see [configuration.md](./configuration.md); for the env/conda setup see
[installation.md](./installation.md); for the paper experiments see
[reproducing.md](./reproducing.md).

A run is a single **config YAML** handed to the driver. The YAML is the source of truth for
the hardware recipe (GPU/FSDP-world-size, memory, offload) and the federation protocol; a
handful of CLI flags override the most-swapped keys (seed, ports, rounds, FedProx).

```bash
python -m fedagent.fed.run_fed --config fedagent/config/<name>.yaml
```

> **Single node, sequential clients.** This runner is **single-node** — `n_gpus_per_node`
> is the FSDP world size on one box; there is **no multi-node (`nnodes`) wiring** in
> `run_fed.py`. Within a round the selected clients train **one after another** (the driver
> loops `for c in selected:` and waits for each subprocess before the next). See
> [Honest scope](#honest-scope) before planning for parallelism or multiple nodes.

## Basics

Run the driver inside the **`fedagent-verl08`** conda env, on a GPU node, from the repo root.
For WebShop/ALFWorld, `run_fed.py` **launches the per-client env services itself** (one
service per client, each in its own service conda env) and tears them down at the end — you
do not start them manually. TinyGuess runs in-process (no service). The mode (federated /
centralized / local) and the algorithm (GRPO / PPO) are **implied by the config**, not by a
flag — see [Run-mode matrix](#run-mode-matrix) and [Algorithm: GRPO vs PPO](#algorithm-grpo-vs-ppo).

The `fedagent/config/fed_*.yaml` (hand-written, repo top of `config/`) are small smokes for
wiring checks; the paper grid lives under `fedagent/config/paper/`. See
[Smoke tests](#smoke-tests) and [Worked examples](#worked-examples).

## CLI flags → config keys

Every flag overrides the matching key from the YAML (which itself overrides the
[`DEFAULTS`](../fed/run_fed.py) in `run_fed.py`). Only these flags exist — everything else is
set in the YAML or via `client_overrides`.

| Flag | Config key it overrides | Default (in `DEFAULTS`) | Use |
|---|---|---|---|
| `--config <yaml>` | — | — | the federated config (you almost always pass this) |
| `--model-path <dir>` | `model_path` | `""` → auto-discover Qwen2.5-0.5B | base model for round 1 (offline: a local HF snapshot) |
| `--output-dir <dir>` | `output_dir` | `/tmp/...tinyguess` | where `round_*/`, logs, checkpoints, summary land |
| `--rounds <T>` | `total_rounds` | `2` | shorten/lengthen the run |
| `--clients <N>` | `total_clients` | `2` | also caps `clients_per_round` to ≤ N |
| `--n-gpus <k>` | `n_gpus_per_node` | `2` | FSDP world size (e.g. `4` for a 4-GPU node, `1` for debug) |
| `--base-seed <s>` | `base_seed` | `42` | seed replication (client selection + per-client env seed) |
| `--port-base <p>` | `webshop_base_port` | `8080` | run two WebShop jobs on one node without port clashes |
| `--fedprox-mu <mu>` | `fedprox_mu` | `0.0` | `>0` enables FedProx (else FedAvg) |
| `--local-client-id <k>` | `local_client_id` | `-1` | Local baseline: pin client k (no federation) |

> `--port-base` overrides **only** `webshop_base_port`. For ALFWorld, set
> `alfworld_base_port` (and the val ports `webshop_val_port` / `alfworld_val_port`) in the
> YAML if you need to deconflict concurrent runs — there is no CLI flag for those.

## Run-mode matrix

The mode is selected by the config, not a flag. `run_fed.py` derives it as: `local` if
`local_client_id ≥ 0`; else `centralized` if `total_clients ≤ 1`; else `federated`.

| Mode | How to select (YAML / flag) | What happens | Services launched |
|---|---|---|---|
| **Federated** (default) | `total_clients: N>1`, `local_client_id: -1` | each round samples `clients_per_round` clients, trains each **sequentially**, then FedAvg → merge → next round starts from the merged model | lazily, per round: one per **selected** client only (the round's sample), torn down after the round; the val service (if eval on) is always-on |
| **Centralized** | `total_clients: 1` (and `clients_per_round: 1`, `partition_strategy: ""`) | one model on the pooled (unpartitioned) data; FedAvg of a single client is the identity, so the loop is just `total_rounds × epochs_per_round` of continued training | one (client 0, full env) |
| **Local** | `local_client_id: k ≥ 0` (with `total_clients: N`) | the paper's "Local Agent Training": pin client `k`'s slice of the N-way partition, train it alone every round, no federation | only the one pinned client `k` |

**Launch each:**

```bash
# Federated (the default — any multi-client config)
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/coverage.yaml

# Centralized (total_clients=1 baked into the config)
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/centralized.yaml

# Local (local_client_id baked into the config…)
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/local.yaml
# …or pin a client of an existing federated config from the CLI:
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/coverage.yaml \
  --clients 2 --local-client-id 0
```

In Local mode only the pinned client's env service is started (`participating_client_ids`
returns `[k]`), so it is cheaper than the matching federated arm.

## Algorithm: GRPO vs PPO

Set by `adv_estimator` in the config (no flag):

- **GRPO** (default, `adv_estimator: grpo`) — group-relative advantage, **no critic**. The
  group size is the rollout `n` (set per config in `client_overrides`, e.g. `rollout.n=2`
  for smokes, `8` for the paper recipe).
- **PPO** (`adv_estimator: gae`) — the value model (critic) is **federated alongside the
  actor** each round. Round-1 critic = the base model (random value head on the backbone);
  thereafter the aggregated critic carries forward via `critic.model.path`. PPO configs
  carry the critic block in `client_overrides` (e.g.
  [`examples/webshop/scaled/ppo.yaml`](../config/examples/webshop/scaled/ppo.yaml)). If any selected
  client fails to emit a critic checkpoint, the round aborts — keep
  `critic.checkpoint.save_contents=[model]` in the overrides.

```bash
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/ppo.yaml
```

## Hardware recipe

`n_gpus_per_node` (or `--n-gpus`) is the **FSDP world size** used for both training and
aggregation: the FedAvg step launches `torchrun --nproc_per_node=<world_size>` to match the
saved shard layout (`model_world_size_<ws>_rank_*.pt`), so the value you train with and the
value you aggregate with are the same key. The paper recipe is **4 GPUs on one node**. Memory
sizing (rollout length, batch, pool, offload) is set per config in `client_overrides`. There
is no separate tensor-parallel knob in the overlay's spine — the per-client subprocess uses
verl's stock FSDP rollout under this world size.

| `n_gpus_per_node` | Typical use | Notes |
|---|---|---|
| `1` | single-GPU debug / wiring check | use a small backbone (0.5B) + small config; lower `rollout.n`, batch, pool; expect offload (below). Not paper-scale. |
| `2` | the smoke default (`DEFAULTS`) | TinyGuess / WebShop smokes on a 2-GPU slice |
| `4` | **the paper recipe** | Qwen2.5-1.5B @ 15 turns; GRPO and PPO both validated here |

### CPU offload and GPU memory (via `client_overrides`)

The spine forwards every `client_overrides` entry verbatim as a Hydra override to the
per-client subprocess, so FSDP offload and the vLLM memory fraction are tuned **per config**,
not by a flag. The keys that matter for fitting a run:

| Override key | What it does | When to set it |
|---|---|---|
| `actor_rollout_ref.actor.fsdp_config.param_offload` | offload trainable **params** to CPU | larger backbone / batch won't fit; trades throughput for capacity |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | offload the **optimizer state** to CPU | same; PPO sets this `true` to free GPU for the resident critic |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM KV-cache fraction of GPU mem | lower it when the actor (and critic, for PPO) crowd the rollout; PPO uses `0.5`, GRPO ~`0.6` |
| `critic.fsdp.param_offload` / `critic.fsdp.optimizer_offload` | PPO critic offload | **verl 0.8** puts the critic FSDP config at `critic.fsdp.*` (not `critic.model.fsdp_config`) |

Example — turn on actor offload and lower the KV fraction for a tighter run:

```yaml
client_overrides:
  - actor_rollout_ref.actor.fsdp_config.param_offload=true
  - actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

Or as a one-off on the command line (each override is a positional arg after the flags):

```bash
python -m fedagent.fed.run_fed --config <...> --n-gpus 1 \
  client_overrides='[actor_rollout_ref.rollout.n=2,actor_rollout_ref.rollout.gpu_memory_utilization=0.4]'
```

## FedProx

```bash
python -m fedagent.fed.run_fed --config <...> --fedprox-mu 0.1
```

`fedprox_mu > 0` sets `FEDPROX_MU` in the client subprocess environment;
[`sitecustomize.py`](../../sitecustomize.py) (repo root, on the client + Ray workers'
`PYTHONPATH`) reads it at interpreter startup and adds the proximal term at the FSDP optimizer
step. `mu = 0` → plain FedAvg. It is injected via `sitecustomize` **not** a Ray
`runtime_env` hook (the hook clobbered verl's per-worker `CUDA_VISIBLE_DEVICES`). Eval passes
strip `FEDPROX_MU`, so validation never enables the term. A ready pair is
[`examples/webshop/scaled/envhet_fedprox.yaml`](../config/examples/webshop/scaled/envhet_fedprox.yaml)
(FedProx, `mu=0.1`) vs its FedAvg twin.

## Seeds

`base_seed` (or `--base-seed`) threads two places, both deterministic on resume:

- **Client selection** — `select_clients` seeds its RNG with `base_seed + round − 1`, so the
  per-round sample is reproducible.
- **Per-client env instance** — each client subprocess gets
  `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id`, so a client re-draws goals from
  its **fixed** shard every round (covering the shard over `T` rounds) while staying distinct
  from other clients.

Three-seed replication is just the same config three times with `--base-seed 42 / 21 / 13`
(use distinct `--output-dir` and, for concurrent WebShop runs, `--port-base`).

## Validation / eval

Eval is **off** unless `val_env_spec` is set (back-compat default `""`). When set, the driver
starts **one shared, unperturbed** val service (`partition_strategy` forced empty / uniform —
no client skew) and scores the aggregated **global** model:

| Key | Effect | Default |
|---|---|---|
| `val_env_spec` | the unperturbed val env-spec; `""` → no eval | `""` |
| `test_freq` | eval the global model every K rounds (+ always the final round) | `5` |
| `val_before_train` | also eval the **base** model before round 1 (the round-0 point) | `true` |
| `val_temperature` | val sampling temperature (`val_kwargs.temperature`) | `0.4` |

The round → success/reward curve is written to `federated_summary.json` (`val_curve`). A
failed eval logs a warning and continues — it never aborts the run.

## Concurrent runs on one node

Two jobs on the same node must not collide on env-service ports. Give each a distinct
`--port-base` (WebShop client `c` → `port_base + c`) and a distinct `--output-dir`:

```bash
python -m fedagent.fed.run_fed --config <...> --base-seed 42 --port-base 8080 --output-dir /tmp/run_s42 &
python -m fedagent.fed.run_fed --config <...> --base-seed 21 --port-base 8120 --output-dir /tmp/run_s21 &
```

For ALFWorld, set `alfworld_base_port` in each YAML (no CLI flag). Remember both jobs share
the node's GPUs — with the 4-GPU recipe, two full runs will not both fit; concurrency is for
small/offloaded runs or different GPU slices.

## Honest scope

- **Clients run sequentially within a round.** The driver loops `for c in selected:` and
  blocks on each client's subprocess (`stream(...)` waits via `proc.wait()`), with a
  `wait_between_clients`-second pause between them to let Ray/GPU fully release. There is **no
  parallel-client execution** — a round's wall-clock is the sum of its clients.
- **Single-node only.** `n_gpus_per_node` is the FSDP world size on **one** box. There is no
  `nnodes` setting and no multi-node launch in `run_fed.py`; the aggregator runs
  `torchrun --nproc_per_node=<ws>` locally. Multi-node is **not implemented**.
- **No legacy launchers.** The old `reproduce.sh` / `run_federated.py` /
  `start_federated.sh` path (and its `parallel_workers` knob) does **not** apply here; the
  only entry is `python -m fedagent.fed.run_fed`.

## Running on SLURM (srun)

The driver is a normal Python process — you do not `sbatch` a special script. Get an
interactive GPU allocation (or attach to an existing job) and run the driver on the node with
`srun --overlap`. Everything runs inside the **`fedagent-verl08`** env; the WebShop/ALFWorld
**services are launched by the driver itself** in their own service envs
(`verl-agent-webshop` / `verl-agent-alfworld`), so those envs must exist on the node but you
do not activate them by hand. See [installation.md](./installation.md) for the three envs.

The real pattern (mirrors [`fedagent/scripts/run_smoke.sh`](../scripts/run_smoke.sh) and
[EXPERIMENTS.md](../EXPERIMENTS.md)) — attach to a running job `<JID>`:

```bash
# 1) get / identify a GPU allocation
#    (e.g. salloc --gres=gpu:4 ... ; or reuse an existing job id)
JID=<your_slurm_job_id>

# 2) run the driver ON the GPU node, in the trainer env
srun --jobid="$JID" --overlap bash -lc '
  source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
  conda activate fedagent-verl08

  cd /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
  export PYTHONPATH="$PWD:$PYTHONPATH"                       # so `import fedagent` resolves (driver + Ray workers)
  export VERL_CFG="$(python -c "import verl,os;print(os.path.join(os.path.dirname(verl.__file__),\"trainer\",\"config\"))")"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1
  export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1   # deep_gemm asserts a CUDA toolkit
  export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0            # point at the CUDA module

  python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/coverage.yaml --n-gpus 4
'
```

Notes:

- `--overlap` lets the `srun` share the existing allocation's resources (so you can run the
  driver inside a job whose main step is idle/holding the node).
- `CUDA_HOME`, the offline flags, `PYTHONPATH`, and `VERL_CFG` are required (the smoke scripts
  set exactly these); `VERL_CFG` points Hydra at verl's stock `trainer/config`.
- Federated checkpoints land on the **compute node's** `/tmp` by default — inspect them with
  another `srun --jobid=<JID> --overlap ls ...`.
- The wrapper scripts in [`fedagent/scripts/`](../scripts) (`run_smoke.sh`,
  `run_tinyguess_fed_smoke.sh`, `run_webshop_fed_smoke.sh CFG …`) bake all of the above and
  are the quickest way to launch under `srun`.

## Smoke tests

The hand-written `fedagent/config/fed_*.yaml` are small (e.g. 2 clients × a few rounds) for
fast wiring checks:

```bash
# In-process, no service — fastest end-to-end check of the federated loop
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml

# WebShop smoke (driver launches 2 services), shortened to 2 rounds
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/homog_long.yaml --rounds 2

# Wrapper (sets env + srun-friendly), forwarding extra flags to run_fed
bash fedagent/scripts/run_webshop_fed_smoke.sh fedagent/config/examples/webshop/scaled/homog.yaml \
  --base-seed 43 --output-dir /tmp/run_s43 --port-base 8090
```

## Worked examples (paper configs)

```bash
# WebShop main, GRPO, Qwen2.5-1.5B, 4 GPUs
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# Same config, second seed + its own output dir + client-service ports.
# NOTE: --port-base moves ONLY webshop_base_port (the per-client services). This paper config
# enables eval, which uses a fixed webshop_val_port (no CLI flag) -- two concurrent runs of the
# SAME config would share that one val port. To deconflict, copy the YAML and change
# webshop_val_port too, or disable eval for the second run (val_env_spec: "").
python -m fedagent.fed.run_fed --config <...same...> \
  --base-seed 21 --output-dir /tmp/run_s21 --port-base 8120

# Environment-level heterogeneity (Catalog Split)
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-0.7_keep-0.7.yaml

# Centralized baseline (total_clients=1)
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

## Resume

The federation owns resume at the **round level**: re-running the same `--output-dir`
continues from the last completed round's aggregated model. Each client's per-run
auto-resume is disabled (`trainer.resume_mode=disable`) so a crashed in-flight round never
FedAvgs partial weights. Consumed FSDP shards are deleted after each merge to keep peak disk
to ~one round (toggle with `cleanup_checkpoints`; an 8-round run otherwise grew to 367 GB).

## Outputs

Under `output_dir/`:

- `round_*/client_*/training.log` + `round_*/client_*/json_logs/metrics.json` (per-client
  reward curve in FedAgent plot format)
- `round_*/aggregated/hf` — the round's global model (HF format; the next round's starting
  point)
- `<env>_service_client*.log`, `<env>_val_service.log` — per-service logs
- `federated_summary.json` — round history, mode/algorithm, final model, and (if eval on)
  the unperturbed `val_curve`

See [architecture.md](./architecture.md#outputs) and [`../fed/README.md`](../fed/README.md).

## See also

- [installation.md](./installation.md) — the three conda envs (`fedagent-verl08` trainer +
  `verl-agent-webshop` / `verl-agent-alfworld` services).
- [configuration.md](./configuration.md) — the full config-key reference and filename decoder.
- [reproducing.md](./reproducing.md) — the paper grid and seeds.
- [heterogeneity.md](./heterogeneity.md) — the task-level and environment-level partition
  strategies.
- [`../fed/README.md`](../fed/README.md) — the driver internals (round loop, FedAvg, merge).
