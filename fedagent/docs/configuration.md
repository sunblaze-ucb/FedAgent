# Configuration

FedAgent is a **thin overlay on unmodified verl 0.8** — there is no trainer fork. Every
run is driven by configuration: a flat YAML the federated runner reads, a Hydra base that
composes verl's stock `ppo_trainer`, an agent-loop registry, and per-episode env specs.
This page is the **config-file decoder** and the **federated-runner key reference**: every
key in `run_fed.py`'s `DEFAULTS` dict, the env-spec row schema, the `paper/` filename
grammar, and the naming gotchas that bite when you read a filename against the code.

See the package overview in [`../README.md`](../README.md), the folder map in
[`../config/README.md`](../config/README.md), and the federated driver in
[`../fed/README.md`](../fed/README.md). For what each heterogeneity arm *does* see
[`./heterogeneity.md`](./heterogeneity.md); for launching the loop,
[`./running.md`](./running.md); for the figure-by-figure matrix,
[`./reproducing.md`](./reproducing.md).

> **No legacy schema.** This is the verl-0.8 runner. The original FedAgent's nested
> `federated:` / `verl:` / `data_preprocess:` blocks are gone — that schema lives only in
> the archived `legacy/docs/` and is **not** read by anything here. A FedAgent config is a
> **flat** key/value file whose keys are `run_fed.py`'s `DEFAULTS`; per-client verl knobs
> are passed through `client_overrides` (see [§ client_overrides](#client_overrides-and-adv_estimator)).

---

## The four config types

| Type | File(s) | Consumed by | Role |
|---|---|---|---|
| **Hydra base config** | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) | `fedagent.main_ppo_fed` (`@hydra.main(config_name="fedagent_ppo")`) | Training config for **one client**: composes verl's stock `ppo_trainer` via `hydra.searchpath` and overrides only the leaves FedAgent needs. |
| **Agent registry** | [`config/agent.yaml`](../config/agent.yaml) | verl's `AgentLoopManager` (via `actor_rollout_ref.rollout.agent.agent_loop_config_path`) | Maps each `agent_name` on a dataset row to its `AgentLoopBase` class. |
| **Env spec** | [`config/envs/*.yaml`](../config/envs/) | `fedagent.data.agentic_dataset.AgenticDataset` (via `data.train_files` / `data.val_files`) | Declares the env pool: one dataset row per episode (`n_envs` rows, distinct seeds). |
| **Federated-runner config** | `config/fed_*.yaml`, `config/paper/**/*.yaml` | `python -m fedagent.fed.run_fed --config <file>` | The **outer** layer: top-level federation knobs; keys == `run_fed.py`'s `DEFAULTS` dict. Drives the round loop, FedAvg, env services, and validation. |

The runner is outermost: `run_fed` reads the flat config, launches per-client env
services, then shells out to `main_ppo_fed` (which loads `fedagent_ppo.yaml`) **once per
client per round**, injecting `data.train_files=<env_spec>`, the model path, and the
`client_overrides` as Hydra CLI overrides.

### `fedagent_ppo.yaml` — the Hydra base

Composes verl's **stock `ppo_trainer`**, resolved through `hydra.searchpath` -> verl's
`trainer/config` dir (exported as `$VERL_CFG`; `run_fed` falls back to
`verl.__file__/trainer/config`):

```yaml
defaults:
  - ppo_trainer
  - _self_
hydra:
  searchpath:
    - file://${oc.env:VERL_CFG}
```

It overrides only FedAgent leaves: **GRPO** (`algorithm.adv_estimator: grpo`,
`use_kl_in_reward: false`), **group size** (`rollout.n: 4` in the base; every arm re-pins
it via `client_overrides` — paper=`8`, smokes=`2`), **async multi-turn rollout**
(`rollout.name: vllm`, `mode: async`, `multi_turn.enable: true`,
`agent.default_agent_loop: gym_text`), the **paper actor objective on every arm**
(`use_kl_loss: true`, `kl_loss_coef: 0.01`, `kl_loss_type: low_var_kl`,
`entropy_coeff: 0.001` — verl 0.8 defaults differ), the **custom dataset**
(`data.custom_cls.name: AgenticDataset`), `reward_model.enable: false`, and
`trainer.logger: [console]`. Machine/run-specific leaves (`model.path`,
`data.{train,val}_files`, `custom_cls.path`, `agent_loop_config_path`,
`default_local_dir`) and the struct-additive
`+actor_rollout_ref.model.override_config.attn_implementation` are supplied on the CLI.

### `agent.yaml`

A list mapping `agent_name` -> `AgentLoopBase` `_target_`; it has two entries: `gym_text` ->
`fedagent.agent_loops.gym_text_agent_loop.GymTextAgentLoop` (the concat-style multi-turn
loop) and `gym_text_windowed` ->
`fedagent.agent_loops.windowed_agent_loop.WindowedGymTextAgentLoop` (the per-turn windowed
variant). The `agent_name` travels on each dataset row (see below), so verl
instantiates the right loop per rollout.

---

## Env specs — the row schema and how `AgenticDataset` consumes them

An env spec (`config/envs/*.yaml`) declares **one or more env pools** under a top-level
`envs:` list. `run_fed` points **both** `data.train_files` and `data.val_files` at the
same spec (`cfg.env_spec`); validation uses a separate spec via `cfg.val_env_spec`.

### Row schema

```yaml
envs:
  - name: WebShop          # env id -> dataset row's env_name / data_source (.lower())
    n_envs: 8              # number of dataset rows emitted for this pool (one per episode)
    max_turns: 15         # per-episode turn cap handed to the agent loop
    agent_name: gym_text  # optional; AgentLoop class to use (default: gym_text)
    config:               # optional per-env kwargs passed verbatim to the env/agent loop
      timeout: 180.0
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `TinyGuess` | Env id. Becomes the row's `env_name` and `data_source` (lowercased); selects the env class in the agent loop. |
| `n_envs` | int | `64` | Number of dataset rows emitted for this pool — **one row per episode** (distinct seed). |
| `max_turns` | int | `6` | Per-episode turn cap, forwarded to the agent loop (`WebShop=15`, `ALFWorld=50`). |
| `agent_name` | str | `gym_text` | Agent-loop key (must exist in `agent.yaml`). |
| `config` | map | `{}` | Per-env kwargs (e.g. `timeout`, or `{lo, hi}` for TinyGuess) passed to the env. WebShop/ALFWorld do **not** pin a service URL here — it comes from the `WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL` env var that `run_fed` sets per client. |

### How `AgenticDataset` turns a spec into rows

[`data/agentic_dataset.py`](../data/agentic_dataset.py) is verl's `custom_cls` dataset. It
loads the spec (`data.train_files[0]`), reads its `envs:` list, and for each pool emits
`n_envs` rows — each a **distinct env instance with a distinct seed**:

```python
seed = base_seed * 100_000 + spec_index * 1_000 + episode_index
# base_seed = int(os.environ["FEDAGENT_BASE_SEED"])  (0 if unset)
```

`run_fed` sets `FEDAGENT_BASE_SEED = base_seed + round*100 + client` per client-round, so
each client re-draws goals from its fixed shard every round (covering the shard over `T`
rounds) while staying reproducible. Each row carries `env_name`, `seed`, `config`,
`max_turns`, `agent_name`, `data_source`, a placeholder `raw_prompt`, and a single dummy
tensor `ds_dummy` (the row carries **no** `input_ids`/`attention_mask`/`position_ids` — the
agent loop generates those; the dummy tensor exists only for batch sizing, because stock
verl `_get_gen_batch` does not pop tensor keys before unioning the agent-loop output back
onto the batch). GRPO grouping is **not** done here: verl's `rollout.n` repeats each row
`n` times downstream, forming one GRPO group per env instance.

### Mapping to `data.train_files`

`run_fed` passes the spec path straight through as Hydra overrides
(`run_client` / `eval_global`):

```
data.train_files=<env_spec>   data.val_files=<env_spec>   data.custom_cls.path=<custom_cls_path>
```

So `data.train_files` is the **env-spec YAML path**, not a parquet file — the verl-0.8
overlay replaced parquet preprocessing with on-the-fly env enumeration.

### Shipped specs

| Spec | `n_envs` | `max_turns` | Used for |
|---|---|---|---|
| [`tiny_guess.yaml`](../config/envs/tiny_guess.yaml) | 64 | 6 | `TinyGuess`, in-process wiring proof (runner default `env_kind=tinyguess`). |
| [`webshop.yaml`](../config/envs/webshop.yaml) | 16 | 6 | WebShop smoke (small budget). |
| [`webshop_15.yaml`](../config/envs/webshop_15.yaml) | 8 | 15 | WebShop **GRPO** train (`n_envs=8` == original GRPO train_data_size; with `train_batch_size=8` that is 1 optimizer step/epoch). |
| [`webshop_15_ppo.yaml`](../config/envs/webshop_15_ppo.yaml) | 64 | 15 | WebShop **PPO** train (`n_envs=64` == original PPO train_data_size, paired with `train_batch_size=64`). |
| [`webshop_15_val.yaml`](../config/envs/webshop_15_val.yaml) | 500 | 15 | WebShop validation: held-out `goals[0:500]` on the full catalog (the whole held-out set; eval sets no `FEDAGENT_BASE_SEED`, so every round scores the same 500 goals). |
| [`alfworld.yaml`](../config/envs/alfworld.yaml) | 8 | 50 | ALFWorld train (game shards; `max_turns=50` == original `max_steps`). |
| [`alfworld_val.yaml`](../config/envs/alfworld_val.yaml) | 140 | 50 | ALFWorld validation: `valid_seen` (140, in-distribution). For the full 274 trials + per-task-type breakdown, run `tools/verl08_migration/eval_alfworld_by_tasktype.py` on the final model. |

---

## Filename decoder — the `paper/` tree

`config/paper/` holds the full paper-scale runs in a family tree that **mirrors the
original FedAgent** `config/` (176 configs). Every leaf is a flat runner config whose name
encodes its protocol:

```
fed_<env>_<algo>_total-<N>_cl-per-rd-<M>_rd-<T>_ep-per-cl-<E>_min-goals-per-cl-<G>_p-<strategy>_<knobs>.yaml
```

| Token | Runner key | Meaning |
|---|---|---|
| `<env>` | `env_kind` | `webshop` or `alfworld`. |
| `<algo>` | `adv_estimator` | `grpo` (no critic) or `ppo` (== `gae`, federates the critic). |
| `total-<N>` | `total_clients` | Client population N (`100`; `1` for centralized). |
| `cl-per-rd-<M>` | `clients_per_round` | Clients selected per round M (`2`; `1` for local/centralized). |
| `rd-<T>` | `total_rounds` | Communication rounds T (`70`). |
| `ep-per-cl-<E>` | `epochs_per_round` | Local epochs per client per round E (`3`). |
| `min-goals-per-cl-<G>` | `min_goals_per_client` | Minimum goals per client's shard (`100`). |
| `p-<strategy>` | `partition_strategy` (see caveat) | `uniform` (== IID, runner key `""`) or a heterogeneity strategy. |
| `<knobs>` | strategy knobs | Strategy params spelled out, e.g. `div-0.7_keep-0.7`, `omega-0.99`, `std-256`, `success_std-1`, `N-4`. |

The constant cell across the matrix is `total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100`
(N=100, M=2, T=70, E=3, G=100), giving `E*T = 210` local epochs. Baselines and
decentralized ablations vary exactly one of these tokens.

### Filename `p-...` token → runner key + knobs

> **The `p-<strategy>` token is the only one that is not a verbatim copy of the runner
> value.** The filename uses the *paper's* spelling; the YAML `partition_strategy` uses the
> code's dispatch key. They diverge for the WebShop env-variant arms — verify the YAML, not
> the filename:

| Filename `p-...` | `partition_strategy` (YAML) | Knob keys (YAML) | Axis / paper name |
|---|---|---|---|
| `p-uniform` | `""` | (none) | IID (homogeneous) |
| `p-preference_omega-<ω>` | `preference` | `omega` | task — Preference |
| `p-coverage_std-<s>` | `coverage` | `size_std` | task — Coverage |
| `p-hardness_success_std-<s>` | `hardness` | `success_std`, `trajectories_file` | task — Hardness |
| `p-catalog_split_div-<d>_keep-<r>` | `catalog_split` | `env_div`, `keep_ratio` | env — Catalog Split |
| `p-field_subset_index_N-<n>` | **`bm25_field_subset`** | `variant_n` | env — Field-Subset Index |
| `p-bm25_reweighting_N-<n>` | **`bm25_reweight`** | `variant_n` | env — BM25 Reweighting |
| `p-lookalike_injection_N-<n>` | **`lookalike`** | `variant_n` | env — Lookalike Injection |
| `p-rank_wrapper_N-<n>` | `rank_wrapper` | `variant_n` | env — Rank Wrapper |

The ALFWorld env-het analogue (not in the filename grammar above; used in the hand-written
`examples/alfworld/paper.yaml`) is `partition_strategy: env_disjoint` — disjoint per-client game
shards. Its WebShop task-only sibling is `task_disjoint` (disjoint goals, full catalog).

### Sweep endpoints

Each het axis is swept between a near-uniform and an extreme endpoint, visible in the
filenames:

| Axis | Near-uniform | Extreme | Note |
|---|---|---|---|
| Preference | `omega-0.01` | `omega-0.99` | larger `omega` => **more** heterogeneity |
| Coverage | `std-256` | `std-1` | high `size_std` (Beta concentration) => near-uniform; low => skewed |
| Hardness | `success_std-256` | `success_std-1` | same Beta-dispersion convention |
| Catalog Split | `div-0.0` | `div-1.0` | swept at fixed `keep-0.7` |
| env-variants | `N-2` | `N-8` | variant-pool size `variant_n` |

### Directory families

| Family | Layout | What varies |
|---|---|---|
| `uniform/<Model>/<setting>/<algo>/` | per-backbone IID + baselines | the **setting** (see below). |
| `env_heterogeneity/<strategy>[_ppo]/` | webshop only | the env-level perturbation strategy (`_ppo` => `adv_estimator: gae`). |
| `task_heterogeneity/<algo>/<env>/` | grpo+ppo × webshop+alfworld | the task-level partition (preference / coverage / hardness). |
| `decentralized/<change>/<algo>/` | webshop+alfworld | one protocol knob (`selected_cl_change` => M∈{1,4}; `ep_per_round_change` => (E,T)∈{(1,210),(5,42)}; `samples_change` => G∈{500,1000}). |

**Backbones** (one `uniform/<Model>/` subdir each): `Qwen2.5-1.5B-Instruct`,
`Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Llama-3.2-3B-Instruct`. The
`env_heterogeneity`, `task_heterogeneity`, and `decentralized` trees are generated for the
1.5B backbone only. `env_heterogeneity` is **webshop-only** (the catalog/BM25/lookalike/rank
arms perturb the WebShop catalog + search engine and have no ALFWorld analogue).

### Uniform settings

| Setting | Differs by | Runner keys |
|---|---|---|
| `main` | the IID anchor (seed 42) | `total_clients: 100`, `clients_per_round: 2`, `base_seed: 42`. |
| `main_seed1` / `main_seed2` | 3-seed replication | `base_seed: 21` / `84` (the original varied the shuffle seed 42/21/84). |
| `centralized` | one model on pooled data | `total_clients: 1`, `clients_per_round: 1` (FedAvg of one client == identity). |
| `local_client1` / `2` / `3` | "Local Agent Training" | `local_client_id: 21` / `42` / `84` (pin one client of 100; `clients_per_round: 1`; no federation). |

So **3-seed replication** = `base_seed` 42 / 21 / 84 across `main`, `main_seed1`,
`main_seed2`; the **Local** baselines pin clients `21`, `42`, `84`. One deliberate
divergence from the original filenames: `centralized` / `local_client*` encode
`rd-70_ep-3` (not the original `rd-1_ep-210`) because the verl-0.8 runner draws goal
variety from *rounds*, so the 210 local epochs are spread over 70 rounds. Regenerate the
whole tree with `tools/verl08_migration/gen_paper_configs.py`.

---

## Federated-runner key reference

Every key below is an entry in `run_fed.py`'s `DEFAULTS` dict; anything omitted from a
config falls back to the default. The CLI flags `--model-path --output-dir --rounds
--clients --n-gpus --base-seed --port-base --fedprox-mu --local-client-id` override the
YAML. Package-relative paths (`env_spec`, `val_env_spec`, `custom_cls_path`,
`agent_config_path`, `webshop_run_service`, `alfworld_run_service`) resolve against
`fedagent/`.

### Core loop

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model_path` | str | `""` | Base HF model dir for round 1; `""` => auto-discover a local Qwen2.5-0.5B-Instruct snapshot. |
| `output_dir` | path | `/tmp/xbb9020_fedagent_fed_tinyguess` | Run root: per-round client/aggregated checkpoints, logs, `federated_summary.json`. |
| `env_spec` | path | `config/envs/tiny_guess.yaml` | Env spec -> `data.{train,val}_files` for every client. |
| `custom_cls_path` | path | `data/agentic_dataset.py` | Path to `AgenticDataset` (-> `data.custom_cls.path`). |
| `agent_config_path` | path | `config/agent.yaml` | Agent-loop registry (-> `rollout.agent.agent_loop_config_path`). |
| `total_clients` | int | `2` | Client population N. |
| `clients_per_round` | int | `2` | Clients selected per round M (deterministic seeded sampling when `M < N`; seed = `base_seed + round - 1`). |
| `total_rounds` | int | `2` | Communication rounds T. |
| `epochs_per_round` | int | `1` | Local epochs E per client per round (-> `trainer.total_epochs`). |
| `base_seed` | int | `42` | Master seed; per-(round,client) env seed = `base_seed + round*100 + client` (also drives client selection). |
| `n_gpus_per_node` | int | `2` | FSDP world size per client run (== aggregator `nproc`). |
| `total_training_steps` | int | `1` | Per-client-round step cap (smokes); `<=0` => emit `null` so verl runs full E epochs (`len(dataloader)*total_epochs`). Emitted explicitly so a stale base value never leaks into paper runs. |
| `save_freq` | int | `1` | verl `trainer.save_freq` (paper configs use a huge value, e.g. `100000`, to save only the round's last step). |
| `weights` | str | `""` | FedAvg weights passed to the aggregator (e.g. by client data size); `""` => uniform average. |
| `wait_between_clients` | int (s) | `5` | Seconds between sequential client runs (let Ray/GPU release). |
| `client_overrides` | list | `[]` | Extra `key=value` Hydra overrides applied to every client (and reused for eval). See [§ below](#client_overrides-and-adv_estimator). |
| `cleanup_checkpoints` | bool | `True` | Delete consumed FSDP shards after each merge (keep HF + logs); disk hygiene. |
| `adv_estimator` | str | `grpo` | `grpo` (no critic) or `gae` (PPO: FedAvg actor **and** critic). |

### Env services

| Key | Type | Default | Meaning |
|---|---|---|---|
| `env_kind` | str | `tinyguess` | `tinyguess` (in-process), `webshop`, or `alfworld` (remote services). |
| `webshop_run_service` | path | `envs/webshop/service/run_service.sh` | Launcher for a WebShop service. |
| `webshop_base_port` | int | `8080` | Client `c`'s service -> `webshop_base_port + c`. |
| `webshop_pool_size` | int | `8` | Env pool per WebShop service (must be `>= gen_batch`). |
| `search_return_n` | int | `200` | `WEBSHOP_SEARCH_RETURN_N`: BM25 top-K. Env-het arms use `200` (engine default `50` drops targets under filtering); non-het baselines keep `50`. |
| `alfworld_run_service` | path | `envs/alfworld/service/run_service.sh` | Launcher for an ALFWorld service. |
| `alfworld_base_port` | int | `8200` | Client `c`'s service -> `alfworld_base_port + c`. |
| `alfworld_pool_size` | int | `4` | TextWorld env pool per ALFWorld service (must be `>= gen_batch`). |
| `alfworld_train_eval` | str | `train` | ALFWorld game split: `train` / `eval_in_distribution` / `eval_out_of_distribution`. |
| `alfworld_task_types` | str | `""` | `""` => all 6 types; else comma-sep IDs (1=Pick..6=Pick2) for the eval breakdown. |
| `service_health_timeout` | int (s) | `900` | Seconds to wait for each service `/health` (pool warmup takes minutes). |

### Heterogeneity

| Key | Type | Default | Meaning |
|---|---|---|---|
| `partition_strategy` | str | `""` | `""` (IID) \| `catalog_split`/`task_disjoint` (WebShop env/task) \| `env_disjoint` (ALFWorld env) \| `preference`/`coverage`/`hardness` (task) \| `bm25_field_subset`/`bm25_reweight`/`lookalike`/`rank_wrapper` (WebShop env variants). |
| `env_div` | float | `0.7` | catalog-split heterogeneity strength. |
| `keep_ratio` | float | `0.7` | catalog-split distractor density. |
| `omega` | float | `0.5` | **preference** (task-het) Dirichlet spread ω — larger ω = more skew. |
| `size_std` | float | `1.0` | **coverage** (task-het) Beta dispersion ξ. |
| `success_std` | float | `1.0` | **hardness** (task-het) Beta dispersion ξ′. |
| `variant_n` | int | `0` | env-variant arms (bm25/lookalike/rank): # variants in the pool (`0` => fn default 2/4). Filename token `N-<n>`. |
| `trajectories_file` | path | `""` | hardness: **required** `task_id`->success-labels file (generate via `tools/verl08_migration/gen_hardness_trajectories.py`). |
| `min_goals_per_client` | int | `100` | Minimum goals per client's shard. Filename token `min-goals-per-cl-<G>`. |

See [`./heterogeneity.md`](./heterogeneity.md) for the full taxonomy and how each knob maps
to an arm.

### Baselines

| Key | Type | Default | Meaning |
|---|---|---|---|
| `local_client_id` | int | `-1` | `>=0` => **Local** baseline: train only this client of `total_clients`, every round, no federation. |

**Mode selection** (all via the same schema): **Federated** = default (`total_clients=N>1`,
`local_client_id<0`); **Centralized** = `total_clients=1` (per-round FedAvg of one client
is the identity, so the loop is `T*E` epochs of centralized training); **Local** =
`local_client_id=k>=0`; **FedProx** = `fedprox_mu>0`; **PPO** = `adv_estimator=gae`.

### Eval (unperturbed global-model validation)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `val_env_spec` | path | `""` | `""` => **no eval**; else the UNPERTURBED val env-spec. |
| `test_freq` | int | `5` | Eval the aggregated global model every K rounds (+ the final round). |
| `val_before_train` | bool | `True` | Also eval the base model before round 1 (the round-0 point). |
| `val_temperature` | float | `0.4` | Val sampling temperature (paper `val_kwargs.temperature=0.4`). |
| `webshop_val_port` | int | `8090` | Shared unperturbed WebShop val service port. |
| `alfworld_val_port` | int | `8290` | Shared unperturbed ALFWorld val service port. |
| `alfworld_val_split` | str | `eval_in_distribution` | ALFWorld val games (the in-distribution `valid_seen` eval set). |

Eval scores the **global** model (base on round 0, else the round's aggregated HF) on one
shared unperturbed val service via a verl `val_only` pass (`adv_estimator=grpo`, no critic,
FedProx off), so every arm is measured on the same fixed set. A failed eval never aborts
the run — it is measurement, not the loop.

### FedProx

| Key | Type | Default | Meaning |
|---|---|---|---|
| `fedprox_mu` | float | `0.0` | `>0` => client-side FedProx proximal term (else FedAvg). |

`fedprox_mu>0` is bridged to each client (and its Ray workers) via the env var
`FEDPROX_MU`, which `sitecustomize.py` reads at interpreter startup to patch
`FSDPEngine.optimizer_step` with the proximal term — chosen over a Ray `runtime_env` hook
so verl's per-worker `CUDA_VISIBLE_DEVICES` isolation is preserved.

---

## `client_overrides` and `adv_estimator`

`client_overrides` is a list of extra `key=value` **Hydra overrides** appended verbatim to
every client's `main_ppo_fed` command (and reused for eval, so the rollout shape matches).
It is where each arm pins the rollout/batch/context shape that the base `fedagent_ppo.yaml`
leaves at smoke defaults. The key ones:

| Override | Role |
|---|---|
| `actor_rollout_ref.rollout.n=8` | **GRPO group size G** (8 in the paper). |
| `data.train_batch_size=8` (PPO: `64`) | Prompts per optimizer step; pair with `actor_rollout_ref.actor.ppo_mini_batch_size`. |
| `data.max_prompt_length` / `max_response_length` (WebShop `4096` / `512`; ALFWorld `2048` / `512`) | Token budgets; mirror on `rollout.prompt_length` / `response_length`. |
| `actor_rollout_ref.rollout.max_model_len` (WebShop `4608`; ALFWorld `2560`) | vLLM context window. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM KV-cache fraction (`0.5`–`0.6`). |

> **`ppo_mini_batch_size` is set in `client_overrides`, not the filename or top-level
> keys.** It is a verl actor knob (`actor_rollout_ref.actor.ppo_mini_batch_size`), so it
> rides in `client_overrides`. verl multiplies it by `rollout.n` internally to form the
> per-update sample count, which is why the paper pairs `ppo_mini_batch_size=8` with
> `rollout.n=8` (GRPO) and `=64` with `rollout.n=8` (PPO). Do not confuse it with
> `min_goals_per_client` (a federation/sharding knob) or `data.train_batch_size` (prompts
> per step).

For **PPO** (`adv_estimator: gae`) the overrides also enable and shape the critic, and
`save_contents=[model]` makes the value-model checkpoint FedAvg-able:

```yaml
adv_estimator: gae
client_overrides:
  - actor_rollout_ref.actor.checkpoint.save_contents=[model]
  - critic.optim.lr=1e-5
  - critic.model.use_remove_padding=true
  - critic.model.enable_gradient_checkpointing=true
  - critic.fsdp.optimizer_offload=true
  - critic.ppo_micro_batch_size_per_gpu=4
  - critic.checkpoint.save_contents=[model]
  - trainer.critic_warmup=0
```

**GRPO vs PPO:** GRPO (the default, `rollout.n=G=8`, no critic) leaves the client command
byte-identical to the verified path. PPO (`adv_estimator=gae`) flips `need_critic` on; the
runner federates the value model **alongside the actor every round** (round-1 critic = the
base model, thereafter the aggregated critic), reusing the same FedAvg + merge machinery —
the merger auto-detects `...ForTokenClassification` vs `...ForCausalLM` from the shard's
`huggingface/config.json`.

---

## Naming gotchas

The single-word arm names hide a few traps. Keep these straight when reading a config:

- **Which knob per task-het axis** — each task axis has its own knob; passing the wrong one
  silently no-ops (the service forwards only the key its strategy needs):
  - **Preference** -> `omega` (Dirichlet spread ω; larger = more skew).
  - **Coverage** -> `size_std` (Beta dispersion ξ).
  - **Hardness** -> `success_std` (Beta dispersion ξ′) **and** the required `trajectories_file`.
- **Filename token ≠ runner `partition_strategy`** for the WebShop env-variant arms. The
  filename spells the paper name; the YAML uses the dispatch key:
  `field_subset_index` -> `bm25_field_subset`, `bm25_reweighting` -> `bm25_reweight`,
  `lookalike_injection` -> `lookalike`, `rank_wrapper` -> `rank_wrapper`. (`catalog_split`
  and the task strategies match in both places.) Always trust the YAML.
- **`variant_n` is the env-variant count**, surfaced as the filename token `N-<n>`. It
  applies only to `bm25_field_subset` / `bm25_reweight` / `lookalike` / `rank_wrapper`;
  `0` => the function's built-in default (2 or 4).
- **`ppo_mini_batch_size` lives in `client_overrides`** (a verl actor leaf, multiplied by
  `rollout.n` internally), not in the filename or the top-level runner keys — see the box
  above.
- **`env_disjoint` (ALFWorld) vs `catalog_split`/`task_disjoint` (WebShop)** are the
  env-level partitions; the ALFWorld one is named differently because it shards game files,
  not a catalog.

---

## A worked config

A real `uniform/main/grpo` WebShop config
(`fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml`):

```yaml
env_kind: webshop
env_spec: config/envs/webshop_15.yaml
val_env_spec: config/envs/webshop_15_val.yaml
output_dir: /tmp/xbb9020_fedpaper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform
model_path: Qwen/Qwen2.5-1.5B-Instruct

total_clients: 100
clients_per_round: 2
total_rounds: 70
epochs_per_round: 3
base_seed: 42

n_gpus_per_node: 4
total_training_steps: 0        # 0 => full E epochs/round (no per-round step cap)
save_freq: 100000              # save only the round's last step
test_freq: 5
val_before_train: true
val_temperature: 0.4
wait_between_clients: 8
min_goals_per_client: 100
webshop_pool_size: 16
webshop_base_port: 10000
webshop_val_port: 10100
search_return_n: 50            # engine default (matches the original non-het baselines)
partition_strategy: ""         # IID

client_overrides:
  - data.train_batch_size=8
  - data.max_prompt_length=4096
  - data.max_response_length=512
  - actor_rollout_ref.actor.ppo_mini_batch_size=8
  - actor_rollout_ref.rollout.n=8
  - actor_rollout_ref.rollout.prompt_length=4096
  - actor_rollout_ref.rollout.response_length=512
  - actor_rollout_ref.rollout.max_model_len=4608
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  - actor_rollout_ref.actor.checkpoint.save_contents=[model]
```

Run it directly:

```bash
python -m fedagent.fed.run_fed \
    --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml \
    --model-path /path/to/Qwen2.5-1.5B-Instruct      # offline: a local snapshot
```

See [`./running.md`](./running.md) for modes, GPUs, and worked examples, and
[`./reproducing.md`](./reproducing.md) for the full matrix mapped to commands.
