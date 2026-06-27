# `config/` — every config FedAgent layers on stock verl 0.8

FedAgent is a **thin overlay on unmodified verl 0.8** — there is no trainer fork. This
folder holds *all* of the configuration that turns stock verl into a federated agent
trainer: the Hydra training config layered on verl's `ppo_trainer`, the agent-loop
registry, the environment specs, the federated-runner configs that drive
[`python -m fedagent.fed.run_fed`](../fed/README.md), and the auto-generated paper config
matrix.

```
config/
├── fedagent_ppo.yaml          # Hydra base: composes verl's stock ppo_trainer (hydra.searchpath)
├── agent.yaml                 # agent-loop registry: agent_name -> AgentLoopBase _target_
├── envs/                      # env specs (one row per episode); -> data.train_files
│   ├── tiny_guess.yaml        webshop.yaml  webshop_15.yaml  webshop_15_ppo.yaml
│   └── webshop_15_val.yaml    alfworld.yaml  alfworld_val.yaml
├── examples/                  # hand-written federated-runner configs, grouped by env:
│   ├── tinyguess_2cl_2rd.yaml         #   in-process wiring smoke
│   ├── webshop/                       #   *_long / probe / fedprox / 2cl_catalog_split smokes
│   │   └── scaled/                    #   the 15 scaled WebShop arms (homog, task, pref, …, ppo)
│   └── alfworld/                      #   smoke.yaml + paper.yaml (env-het, 8cl x 70rd)
└── paper/                     # generated paper matrix (176): uniform/ env_heterogeneity/ task_heterogeneity/ decentralized/
```

See the top-level [`../README.md`](../README.md) for the project overview,
[`../fed/README.md`](../fed/README.md) for the federated driver, and
[`../docs/configuration.md`](../docs/configuration.md) /
[`../docs/reproducing.md`](../docs/reproducing.md) for the full config reference and the
figure-by-figure reproduction guide.

---

## The four config types

| Type | File(s) | Consumed by | Role |
|---|---|---|---|
| **Hydra base config** | `fedagent_ppo.yaml` | `fedagent.main_ppo_fed` (`@hydra.main(config_name="fedagent_ppo")`) | The single training config for one client; composes verl's stock `ppo_trainer` and overrides only the leaves FedAgent needs. |
| **Agent registry** | `agent.yaml` | verl's `AgentLoopManager` (via `actor_rollout_ref.rollout.agent.agent_loop_config_path`) | Maps each `agent_name` carried on dataset rows to its `AgentLoopBase` class. |
| **Env spec** | `envs/*.yaml` | `fedagent.data.agentic_dataset.AgenticDataset` (via `data.train_files` / `data.val_files`) | Declares the env pool: one dataset row per episode (`n_envs` rows, distinct seeds). |
| **Federated-runner config** | `examples/**/*.yaml`, `paper/**/*.yaml` | `python -m fedagent.fed.run_fed --config <file>` | Top-level federation knobs; keys map to `run_fed.py`'s `DEFAULTS` dict. Drives the round loop, FedAvg, env services, and validation. |

The runner config is the *outer* layer: `run_fed` reads it, launches per-client env
services, then shells out to `main_ppo_fed` (which loads `fedagent_ppo.yaml`) once per
client per round, injecting `data.train_files=<env_spec>`, the model path, and the
`client_overrides` as Hydra CLI overrides.

---

## `fedagent_ppo.yaml` — the Hydra base config

Composes verl's **stock `ppo_trainer`** config, resolved through `hydra.searchpath` ->
verl's `trainer/config` dir (exported as `$VERL_CFG`; `run_fed` falls back to
`verl.__file__/trainer/config`):

```yaml
defaults:
  - ppo_trainer
  - _self_
hydra:
  searchpath:
    - file://${oc.env:VERL_CFG}
```

It then overrides only the FedAgent-specific leaves. Notable defaults:

- **Algorithm: GRPO.** `algorithm.adv_estimator: grpo`, `use_kl_in_reward: false`.
- **GRPO group size.** `actor_rollout_ref.rollout.n: 4` in the base; the runner's
  `client_overrides` re-pin `rollout.n` per arm (paper arms = 8, smokes = 2). PPO arms
  switch to `adv_estimator: gae` and federate a critic — see below.
- **Async multi-turn rollout:** `rollout.name: vllm`, `mode: async`,
  `multi_turn.enable: true`, with `agent.default_agent_loop: gym_text` and
  `agent_loop_config_path` (-> `agent.yaml`) supplied on the CLI.
- **Paper actor objective on every arm:** `actor.use_kl_loss: true`,
  `kl_loss_coef: 0.01`, `kl_loss_type: low_var_kl`, `entropy_coeff: 0.001` — set in the
  base because verl 0.8's defaults differ, so GRPO smokes, the paper matrix, and PPO all
  inherit the correct loss.
- **Custom dataset:** `data.custom_cls.name: AgenticDataset` (its `path` is set on the CLI).
- **`reward_model.enable: false`** (reward comes from the env), and `trainer.logger: [console]`.

Machine/run-specific leaves — `model.path`, `data.{train,val}_files`,
`data.custom_cls.path`, `agent_loop_config_path`, `trainer.default_local_dir` — and the
struct-additive `+actor_rollout_ref.model.override_config.attn_implementation` are **not
pinned here**; `run_fed` / the smoke launcher supply them on the CLI.

---

## `agent.yaml` — the agent-loop registry

A list mapping `agent_name` (a column on each dataset row) to the agent-loop class verl
instantiates per rollout:

```yaml
- name: gym_text
  _target_: fedagent.agent_loops.gym_text_agent_loop.GymTextAgentLoop
```

`gym_text` is the single concat-style multi-turn loop used by every env in this repo.

---

## `envs/` — environment specs

Each spec lists one or more env pools; `AgenticDataset` emits **`n_envs` rows per pool**,
each a distinct episode (distinct seed). `data.train_files` and `data.val_files` both
point at the *same* spec (`run_fed` sets both to `cfg.env_spec`); validation uses a
separate `*_val.yaml` via `cfg.val_env_spec`.

| Spec | `n_envs` | `max_turns` | Used for |
|---|---|---|---|
| `tiny_guess.yaml` | 64 | 6 | `TinyGuess`, in-process wiring proof (runner default `env_kind=tinyguess`). |
| `webshop.yaml` | 16 | 6 | WebShop smoke (small budget). |
| `webshop_15.yaml` | 8 | 15 | WebShop **GRPO** train (`n_envs=8` == original GRPO train_data_size). |
| `webshop_15_ppo.yaml` | 64 | 15 | WebShop **PPO** train (`n_envs=64` == original PPO train_data_size). |
| `webshop_15_val.yaml` | 500 | 15 | WebShop validation: held-out `goals[0:500]` on the full catalog. |
| `alfworld.yaml` | 8 | 50 | ALFWorld train (game shards). |
| `alfworld_val.yaml` | 140 | 50 | ALFWorld validation: `valid_seen` (in-distribution). |

Common row fields: `name`, `n_envs`, `max_turns`, `agent_name` (default `gym_text`), and a
per-env `config` block (e.g. `timeout`). WebShop/ALFWorld are HTTP clients to per-client
services; the URL comes from `WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL` (set per
client by `run_fed`), so it is **not** pinned in the spec.

---

## `examples/` — hand-written federated-runner configs

These are inputs to `python -m fedagent.fed.run_fed --config config/examples/<...>.yaml`. Every
key corresponds to an entry in `run_fed.py`'s `DEFAULTS` dict; anything omitted falls back to
the default. Most configs under `examples/` are fast **smokes** (typically 2 clients x 4
rounds); the full paper-scale runs live under `paper/` (a couple — `webshop/*_long`,
`alfworld/paper` — are longer demos, noted below).

Representative keys (see [`../fed/README.md`](../fed/README.md) for the **full** reference):

```yaml
env_kind: webshop                 # tinyguess (in-process) | webshop | alfworld (remote services)
env_spec: config/envs/webshop_15.yaml
val_env_spec: config/envs/webshop_15_val.yaml   # "" => no eval; else unperturbed val spec
model_path: /path/to/Qwen2.5-1.5B-Instruct
adv_estimator: grpo               # grpo (no critic) | gae (PPO: FedAvg actor + critic)

total_clients: 2                  # client population N
clients_per_round: 2              # selected per round M
total_rounds: 4                   # rounds T
epochs_per_round: 3               # local epochs E
base_seed: 42

partition_strategy: ""            # ""=IID | catalog_split/task_disjoint (env-het) |
                                  #   preference/coverage/hardness (task-het) |
                                  #   bm25_field_subset/bm25_reweight/lookalike/rank_wrapper (env variants)
# + strategy knobs: env_div, keep_ratio, omega, size_std, success_std, variant_n, trajectories_file
fedprox_mu: 0.0                   # >0 => client-side FedProx proximal term (else FedAvg)
test_freq: 5                      # eval the aggregated model every K rounds; val_temperature: 0.4
local_client_id: -1               # >=0 => Local baseline (train one client alone, no federation)

client_overrides:                 # extra `key=value` Hydra overrides applied to every client
  - data.train_batch_size=8
  - actor_rollout_ref.rollout.n=2
  - actor_rollout_ref.rollout.max_model_len=8192
```

**Families under `examples/`** (all smokes unless noted):

- **`examples/tinyguess_2cl_2rd.yaml`** — in-process wiring smoke.
- **`examples/webshop/`** — `homog_long`, `envhet_long`, `probe_signal`, `fedprox_test`,
  `2cl_catalog_split`: early WebShop smokes / probes.
- **`examples/webshop/scaled/`** — the scaled WebShop arms at the 15-turn budget:
  `homog` (IID anchor), `task`/`pref` (task-het), `coverage`, `hardness`, `catalog`
  (env-het), `envhet_fedprox`, `local`, `centralized`, `lookalike`, `rank`, `bm25field`,
  `bm25reweight`, `ppo`, `ppo_lookalike`. (`hardness` requires a `trajectories_file` —
  generate one with `tools/verl08_migration/gen_hardness_trajectories.py`.)
- **`examples/alfworld/`** — `smoke.yaml` and `paper.yaml`
  (game-shard env-het, `partition_strategy: env_disjoint`, 8 clients x 70 rounds).

Baseline modes are selected via the same schema: **Centralized** = `total_clients: 1`;
**Local** = `local_client_id: k>=0`; **FedProx** = `fedprox_mu > 0`; **PPO** =
`adv_estimator: gae` (federates the value model too).

---

## `paper/` — the generated paper config matrix

`paper/` holds the **full paper-scale runs** (federated `N=100 / M=2 / E=3 / T=70`, 4-GPU
FSDP, unperturbed validation). It mirrors the **original FedAgent `config/` tree 1:1 in
structure + naming** — the four experiment families and the descriptive
`fed_<env>_<algo>_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-<strategy>_<knobs>.yaml`
filenames; only the file *contents* are verl-0.8 `run_fed.py` configs (the migration changed
the runner, not the experiment design). **176 configs** total:

```
paper/
├── uniform/<Model>/{main,main_seed1,main_seed2,centralized,local_client1-3}/{grpo,ppo}/   112
│       4 backbones × 7 settings × {grpo,ppo} × {webshop,alfworld}; p-uniform
├── env_heterogeneity/<strategy>[_ppo]/                                                      16
│       Qwen2.5-1.5B, WebShop only: catalog_split, bm25_reweighting, field_subset_index,
│       lookalike_injection, rank_wrapper
├── task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/                                        24
│       Qwen2.5-1.5B: preference(ω), coverage(ξ), hardness(ξ′)
└── decentralized/{ep_per_round_change,samples_change,selected_cl_change}/{grpo,ppo}/        24
        Qwen2.5-1.5B: each varies one protocol knob (M, |Xᵢ|, or E×T) on the homog baseline
```

- The **main table** is the 4-backbone uniform sweep (`Qwen2.5-1.5B/3B/7B-Instruct`,
  `Llama-3.2-3B-Instruct`) across WebShop + ALFWorld, 3 seeds (`main`/`main_seed1`/`main_seed2`
  → base_seed 42/21/84), GRPO + PPO. The het / decentralized families use a single backbone
  (Qwen2.5-1.5B), matching the paper.
- ALFWorld appears only where it has an analogue (uniform, task-het, decentralized); the
  env-het arms perturb the WebShop catalog + search engine and have no ALFWorld counterpart.
- One deliberate divergence from the original filenames: `centralized`/`local_client*` encode
  `rd-70_ep-3` (not the original `rd-1_ep-210`) — the verl-0.8 runner draws goal variety from
  *rounds*, so 210 local epochs are spread over 70 rounds to re-draw goals each round.

**Regenerate** the whole matrix with one command:

```bash
python tools/verl08_migration/gen_paper_configs.py                # all 176 -> fedagent/config/paper
python tools/verl08_migration/gen_paper_configs.py --group-size 2 # cheap smoke (lower G)
```

Every generated config runs directly with `python -m fedagent.fed.run_fed --config <path>`.
Per-table reproduction recipes (which config → which paper number) are in
[`../docs/reproducing.md`](../docs/reproducing.md).
