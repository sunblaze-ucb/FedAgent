# `fedagent/` — FedAgent on stock verl 0.8

The verl-0.8 home for FedAgent: federated reinforcement learning for LLM agents.
A **thin overlay** — it imports **stock verl 0.8** as a library and adds only what
FedAgent needs on top of verl's **stock PPO/GRPO trainer** and **native async
agent-loop rollout**. **No trainer fork**, no patched verl tree.

Everything FedAgent contributes — environments, the multi-turn agent loop, the
dataset adapter, the two-level heterogeneity suite, FedProx, the JSON metrics
logger, and the federated round loop — lives in this package and is wired into
verl through its public extension points (`data.custom_cls`, the agent-loop
registry, Hydra `searchpath`).

## How it fits together

| Layer | Where | Role |
|---|---|---|
| **Federated control plane** | [`fed/run_fed.py`](fed/) | The round loop: one training subprocess per (client, round) → FedAvg the FSDP checkpoints → merge → re-enter the next round from the aggregated model. verl-agnostic (never imports verl). |
| **In-framework hooks** | [`envs/`](envs/), [`agent_loops/`](agent_loops/), [`data/`](data/), [`fedprox.py`](fedprox.py) | Plugged into verl's stock trainer/rollout via its extension points. |
| **Heterogeneity suite** | [`hetero/`](hetero/) | The two-level (task vs environment) partitioning that is the paper's core contribution. |
| **Remote env services** | [`envs/webshop/service/`](envs/webshop/service/), [`envs/alfworld/service/`](envs/alfworld/service/) | One HTTP env service per client — each owns that client's environment / data shard. Co-located with its trainer-side client under `envs/<name>/`. |
| **verl 0.8** | installed package | Used as a library: trainer, FSDP engine, async rollout, model merger. Unmodified. |

The single verl-side entry a client runs is `python -m fedagent.main_ppo_fed`
(verl's stock `run_ppo` with FedAgent's config + hooks); the federated driver
[`fed/run_fed.py`](fed/) orchestrates many of those subprocesses into rounds.

## Layout

```
fedagent/
├── fed/                 federated round loop (run_fed.py) + JSON metrics logger   → fed/README.md
├── agent_loops/         GymTextAgentLoop: multi-turn text rollout (verl AgentLoopBase)  → agent_loops/README.md
├── envs/                env contract + registry, one package per environment   → envs/README.md
│   ├── base.py            BaseTextEnv async contract
│   ├── registry.py        env_name -> env class
│   ├── tiny_guess.py      in-process smoke env
│   ├── webshop/           WebShopEnv client + service/ backend (verl-agent-webshop)
│   └── alfworld/          AlfworldEnv client + service/ backend (verl-agent-alfworld)
├── hetero/              two-level heterogeneity constructions (task + environment)  → hetero/README.md
├── data/                AgenticDataset: verl custom_cls emitting env-spec rows   → data/README.md
├── config/              Hydra config, agent registry, env specs, paper matrix    → config/README.md
├── fedprox.py           client-side FedProx proximal term (see sitecustomize.py)
├── main_ppo_fed.py      the per-client verl entry (stock run_ppo + FedAgent hooks)
├── EXPERIMENTS.md       running experiment log + migration-fidelity record
└── docs/                full documentation suite   → docs/README.md
```

Each subfolder has its own `README.md` (linked above). For end-to-end guides see
[`docs/`](docs/README.md).

## What's implemented

- **Algorithms** — **GRPO** (default; `adv_estimator=grpo`, group size **G=8** via
  `rollout.n=8`) and **PPO** (`adv_estimator=gae`, which federates the value model
  alongside the actor each round).
- **Federation** — FedAvg over FSDP-sharded checkpoints, with optional client-side
  **FedProx** (proximal term, enabled by `fedprox_mu>0`). Configurable protocol:
  clients `N`, clients/round `M`, local epochs `E`, rounds `T`, tasks/client.
- **Baselines** — `federated` (default), `centralized` (`total_clients=1`), and
  `local` (`local_client_id>=0`: one pinned client, no federation).
- **Environments** — `tinyguess` (in-process smoke), **WebShop** and **ALFWorld**
  (remote HTTP env services, one per client).
- **Two-level heterogeneity** — environment-level (`catalog_split`, `task_disjoint`)
  and task-level (`preference`/`omega`, `coverage`/`size_std`, `hardness`/`success_std`),
  plus WebShop env-variant arms (`bm25_field_subset`, `bm25_reweight`, `lookalike`,
  `rank_wrapper`). See [`docs/heterogeneity.md`](docs/heterogeneity.md).
- **Evaluation** — a shared **unperturbed** validation service scores the aggregated
  global model every `test_freq` rounds (plus the base model at round 0).
- **Backbones** — any HuggingFace causal-LM id (paper: Qwen2.5-1.5B/3B/7B-Instruct,
  Llama-3.2-3B-Instruct).

## Quick start

Run inside the **`fedagent-verl08`** conda env on a GPU node. WebShop and ALFWorld
additionally need their own service env (`verl-agent-webshop` / `verl-agent-alfworld`)
— see [`docs/installation.md`](docs/installation.md).

```bash
# 1) In-process smoke (no remote service) — verifies the federated loop end-to-end
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml

# 2) WebShop, homogeneous, GRPO
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/homog_long.yaml

# 3) WebShop with task-level heterogeneity (Preference, omega=0.5)
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/pref.yaml

# Common CLI overrides (win over the YAML):
#   --rounds N  --clients N  --n-gpus 4  --base-seed S  --fedprox-mu 0.1  --local-client-id K
```

Every config key is documented in [`fed/README.md`](fed/README.md) (built from the
`DEFAULTS` dict in `run_fed.py`); the config families are in
[`config/README.md`](config/README.md); per-table reproduction recipes are in
[`docs/reproducing.md`](docs/reproducing.md).

## Documentation

| Doc | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | The overlay design: how the round loop + hooks sit on stock verl 0.8. |
| [`docs/running.md`](docs/running.md) | Running `run_fed.py`: modes, GPUs, baselines, FedProx, eval, worked examples. |
| [`docs/configuration.md`](docs/configuration.md) | Config-file decoder and the federated-runner key reference. |
| [`docs/heterogeneity.md`](docs/heterogeneity.md) | The two-level taxonomy and how to construct/select each arm. |
| [`docs/reproducing.md`](docs/reproducing.md) | The paper config matrix (176 configs) mapped to commands. |
| [`docs/installation.md`](docs/installation.md) | The conda envs (orchestrator + WebShop + ALFWorld), data, and models. |
| [`docs/migration.md`](docs/migration.md) | What changed from the verl-agent-0.3.1 fork to stock verl 0.8, and the equivalence checks. |

See [`docs/README.md`](docs/README.md) for the index.
