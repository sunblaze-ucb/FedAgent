# FedAgent documentation

User-facing documentation for the **FedAgent verl-0.8 overlay** — federated reinforcement
learning for LLM agents, built as a thin overlay on stock verl 0.8. Start with the package
overview in [`../README.md`](../README.md), then:

| Doc | Read it for |
|---|---|
| [architecture.md](./architecture.md) | How the overlay works: the federated round loop, the in-framework hooks, the remote env services, FedProx, eval. |
| [installation.md](./installation.md) | The three conda envs (trainer + WebShop + ALFWorld services), data, and models. |
| [running.md](./running.md) | Running `run_fed.py`: run modes, GPUs, baselines, FedProx, validation, worked examples. |
| [configuration.md](./configuration.md) | The config-file decoder and the full federated-runner key reference. |
| [features.md](./features.md) | Each capability → its config key → its source file (a navigation map). |
| [heterogeneity.md](./heterogeneity.md) | The two-level (task vs environment) heterogeneity suite, with the construction math for each arm. |
| [reproducing.md](./reproducing.md) | The paper's 176-config matrix mapped to run commands; 3-seed replication; baselines. |
| [extending.md](./extending.md) | Extension points: a new dataset/env, heterogeneity strategy, RL algorithm, or aggregation rule. |
| [migration.md](./migration.md) | **Migration** — the condensed fidelity record: what changed from the verl-agent-0.3.1 fork, the science-critical alignments, and the verification status. |
| [acceleration.md](./acceleration.md) | The acceleration analysis: bottleneck decomposition, the lever stack, and the measured recipes. |

> **Engineering archive.** The full documentation set — Chinese twins of every doc, the dated
> acceleration/validation reports (tier-1/tier-2/frontier/final), the complete migration
> walkthrough (`migration_report.md`), the agent-RL engine design doc, and the NanoRollout
> comparison — is preserved intact on the
> [`migrate/verl-0.8.0`](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs) branch; `main` keeps the core English set only.

## Per-component references

Each `fedagent/` subpackage has its own README with code-level detail:

- [`../fed/`](../fed/README.md) — federated round loop + metrics logger
- [`../agent_loops/`](../agent_loops/README.md) — multi-turn agent rollout (`GymTextAgentLoop`)
- [`../envs/`](../envs/README.md) — `BaseTextEnv` contract + registry; TinyGuess / WebShop / ALFWorld clients
- [`../hetero/`](../hetero/README.md) — the heterogeneity constructions
- [`../envs/webshop/service/`](../envs/webshop/service/README.md) · [`../envs/alfworld/service/`](../envs/alfworld/service/README.md) — remote env services
- [`../data/`](../data/README.md) — `AgenticDataset` (verl `custom_cls`)
- [`../config/`](../config/README.md) — configs + the paper matrix
- [`../EXPERIMENTS.md`](../EXPERIMENTS.md) — the running experiment log

## Scope

These docs describe the **verl-0.8 overlay** (the live system, under `fedagent/`). The repo's
top-level [`README.md`](../../README.md) is the landing page for this system; the *original*
verl-agent-0.3.1 artifact is preserved on the `paper-reproduce-verl-agent` branch as
historical reference. See [migration.md](./migration.md) for the relationship.
