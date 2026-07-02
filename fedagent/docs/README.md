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
| [agent_rl_design.md](./agent_rl_design.md) | **Agent-RL engine design** — rollout (windowed vs concat, the per-row async contract), the three-layer async model and where its bounds are (pool / `_TW_LOCK` / GIL), the HTTP boundary contract (retries, idempotency, blocking `/create`, replica routing), env-service + trainer-plane seams, lifecycle modes, the acceleration lever stack, and SLURM ops patterns. |
| [acceleration_report.md](./acceleration_report.md) | **Acceleration & validation — the complete walkthrough**: every lever & feature in depth (persistent trainer, eval modes, client-parallel #3, equivalence), the investigations + corrections, and how to run. Companions: [acceleration_results.md](./acceleration_results.md) (numbers at a glance) · [acceleration.md](./acceleration.md) (the original analysis & plan). |
| [alfworld_testing.md](./alfworld_testing.md) | **Why ALFWorld is tested *this* way**: the env-agnostic fix boundary (correctness already covered), the wall-clock economics that ALFWorld's longer/heavier/larger rollout changes, and the Tier-1/Tier-2 test plan — the two WebShop conclusions (1-GPU penalty, eval-hiding) that may *flip* on ALFWorld. |
| [acceleration_cross_env.md](./acceleration_cross_env.md) | **WebShop vs ALFWorld — acceleration findings, side by side**: one master table + the principle for *which choices transfer and which flip* across environments (eval-mode ranking flips; ~+38% 1-GPU penalty transfers; ALFWorld rollout is env-bound; the concurrency fix is env-agnostic). Self-contained synthesis. |
| [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md) | **Deep-validation report (2026-07-01)** — env-service **replica sharding** (`*_replicas`): the diagnosis (86 ms × 3200 steps = the `_TW_LOCK` floor), design decisions, the four-level validation chain (mechanism → control → component → end-to-end: ALFWorld step −57 %, run −31 %), WebShop's first decomposition (GPU-bound mirror image, −12 %), prediction scorecard incl. the misses, and how it composes with the prior lever stack. |
| [extending.md](./extending.md) | Extension points: a new dataset/env, heterogeneity strategy, RL algorithm, or aggregation rule. |
| [migration.md](./migration.md) · [migration_report.md](./migration_report.md) | **Migration** — `migration.md` is the condensed fidelity record (what changed from the verl-agent-0.3.1 fork + the science-critical alignments); `migration_report.md` is the **complete engineering walkthrough** (route decision, the dependency saga, and the checkpoint / agent-loop / env-service / windowed deep-dives). |

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
verl-agent-0.3.1 artifact is archived under [`legacy/`](../../legacy/README.md) as historical
reference. See [migration.md](./migration.md) for the relationship.
