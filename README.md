<p align="center">
  <img src="assets/logo_w_text_horizontal.png" alt="FedAgent logo" width="620"/>
</p>

<h1 align="center">FedAgent: A Library for Decentralized Agent Learning</h1>

<p align="center">
  <!-- <em>Train LLM agents collaboratively across decentralized clients, without sharing local data.</em> -->
  <em>Train agents on everyone’s experience without anyone sharing it.</em>
</p>

<p align="center">
  <a href="https://fed-agent.github.io/"><img src="https://img.shields.io/badge/🏠_Homepage-FF5722?style=for-the-badge&logoColor=white" alt="Homepage"></a>
  <a href="https://fed-agent.github.io/pdf/FedAgent.pdf"><img src="https://img.shields.io/badge/📄_Paper-DC143C?style=for-the-badge&logoColor=white" alt="Paper (PDF)"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/⚖️_License-Apache_2.0-4285F4?style=for-the-badge&logoColor=white" alt="License: Apache 2.0"></a>
</p>

---

## Updates

- **[Jun 2026]** Initial release of the FedAgent library, federated PPO/GRPO
  trainer, two-level heterogeneity suite, and full WebShop + ALFWorld reproduction.
- **[Jun 2026]** Paper online: **[Is Decentralized LLM Agent RL Robust to
  Heterogeneity? An Asymmetric Tale](https://fed-agent.github.io/)**, by [Canyu Chen](https://canyuchen.com/)\*, [Kangyu Zhu](https://scholar.google.com/citations?user=55J-zgwAAAAJ&hl=en)\*, [Zhaorun Chen](https://billchan226.github.io/),
  [Zhanhui Zhou](https://scholar.google.com/citations?user=SbACfYQAAAAJ), [Shizhe Diao](https://shizhediao.github.io/), [Yiping Lu](https://2prime.github.io/), [Tian Li](https://litian96.github.io/), [Manling Li](https://limanling.github.io/)+, and [Dawn Song](https://dawnsong.io/)+
  (*Equal contribution, +Equal Advising, Homepage: [https://fed-agent.github.io](https://fed-agent.github.io/), [PDF](https://fed-agent.github.io/pdf/FedAgent.pdf)). This work is honored to receive the 🏆 **[Best Paper Award](https://drive.google.com/file/d/1S13-8c382w8urNDCLwli1PNdc3ABnW1H/view?usp=sharing)** in the *AAAI 2026
  Workshop on Trust and Control in Agentic AI* and 🏆 **[Outstanding Paper Award](https://drive.google.com/file/d/1Z1x_cvT2I6aFdsF0ai25ZksLRcQ_VLMo/view?usp=sharing)**
  in the *AAAI 2026 Workshop on Personalization in the Era of Large Foundation
  Models*.

<!-- Add new entries on top. -->

---

## Overview

FedAgent is a library for **federated RL training of LLM agents**. It implements a
federated training loop with **FedAvg** aggregation (plus optional client-side
**FedProx**), a **two-level heterogeneity suite** (task vs environment partitioning),
and federated **PPO/GRPO** trainers — built as a **thin overlay on stock
[verl](https://github.com/volcengine/verl) 0.8** (no trainer fork: verl is imported as a
library and driven through its public extension points). You can reproduce the paper's
experiments or extend the framework with your own datasets, environments, and algorithms.

FedAgent is the reference implementation for the paper, which formalizes agent
heterogeneity at two structurally distinct levels (task vs environment) and derives an
**asymmetric robustness** result: federated training is robust to task-level heterogeneity
but worst-case non-robust to environment-level heterogeneity. See
[`fedagent/docs/heterogeneity.md`](fedagent/docs/heterogeneity.md) for the full construction.

> **The maintained code lives in [`fedagent/`](fedagent/README.md)** — this README and the
> [`fedagent/docs/`](fedagent/docs/README.md) suite document it. The original
> verl-agent-0.3.1 implementation is archived under [`legacy/`](legacy/README.md) for
> reference; what changed and why is in [`fedagent/docs/migration.md`](fedagent/docs/migration.md).

---

## Key Features

- **Federated GRPO and PPO** on stock verl 0.8 — GRPO is the default (group size **G=8**
  via `rollout.n=8`, no critic); PPO (`adv_estimator=gae`) additionally federates the value
  model alongside the actor each round.
- **Two-level heterogeneity suite** — task-level (Preference / Coverage / Hardness) and
  environment-level (Catalog-Split + 4 WebShop transition variants: BM25 field-subset,
  BM25 reweight, lookalike, rank-wrapper), the first systematic decomposition for agent FL.
- **FedAvg aggregation** over FSDP-sharded checkpoints, plus optional client-side
  **FedProx** (a proximal term added to local training, injected non-fork via the repo-root
  `sitecustomize.py` — not a server rule).
- **Baselines built in** — `federated` (default), `centralized` (one client on pooled data),
  and `local` (one pinned client, no federation), selectable from the same config.
- **Fully configurable protocol** — clients `N`, clients/round `M`, local epochs `E`, rounds
  `T`, tasks/client `|Xᵢ|` — with a ready-made **176-config paper matrix**.
- **Any HuggingFace backbone** (paper: Qwen2.5-1.5B/3B/7B-Instruct, Llama-3.2-3B-Instruct);
  **WebShop** and **ALFWorld** benchmarks out of the box, each behind a per-client HTTP env
  service so their conflicting dependencies stay isolated from the trainer.
- **FSDP** sharding (single-GPU to 4-GPU), W&B-free (metrics go to JSON / console).

Within a round, clients are trained **sequentially** (one subprocess per client, then
FedAvg); the loop is verl-agnostic and resumable. Extension points for new datasets,
environments, heterogeneity strategies, and aggregation rules are documented in
[`fedagent/docs/extending.md`](fedagent/docs/extending.md); the capability→config→source map
is in [`fedagent/docs/features.md`](fedagent/docs/features.md).

---

## Repository layout

```
fedagent/                      ← the maintained verl-0.8 overlay (start here)
├── fed/                       federated round loop (run_fed.py) + JSON metrics logger
├── envs/                      BaseTextEnv contract + registry; tiny_guess + per-env packages:
│   └── {webshop,alfworld}/    └── <env>_env.py (client) + service/ (HTTP backend) + engine/ (vendored sim)
├── agent_loops/               GymTextAgentLoop — multi-turn rollout (verl AgentLoopBase)
├── hetero/                    two-level heterogeneity constructions (task + environment)
├── data/                      AgenticDataset (verl custom_cls) + per-client partitioning
├── config/                    Hydra base, agent registry, env specs, + the 176-config paper matrix
├── docs/                      full documentation suite (architecture … migration)
├── fedprox.py                 client-side FedProx proximal term
└── main_ppo_fed.py            per-client entry: stock verl run_ppo + FedAgent hooks

sitecustomize.py               repo-root FedProx hook (auto-imported on PYTHONPATH)
tools/verl08_migration/        FedAvg aggregator, paper-config generator, hardness-traj generator, helpers
data/env_heterogeneity/        shipped env-level heterogeneity data (holdout / lookalike sets)
legacy/                        the original verl-agent-0.3.1 artifact (archived; do not run)
LICENSE · NOTICE · CITATION.cff
```

Per-subpackage READMEs live alongside the code (e.g. [`fedagent/fed/`](fedagent/fed/),
[`fedagent/envs/`](fedagent/envs/), [`fedagent/hetero/`](fedagent/hetero/)); the design and
the file→role map are in [`fedagent/docs/architecture.md`](fedagent/docs/architecture.md).

---

## Installation

FedAgent uses **three conda envs** because the trainer, WebShop, and ALFWorld have mutually
conflicting dependencies — they communicate over HTTP, so each stays isolated:

| Env | Role | Key deps |
|---|---|---|
| `fedagent-verl08` | the trainer / federated runner | Python 3.12, verl 0.8, vLLM, FSDP |
| `verl-agent-webshop` | the WebShop env service | Python 3.10, `gym 0.24`, `pyserini` (JDK/Lucene), `torch 2.6` |
| `verl-agent-alfworld` | the ALFWorld env service | Python 3.10, `alfworld`, `textworld`, `gymnasium` |

Full step-by-step setup (env creation, the JDK for WebShop, ALFWorld game files via
`alfworld-download`) is in **[`fedagent/docs/installation.md`](fedagent/docs/installation.md)**.
W&B logging is **removed** — no tracking account or key needed.

---

## Data

The default WebShop configs run **out of the box**: the small WebShop catalog files are
already shipped inside the vendored env package
(`fedagent/envs/webshop/engine/webshop/data/`), and the env-level heterogeneity holdout/lookalike
sets are tracked under [`data/env_heterogeneity/`](data/env_heterogeneity/). Two things are
fetched/generated separately:

- **ALFWorld game files** — one-time `alfworld-download -f` → `~/.cache/alfworld/` (see
  [`fedagent/docs/installation.md`](fedagent/docs/installation.md)).
- **Hardness arm trajectories** — the Hardness heterogeneity configs require per-backbone
  task-difficulty labels at `data/hardness/*.json`; generate them **before** any hardness run
  with `python tools/verl08_migration/gen_hardness_trajectories.py` (see
  [`fedagent/docs/reproducing.md`](fedagent/docs/reproducing.md)).

## Models

Backbones are **HuggingFace model ids** (default `Qwen/Qwen2.5-1.5B-Instruct`) and
**auto-download** on first run to `~/.cache/huggingface` (set `HF_HOME` to relocate). The main
table's **`Llama-3.2-3B-Instruct` is gated** (accept its license + `huggingface-cli login`);
on **offline / air-gapped** clusters, pre-fetch on a login node and pass `--model-path <local
snapshot>`. See [`fedagent/docs/installation.md`](fedagent/docs/installation.md#models).

---

## Quick Start

Run a FedAgent experiment **directly** with the federated runner, from the repo root inside
the `fedagent-verl08` env (WebShop/ALFWorld runs also need their service env available):

```bash
# 0) In-process smoke — verifies the federated loop end-to-end, no remote service
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml

# 1) WebShop, homogeneous, GRPO
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/homog_long.yaml

# 2) A paper cell (WebShop, Qwen2.5-1.5B, main, GRPO) from the 176-config matrix
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

CLI flags override the YAML: `--rounds N` · `--clients N` · `--n-gpus 4` · `--base-seed S`
· `--fedprox-mu 0.1` · `--local-client-id K`. Every config key is documented in
[`fedagent/fed/README.md`](fedagent/fed/README.md); hardware/run modes are in
[`fedagent/docs/running.md`](fedagent/docs/running.md).

---

## Reproducing the paper

The paper's experiments are the **176-config matrix** under
[`fedagent/config/paper/`](fedagent/config/README.md), mirroring the original tree 1:1
(`uniform/` main table across 4 backbones × WebShop + ALFWorld; `env_heterogeneity/`,
`task_heterogeneity/`, `decentralized/` on Qwen2.5-1.5B). Each cell is one command:

```bash
python -m fedagent.fed.run_fed --config fedagent/config/paper/<family>/<...>.yaml
```

Per-table recipes, seeds, and compute estimates (**~1,800 H100 GPU-hours** total) are in
**[`fedagent/docs/reproducing.md`](fedagent/docs/reproducing.md)**, covering the main table
(Local / Centralized / FedAgent × four backbones × WebShop + ALFWorld), the task- and
environment-level heterogeneity studies, and the decentralized ablations.

---

## Documentation

| Doc | Contents |
|---|---|
| [`fedagent/docs/architecture.md`](fedagent/docs/architecture.md) | The overlay design: the round loop + hooks on stock verl 0.8, and the file→role map. |
| [`fedagent/docs/installation.md`](fedagent/docs/installation.md) | The three-conda-env setup (trainer + WebShop + ALFWorld), JDK / game-file notes. |
| [`fedagent/docs/running.md`](fedagent/docs/running.md) | Running `run_fed.py`: modes, GPUs, baselines, FedProx, eval, worked examples. |
| [`fedagent/docs/reproducing.md`](fedagent/docs/reproducing.md) | Per-experiment reproduction recipes, the 176-config matrix, compute, seeds. |
| [`fedagent/docs/heterogeneity.md`](fedagent/docs/heterogeneity.md) | The two-level taxonomy and how to construct/select each arm. |
| [`fedagent/docs/configuration.md`](fedagent/docs/configuration.md) | Config-file decoder and the federated-runner key reference. |
| [`fedagent/docs/features.md`](fedagent/docs/features.md) | Each capability → its config key → its source file. |
| [`fedagent/docs/extending.md`](fedagent/docs/extending.md) | Extension points: new dataset/env, heterogeneity strategy, RL algorithm, aggregation rule. |
| [`fedagent/docs/migration.md`](fedagent/docs/migration.md) | What changed from the verl-agent-0.3.1 fork to stock verl 0.8, and the equivalence checks. |

---

## Citation

If you use FedAgent in your research, please cite:

```bibtex
@article{fedagent2026,
  title   = {Is Decentralized LLM Agent RL Robust to Heterogeneity? An Asymmetric Tale},
  author  = {Chen, Canyu and Zhu, Kangyu and Chen, Zhaorun and Zhou, Zhanhui and Diao, Shizhe and Lu, Yiping and Li, Tian and Li, Manling and Song, Dawn},
  journal = {arXiv preprint arXiv:},
  year    = {2026}
}
```

---

## License

This project is released under the **Apache License 2.0**: see [`LICENSE`](LICENSE).

## Acknowledgements

FedAgent is a thin overlay on **stock verl 0.8** and reuses the **verl-agent** environment
packages. We gratefully acknowledge:

- **veRL**: © ByteDance / the veRL authors (Apache-2.0): the base RL training framework that
  FedAgent imports as a library. <https://github.com/volcengine/verl>
- **verl-agent / GiGPO**: Feng et al., *Group-in-Group Policy Optimization for LLM Agent
  Reinforcement Learning* ([arXiv:2505.10978](https://arxiv.org/abs/2505.10978)): the agent-RL
  environment integrations FedAgent builds on. <https://github.com/langfengQ/verl-agent>
- **WebShop**: Yao et al., Princeton NLP (MIT License): the e-commerce agent benchmark.
- **ALFWorld**: Shridhar et al., Microsoft Research (MIT License): the embodied household
  agent benchmark.

Full per-component attributions and license texts are aggregated in [`NOTICE`](NOTICE).
