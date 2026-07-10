<p align="center">
  <img src="assets/logo_w_text_horizontal.png" alt="FedAgent logo" width="620"/>
</p>

<h1 align="center">FedAgent: A Library for Decentralized Agent Learning</h1>

<p align="center">
  <em>Train LLM agents collaboratively across decentralized clients, without sharing local data.</em>
</p>

<p align="center">
  <a href="https://fed-agent.github.io/"><img src="https://img.shields.io/badge/🏠_Homepage-FF5722?style=for-the-badge&logoColor=white" alt="Homepage"></a>
  <a href="https://fed-agent.github.io/pdf/FedAgent.pdf"><img src="https://img.shields.io/badge/📄_Paper-DC143C?style=for-the-badge&logoColor=white" alt="Paper (PDF)"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/⚖️_License-Apache_2.0-4285F4?style=for-the-badge&logoColor=white" alt="License: Apache 2.0"></a>
</p>

---

## Updates

- **[Jun 2026]** FedAgent is gradually migrating to **verl 0.8.0** on the
  [`migrate/verl-0.8.0`](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0)
  branch, which is a thin overlay (no trainer fork) running federated LLM agent GRPO/PPO on stock
  verl 0.8, with WebShop + ALFWorld acceleration studies.
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

FedAgent is a library for **federated RL training of LLM agents**.
It implements a federated training server with **FedAvg** aggregation (plus
optional client-side **FedProx**), a **two-level heterogeneity suite** (task vs
environment partitioning), and
federated **PPO/GRPO** trainers built on
[verl-agent](https://github.com/langfengQ/verl-agent).
You can reproduce the paper's experiments or extend the framework with your own
datasets, environments, and algorithms.

FedAgent is also the reference implementation for the paper, which formalizes
agent heterogeneity at two structurally distinct levels (task vs environment)
and derives an **asymmetric robustness** result: federated training is robust to
task-level heterogeneity but worst-case non-robust to environment-level
heterogeneity. See [`docs/heterogeneity.md`](docs/heterogeneity.md) for
the full construction.

---

## Key Features

- **Federated PPO and GRPO trainers**: drop-in federated counterparts of the
  verl-agent trainers; swap one config to go from single-client to federated
- **Two-level heterogeneity suite**: task-level (Preference / Coverage /
  Hardness) and environment-level (5 WebShop transition variants), the first
  systematic decomposition for agent FL
- **FedAvg aggregation** with FSDP-sharded model support, pluggable for custom
  rules, plus optional **client-side FedProx** (a proximal term added to local
  training, not a server rule)
- Fully configurable federation protocol (clients `N`, clients/round `M`,
  local epochs `E`, rounds `T`, tasks/client `|Xᵢ|`) with ready-made sweeps
- Any HuggingFace backbone (paper uses Qwen2.5-1.5B/3B/7B-Instruct,
  Llama-3.2-3B-Instruct); WebShop and ALFWorld benchmarks out of the box
- FSDP sharding, single-GPU to multi-node, SLURM / torchrun launch paths

Clients can run **serially** or **in parallel** across GPUs; the library is
W&B-free (metrics go to JSON / console) and exposes extension points for new
datasets, environments, heterogeneity strategies, and aggregation rules
(see [`docs/extending.md`](docs/extending.md)).
Full details in [`docs/features.md`](docs/features.md).

---

## Repository layout

```
fedagent/
├── README.md                  # this file
├── LICENSE                    # Apache-2.0
├── NOTICE                     # third-party attributions
├── CITATION.cff               # how to cite (TODO: finalize once published)
├── reproduce.sh               # one-command reproduction entry point
├── evaluate.sh                # evaluate a trained checkpoint + collect trajectories
├── download_data.sh           # fetch WebShop / ALFWorld data (not shipped)
├── .env.example               # optional environment variables (W&B removed)
├── .gitignore
├── core/                      # federated server + aggregation + trainers (contribution)
├── utils/                     # model aggregation (FedAvg, incl. FSDP)
├── tools/                     # run_federated.py, resolve_paths.py, checkpoint monitor
│   └── aggregation/           # aggregation verification / diagnostic toolbox
├── scripts/                   # setup_env.sh, runners, verl-agent base launch scripts
├── config/                    # curated experiment configs (W&B stripped)
│   ├── paths.yaml.example     # path template consumed by tools/resolve_paths.py
│   └── example.yaml           # fully annotated example config
├── docs/                      # user-facing documentation (see below)
├── eval/                      # checkpoint evaluation + trajectory collection
└── third_party/
    └── verl-agent/            # vendored upstream (Apache-2.0), no bundled 5.6 GB data
```

### FedAgent code map

FedAgent is a framework extension, so first-party code spans two layers: a
top-level control plane and in-framework hooks that live inside the vendored
tree (verl-agent imports/runs them). Everything else under
`third_party/verl-agent/` is unmodified upstream (Apache-2.0). Per-file detail:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); exhaustive edit list:
[`CHANGES.md`](third_party/verl-agent/CHANGES.md).

```text
fedagent/                                       ── first-party (this work) ──
├── core/                 control plane: federated server, round orchestration, aggregation
├── utils/                model aggregation (FedAvg, incl. FSDP)
├── tools/                run_federated.py, resolve_paths.py, aggregation/, env_heterogeneity/, heterogeneity_test/, monitor/
├── eval/                 checkpoint evaluation + trajectory collection
├── scripts/              setup_env.sh, federated runners, verl-agent launchers, plotting/
├── config/, docs/        experiment configs (W&B stripped) + documentation
│
└── third_party/verl-agent/    ── vendored upstream (Apache-2.0); our hooks woven in ──
    ├── agent_system/environments/partition_strategy.py        core heterogeneity constructions
    ├── agent_system/environments/fed_env_manager.py           federated env managers
    ├── verl/trainer/main_ppo_fed.py                           federated PPO/GRPO entry point
    ├── verl/trainer/ppo/ray_trainer_fed.py                    Ray federated trainer
    ├── verl/utils/checkpoint/fsdp_checkpoint_manager_fed.py   federated checkpoint manager
    └── verl/utils/tracking_fed.py                             per-round / per-client tracking
```

---

## Installation

FedAgent runs on **Python 3.10**. **WebShop and ALFWorld have conflicting
dependencies** (WebShop needs a Java/Lucene search stack via `pyserini`/`pyjnius`;
ALFWorld needs the TextWorld + Fast-Downward planning stack). Following
[verl-agent's own guidance](https://github.com/langfengQ/verl-agent#install-supported-environments),
**each benchmark gets its own conda env**:

```bash
# WebShop  -> conda env `fedagent-webshop` (Python 3.10), incl. vendored verl-agent
bash scripts/setup_env.sh create webshop
conda activate fedagent-webshop

# ALFWorld -> conda env `fedagent-alfworld`
bash scripts/setup_env.sh create alfworld
conda activate fedagent-alfworld
alfworld-download -f          # one-time: PDDL + game files -> ~/.cache/alfworld/

# Path template (both envs)
cp config/paths.yaml.example config/paths.yaml && $EDITOR config/paths.yaml
```

WebShop additionally needs a **JDK** on PATH (for `pyserini`). Each `reproduce.sh` /
`evaluate.sh` run must happen **inside the matching env**. Full step-by-step setup (both envs, data, and the upstream env packages) is in
**[`docs/installation.md`](docs/installation.md)**.

> W&B logging is **removed** from this release, no tracking account or key needed.

---

## Data

The default configs run **out of the box**: the three small WebShop catalog files
(`items_shuffle_1000.json`, `items_ins_v2_1000.json`, `items_human_ins.json`,
backing `webshop.use_small: true`) are **already shipped** in the repo, where the
WebShop env loads them from
`third_party/verl-agent/agent_system/environments/env_package/webshop/webshop/data/`
(**not** the top-level `data/`, which ships only a README).

Two things are fetched **separately**:

```bash
bash download_data.sh           # ALFWorld game files (auto) + WebShop full-catalog instructions
```

- **ALFWorld game files**: auto-downloaded by the script (`alfworld-download`) to
  `~/.cache/alfworld`, where the env reads them.
- **WebShop full catalog** (`items_shuffle.json` ~5.2 GB + `items_ins_v2.json`),
  needed only for full-scale `webshop.use_small: false` runs, and fetched
  **manually**: the script prints instructions to download them from
  [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop) into the same
  WebShop `data/` directory. The shipped small files already reproduce the paper's
  WebShop results. See [`docs/configuration.md`](docs/configuration.md) for the
  `use_small` switch.

## Models

Backbones are **HuggingFace model ids** (default `Qwen/Qwen2.5-1.5B-Instruct`) and
**auto-download** on first run to `~/.cache/huggingface` (set `HF_HOME` to relocate;
~3 GB for 1.5B up to ~15 GB for 7B). Two caveats: the main table's
**`Llama-3.2-3B-Instruct` is gated**: accept its HuggingFace license and
`huggingface-cli login` (or set `HF_TOKEN`) first; and on **offline / air-gapped
clusters** pre-fetch on a login node and set `HF_HUB_OFFLINE=1`. See
[`docs/installation.md`](docs/installation.md#models) for details.

---

## Quick Start

Run a FedAgent experiment **directly** with the federated runner: give it a config
name (its path under `config/`, without the `.yaml`) and a round count, from the
repository root inside the matching conda env.

```bash
# WebShop main run, 70 rounds. The config sets the backbone, GPU count, and protocol:
python tools/run_federated.py --restart-resume \
  uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform 70
```

The runner resolves the config, creates the run's `./output/` directory, and
launches per-client training; it is re-runnable and resumes where it left off.
**Hardware** (GPU count, FSDP offload, serial vs parallel clients) is read from the
config (`verl.trainer.*`, `federated.training.*`); to change it, edit those keys or
follow the **[running guide](docs/running.md)**, which also documents the
lower-level launcher `scripts/start_federated.sh`.

Evaluate a trained checkpoint and collect trajectories:

```bash
bash evaluate.sh webshop /path/to/checkpoint
```

Trained checkpoints are saved as FSDP shards; `evaluate.sh` merges them to
HuggingFace format on first use (see [eval/README.md](eval/README.md)).

---

## Reproducing the paper

To reproduce the paper, **`reproduce.sh`** wraps the runner with named experiments
and hardware flags: it resolves the canonical config, applies any overrides, and
launches it.

```bash
bash reproduce.sh webshop-main                  # WebShop main table, GRPO, 4 GPUs
bash reproduce.sh alfworld-main --single-gpu    # ALFWorld main, 1-GPU debug run
bash reproduce.sh webshop-main --mode serial    # clients run one at a time
bash reproduce.sh webshop-main --slurm          # submit via SLURM (cluster)
```

The full guide is in **[`docs/reproducing.md`](docs/reproducing.md)**: every table
and figure mapped to its config directory, with run commands, seeds, and compute
estimates (**~1,800 H100 GPU-hours** total). It covers the main table (Local /
Centralized / FedAgent across four backbones × WebShop + ALFWorld), the task- and
environment-level heterogeneity studies, and the decentralized ablations.

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/features.md`](docs/features.md) | **Key features in depth**: the config keys, flags, and files behind each headline capability. |
| [`docs/installation.md`](docs/installation.md) | **Two-conda-env setup** (WebShop vs ALFWorld), full step-by-step, JDK / game-file notes. |
| [`docs/running.md`](docs/running.md) | **Running FedAgent**: the run-mode matrix (parallel vs serial, FSDP on/off, single-GPU, variable GPU count, multi-node, SLURM), flag-to-knob table, and worked examples. |
| [`docs/reproducing.md`](docs/reproducing.md) | Per-experiment reproduction recipes, compute estimates, and seeds. |
| [`docs/heterogeneity.md`](docs/heterogeneity.md) | The two-level heterogeneity taxonomy and how to construct/select each variant. |
| [`docs/configuration.md`](docs/configuration.md) | Config filename decoder and field reference for the `federated:` and `verl:` blocks. |
| [`docs/extending.md`](docs/extending.md) | Extension points: new dataset/env, new heterogeneity strategy, new RL algorithm, new aggregation strategy. |

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

This project is released under the **Apache License 2.0**: see
[`LICENSE`](LICENSE).

## Acknowledgements

FedAgent builds on a vendored, modified fork of **verl-agent**, which itself
extends **veRL**. We gratefully acknowledge:

- **veRL**: © ByteDance / the veRL authors (Apache-2.0): the base RL training
  framework. <https://github.com/volcengine/verl>
- **verl-agent / GiGPO**: Feng et al., *Group-in-Group Policy Optimization for
  LLM Agent Reinforcement Learning* ([arXiv:2505.10978](https://arxiv.org/abs/2505.10978)):
  the agent-RL fork FedAgent is built on. <https://github.com/langfengQ/verl-agent>
- **WebShop**: Yao et al., Princeton NLP (MIT License): the e-commerce agent
  benchmark.
- **ALFWorld**: Shridhar et al., Microsoft Research (MIT License): the embodied
  household agent benchmark.

Full per-component attributions and license texts are aggregated in
[`NOTICE`](NOTICE).
