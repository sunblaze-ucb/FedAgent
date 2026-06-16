# FedAgent: A Library for Decentralized LLM Agent RL

> Train LLM agents collaboratively across decentralized clients — **without sharing local data**.

**FedAgent is an extendable library for federated reinforcement learning of LLM
agents.** It provides reusable, pluggable building blocks — a federated training
server, model **aggregation strategies** (FedAvg / FedProx / your own), client
data & environment **partitioning** (including a two-level *task* vs *environment*
heterogeneity suite), and a federated **PPO / GRPO trainer** built on
[verl-agent](https://github.com/langfengQ/verl-agent) — so you can run, study, and
**extend** federated agent RL on your own datasets, environments, and algorithms.

It is *also* the reference implementation for the FedAgent paper *"Is Decentralized
LLM Agent RL Robust to Heterogeneity? An Asymmetric Tale"* — **reproducing the paper
is one supported use case** (see [Reproducing the paper](#reproducing-the-paper)),
not the whole library.

### Extend it
| You want to add… | Where | Guide |
|---|---|---|
| a new **environment / dataset** | `third_party/verl-agent` env package | [extending.md](docs/extending.md) |
| a new **heterogeneity** (client partition) | `partition_strategy.py` | [heterogeneity.md](docs/heterogeneity.md) |
| a new **RL algorithm** (beyond PPO/GRPO) | verl-agent trainer | [extending.md](docs/extending.md) |
| a new **aggregation** (beyond FedAvg/FedProx) | `utils/model_aggregation.py` | [extending.md](docs/extending.md) |

---

## Abstract

Training AI agents powered by Large Language Models (LLMs) typically requires
centralized access to user data, raising privacy and scalability concerns. We
explore **FedAgent**, a decentralized RL paradigm that collaboratively trains
LLM agents across distributed clients without sharing local data. The central
reliability question is twofold: is FedAgent effective under a *uniform* client
distribution, and — more importantly — is it robust to client *heterogeneity*?

For the former, we provide the first empirical evidence that FedAgent matches
**Centralized Agent Training** and outperforms **Local Agent Training**. For the
latter, we formalize **Agent Heterogeneity** at two structurally distinct
levels: **task-level** (what clients ask the agent to do) and
**environment-level** (the dynamics in which the agent acts). This split is
anchored on the **Input-Dynamics Asymmetry** of task-augmented MDPs: tasks enter
the policy through its input channel, while environments do not. From this we
derive an **Asymmetric Robustness Mechanism**: FedAgent is robust to task-level
heterogeneity but worst-case non-robust to environment-level heterogeneity, with
three sufficient conditions that recover robustness. On the real-world agent
benchmarks **WebShop** and **ALFWorld**, FedAgent remains robust under extreme
task-level heterogeneity and traces a *stable–degrade–collapse* spectrum under
environment-level heterogeneity.

---

## Key idea: two-level heterogeneity and asymmetric robustness

The conceptual core of FedAgent is a **two-level taxonomy of agent
heterogeneity** and the **asymmetric robustness** it implies:

| Level | What varies across clients | Observable to policy? | Robustness |
|---|---|---|---|
| **Task-level**  | the per-client task distribution $\mathcal{D}_{\tau_i}$ (Preference / Coverage / Hardness) | **Yes** — $\tau$ is part of the prompt | **Robust** (≈ centralized on the task mixture) |
| **Environment-level** | the transition kernel $P_i$ (5 WebShop variants along 4 pipeline stages) | **No** — only sensed through successor states | **Worst-case non-robust** (stable→degrade→collapse) |

Because a task descriptor enters the policy as input, a single set of weights can
encode "different prompt → different behavior." Because the transition kernel is
implicit in the dynamics, a single set of weights *cannot* encode an
environment-conditional function — when two clients disagree on the optimal
action for the same state, the aggregated model must commit to one and be wrong
for the other. This asymmetry is what the framework is built to study. See
[`docs/heterogeneity.md`](docs/heterogeneity.md) for the full construction.

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
├── utils/                     # model aggregation (FedAvg / FedProx, incl. FSDP)
├── tools/                     # run_federated.py, resolve_paths.py, checkpoint monitor
│   └── aggregation/           # aggregation verification / diagnostic toolbox
├── scripts/                   # setup_env.sh, runners, verl-agent base launch scripts
├── config/                    # curated experiment configs (W&B stripped)
│   ├── paths.yaml.example     # path template consumed by tools/resolve_paths.py
│   └── example.yaml           # fully annotated example config
├── docs/                      # user-facing documentation (see below)
├── tests/                     # partition demos + conversion utilities
├── eval/                      # checkpoint evaluation + trajectory collection
└── third_party/
    └── verl-agent/            # vendored upstream (Apache-2.0), no bundled 5.6 GB data
```

### FedAgent code map

FedAgent is a **framework extension**, so first-party code spans two layers: a
top-level **control plane** and **in-framework hooks** that live inside the
vendored tree because verl-agent imports/runs them. Everything else under
`third_party/verl-agent/` is unmodified upstream (Apache-2.0). Per-file detail:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**; exhaustive edit list:
[`CHANGES.md`](third_party/verl-agent/CHANGES.md).

```text
fedagent/                                       ── first-party (this work) ──
├── core/                 control plane: federated server, round orchestration, aggregation
├── utils/                model aggregation (FedAvg / FedProx, incl. FSDP)
├── tools/                run_federated.py, resolve_paths.py, aggregation/, env_heterogeneity/, monitor/
├── eval/                 checkpoint evaluation + trajectory collection
├── scripts/              setup_env.sh, federated runners, verl-agent launchers, plotting/
├── tests/heterogenous/   partition simulations + sharding smoke test
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
ALFWorld needs the TextWorld + Fast-Downward planning stack), so — following
[verl-agent's own guidance](https://github.com/langfengQ/verl-agent#install-supported-environments)
— **each benchmark gets its own conda env**:

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
`evaluate.sh` run must happen **inside the matching env**. Full step-by-step setup —
both envs, data, and the upstream env packages — is in
**[`docs/installation.md`](docs/installation.md)**.

> W&B logging is **removed** from this release — no tracking account or key needed.

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

- **ALFWorld game files** — auto-downloaded by the script (`alfworld-download`) to
  `~/.cache/alfworld`, where the env reads them.
- **WebShop full catalog** (`items_shuffle.json` ~5.2 GB + `items_ins_v2.json`) —
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
**`Llama-3.2-3B-Instruct` is gated** — accept its HuggingFace license and
`huggingface-cli login` (or set `HF_TOKEN`) first; and on **offline / air-gapped
clusters** pre-fetch on a login node and set `HF_HUB_OFFLINE=1`. See
[`docs/installation.md`](docs/installation.md#models) for details.

---

## Quickstart

A complete, paper-default run (the WebShop main table, GRPO, 4 × H100) is a
single command:

```bash
bash reproduce.sh webshop-main
```

This resolves the canonical config, launches the federated runner, and writes
checkpoints and metrics under `./output/`. Flags for other hardware / parallel
modes (serial clients, FSDP off, single GPU, multi-node, SLURM) are documented
in [`docs/running_experiments.md`](docs/running_experiments.md):

```bash
bash reproduce.sh webshop-main --gpus 4            # default
bash reproduce.sh webshop-main --mode serial       # clients run serially
bash reproduce.sh webshop-main --single-gpu        # 1-GPU debug run
bash reproduce.sh webshop-main --fsdp off          # disable FSDP param offload
bash reproduce.sh alfworld-main --slurm            # submit via SLURM (cluster)
```

To evaluate a trained checkpoint and collect trajectories:

```bash
bash evaluate.sh webshop /path/to/checkpoint
```

---

## Reproducing the paper

The main results (Table 1, GRPO) compare **Local / Centralized / FedAgent**
across **four backbones** (Qwen2.5-1.5B/3B/7B-Instruct, Llama-3.2-3B-Instruct)
on **WebShop** and **ALFWorld**. The shared federation protocol is
$N=100$ clients, $M=2$ clients/round, $E=3$ local epochs, $T=70$ rounds,
$|X_i|=100$ tasks/client; runs are averaged over **3 seeds**.

| Paper artifact | Config directory (GRPO) | What it backs |
|---|---|---|
| **Table 1** (GRPO main; PPO appendix table) | `config/uniform/<model>/{local_clientN,centralized,main}/{grpo,ppo}/` | Local / Centralized / FedAgent × 4 models × 2 envs |
| **Fig. training-dynamics-main** (`main_combined_val_success_rate.pdf`) | `config/uniform/Qwen2.5-1.5B-Instruct/{main,centralized}/grpo/` | FedAgent vs Centralized validation curves (Qwen2.5-1.5B) |
| **Fig. heterogeneity-challenges** (`heterogeneous_combined_val_success_rate.pdf`, 6 panels) | `config/task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/` | (a,b) Preference `omega` 0.01→0.99 · (c,d) Coverage `std` 256→1 · (e,f) Hardness `success_std` 256→1 |
| **Fig. env-heterogeneity** (`webshop_env_variants_combined_val_success_rate.pdf`, GRPO left / PPO right) | `config/env_heterogeneity/{catalog_split,field_subset_index,bm25_reweighting,lookalike_injection,rank_wrapper}{,_ppo}/` | 5 WebShop variants (see per-variant table below) |
| **Fig. decentralized** / hyperparameter-sensitivity (`decentralized_setting.pdf`) | `config/decentralized/{ep_per_round_change,samples_change,selected_cl_change}/{grpo,ppo}/` | $E$, $\|X_i\|$, $M$ sweeps |

Within `uniform/<model>/`, `main/` (and `main_seed{1,2}/`) is FedAgent
(`total-100`), `centralized/` is the centralized baseline (`total-1`), and
`local_client{1,2,3}/` are the single-client Local baselines (paper reports
client indices 21, 42, 84). Across every group the **GRPO** configs live under
`grpo/` and the **PPO** appendix counterparts under `ppo/` (or the `*_ppo`
sibling dirs for env-heterogeneity).

The five WebShop env-heterogeneity variants (each its own `main_*` dir) map to
the env-heterogeneity figure and the *stable→degrade→collapse* spectrum as
follows:

| `main_*` dir | Paper name | Perturbed pipeline stage | Strategy (`env.partition.strategy`) | Spectrum (GRPO → PPO) |
|---|---|---|---|---|
| `catalog_split` | Catalog Split | content | `catalog_split` | Pattern B/C |
| `field_subset_index` | Field-Subset Index | encoding | `bm25_variant` (variant_pool: fields_only) | Pattern C |
| `bm25_reweighting` | BM25 Reweighting | matching | `bm25_variant` (default pool) | Pattern C |
| `lookalike_injection` | Lookalike Injection | content + matching | `lookalike_injection` | Pattern D → rescued to C |
| `rank_wrapper` | Rank Wrapper | rendering | `rank_wrapper` | Pattern D → rescued to C |

**Compute.** Total reported compute is **~1,800 H100 GPU-hours**; a single
WebShop-GRPO federated sweep is **~93 GPU-hours** (~24 h on 4 × H100). Per-sweep
estimates for every env/algorithm pair are in
[`docs/reproducing.md`](docs/reproducing.md).

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/installation.md`](docs/installation.md) | **Two-conda-env setup** (WebShop vs ALFWorld) — full step-by-step, JDK / game-file notes. |
| [`docs/running_experiments.md`](docs/running_experiments.md) | Hardware & scaling matrix: parallel vs serial clients, FSDP on/off, single-GPU, multi-node, SLURM vs torchrun, mapped to `reproduce.sh` flags. |
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
  author  = {TODO(author): author list},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

---

## License

This project is released under the **Apache License 2.0** — see
[`LICENSE`](LICENSE).

## Acknowledgements

FedAgent builds on a vendored, modified fork of **verl-agent**, which itself
extends **veRL**. We gratefully acknowledge:

- **veRL** — © ByteDance / the veRL authors (Apache-2.0): the base RL training
  framework. <https://github.com/volcengine/verl>
- **verl-agent / GiGPO** — Feng et al., *Group-in-Group Policy Optimization for
  LLM Agent Reinforcement Learning* ([arXiv:2505.10978](https://arxiv.org/abs/2505.10978)):
  the agent-RL fork FedAgent is built on. <https://github.com/langfengQ/verl-agent>
- **WebShop** — Yao et al., Princeton NLP (MIT License): the e-commerce agent
  benchmark.
- **ALFWorld** — Shridhar et al., Microsoft Research (MIT License): the embodied
  household agent benchmark.

Full per-component attributions and license texts are aggregated in
[`NOTICE`](NOTICE).
