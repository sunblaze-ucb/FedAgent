# Installation

FedAgent runs on **Python 3.10** with **NVIDIA GPUs** (paper default: 4 × H100
80 GB, but smaller setups work — see [running_experiments.md](running_experiments.md)).

## Why two conda environments

The two bundled agent benchmarks have **mutually incompatible dependencies**, so
each lives in its own conda environment. This follows
[verl-agent's own setup guidance](https://github.com/langfengQ/verl-agent#install-supported-environments),
which installs every environment separately.

| Benchmark | conda env | Key extra stack | Special requirement |
|---|---|---|---|
| **WebShop**  | `fedagent-webshop`  | `pyserini` + `pyjnius` (Lucene/BM25 over the product catalog), `faiss`, `spacy` | a **JDK** on `PATH` (pyserini uses Java) |
| **ALFWorld** | `fedagent-alfworld` | `alfworld==0.4.2`, `textworld`, `fast-downward` (PDDL planning) | **game files** via `alfworld-download` |

Both envs share the same vendored verl-agent + torch/vLLM core; only the
benchmark-specific packages differ. You only need the env for the benchmark you run.

> `setup_env.sh` creates a fresh Python 3.10 env named `fedagent-<task>` and installs
> `<task>_requirements.txt` (which includes the vendored verl-agent via
> `pip install -e ./third_party/verl-agent`).

## WebShop

```bash
bash scripts/setup_env.sh create webshop      # -> conda env `fedagent-webshop`
conda activate fedagent-webshop
```

- **JDK required** for `pyserini`/`pyjnius`. e.g. `conda install -c conda-forge openjdk=21`, or use a system JDK and set `JAVA_HOME`.
- **Data:** the three small WebShop catalog files are **already shipped**
  (`items_shuffle_1000.json`, `items_ins_v2_1000.json`, `items_human_ins.json`),
  backing the `webshop.use_small: true` code path — so the default configs run with
  **no WebShop download**. They are not in the top-level `data/` directory (which
  ships only a README); they live where the WebShop environment loads them from:
  `third_party/verl-agent/agent_system/environments/env_package/webshop/webshop/data/`.
  The full ~5.2 GB catalog (`items_shuffle.json` + `items_ins_v2.json`, used only by
  `webshop.use_small: false`) is **not** auto-downloaded — fetch it manually from
  [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop) into that same
  directory. (`bash download_data.sh` prints these exact instructions; it does not
  download the WebShop catalog itself.)
- The `spacy` / `typer` version-conflict warning during install is **benign**
  (noted upstream by verl-agent) — ignore it.
- **`flash-attn`** builds from source and `import`s `torch` at build time. If
  `pip install -r ...` fails on `flash_attn` with `ModuleNotFoundError: No module
  named 'torch'`, install it after torch with build isolation off:
  `pip install torch==2.6.0 && pip install flash_attn==2.7.4.post1 --no-build-isolation`.

## ALFWorld

```bash
bash scripts/setup_env.sh create alfworld     # -> conda env `fedagent-alfworld`
conda activate fedagent-alfworld

# One-time: download PDDL + game files + the MaskRCNN detector to ~/.cache/alfworld/
alfworld-download -f
```

## Models

Backbones are specified as **HuggingFace model ids** (e.g.
`actor_rollout_ref.model.path: Qwen/Qwen2.5-1.5B-Instruct`), so they
**auto-download** from the Hub on first run — no manual step for the default setup.

- **Cache / disk.** Models land in `~/.cache/huggingface` (override with `HF_HOME`).
  Budget roughly ~3 GB for Qwen2.5-1.5B up to ~15 GB for Qwen2.5-7B.
- **Gated backbone.** The main table's `Llama-3.2-3B-Instruct` is **gated** on
  HuggingFace: accept its license on the model page, then authenticate
  (`huggingface-cli login`, or export `HF_TOKEN`) before using that backbone. The
  Qwen backbones are ungated.
- **Offline / air-gapped clusters** (compute nodes without internet). Pre-fetch on a
  login node (`huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`), then set
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` on the compute nodes — or point
  `actor_rollout_ref.model.path` / `tokenizer_path` at a local snapshot directory.

## Path configuration (both envs)

FedAgent resolves machine-specific roots through `config/paths.yaml`:

```bash
cp config/paths.yaml.example config/paths.yaml
$EDITOR config/paths.yaml      # set project_root, data dirs, ...
```

## Running

`reproduce.sh` / `evaluate.sh` must run **inside the matching env** — activate
`fedagent-webshop` for WebShop, `fedagent-alfworld` for ALFWorld:

```bash
conda activate fedagent-webshop
bash reproduce.sh webshop-main

conda activate fedagent-alfworld
bash reproduce.sh alfworld-main
```

## Reference

The underlying environment packages (and the other verl-agent benchmarks — Sokoban,
Gym Cards, AppWorld) are documented in the vendored
[`third_party/verl-agent/README.md`](../third_party/verl-agent/README.md).

> Weights & Biases logging is removed from this release; no W&B account or key is needed.
