# Installation

FedAgent is a **thin overlay on stock verl 0.8** (it imports verl as a library —
no fork; see [`../README.md`](../README.md)). It runs on **NVIDIA GPUs** (paper
default: 4 × H100 80 GB; 2 GPUs suffice for the smokes — see
[`./running.md`](./running.md)).

## Why three conda environments

FedAgent uses **three** conda environments because the trainer and the two bundled
agent benchmarks have **mutually incompatible dependencies**. The separation is
load-bearing: the env services pin their own `torch` / `gym` / `numpy` and a
Java/Lucene or planner stack that cannot coexist with verl 0.8's stack. The
service packages document this directly:

> *"WebShop remote env service (runs in the verl-agent-webshop conda env, NOT the
> trainer env). Kept separate from `fedagent.envs` so importing the package in the
> trainer env never pulls WebShop's conflicting deps (gym 0.24 / pyserini / torch
> 2.6). Only the HTTP client `fedagent.envs.webshop.WebShopEnv` is imported
> trainer-side."* — [`../envs/webshop/service/__init__.py`](../envs/webshop/service/__init__.py)

> *"ALFWorld remote env service ... Kept separate from `fedagent.envs` so importing
> the package in the trainer env never pulls ALFWorld's heavy/conflicting deps
> (alfworld / textworld / gymnasium / torch + torchvision pinned for the env).
> Only the HTTP client `fedagent.envs.alfworld.AlfworldEnv` is imported
> trainer-side."* — [`../envs/alfworld/service/__init__.py`](../envs/alfworld/service/__init__.py)

The trainer therefore only ever imports the thin HTTP **client** for an
environment; the heavy **engine** runs in its own env behind a FastAPI service, and
the two talk over HTTP. You only need the service env for the benchmark you run
(`tinyguess` runs in-process in the trainer env and needs neither service).

| conda env | Purpose | What runs in it | Key deps |
|---|---|---|---|
| `fedagent-verl08` | Trainer / orchestrator | `python -m fedagent.fed.run_fed` (the federated driver) and each per-client `python -m fedagent.main_ppo_fed` | **Python 3.12**, stock **verl 0.8**, vLLM, flash-attn, ray, torch (cu12) |
| `verl-agent-webshop` | WebShop remote env service | `uvicorn fedagent.envs.webshop.service.server:app`, launched by [`../envs/webshop/service/run_service.sh`](../envs/webshop/service/run_service.sh) | **Python 3.10**, `gym==0.24.0`, `pyserini==0.17.0` + `pyjnius` (Lucene/BM25), `torch==2.6.0`, `numpy==1.26.4`, `spacy`; **a JDK on `PATH`** |
| `verl-agent-alfworld` | ALFWorld remote env service | `uvicorn fedagent.envs.alfworld.service.server:app`, launched by [`../envs/alfworld/service/run_service.sh`](../envs/alfworld/service/run_service.sh) | **Python 3.10**, `alfworld==0.4.2`, `textworld==1.6.2`, `fast_downward_textworld` (PDDL planner), `gymnasium==0.29.1`, `torch==2.6.0` + `torchvision==0.21.0`; **game files** via `alfworld-download` |

All three envs are created with the cluster conda; activate via:

```bash
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate <env-name>
```

## 1. Trainer env — `fedagent-verl08` (verl 0.8, Python 3.12)

This is **stock verl 0.8 imported as a library** — there is no verl fork and no
patched verl tree. Create a Python 3.12 env and install verl 0.8 with its FSDP
inference stack (vLLM + flash-attn); FedAgent itself ships no `setup.py` — it is
used in-place from the repo with the repo root on `PYTHONPATH`.

```bash
conda create -n fedagent-verl08 python=3.12 -y
conda activate fedagent-verl08

# Install verl 0.8 + the vLLM/SGLang inference stack (FSDP-only; no Megatron).
# verl ships an installer for the GPU stack:
bash /path/to/verl/scripts/install_vllm_sglang_mcore.sh   # USE_MEGATRON=0
pip install -e /path/to/verl                              # verl 0.8 as a library
```

- **Python 3.12** is required for this env (the WebShop/ALFWorld service envs use
  3.10).
- **flash-attn is mandatory.** verl 0.8 calls into `flash_attn.bert_padding`
  unconditionally during training (using `attn_implementation=sdpa` does *not*
  avoid it). If a prebuilt wheel is incompatible with your glibc / CUDA, build it
  from source against your toolchain (e.g. `flash_attn==2.7.4.post1`,
  `--no-build-isolation`) after `torch` is installed.
- **Do not** `pip install --force-reinstall` without `--no-deps`: it cascades into
  a bare `torch` dependency and can pull a mismatched CUDA build that breaks the
  env.

No FedAgent install step is needed: the driver and the per-client entry add the
repo root to `PYTHONPATH` themselves and import `verl` from the active env
(`fedagent/fed/run_fed.py` sets `PYTHONPATH=<repo root>` and resolves verl's stock
config dir via `import verl`).

## 2. WebShop service env — `verl-agent-webshop` (Python 3.10)

[`../envs/webshop/service/run_service.sh`](../envs/webshop/service/run_service.sh) does
`conda activate verl-agent-webshop` and launches the service with `uvicorn`. The
env holds WebShop's conflicting stack (`gym 0.24` / `pyserini` / `torch 2.6` /
`numpy 1.26`).

```bash
conda create -n verl-agent-webshop python=3.10 -y
conda activate verl-agent-webshop
pip install -r webshop_requirements.txt      # repo root; pins the WebShop stack
```

- **A JDK must be on `PATH`.** `pyserini` / `pyjnius` drive a Java/Lucene BM25 index
  over the product catalog. Install one, e.g. `conda install -c conda-forge
  openjdk=21`, or use a system JDK and export `JAVA_HOME`.
- The WebShop engine and goal data are **vendored in-tree** at
  `fedagent/envs/webshop/engine/` (see §4); nothing is fetched from PyPI for it and
  `webshop_requirements.txt` no longer needs an editable verl-agent install.
- `server.py` additionally injects the WebShop engine onto `sys.path` at startup
  and pre-warms a pool of `WebAgentTextEnv` instances (each `gym.make` is ~26 s,
  JVM + index startup), so the trainer never imports WebShop.

## 3. ALFWorld service env — `verl-agent-alfworld` (Python 3.10)

[`../envs/alfworld/service/run_service.sh`](../envs/alfworld/service/run_service.sh) does
`conda activate verl-agent-alfworld`, exports `ALFWORLD_DATA`, and launches the
service with `uvicorn`. The env holds ALFWorld's stack (`alfworld 0.4.2` /
`textworld` / a Fast-Downward PDDL planner / `torchvision`).

```bash
conda create -n verl-agent-alfworld python=3.10 -y
conda activate verl-agent-alfworld
pip install -r alfworld_requirements.txt     # repo root; pins the ALFWorld stack

# One-time: download the PDDL + textworld game files (and detector) into the cache.
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download -f
```

- **Game files are required.** `alfworld-download` populates `ALFWORLD_DATA` (the
  solvable `game.tw-pddl` files the service walks at startup). `run_service.sh`
  exports `ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"`; **this var must
  be exported** because the bundled `config_tw.yaml` references game/logic/detector
  paths as `$ALFWORLD_DATA/...` (expanded at runtime). Set it to the same directory
  you downloaded into.
- The ALFWorld engine is **vendored in-tree** at `fedagent/envs/alfworld/engine/`
  (see §4); `alfworld_requirements.txt` no longer needs an editable verl-agent
  install. The service builds the `AlfredTWEnv` interface once and pools
  single-instance textworld envs; the trainer never imports ALFWorld.

## 4. Vendored engines — `fedagent/envs/<name>/engine/`

The real WebShop and ALFWorld engines (and the original action parsers / partition
code) are **vendored in-tree**, each beside its service:
- WebShop: [`../envs/webshop/engine/`](../envs/webshop/engine/) — `web_agent_site` +
  the shipped catalog data.
- ALFWorld: [`../envs/alfworld/engine/`](../envs/alfworld/engine/) — the `AlfredTWEnv`
  wrapper under a preserved `agent_system/environments/` import anchor, plus
  `partition_strategy.py`.

They are **not fetched from PyPI** and need **no editable install**. At runtime each
service injects its engine onto `sys.path` and loads the action parser in isolation.
The vendored `agent_system/environments/__init__.py` is intentionally **empty** — the
upstream one imported verl-agent's `env_manager` (→ the old verl 0.3.x); neutralised so
the engines carry **no verl-agent dependency**. Nothing here needs a separate install
step beyond the `-r *_requirements.txt` above.

## 5. Models

Backbones are specified as **HuggingFace model ids** (the paper configs set
`actor_rollout_ref.model.path` to ids such as `Qwen/Qwen2.5-1.5B-Instruct`), so
they **auto-download** from the Hub on first run — no manual step for the default
setup.

- **Cache / disk.** Models land in `~/.cache/huggingface` (override with `HF_HOME`).
  Budget ~3 GB for Qwen2.5-1.5B up to ~15 GB for Qwen2.5-7B.
- **Gated backbone.** `Llama-3.2-3B-Instruct` is **gated**: accept its license on
  the model page, then authenticate (`huggingface-cli login`, or export `HF_TOKEN`)
  before using it. The Qwen backbones are ungated.
- **Offline / air-gapped clusters.** Pre-fetch on a login node
  (`huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`), then export
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` on the compute node and point the
  config at a local snapshot — either override `actor_rollout_ref.model.path` in
  the YAML or pass `--model-path /path/to/snapshot` to `run_fed.py`. (Strip any
  trailing `/` from the snapshot path; verl's `copy_to_local` rejects it.)

## 6. CUDA note

On clusters where the CUDA toolkit is a module (rather than installed in the env),
export `CUDA_HOME` so vLLM's deep-GEMM check is satisfied. This repo's smoke
scripts use the cuda-12.1 module and disable the Hopper deep-GEMM path (not needed
for bf16):

```bash
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
```

Adjust `CUDA_HOME` to your cluster's CUDA module. This is only relevant in the
`fedagent-verl08` trainer env.

## Next steps

With the trainer env active you can run the in-process smoke immediately (no
service needed); WebShop / ALFWorld runs auto-launch their per-client services in
the matching env. See [`./running.md`](./running.md) for invocation, GPUs, and
baselines, and [`../README.md`](../README.md) for the overlay design.
