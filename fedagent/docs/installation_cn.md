# 安装

FedAgent 是**架在原生 verl 0.8 之上的薄 overlay**（它把 verl 作为库 import ——
不 fork；见 [`../README.md`](../README.md)）。它运行在 **NVIDIA GPU** 上（论文
默认：4 × H100 80 GB；smoke 只需 2 张 GPU —— 见
[`./running.md`](./running.md)）。

## 为什么要三个 conda 环境

FedAgent 使用**三个** conda 环境，因为 trainer 与两个内置的
agent benchmark 有**互不兼容的依赖**。这种隔离是**承重的**：env 服务各自钉死自己的
`torch` / `gym` / `numpy`，以及一套无法与 verl 0.8 的栈共存的
Java/Lucene 或 planner 栈。这些服务包的文档直接说明了这一点：

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

因此 trainer 永远只 import 某个环境的薄 HTTP **client**；沉重的 **engine**
跑在它自己的环境里、藏在一个 FastAPI 服务后面，二者通过 HTTP 通信。你只需为你要跑的那个
benchmark 准备它的服务环境（`tinyguess` 在 trainer 环境里**进程内**运行，两个服务都不需要）。

| conda env | 用途 | 里面跑什么 | 关键依赖 |
|---|---|---|---|
| `fedagent-verl08` | Trainer / orchestrator | `python -m fedagent.fed.run_fed`（联邦 driver）以及每个 per-client 的 `python -m fedagent.main_ppo_fed` | **Python 3.12**、原生 **verl 0.8**、vLLM、flash-attn、ray、torch (cu12) |
| `verl-agent-webshop` | WebShop remote env service | `uvicorn fedagent.envs.webshop.service.server:app`，由 [`../envs/webshop/service/run_service.sh`](../envs/webshop/service/run_service.sh) 启动 | **Python 3.10**、`gym==0.24.0`、`pyserini==0.17.0` + `pyjnius`（Lucene/BM25）、`torch==2.6.0`、`numpy==1.26.4`、`spacy`；**`PATH` 上要有一个 JDK** |
| `verl-agent-alfworld` | ALFWorld remote env service | `uvicorn fedagent.envs.alfworld.service.server:app`，由 [`../envs/alfworld/service/run_service.sh`](../envs/alfworld/service/run_service.sh) 启动 | **Python 3.10**、`alfworld==0.4.2`、`textworld==1.6.2`、`fast_downward_textworld`（PDDL planner）、`gymnasium==0.29.1`、`torch==2.6.0` + `torchvision==0.21.0`；**game files** 经由 `alfworld-download` |

三个环境都用集群的 conda 创建；激活方式：

```bash
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate <env-name>
```

## 1. Trainer 环境 —— `fedagent-verl08`（verl 0.8，Python 3.12）

这是**作为库 import 的原生 verl 0.8** —— 没有 verl fork，也没有
打过 patch 的 verl 树。创建一个 Python 3.12 环境，并安装带 FSDP
inference 栈（vLLM + flash-attn）的 verl 0.8；FedAgent 本身不带 `setup.py` —— 它
就在 repo 里原地使用，把 repo root 放到 `PYTHONPATH` 上即可。

```bash
conda create -n fedagent-verl08 python=3.12 -y
conda activate fedagent-verl08

# Install verl 0.8 + the vLLM/SGLang inference stack (FSDP-only; no Megatron).
# verl ships an installer for the GPU stack:
bash /path/to/verl/scripts/install_vllm_sglang_mcore.sh   # USE_MEGATRON=0
pip install -e /path/to/verl                              # verl 0.8 as a library
```

- 这个环境**必须用 Python 3.12**（WebShop/ALFWorld 的服务环境用 3.10）。
- **flash-attn 是强制的。** verl 0.8 在训练期间无条件调用
  `flash_attn.bert_padding`（用 `attn_implementation=sdpa` *并不能*绕开它）。如果某个预编译 wheel
  与你的 glibc / CUDA 不兼容，请在装好 `torch` 之后用你的 toolchain
  从源码编译（例如 `flash_attn==2.7.4.post1`，
  `--no-build-isolation`）。
- **不要**在没有 `--no-deps` 的情况下 `pip install --force-reinstall`：它会级联出一个
  裸的 `torch` 依赖，可能拉进一个不匹配的 CUDA 构建，从而搞坏环境。

不需要单独的 FedAgent 安装步骤：driver 和 per-client 入口会自己把
repo root 加到 `PYTHONPATH`，并从激活的环境里 import `verl`
（`fedagent/fed/run_fed.py` 设置 `PYTHONPATH=<repo root>`，并通过 `import verl` 解析出
verl 的原生 config 目录）。

## 2. WebShop 服务环境 —— `verl-agent-webshop`（Python 3.10）

[`../envs/webshop/service/run_service.sh`](../envs/webshop/service/run_service.sh) 会做
`conda activate verl-agent-webshop`，并用 `uvicorn` 启动该服务。这个
环境装着 WebShop 的冲突栈（`gym 0.24` / `pyserini` / `torch 2.6` /
`numpy 1.26`）。

```bash
conda create -n verl-agent-webshop python=3.10 -y
conda activate verl-agent-webshop
pip install -r webshop_requirements.txt      # repo root; pins the WebShop stack
```

- **`PATH` 上必须有一个 JDK。** `pyserini` / `pyjnius` 在产品 catalog 上驱动一个
  Java/Lucene BM25 索引。装一个，例如 `conda install -c conda-forge
  openjdk=21`，或者用系统 JDK 并 export `JAVA_HOME`。
- WebShop engine 与 goal data **就地 vendored 在 in-tree** 的
  `fedagent/envs/webshop/engine/`（见 §4）；它不从 PyPI 拉任何东西，且
  `webshop_requirements.txt` 也不再需要 editable 的 verl-agent 安装。
- `server.py` 还会在启动时把 WebShop engine 注入到 `sys.path` 上，并
  预热一池 `WebAgentTextEnv` 实例（每个 `gym.make` 约 26 s，
  JVM + 索引启动），从而让 trainer 永不 import WebShop。

## 3. ALFWorld 服务环境 —— `verl-agent-alfworld`（Python 3.10）

[`../envs/alfworld/service/run_service.sh`](../envs/alfworld/service/run_service.sh) 会做
`conda activate verl-agent-alfworld`，export `ALFWORLD_DATA`，并用 `uvicorn` 启动
该服务。这个环境装着 ALFWorld 的栈（`alfworld 0.4.2` /
`textworld` / 一个 Fast-Downward PDDL planner / `torchvision`）。

```bash
conda create -n verl-agent-alfworld python=3.10 -y
conda activate verl-agent-alfworld
pip install -r alfworld_requirements.txt     # repo root; pins the ALFWorld stack

# One-time: download the PDDL + textworld game files (and detector) into the cache.
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download -f
```

- **Game files 是必需的。** `alfworld-download` 会填充 `ALFWORLD_DATA`（服务在启动时
  遍历的那些可解的 `game.tw-pddl` 文件）。`run_service.sh` 会
  export `ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"`；**这个变量必须被
  export**，因为内置的 `config_tw.yaml` 把 game/logic/detector 路径写成
  `$ALFWORLD_DATA/...`（在运行时展开）。把它设成你下载进去的那个目录。
- ALFWorld engine **就地 vendored 在 in-tree** 的 `fedagent/envs/alfworld/engine/`
  （见 §4）；`alfworld_requirements.txt` 也不再需要 editable 的 verl-agent
  安装。该服务把 `AlfredTWEnv` 接口构建一次，并池化
  单实例 textworld env；trainer 永不 import ALFWorld。

## 4. Vendored engine —— `fedagent/envs/<name>/engine/`

真正的 WebShop 与 ALFWorld engine（以及原始的 action parser / partition
代码）都**就地 vendored 在 in-tree**，每个都紧挨着它的服务：
- WebShop：[`../envs/webshop/engine/`](../envs/webshop/engine/) —— `web_agent_site` +
  随附的 catalog 数据。
- ALFWorld：[`../envs/alfworld/engine/`](../envs/alfworld/engine/) —— 处于一个保留的
  `agent_system/environments/` import 锚点之下的 `AlfredTWEnv` wrapper，外加
  `partition_strategy.py`。

它们**不从 PyPI 拉取**，也**不需要 editable 安装**。运行时每个
服务都把它的 engine 注入到 `sys.path` 上，并隔离地加载 action parser。
vendored 的 `agent_system/environments/__init__.py` 有意保持**空**的 —— 上游那个
import 了 verl-agent 的 `env_manager`（→ 旧的 verl 0.3.x）；这里把它中和掉，从而让
engine **不带任何 verl-agent 依赖**。这里没有任何东西需要超出上面 `-r *_requirements.txt`
之外的单独安装步骤。

## 5. 模型

Backbone 用 **HuggingFace model id** 指定（论文配置把
`actor_rollout_ref.model.path` 设成诸如 `Qwen/Qwen2.5-1.5B-Instruct` 这样的 id），所以
它们会在首次运行时**自动从 Hub 下载** —— 默认设置下无需手动步骤。

- **Cache / 磁盘。** 模型落在 `~/.cache/huggingface`（用 `HF_HOME` 覆盖）。
  为 Qwen2.5-1.5B 预留约 3 GB，最高到 Qwen2.5-7B 的约 15 GB。
- **受限 backbone。** `Llama-3.2-3B-Instruct` 是**受限（gated）**的：先在模型页面
  接受它的 license，然后在使用前完成认证（`huggingface-cli login`，或 export `HF_TOKEN`）。
  Qwen 系列 backbone 是不受限的。
- **离线 / 隔离网集群。** 在 login node 上预先取好
  （`huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`），然后在 compute node 上 export
  `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，并把配置指向一个本地
  snapshot —— 要么在 YAML 里覆盖 `actor_rollout_ref.model.path`，要么给 `run_fed.py`
  传 `--model-path /path/to/snapshot`。（去掉 snapshot 路径末尾任何
  结尾的 `/`；verl 的 `copy_to_local` 会拒绝它。）

## 6. CUDA 注意事项

在那些把 CUDA toolkit 作为 module（而非装在环境里）的集群上，请
export `CUDA_HOME`，以满足 vLLM 的 deep-GEMM 检查。本 repo 的 smoke
脚本用 cuda-12.1 module，并禁用 Hopper 的 deep-GEMM 路径（bf16 不需要它）：

```bash
export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0
export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1
```

把 `CUDA_HOME` 调成你集群的 CUDA module。这只在
`fedagent-verl08` trainer 环境里才相关。

## 下一步

激活 trainer 环境后，你可以立刻跑进程内的 smoke（不需要
服务）；WebShop / ALFWorld 的 run 会在匹配的环境里自动启动各自的 per-client 服务。
关于调用方式、GPU 与 baseline，见 [`./running.md`](./running.md)；关于 overlay
设计，见 [`../README.md`](../README.md)。
