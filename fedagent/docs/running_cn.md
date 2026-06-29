# 运行 FedAgent

如何用 [`fed/run_fed.py`](../fed/README.md) 运行联邦循环 —— 它是 **stock verl 0.8 上那层
thin overlay** 的脊梁。没有 trainer fork：每个 client 都是一个普通的子进程
（`python -m fedagent.main_ppo_fed`）；由 driver 编排各个 round。config-key 参考见
[configuration.md](./configuration.md)；env/conda 安装见 [installation.md](./installation.md)；
论文实验见 [reproducing.md](./reproducing.md)。

一次 run 就是交给 driver 的一份 **config YAML**。该 YAML 是硬件配方（GPU/FSDP-world-size、
显存、offload）和联邦协议的唯一真相来源；少量几个 CLI flag 覆盖最常被调换的那几个 key
（seed、ports、rounds、FedProx）。

```bash
python -m fedagent.fed.run_fed --config fedagent/config/<name>.yaml
```

> **单节点，串行 client。** 这个 runner 是**单节点**的 —— `n_gpus_per_node` 是单台机器上的
> FSDP world size；`run_fed.py` 中**没有多节点（`nnodes`）接线**。一轮内被选中的 client
> **一个接一个**训练（driver 循环 `for c in selected:`，并在进入下一个之前等待每个子进程）。
> 在规划并行或多节点之前请看 [Honest scope](#honest-scope)。

## Basics

在 GPU 节点上、从 repo 根目录、于 **`fedagent-verl08`** conda env 中运行 driver。
对 WebShop/ALFWorld，`run_fed.py` **自己启动每个 client 的 env service**（每个 client 一个
service，各自在自己的 service conda env 里），并在结束时拆除 —— 你**不需要**手动启动它们。
TinyGuess 在进程内运行（无 service）。模式（federated / centralized / local）与算法
（GRPO / PPO）由 **config 隐含决定**，而非由 flag —— 见 [Run-mode matrix](#run-mode-matrix)
和 [Algorithm: GRPO vs PPO](#algorithm-grpo-vs-ppo)。

位于 `fedagent/config/examples/` 下的手写 smoke config
（`examples/tinyguess_2cl_2rd.yaml`、`examples/webshop/`、`examples/alfworld/`）是用于接线检查的
小型 smoke；论文网格位于 `fedagent/config/paper/` 下。见 [Smoke tests](#smoke-tests) 和
[Worked examples](#worked-examples-paper-configs)。

## CLI flags → config keys

每个 flag 都覆盖 YAML 中对应的 key（YAML 本身又覆盖 `run_fed.py` 中的
[`DEFAULTS`](../fed/run_fed.py)）。只有这些 flag 存在 —— 其余一切都在 YAML 里设置，
或通过 `client_overrides`。

| Flag | 它覆盖的 config key | 默认值（在 `DEFAULTS` 中） | 用途 |
|---|---|---|---|
| `--config <yaml>` | — | — | 联邦 config（几乎总要传它） |
| `--model-path <dir>` | `model_path` | `""` → 自动发现 Qwen2.5-0.5B | round 1 的基础模型（离线：一份本地 HF snapshot） |
| `--output-dir <dir>` | `output_dir` | `/tmp/...tinyguess` | `round_*/`、日志、checkpoint、summary 落地处 |
| `--rounds <T>` | `total_rounds` | `2` | 缩短/拉长这次 run |
| `--clients <N>` | `total_clients` | `2` | 同时把 `clients_per_round` 也限到 ≤ N |
| `--n-gpus <k>` | `n_gpus_per_node` | `2` | FSDP world size（如 4-GPU 节点用 `4`，debug 用 `1`） |
| `--base-seed <s>` | `base_seed` | `42` | seed 复现（client 选择 + 每个 client 的 env seed） |
| `--port-base <p>` | `webshop_base_port` | `8080` | 在一个节点上跑两个 WebShop 作业而不撞端口 |
| `--fedprox-mu <mu>` | `fedprox_mu` | `0.0` | `>0` 启用 FedProx（否则 FedAvg） |
| `--local-client-id <k>` | `local_client_id` | `-1` | Local baseline：钉住 client k（无联邦） |

> `--port-base` **只**覆盖 `webshop_base_port`。对 ALFWorld，若你需要让并发的 run 互不冲突，
> 请在 YAML 里设 `alfworld_base_port`（以及 val 端口 `webshop_val_port` / `alfworld_val_port`）
> —— 它们没有 CLI flag。

## Run-mode matrix

模式由 config 选择，而非 flag。`run_fed.py` 这样推导它：若 `local_client_id ≥ 0` 则为
`local`；否则若 `total_clients ≤ 1` 则为 `centralized`；否则为 `federated`。

| 模式 | 如何选择（YAML / flag） | 发生了什么 | 启动的 service |
|---|---|---|---|
| **Federated**（默认） | `total_clients: N>1`，`local_client_id: -1` | 每轮采样 `clients_per_round` 个 client，**串行**训练每一个，然后 FedAvg → merge → 下一轮从 merged 模型开始 | 惰性、按轮：仅为该轮**被选中**的 client 各启一个（该轮的采样），训练后拆除；val service（若开了 eval）则常驻 |
| **Centralized** | `total_clients: 1`（且 `clients_per_round: 1`，`partition_strategy: ""`） | 在汇聚的（未分区）数据上训一个模型；单个 client 的 FedAvg 是恒等映射，所以这个循环就是 `total_rounds × epochs_per_round` 的持续训练 | 一个（client 0，完整 env） |
| **Local** | `local_client_id: k ≥ 0`（配 `total_clients: N`） | 论文的 "Local Agent Training"：钉住 client `k` 在 N-way 分区中的那片，每轮单独训它，无联邦 | 只有那一个被钉住的 client `k` |

**各自如何启动：**

```bash
# Federated（默认 —— 任何多 client config）
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/coverage.yaml

# Centralized（config 里烤死 total_clients=1）
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/centralized.yaml

# Local（config 里烤死 local_client_id…）
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/local.yaml
# …或者从 CLI 给一个已有的 federated config 钉住某个 client：
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/coverage.yaml \
  --clients 2 --local-client-id 0
```

在 Local 模式下，只启动被钉住那个 client 的 env service（`participating_client_ids`
返回 `[k]`），所以它比对应的 federated 那一支更便宜。

## Algorithm: GRPO vs PPO

由 config 中的 `adv_estimator` 设定（无 flag）：

- **GRPO**（默认，`adv_estimator: grpo`）—— group-relative advantage，**无 critic**。
  group size 即 rollout `n`（在 `client_overrides` 中按 config 设置，例如 smoke 用 `rollout.n=2`、
  论文配方用 `8`）。
- **PPO**（`adv_estimator: gae`）—— value model（critic）每轮**与 actor 一起被联邦**。
  Round-1 critic = 基础模型（在 backbone 上加一个随机的 value head）；此后聚合后的 critic
  经由 `critic.model.path` 携带前行。PPO config 在 `client_overrides` 里带上 critic 块（例如
  [`examples/webshop/scaled/ppo.yaml`](../config/examples/webshop/scaled/ppo.yaml)）。若任何被选中的
  client 没能产出 critic checkpoint，该轮会 abort —— 请在 overrides 中保留
  `critic.checkpoint.save_contents=[model]`。

```bash
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/ppo.yaml
```

## Hardware recipe

`n_gpus_per_node`（或 `--n-gpus`）是训练与聚合**共用**的 **FSDP world size**：FedAvg 步会启动
`torchrun --nproc_per_node=<world_size>` 以匹配保存的分片布局（`model_world_size_<ws>_rank_*.pt`），
所以你训练时用的值和聚合时用的值是同一个 key。论文配方是**单节点 4 GPU**。显存大小
（rollout length、batch、pool、offload）在 `client_overrides` 中按 config 设置。overlay 的脊梁里
没有单独的 tensor-parallel 旋钮 —— 每个 client 的子进程在这个 world size 下使用 verl 原生的
FSDP rollout。

| `n_gpus_per_node` | 典型用途 | 备注 |
|---|---|---|
| `1` | 单 GPU debug / 接线检查 | 用小 backbone（0.5B）+ 小 config；降低 `rollout.n`、batch、pool；预期会 offload（见下）。非论文规模。 |
| `2` | smoke 默认值（`DEFAULTS`） | 2-GPU 切片上的 TinyGuess / WebShop smoke |
| `4` | **论文配方** | Qwen2.5-1.5B @ 15 turns；GRPO 与 PPO 都在此验证过 |

### CPU offload 与 GPU 显存（经由 `client_overrides`）

脊梁把每一条 `client_overrides` 原样作为 Hydra override 转发给每个 client 的子进程，所以
FSDP offload 和 vLLM 显存比例都是**按 config** 调，而非靠 flag。对装下一次 run 至关重要的 key：

| Override key | 它做什么 | 何时设置 |
|---|---|---|
| `actor_rollout_ref.actor.fsdp_config.param_offload` | 把可训练的**参数**offload 到 CPU | 更大的 backbone / batch 装不下；以吞吐换容量 |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | 把**optimizer state** offload 到 CPU | 同上；PPO 设它为 `true` 以腾出 GPU 给常驻的 critic |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM KV-cache 占 GPU 显存的比例 | 当 actor（PPO 还有 critic）挤占 rollout 时调低它；PPO 用 `0.5`，GRPO 约 `0.6` |
| `critic.fsdp.param_offload` / `critic.fsdp.optimizer_offload` | PPO critic offload | **verl 0.8** 把 critic FSDP config 放在 `critic.fsdp.*`（而非 `critic.model.fsdp_config`） |

示例 —— 开启 actor offload 并调低 KV 比例，以让 run 更紧凑：

```yaml
client_overrides:
  - actor_rollout_ref.actor.fsdp_config.param_offload=true
  - actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

或者作为命令行上的一次性设置（每个 override 是 flag 之后的一个位置参数）：

```bash
python -m fedagent.fed.run_fed --config <...> --n-gpus 1 \
  client_overrides='[actor_rollout_ref.rollout.n=2,actor_rollout_ref.rollout.gpu_memory_utilization=0.4]'
```

## FedProx

```bash
python -m fedagent.fed.run_fed --config <...> --fedprox-mu 0.1
```

`fedprox_mu > 0` 会在 client 子进程的环境里设 `FEDPROX_MU`；
[`sitecustomize.py`](../../sitecustomize.py)（repo 根目录，在 client + Ray worker 的
`PYTHONPATH` 上）在解释器启动时读取它，并在 FSDP optimizer step 处加上 proximal 项。
`mu = 0` → 纯 FedAvg。它经由 `sitecustomize` 注入，**而非** Ray `runtime_env` hook
（那个 hook 会破坏 verl 的 per-worker `CUDA_VISIBLE_DEVICES`）。Eval pass 会剥掉
`FEDPROX_MU`，所以 validation 永远不会启用该项。一对现成的配置是
[`examples/webshop/scaled/envhet_fedprox.yaml`](../config/examples/webshop/scaled/envhet_fedprox.yaml)
（FedProx，`mu=0.1`）对它的 FedAvg 双胞胎。

## Seeds

`base_seed`（或 `--base-seed`）穿过两个地方，两者在 resume 时都确定：

- **Client selection** —— `select_clients` 用 `base_seed + round − 1` 给它的 RNG 播种，
  所以每轮的采样可复现。
- **Per-client env instance** —— 每个 client 子进程拿到
  `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id`，所以一个 client 每轮都从它**固定**的
  那片 shard 里重新抽 goal（在 `T` 轮内覆盖该 shard），同时与其他 client 保持区分。

三 seed 复现就是把同一个 config 跑三次，配 `--base-seed 42 / 21 / 13`（用各自不同的
`--output-dir`，对并发的 WebShop run 还要用 `--port-base`）。

## Validation / eval

除非设置了 `val_env_spec`，否则 eval 是**关闭**的（向后兼容默认值 `""`）。设置后，driver 启动
**一个共享、未扰动**的 val service（`partition_strategy` 强制为空 / uniform —— 无 client 偏斜），
并对聚合后的**全局**模型打分：

| Key | 效果 | 默认值 |
|---|---|---|
| `val_env_spec` | 未扰动的 val env-spec；`""` → 不 eval | `""` |
| `test_freq` | verl 的**作业内**步频（每个 client 的 circle 标记），**并非**全局 eval 的门控 —— 聚合后的全局模型**每轮**都打分 | `5` |
| `val_before_train` | 在 round 1 之前也 eval 一次**基础**模型（round-0 那个点） | `true` |
| `val_temperature` | val 采样温度（`val_kwargs.temperature`） | `0.4` |

round → success/reward 曲线写入 `federated_summary.json`（`val_curve`）。一次失败的 eval 会记一条
warning 并继续 —— 它绝不会 abort 整个 run。

## 一个节点上的并发 run

同一节点上的两个作业绝不能在 env-service 端口上撞车。给每个作业一个不同的 `--port-base`
（WebShop client `c` → `port_base + c`）和一个不同的 `--output-dir`：

```bash
python -m fedagent.fed.run_fed --config <...> --base-seed 42 --port-base 8080 --output-dir /tmp/run_s42 &
python -m fedagent.fed.run_fed --config <...> --base-seed 21 --port-base 8120 --output-dir /tmp/run_s21 &
```

对 ALFWorld，请在每份 YAML 里设 `alfworld_base_port`（无 CLI flag）。记住两个作业共享该节点的
GPU —— 在 4-GPU 配方下，两个完整的 run 不会都装得下；并发是给小型/offload 的 run 或不同的
GPU 切片用的。

## Honest scope

- **一轮内 client 串行运行。** driver 循环 `for c in selected:` 并阻塞在每个 client 的子进程上
  （`stream(...)` 经由 `proc.wait()` 等待），client 之间有一段 `wait_between_clients` 秒的暂停以让
  Ray/GPU 完全释放。**没有并行 client 执行** —— 一轮的墙钟是它各 client 之和。
- **仅单节点。** `n_gpus_per_node` 是**一台**机器上的 FSDP world size。`run_fed.py` 里没有
  `nnodes` 设置、也没有多节点启动；聚合器在本地跑 `torchrun --nproc_per_node=<ws>`。多节点
  **未实现**。
- **无 legacy 启动器。** 旧的 `reproduce.sh` / `run_federated.py` / `start_federated.sh` 路径
  （及其 `parallel_workers` 旋钮）在这里**不适用**；唯一的入口是 `python -m fedagent.fed.run_fed`。

## 在 SLURM 上运行（srun）

driver 是一个普通的 Python 进程 —— 你**不**用 `sbatch` 一个特殊脚本。拿到一个交互式 GPU 分配
（或挂到一个已有的作业上），用 `srun --overlap` 在节点上跑 driver。一切都在 **`fedagent-verl08`**
env 内运行；WebShop/ALFWorld **service 由 driver 自己**在它们各自的 service env
（`verl-agent-webshop` / `verl-agent-alfworld`）里启动，所以这些 env 必须存在于节点上，但你**不需要**
手动 activate 它们。三个 env 见 [installation.md](./installation.md)。

真实的范式（对照 [`fedagent/scripts/run_smoke.sh`](../scripts/run_smoke.sh) 和
[EXPERIMENTS.md](../EXPERIMENTS.md)）—— 挂到一个正在运行的作业 `<JID>`：

```bash
# 1) 拿到 / 确认一个 GPU 分配
#    （例如 salloc --gres=gpu:4 ... ；或复用一个已有的 job id）
JID=<your_slurm_job_id>

# 2) 在 GPU 节点上、于 trainer env 里跑 driver
srun --jobid="$JID" --overlap bash -lc '
  source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
  conda activate fedagent-verl08

  cd /gpfs/projects/b1222/userdata/canyu/kangyu/fedagent
  export PYTHONPATH="$PWD:$PYTHONPATH"                       # 让 `import fedagent` 能解析（driver + Ray worker）
  export VERL_CFG="$(python -c "import verl,os;print(os.path.join(os.path.dirname(verl.__file__),\"trainer\",\"config\"))")"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1
  export VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1   # deep_gemm 会断言需要 CUDA toolkit
  export CUDA_HOME=/hpc/software/cuda/cuda-12.1.0            # 指向 CUDA 模块

  python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/scaled/coverage.yaml --n-gpus 4
'
```

注意：

- `--overlap` 让这个 `srun` 共享已有分配的资源（这样你就能在一个主 step 处于空闲/占着节点的作业里
  跑 driver）。
- `CUDA_HOME`、那些 offline flag、`PYTHONPATH` 和 `VERL_CFG` 都是必需的（smoke 脚本设的正是这些）；
  `VERL_CFG` 把 Hydra 指向 verl 原生的 `trainer/config`。
- 联邦 checkpoint 默认落在**计算节点的** `/tmp` 上 —— 用另一条 `srun --jobid=<JID> --overlap ls ...`
  去查看它们。
- [`fedagent/scripts/`](../scripts) 里的 wrapper 脚本（`run_smoke.sh`、
  `run_tinyguess_fed_smoke.sh`、`run_webshop_fed_smoke.sh CFG …`）把上面这一切都烤了进去，
  是在 `srun` 下启动的最快方式。

## Smoke tests

位于 `fedagent/config/examples/` 下的手写 smoke config 很小（例如 2 client × 几轮），用于快速接线检查：

```bash
# 进程内、无 service —— 对联邦循环最快的端到端检查
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml

# WebShop smoke（driver 启动 2 个 service），缩短到 2 轮
python -m fedagent.fed.run_fed --config fedagent/config/examples/webshop/homog_long.yaml --rounds 2

# Wrapper（设好 env + 适配 srun），把额外的 flag 转发给 run_fed
bash fedagent/scripts/run_webshop_fed_smoke.sh fedagent/config/examples/webshop/scaled/homog.yaml \
  --base-seed 43 --output-dir /tmp/run_s43 --port-base 8090
```

## Worked examples (paper configs)

```bash
# WebShop 主实验，GRPO，Qwen2.5-1.5B，4 GPU
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml

# 同一份 config，第二个 seed + 它自己的 output dir + client-service 端口。
# 注意：--port-base 只移动 webshop_base_port（每个 client 的 service）。这份论文 config
# 开了 eval，它用一个固定的 webshop_val_port（无 CLI flag）—— 同一份 config 的两次并发 run
# 会共享那一个 val 端口。要去冲突，复制该 YAML 并也改掉 webshop_val_port，或者给第二个 run
# 关掉 eval（val_env_spec: ""）。
python -m fedagent.fed.run_fed --config <...same...> \
  --base-seed 21 --output-dir /tmp/run_s21 --port-base 8120

# 环境级异质性（Catalog Split）
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-0.7_keep-0.7.yaml

# Centralized baseline（total_clients=1）
python -m fedagent.fed.run_fed \
  --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
```

## Resume

联邦在 **round 层面**掌管 resume：用同一个 `--output-dir` 重新运行会从上一个已完成 round 的
聚合模型继续。每个 client 的 per-run 自动 resume 被禁用（`trainer.resume_mode=disable`），
这样一个中途崩溃的 in-flight round 永远不会被 FedAvg 进半成品权重。被消耗掉的 FSDP 分片在每次
merge 后删除，以把峰值磁盘维持在约一轮的量（用 `cleanup_checkpoints` 切换；否则一次 8 轮的 run
曾涨到 367 GB）。

## Outputs

在 `output_dir/` 下：

- `round_*/client_*/training.log` + `round_*/client_*/json_logs/metrics.json`
  （FedAgent plot 格式的每个 client 的 reward 曲线）
- `round_*/aggregated/hf` —— 该轮的全局模型（HF 格式；下一轮的起点）
- `<env>_service_client*.log`、`<env>_val_service.log` —— 每个 service 的日志
- `federated_summary.json` —— round 历史、mode/algorithm、最终模型，以及（若开了 eval）
  未扰动的 `val_curve`

见 [architecture.md](./architecture.md#outputs) 和 [`../fed/README.md`](../fed/README.md)。

## See also

- [installation.md](./installation.md) —— 三个 conda env（`fedagent-verl08` trainer +
  `verl-agent-webshop` / `verl-agent-alfworld` service）。
- [configuration.md](./configuration.md) —— 完整的 config-key 参考和文件名解码器。
- [reproducing.md](./reproducing.md) —— 论文网格与 seed。
- [heterogeneity.md](./heterogeneity.md) —— task 级与环境级的分区策略。
- [`../fed/README.md`](../fed/README.md) —— driver 内部机制（round 循环、FedAvg、merge）。
