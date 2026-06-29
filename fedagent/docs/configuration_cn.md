# Configuration（配置）

FedAgent 是**叠加在未经修改的 verl 0.8 之上的一层薄 overlay** —— 没有 trainer fork。每一次
run 都由配置驱动：联邦 runner 读取的一份扁平 YAML、一份组合 verl 原生 `ppo_trainer` 的 Hydra
base、一份 agent-loop registry，以及 per-episode 的 env spec。本页是**配置文件解码器**与
**联邦 runner key 参考**：`run_fed.py` 的 `DEFAULTS` dict 里的每一个 key、env-spec 行 schema、
`paper/` 文件名语法，以及当你对照代码读文件名时会踩到的命名陷阱。

包级总览见 [`../README.md`](../README.md)，文件夹地图见
[`../config/README.md`](../config/README.md)，联邦 driver 见
[`../fed/README.md`](../fed/README.md)。每个异质性 arm *做什么* 见
[`./heterogeneity.md`](./heterogeneity.md)；如何启动循环见
[`./running.md`](./running.md)；逐图复现矩阵见
[`./reproducing.md`](./reproducing.md)。

> **没有 legacy schema。** 这是 verl-0.8 的 runner。原始 FedAgent 那套嵌套的
> `federated:` / `verl:` / `data_preprocess:` block 已经没有了 —— 那套 schema 只存在于
> 归档的 `legacy/docs/` 里，**这里没有任何东西**会读它。一份 FedAgent 配置是一个
> **扁平**的 key/value 文件，其 key 即 `run_fed.py` 的 `DEFAULTS`；per-client 的 verl 旋钮
> 通过 `client_overrides` 传入（见 [§ client_overrides](#client_overrides-and-adv_estimator)）。

---

## 四种配置类型

| 类型 | 文件 | 被谁消费 | 角色 |
|---|---|---|---|
| **Hydra base config** | [`config/fedagent_ppo.yaml`](../config/fedagent_ppo.yaml) | `fedagent.main_ppo_fed`（`@hydra.main(config_name="fedagent_ppo")`） | **单个 client** 的训练配置：经由 `hydra.searchpath` 组合 verl 原生 `ppo_trainer`，只覆盖 FedAgent 需要的那些叶子。 |
| **Agent registry** | [`config/agent.yaml`](../config/agent.yaml) | verl 的 `AgentLoopManager`（经由 `actor_rollout_ref.rollout.agent.agent_loop_config_path`） | 把数据集行上的每个 `agent_name` 映射到它的 `AgentLoopBase` 类。 |
| **Env spec** | [`config/envs/*.yaml`](../config/envs/) | `fedagent.data.agentic_dataset.AgenticDataset`（经由 `data.train_files` / `data.val_files`） | 声明 env 池：每个 episode 一个数据集行（`n_envs` 行，各自不同 seed）。 |
| **联邦 runner 配置** | `config/fed_*.yaml`、`config/paper/**/*.yaml` | `python -m fedagent.fed.run_fed --config <file>` | **最外**层：顶层联邦旋钮；key == `run_fed.py` 的 `DEFAULTS` dict。驱动轮次循环、FedAvg、env 服务与验证。 |

Runner 在最外层：`run_fed` 读取扁平配置，启动 per-client 的 env 服务，然后 **每 client 每轮一次**
shell 调出 `main_ppo_fed`（它加载 `fedagent_ppo.yaml`），把 `data.train_files=<env_spec>`、模型路径、
以及 `client_overrides` 作为 Hydra CLI override 注入进去。

### `fedagent_ppo.yaml` —— Hydra base

组合 verl 的**原生 `ppo_trainer`**，经由 `hydra.searchpath` -> verl 的
`trainer/config` 目录（导出为 `$VERL_CFG`；`run_fed` 回退到
`verl.__file__/trainer/config`）：

```yaml
defaults:
  - ppo_trainer
  - _self_
hydra:
  searchpath:
    - file://${oc.env:VERL_CFG}
```

它只覆盖 FedAgent 叶子：**GRPO**（`algorithm.adv_estimator: grpo`，
`use_kl_in_reward: false`）、**group size**（base 里 `rollout.n: 4`；每个 arm 经由
`client_overrides` 重新钉它 —— paper=`8`，smoke=`2`）、**async multi-turn rollout**
（`rollout.name: vllm`，`mode: async`，`multi_turn.enable: true`，
`agent.default_agent_loop: gym_text`）、**每个 arm 上的论文 actor 目标**
（`use_kl_loss: true`，`kl_loss_coef: 0.01`，`kl_loss_type: low_var_kl`，
`entropy_coeff: 0.001` —— verl 0.8 的默认值与此不同）、**自定义数据集**
（`data.custom_cls.name: AgenticDataset`）、`reward_model.enable: false`，以及
`trainer.logger: [console]`。机器/run 相关的叶子（`model.path`、
`data.{train,val}_files`、`custom_cls.path`、`agent_loop_config_path`、
`default_local_dir`）以及 struct-additive 的
`+actor_rollout_ref.model.override_config.attn_implementation` 在 CLI 上提供。

### `agent.yaml`

一份把 `agent_name` -> `AgentLoopBase` `_target_` 的列表；它有两个条目：`gym_text` ->
`fedagent.agent_loops.gym_text_agent_loop.GymTextAgentLoop`（concat 风格的 multi-turn
循环）以及 `gym_text_windowed` ->
`fedagent.agent_loops.windowed_agent_loop.WindowedGymTextAgentLoop`（per-turn 的 windowed
变体）。`agent_name` 随每个数据集行而来（见下文），于是 verl 会 per-rollout
实例化正确的循环。

---

## Env specs —— 行 schema 以及 `AgenticDataset` 如何消费它们

一份 env spec（`config/envs/*.yaml`）在顶层 `envs:` 列表下声明**一个或多个 env 池**。`run_fed`
把 `data.train_files` 和 `data.val_files` **两者**都指向同一份 spec（`cfg.env_spec`）；验证经由
`cfg.val_env_spec` 用一份单独的 spec。

### 行 schema

```yaml
envs:
  - name: WebShop          # env id -> 数据集行的 env_name / data_source (.lower())
    n_envs: 8              # 该池发出的数据集行数（每 episode 一行）
    max_turns: 15         # 交给 agent loop 的 per-episode turn 上限
    agent_name: gym_text  # 可选；要用的 AgentLoop 类（默认：gym_text）
    config:               # 可选 per-env kwargs，原样传给 env/agent loop
      timeout: 180.0
```

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `name` | str | `TinyGuess` | Env id。成为该行的 `env_name` 与 `data_source`（小写）；在 agent loop 中选取 env 类。 |
| `n_envs` | int | `64` | 该池发出的数据集行数 —— **每 episode 一行**（各自不同 seed）。 |
| `max_turns` | int | `6` | Per-episode turn 上限，转发给 agent loop（`WebShop=15`，`ALFWorld=50`）。 |
| `agent_name` | str | `gym_text` | Agent-loop key（必须存在于 `agent.yaml`）。 |
| `config` | map | `{}` | Per-env kwargs（如 `timeout`，或 TinyGuess 的 `{lo, hi}`）传给 env。WebShop/ALFWorld **不**在这里钉服务 URL —— 它来自 `run_fed` 为每个 client 设置的 `WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL` env var。 |

### `AgenticDataset` 如何把一份 spec 变成行

[`data/agentic_dataset.py`](../data/agentic_dataset.py) 是 verl 的 `custom_cls` 数据集。它
加载 spec（`data.train_files[0]`），读取其 `envs:` 列表，对每个池发出
`n_envs` 行 —— 每行都是一个**带不同 seed 的不同 env 实例**：

```python
seed = base_seed * 100_000 + spec_index * 1_000 + episode_index
# base_seed = int(os.environ["FEDAGENT_BASE_SEED"])  (0 if unset)
```

`run_fed` 为每个 client-round 设置 `FEDAGENT_BASE_SEED = base_seed + round*100 + client`，于是
每个 client 每轮都从它固定的 shard 重新抽取目标（在 `T` 轮内覆盖该 shard），同时保持可复现。每行
携带 `env_name`、`seed`、`config`、`max_turns`、`agent_name`、`data_source`、一个占位
`raw_prompt`，以及单个 dummy tensor `ds_dummy`（该行**不**携带
`input_ids`/`attention_mask`/`position_ids` —— 这些由 agent loop 生成；dummy tensor 仅为 batch
sizing 而存在，因为原生 verl 的 `_get_gen_batch` 在把 agent-loop 输出 union 回 batch 之前不会
pop tensor key）。GRPO 分组**不**在这里做：verl 的 `rollout.n` 把每行在下游重复
`n` 次，为每个 env 实例形成一个 GRPO group。

### 映射到 `data.train_files`

`run_fed` 把 spec 路径作为 Hydra override 原样传过去
（`run_client` / `eval_global`）：

```
data.train_files=<env_spec>   data.val_files=<env_spec>   data.custom_cls.path=<custom_cls_path>
```

所以 `data.train_files` 是 **env-spec YAML 路径**，不是 parquet 文件 —— verl-0.8
overlay 用即时的 env 枚举取代了 parquet 预处理。

### 随附的 specs

| Spec | `n_envs` | `max_turns` | 用于 |
|---|---|---|---|
| [`tiny_guess.yaml`](../config/envs/tiny_guess.yaml) | 64 | 6 | `TinyGuess`，in-process 接线验证（runner 默认 `env_kind=tinyguess`）。 |
| [`webshop.yaml`](../config/envs/webshop.yaml) | 16 | 6 | WebShop smoke（小预算）。 |
| [`webshop_15.yaml`](../config/envs/webshop_15.yaml) | 8 | 15 | WebShop **GRPO** 训练（`n_envs=8` == 原始 GRPO train_data_size；配合 `train_batch_size=8` 即每 epoch 1 个 optimizer step）。 |
| [`webshop_15_ppo.yaml`](../config/envs/webshop_15_ppo.yaml) | 64 | 15 | WebShop **PPO** 训练（`n_envs=64` == 原始 PPO train_data_size，配 `train_batch_size=64`）。 |
| [`webshop_15_val.yaml`](../config/envs/webshop_15_val.yaml) | 500 | 15 | WebShop 验证：在完整 catalog 上的 held-out `goals[0:500]`（整个 held-out 集；eval 不设 `FEDAGENT_BASE_SEED`，所以每轮都给同样的 500 个目标打分）。 |
| [`alfworld.yaml`](../config/envs/alfworld.yaml) | 8 | 50 | ALFWorld 训练（game shard；`max_turns=50` == 原始 `max_steps`）。 |
| [`alfworld_val.yaml`](../config/envs/alfworld_val.yaml) | 140 | 50 | ALFWorld 验证：`valid_seen`（140，in-distribution）。要拿到完整 274 trial + 按 task-type 拆分，在最终模型上跑 `tools/verl08_migration/eval_alfworld_by_tasktype.py`。 |

---

## 文件名解码器 —— `paper/` 树

`config/paper/` 装着完整 paper-scale 的 run，组织成一棵**镜像原始 FedAgent** `config/` 的
家族树（176 个配置）。每个叶子都是一份扁平 runner 配置，其名字
编码了它的 protocol：

```
fed_<env>_<algo>_total-<N>_cl-per-rd-<M>_rd-<T>_ep-per-cl-<E>_min-goals-per-cl-<G>_p-<strategy>_<knobs>.yaml
```

| Token | Runner key | 含义 |
|---|---|---|
| `<env>` | `env_kind` | `webshop` 或 `alfworld`。 |
| `<algo>` | `adv_estimator` | `grpo`（无 critic）或 `ppo`（== `gae`，联邦化 critic）。 |
| `total-<N>` | `total_clients` | Client 总数 N（`100`；centralized 时为 `1`）。 |
| `cl-per-rd-<M>` | `clients_per_round` | 每轮选取的 client 数 M（`2`；local/centralized 时为 `1`）。 |
| `rd-<T>` | `total_rounds` | 通信轮数 T（`70`）。 |
| `ep-per-cl-<E>` | `epochs_per_round` | 每 client 每轮的 local epoch 数 E（`3`）。 |
| `min-goals-per-cl-<G>` | `min_goals_per_client` | 每个 client 的 shard 的最少目标数（`100`）。 |
| `p-<strategy>` | `partition_strategy`（见 caveat） | `uniform`（== IID，runner key `""`）或某个异质性策略。 |
| `<knobs>` | strategy knobs | 策略参数明示，如 `div-0.7_keep-0.7`、`omega-0.99`、`std-256`、`success_std-1`、`N-4`。 |

矩阵上恒定的那格是 `total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100`
（N=100，M=2，T=70，E=3，G=100），给出 `E*T = 210` 个 local epoch。Baseline 与
decentralized ablation 恰好改变这些 token 之一。

### 文件名 `p-...` token → runner key + knobs

> **`p-<strategy>` token 是唯一一个不是 runner 值逐字拷贝的 token。** 文件名用的是
> *论文的*拼写；YAML 的 `partition_strategy` 用的是代码的 dispatch key。它们在 WebShop
> env-variant arm 上分歧 —— 核对 YAML，别核对文件名：

| 文件名 `p-...` | `partition_strategy`（YAML） | Knob key（YAML） | 轴 / 论文名 |
|---|---|---|---|
| `p-uniform` | `""` | （无） | IID（homogeneous） |
| `p-preference_omega-<ω>` | `preference` | `omega` | task —— Preference |
| `p-coverage_std-<s>` | `coverage` | `size_std` | task —— Coverage |
| `p-hardness_success_std-<s>` | `hardness` | `success_std`、`trajectories_file` | task —— Hardness |
| `p-catalog_split_div-<d>_keep-<r>` | `catalog_split` | `env_div`、`keep_ratio` | env —— Catalog Split |
| `p-field_subset_index_N-<n>` | **`bm25_field_subset`** | `variant_n` | env —— Field-Subset Index |
| `p-bm25_reweighting_N-<n>` | **`bm25_reweight`** | `variant_n` | env —— BM25 Reweighting |
| `p-lookalike_injection_N-<n>` | **`lookalike`** | `variant_n` | env —— Lookalike Injection |
| `p-rank_wrapper_N-<n>` | `rank_wrapper` | `variant_n` | env —— Rank Wrapper |

ALFWorld 的 env-het 类比物（不在上面的文件名语法里；用于手写的
`examples/alfworld/paper.yaml`）是 `partition_strategy: env_disjoint` —— per-client 不相交的 game
shard。它的 WebShop task-only 同胞是 `task_disjoint`（不相交目标，完整 catalog）。

### Sweep 端点

每条 het 轴在一个近-uniform 端点与一个极端端点之间被 sweep，可在
文件名里看到：

| 轴 | 近-uniform | 极端 | 备注 |
|---|---|---|---|
| Preference | `omega-0.01` | `omega-0.99` | 更大 `omega` => **更多**异质性 |
| Coverage | `std-256` | `std-1` | 高 `size_std`（Beta 集中度）=> 近-uniform；低 => 偏斜 |
| Hardness | `success_std-256` | `success_std-1` | 同一 Beta-dispersion 约定 |
| Catalog Split | `div-0.0` | `div-1.0` | 在固定 `keep-0.7` 下 sweep |
| env-variants | `N-2` | `N-8` | variant-pool 大小 `variant_n` |

### 目录家族

| 家族 | 布局 | 变化的是什么 |
|---|---|---|
| `uniform/<Model>/<setting>/<algo>/` | per-backbone 的 IID + baseline | **setting**（见下）。 |
| `env_heterogeneity/<strategy>[_ppo]/` | 仅 webshop | env 级扰动策略（`_ppo` => `adv_estimator: gae`）。 |
| `task_heterogeneity/<algo>/<env>/` | grpo+ppo × webshop+alfworld | task 级 partition（preference / coverage / hardness）。 |
| `decentralized/<change>/<algo>/` | webshop+alfworld | 单个 protocol 旋钮（`selected_cl_change` => M∈{1,4}；`ep_per_round_change` => (E,T)∈{(1,210),(5,42)}；`samples_change` => G∈{500,1000}）。 |

**Backbones**（各一个 `uniform/<Model>/` 子目录）：`Qwen2.5-1.5B-Instruct`、
`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Llama-3.2-3B-Instruct`。
`env_heterogeneity`、`task_heterogeneity`、`decentralized` 三棵树只为
1.5B backbone 生成。`env_heterogeneity` 是 **webshop-only**（catalog/BM25/lookalike/rank
arm 扰动 WebShop catalog + 搜索引擎，没有 ALFWorld 类比物）。

### Uniform settings

| Setting | 差异在 | Runner key |
|---|---|---|
| `main` | IID 锚点（seed 42） | `total_clients: 100`、`clients_per_round: 2`、`base_seed: 42`。 |
| `main_seed1` / `main_seed2` | 3-seed 复制 | `base_seed: 21` / `84`（原始版本改变 shuffle seed 42/21/84）。 |
| `centralized` | 在 pool 数据上的单模型 | `total_clients: 1`、`clients_per_round: 1`（单 client 的 FedAvg == identity）。 |
| `local_client1` / `2` / `3` | "Local Agent Training" | `local_client_id: 21` / `42` / `84`（钉住 100 个里的一个 client；`clients_per_round: 1`；无联邦）。 |

所以 **3-seed 复制** = `base_seed` 42 / 21 / 84，横跨 `main`、`main_seed1`、
`main_seed2`；**Local** baseline 钉住 client `21`、`42`、`84`。与原始文件名的一处有意分歧：
`centralized` / `local_client*` 编码的是 `rd-70_ep-3`（而非原始的 `rd-1_ep-210`），因为
verl-0.8 runner 从*轮次*中抽取目标多样性，所以那 210 个 local epoch 被铺在 70 轮上。用
`tools/verl08_migration/gen_paper_configs.py` 重新生成整棵树。

---

## 联邦 runner key 参考

下面每个 key 都是 `run_fed.py` 的 `DEFAULTS` dict 里的一个条目；配置里省略的任何东西
都回退到默认值。CLI flag `--model-path --output-dir --rounds
--clients --n-gpus --base-seed --port-base --fedprox-mu --local-client-id` 覆盖
YAML。包相对路径（`env_spec`、`val_env_spec`、`custom_cls_path`、
`agent_config_path`、`webshop_run_service`、`alfworld_run_service`）相对
`fedagent/` 解析。

### 核心循环

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `model_path` | str | `""` | 第 1 轮的基础 HF model 目录；`""` => 自动发现一个本地 Qwen2.5-0.5B-Instruct 快照。 |
| `output_dir` | path | `/tmp/xbb9020_fedagent_fed_tinyguess` | Run 根目录：per-round 的 client/聚合 checkpoint、日志、`federated_summary.json`。 |
| `env_spec` | path | `config/envs/tiny_guess.yaml` | Env spec -> 每个 client 的 `data.{train,val}_files`。 |
| `custom_cls_path` | path | `data/agentic_dataset.py` | `AgenticDataset` 的路径（-> `data.custom_cls.path`）。 |
| `agent_config_path` | path | `config/agent.yaml` | Agent-loop registry（-> `rollout.agent.agent_loop_config_path`）。 |
| `total_clients` | int | `2` | Client 总数 N。 |
| `clients_per_round` | int | `2` | 每轮选取的 client 数 M（当 `M < N` 时确定性 seeded 采样；seed = `base_seed + round - 1`）。 |
| `total_rounds` | int | `2` | 通信轮数 T。 |
| `epochs_per_round` | int | `1` | 每 client 每轮的 local epoch E（-> `trainer.total_epochs`）。 |
| `base_seed` | int | `42` | 主 seed；per-(round,client) 的 env seed = `base_seed + round*100 + client`（也驱动 client 选择）。 |
| `n_gpus_per_node` | int | `2` | 每个 client run 的 FSDP world size（== 聚合器 `nproc`）。 |
| `total_training_steps` | int | `1` | Per-client-round 的 step 上限（smoke）；`<=0` => 发出 `null` 让 verl 跑满 E 个 epoch（`len(dataloader)*total_epochs`）。显式发出，以免某个过期 base 值泄漏进 paper run。 |
| `save_freq` | int | `1` | verl `trainer.save_freq`（paper 配置用一个巨大值，如 `100000`，以只保存该轮最后一步）。 |
| `weights` | str | `""` | 传给聚合器的 FedAvg 权重（如按 client 数据量）；`""` => uniform 平均。 |
| `wait_between_clients` | int (s) | `5` | 顺序 client run 之间的秒数（让 Ray/GPU 释放）。 |
| `client_overrides` | list | `[]` | 应用到每个 client 的额外 `key=value` Hydra override（eval 也复用）。见 [§ 下文](#client_overrides-and-adv_estimator)。 |
| `cleanup_checkpoints` | bool | `True` | 每次 merge 之后删掉已消费的 FSDP 分片（保留 HF + 日志）；磁盘卫生。 |
| `adv_estimator` | str | `grpo` | `grpo`（无 critic）或 `gae`（PPO：FedAvg actor **和** critic）。 |

### Env 服务

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `env_kind` | str | `tinyguess` | `tinyguess`（in-process）、`webshop` 或 `alfworld`（远程服务）。 |
| `webshop_run_service` | path | `envs/webshop/service/run_service.sh` | WebShop 服务的 launcher。 |
| `webshop_base_port` | int | `8080` | Client `c` 的服务 -> `webshop_base_port + c`。 |
| `webshop_pool_size` | int | `8` | 每个 WebShop 服务的 env 池（必须 `>= gen_batch`）。 |
| `search_return_n` | int | `200` | `WEBSHOP_SEARCH_RETURN_N`：BM25 top-K。Env-het arm 用 `200`（引擎默认 `50` 在 filtering 下会丢掉 target）；non-het baseline 保持 `50`。 |
| `alfworld_run_service` | path | `envs/alfworld/service/run_service.sh` | ALFWorld 服务的 launcher。 |
| `alfworld_base_port` | int | `8200` | Client `c` 的服务 -> `alfworld_base_port + c`。 |
| `alfworld_pool_size` | int | `4` | 每个 ALFWorld 服务的 TextWorld env 池（必须 `>= gen_batch`）。 |
| `alfworld_train_eval` | str | `train` | ALFWorld game split：`train` / `eval_in_distribution` / `eval_out_of_distribution`。 |
| `alfworld_task_types` | str | `""` | `""` => 全部 6 种类型；否则逗号分隔的 ID（1=Pick..6=Pick2）用于 eval 拆分。 |
| `service_health_timeout` | int (s) | `900` | 等待每个服务 `/health` 的秒数（池预热要几分钟）。 |

### 异质性

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `partition_strategy` | str | `""` | `""`（IID）\| `catalog_split`/`task_disjoint`（WebShop env/task）\| `env_disjoint`（ALFWorld env）\| `preference`/`coverage`/`hardness`（task）\| `bm25_field_subset`/`bm25_reweight`/`lookalike`/`rank_wrapper`（WebShop env variant）。 |
| `env_div` | float | `0.7` | catalog-split 异质性强度。 |
| `keep_ratio` | float | `0.7` | catalog-split distractor 密度。 |
| `omega` | float | `0.5` | **preference**（task-het）Dirichlet spread ω —— 更大 ω = 更偏斜。 |
| `size_std` | float | `1.0` | **coverage**（task-het）Beta dispersion ξ。 |
| `success_std` | float | `1.0` | **hardness**（task-het）Beta dispersion ξ′。 |
| `variant_n` | int | `0` | env-variant arm（bm25/lookalike/rank）：池中的 variant 数（`0` => fn 默认 2/4）。文件名 token `N-<n>`。 |
| `trajectories_file` | path | `""` | hardness：**必需**的 `task_id`->success-labels 文件（用 `tools/verl08_migration/gen_hardness_trajectories.py` 生成）。 |
| `min_goals_per_client` | int | `100` | 每个 client 的 shard 的最少目标数。文件名 token `min-goals-per-cl-<G>`。 |

完整分类法以及每个旋钮如何映射到一个 arm，见
[`./heterogeneity.md`](./heterogeneity.md)。

### Baselines

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `local_client_id` | int | `-1` | `>=0` => **Local** baseline：只训练 `total_clients` 里的这个 client，每轮如此，无联邦。 |

**模式选择**（全部经由同一套 schema）：**Federated** = 默认（`total_clients=N>1`，
`local_client_id<0`）；**Centralized** = `total_clients=1`（单 client 的 per-round FedAvg
是 identity，所以循环是 `T*E` 个 epoch 的 centralized 训练）；**Local** =
`local_client_id=k>=0`；**FedProx** = `fedprox_mu>0`；**PPO** = `adv_estimator=gae`。

### Eval（无扰动的 global-model 验证）

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `val_env_spec` | path | `""` | `""` => **不 eval**；否则即无扰动的 val env-spec。 |
| `test_freq` | int | `5` | 每 K 轮（+ 最后一轮）评一次聚合后的 global 模型。 |
| `val_before_train` | bool | `True` | 第 1 轮之前也评一次 base 模型（round-0 点）。 |
| `val_temperature` | float | `0.4` | Val 采样温度（paper `val_kwargs.temperature=0.4`）。 |
| `webshop_val_port` | int | `8090` | 共享的无扰动 WebShop val 服务端口。 |
| `alfworld_val_port` | int | `8290` | 共享的无扰动 ALFWorld val 服务端口。 |
| `alfworld_val_split` | str | `eval_in_distribution` | ALFWorld val game（in-distribution 的 `valid_seen` eval 集）。 |

Eval 给 **global** 模型（round 0 用 base，否则用该轮聚合后的 HF）在一个
共享的无扰动 val 服务上、经由 verl `val_only` pass（`adv_estimator=grpo`，无 critic，
FedProx 关）打分，于是每个 arm 都在同一个固定集上被测量。一次失败的 eval 永远不会中止
run —— 它是测量，不是循环。

### FedProx

| Key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `fedprox_mu` | float | `0.0` | `>0` => client 侧的 FedProx proximal 项（否则 FedAvg）。 |

`fedprox_mu>0` 经由 env var `FEDPROX_MU` 桥接到每个 client（及其 Ray worker），
`sitecustomize.py` 在解释器启动时读取它，用 proximal 项 patch
`FSDPEngine.optimizer_step` —— 选这个而非 Ray `runtime_env` hook，是为了保住
verl 的 per-worker `CUDA_VISIBLE_DEVICES` 隔离。

---

## `client_overrides` and `adv_estimator`

`client_overrides` 是一份额外的 `key=value` **Hydra override** 列表，原样追加到
每个 client 的 `main_ppo_fed` 命令（eval 也复用，以让 rollout 形状一致）。
它是每个 arm 钉住 rollout/batch/context 形状的地方 —— 这些形状 base `fedagent_ppo.yaml`
留在 smoke 默认值上。关键的几个：

| Override | 角色 |
|---|---|
| `actor_rollout_ref.rollout.n=8` | **GRPO group size G**（论文里是 8）。 |
| `data.train_batch_size=8`（PPO：`64`） | 每个 optimizer step 的 prompt 数；与 `actor_rollout_ref.actor.ppo_mini_batch_size` 配对。 |
| `data.max_prompt_length` / `max_response_length`（WebShop `4096` / `512`；ALFWorld `2048` / `512`） | Token 预算；在 `rollout.prompt_length` / `response_length` 上镜像。 |
| `actor_rollout_ref.rollout.max_model_len`（WebShop `4608`；ALFWorld `2560`） | vLLM context window。 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM KV-cache 比例（`0.5`–`0.6`）。 |

> **`ppo_mini_batch_size` 设在 `client_overrides` 里，不在文件名或顶层
> key 里。** 它是 verl 的 actor 旋钮（`actor_rollout_ref.actor.ppo_mini_batch_size`），所以
> 它搭在 `client_overrides` 上。verl 内部把它乘以 `rollout.n` 形成 per-update 的 sample
> 数，这就是为什么论文把 `ppo_mini_batch_size=8` 与 `rollout.n=8`（GRPO）配对、把 `=64` 与
> `rollout.n=8`（PPO）配对。别把它跟 `min_goals_per_client`（联邦/分片旋钮）或
> `data.train_batch_size`（每 step 的 prompt 数）搞混。

对于 **PPO**（`adv_estimator: gae`），override 还会启用并塑形 critic，且
`save_contents=[model]` 让 value-model checkpoint 可被 FedAvg：

```yaml
adv_estimator: gae
client_overrides:
  - actor_rollout_ref.actor.checkpoint.save_contents=[model]
  - critic.optim.lr=1e-5
  - critic.model.use_remove_padding=true
  - critic.model.enable_gradient_checkpointing=true
  - critic.fsdp.optimizer_offload=true
  - critic.ppo_micro_batch_size_per_gpu=4
  - critic.checkpoint.save_contents=[model]
  - trainer.critic_warmup=0
```

**GRPO vs PPO：** GRPO（默认，`rollout.n=G=8`，无 critic）让 client 命令
与已验证路径逐字节一致。PPO（`adv_estimator=gae`）把 `need_critic` 翻开；runner
**每轮把 value 模型与 actor 一起**联邦化（第 1 轮 critic = base 模型，此后是聚合后的
critic），复用同一套 FedAvg + merge 机制 —— merger 会从分片的
`huggingface/config.json` 自动检测 `...ForTokenClassification` vs `...ForCausalLM`。

---

## 命名陷阱

那些单词式的 arm 名字藏了几个坑。读配置时把这些理清：

- **每条 task-het 轴用哪个旋钮** —— 每条 task 轴有自己的旋钮；传错的会静默 no-op
  （服务只转发其策略需要的那个 key）：
  - **Preference** -> `omega`（Dirichlet spread ω；更大 = 更偏斜）。
  - **Coverage** -> `size_std`（Beta dispersion ξ）。
  - **Hardness** -> `success_std`（Beta dispersion ξ′）**以及**必需的 `trajectories_file`。
- **文件名 token ≠ runner `partition_strategy`**，对 WebShop env-variant arm 而言。文件名
  拼论文名；YAML 用 dispatch key：
  `field_subset_index` -> `bm25_field_subset`，`bm25_reweighting` -> `bm25_reweight`，
  `lookalike_injection` -> `lookalike`，`rank_wrapper` -> `rank_wrapper`。（`catalog_split`
  和 task 策略在两处一致。）永远信 YAML。
- **`variant_n` 是 env-variant 数**，作为文件名 token `N-<n>` 呈现。它
  只适用于 `bm25_field_subset` / `bm25_reweight` / `lookalike` / `rank_wrapper`；
  `0` => 函数内建默认值（2 或 4）。
- **`ppo_mini_batch_size` 住在 `client_overrides` 里**（一个 verl actor 叶子，内部
  乘以 `rollout.n`），不在文件名或顶层 runner key 里 —— 见上方的 box。
- **`env_disjoint`（ALFWorld）vs `catalog_split`/`task_disjoint`（WebShop）** 都是
  env 级的 partition；ALFWorld 那个名字不同，因为它分片的是 game 文件，
  而非 catalog。

---

## 一份走查过的配置

一份真实的 `uniform/main/grpo` WebShop 配置
（`fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml`）：

```yaml
env_kind: webshop
env_spec: config/envs/webshop_15.yaml
val_env_spec: config/envs/webshop_15_val.yaml
output_dir: /tmp/xbb9020_fedpaper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform
model_path: Qwen/Qwen2.5-1.5B-Instruct

total_clients: 100
clients_per_round: 2
total_rounds: 70
epochs_per_round: 3
base_seed: 42

n_gpus_per_node: 4
total_training_steps: 0        # 0 => full E epochs/round (no per-round step cap)
save_freq: 100000              # save only the round's last step
test_freq: 5
val_before_train: true
val_temperature: 0.4
wait_between_clients: 8
min_goals_per_client: 100
webshop_pool_size: 16
webshop_base_port: 10000
webshop_val_port: 10100
search_return_n: 50            # engine default (matches the original non-het baselines)
partition_strategy: ""         # IID

client_overrides:
  - data.train_batch_size=8
  - data.max_prompt_length=4096
  - data.max_response_length=512
  - actor_rollout_ref.actor.ppo_mini_batch_size=8
  - actor_rollout_ref.rollout.n=8
  - actor_rollout_ref.rollout.prompt_length=4096
  - actor_rollout_ref.rollout.response_length=512
  - actor_rollout_ref.rollout.max_model_len=4608
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  - actor_rollout_ref.actor.checkpoint.save_contents=[model]
```

直接运行它：

```bash
python -m fedagent.fed.run_fed \
    --config fedagent/config/paper/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml \
    --model-path /path/to/Qwen2.5-1.5B-Instruct      # offline: a local snapshot
```

模式、GPU 与走查示例见 [`./running.md`](./running.md)，完整矩阵映射到命令见
[`./reproducing.md`](./reproducing.md)。
