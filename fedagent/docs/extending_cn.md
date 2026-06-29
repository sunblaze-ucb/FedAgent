# 扩展 FedAgent

FedAgent 是**原生 verl 0.8 之上的一层薄薄的 overlay**，并且它*首先是一个 library*：联邦控制
循环、各 environment、异质性构造器以及模型聚合都被刻意解耦，所以你可以替换其中任意
一个而不触碰其余部分。本文档正是做这件事的参考。

维护的入口点永不改变 —— 每一个扩展都可以通过

```bash
python -m fedagent.fed.run_fed --config <yaml>
```

触达。

driver（`fedagent/fed/run_fed.py`）是**与 verl 无关的**：一个 client 只是一个
子进程（`python -m fedagent.main_ppo_fed`）。它**串行地**训练每个被选中的 client，
对得到的 FSDP 分片做 FedAvg，把它们 merge 回一个 HuggingFace 模型，并从那个聚合后的模型
重新进入下一轮。正因为有这道分界，每个扩展点都被隔离到少数几个文件中：

| # | 扩展点 | 主要文件 | 由谁选择 |
|---|---|---|---|
| 1 | **新 environment** | `fedagent/envs/<name>/` + `fedagent/envs/registry.py` + 一份 `config/envs/<name>.yaml` | env-spec 行的 `name:` |
| 2 | **新异质性策略** | `fedagent/hetero/<name>.py` + 服务的环境变量桥接 + `run_fed.py` | `partition_strategy`（YAML / env `PARTITION_STRATEGY`） |
| 3 | **新 RL 算法** | verl trainer（`algorithm.adv_estimator`）；FedAgent 只携带 checkpoint | `adv_estimator`（YAML → Hydra） |
| 4 | **新聚合规则** | `tools/verl08_migration/aggregate_fedavg_fsdp.py`（服务端 FedAvg）· `sitecustomize.py` + `fedagent/fedprox.py`（client 侧 hook） | aggregator CLI / `fedprox_mu` |

> **各层如何拼合。** overlay 拥有*编排*（round 循环、per-client env 服务、聚合、eval）
> 与 *agent rollout*（`fedagent/agent_loops/gym_text_agent_loop.py` 为每个 dataset 行驱动
> 一个 `BaseTextEnv`）。原生 verl 拥有 *RL 更新*（advantage 估计、actor/critic FSDP
> worker、optimizer）。扩展点 1 和 2 完全活在 overlay 里；点 3 活在 verl 里（overlay 只是
> 选中它并携带额外的 checkpoint）；点 4 横跨两边 —— 服务端规则是一个 overlay 工具，而
> client 侧的近端项是一个在解释器启动时加载的 verl monkeypatch。

> **开始之前。** 阅读 [`./architecture.md`](./architecture.md) 了解 round 循环，阅读
> [`./installation.md`](./installation.md) 了解**三个** conda env（trainer env
> `fedagent-verl08` 加上 per-service 的 `verl-agent-webshop` / `verl-agent-alfworld`
> env）。那些 Python 依赖与 verl 0.8 冲突的 environment（WebShop 的 pyserini/Java、
> ALFWorld 的 TextWorld）运行在它们各自的 conda env 中、藏在一个 HTTP 服务后面；
> 进程内的 environment（TinyGuess）不需要这些。

---

## 1. 添加一个 environment

### 在哪里

每个 FedAgent agent-loop 驱动的 environment 都实现一份 async 契约
`BaseTextEnv`（`fedagent/envs/base.py`），并通过名字从 `fedagent/envs/registry.py` 查找。
agent-loop **为每个 dataset 行实例化一个 env 实例**，并 `await` 它的 reset/step（verl-0.8
的 agent-loop 是 per-row async 的，而非旧的批处理、同步的 `EnvironmentManager`）。

### 契约

`BaseTextEnv` 就是全部接口 —— 四个方法，其中三个是抽象的。observation 约定是一个 dict，
至少包含 `obs_str`（展示给模型的文本）：

```python
# fedagent/envs/base.py
Obs = Dict[str, Any]

class BaseTextEnv(ABC):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        self.env_config: Dict[str, Any] = dict(env_config or {})

    @abstractmethod
    async def system_prompt(self) -> Obs:
        """The system message ({"obs_str": ...}) shown once at episode start."""

    @abstractmethod
    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        """Reset to a fresh episode, deterministically in `seed`. Returns (obs, info)."""

    @abstractmethod
    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        """Apply the model's decoded text action. Returns (obs, reward, done, info)."""

    async def close(self) -> None:
        """Release any resources held by this instance (override if needed)."""
        return None
```

现有 env 遵守、你也必须遵守的三条不变式：

- **`info` 携带 `success`（bool）。** agent-loop 从 `info["success"]` 记录 episode 结果；
  它会成为 FedAgent 的头牌指标 `val/success_rate`。（看 `TinyGuessEnv.step` 如何返回
  `{"success": self.solved, ...}`，以及 `WebShopEnv.step` 如何映射服务的 `success` 字段。）
- **`reset` 在 `seed` 上是确定性的。** 每个 dataset 行都是一个带不同 seed 的不同实例；
  可复现性（以及 per-`(round, client)` 的重抽）只系于 `seed` 一项。联邦 driver 把
  `FEDAGENT_BASE_SEED = base_seed + round*100 + client_id` 穿进 dataset，它设定每一行的 seed。
- **`obs_str` 是唯一必需的 observation key。** 图像/多模态 env 之后可以添加
  `multi_modal_data` 而不改变契约。

`TinyGuessEnv`（`fedagent/envs/tiny_guess.py`，~70 行）是最干净的参考 —— 完全进程内、
无依赖、用一个正则解析 action、返回 higher/lower。先读它。

### 步骤

1. **继承 `BaseTextEnv`。** 对一个进程内的 env，直接在 `reset`/`step` 里做工作
   （照搬 `TinyGuessEnv`）。对一个依赖与 verl 0.8 冲突的 env，把子类做成一个**轻量
   HTTP client**（见 §1b），并把真正的 env 放进一个服务里。

2. **注册它。** 把类加到 `fedagent/envs/registry.py` 的 `ENV_REGISTRY` 里：

   ```python
   # fedagent/envs/registry.py
   from fedagent.envs.myenv import MyEnv

   ENV_REGISTRY: Dict[str, Type[BaseTextEnv]] = {
       "TinyGuess": TinyGuessEnv,
       "WebShop": WebShopEnv,
       "ALFWorld": AlfworldEnv,
       "MyEnv": MyEnv,          # <- the name the agent-loop looks up
   }
   ```

   dict 的 key 就是 `env_name`；`make_env(env_name, env_config)` 对任何未注册的名字
   都会抛 `KeyError`。

3. **写一份 env-spec YAML** 放在 `fedagent/config/envs/` 下。这就是
   `fedagent.data.agentic_dataset.AgenticDataset`（verl 的 `data.custom_cls`）读取并据以
   为每个 env 实例发出一行的内容。schema 是一个扁平的 env 块列表：

   ```yaml
   # fedagent/config/envs/myenv.yaml
   envs:
     - name: MyEnv          # MUST match the ENV_REGISTRY key
       n_envs: 8            # rows (= distinct instances/seeds) emitted for this spec
       max_turns: 15        # per-episode turn budget enforced by the agent-loop
       agent_name: gym_text # optional (default: gym_text)
       config:              # forwarded verbatim as `env_config` to the constructor
         timeout: 180.0
   ```

   dataset 发出 `n_envs` 行，每行带一个不同的 `seed`；GRPO 的分组随后由 verl 的
   `rollout.n` 在下游处理（每行重复 `n` 次 = 每个实例一个 GRPO 组）。保持
   `n_envs == data.train_batch_size`，使得这个 batch 每步恰好容纳一个组（看
   `webshop_15.yaml` 的文件头了解原因）。

4. **把联邦配置指向该 spec。** 在你的 `run_fed` YAML 里设
   `env_spec: config/envs/myenv.yaml`（相对于该 package 解析）。对一个进程内的 env，
   这就够了 —— 把 `env_kind` 设成 `webshop`/`alfworld` 以外的值，driver 就**不会**启动
   任何服务（`tinyguess` 是进程内的哨兵）。见文末的 smoke 配方。

### 1b. 添加一个 per-client HTTP 服务（仅当依赖冲突时）

WebShop 和 ALFWorld 无法与 verl 0.8 同时 import（pyserini/Java/gym 0.24；TextWorld），所以
真正的 env 运行在它自己的 conda env 里、藏在一个 FastAPI 服务后面，而 `BaseTextEnv` 子类
是一个轻量 client。照搬 `fedagent/envs/webshop/service/`（最完整的参考）或
`fedagent/envs/alfworld/service/`。

**client 侧**从一个 per-env 环境变量读取它的服务 URL，该变量由 driver *按 client* 设置，
然后把文本送进、把 observation 送出：

```python
# fedagent/envs/myenv/myenv_env.py  (mirrors fedagent/envs/webshop/webshop_env.py)
class MyEnv(BaseTextEnv):
    def __init__(self, env_config=None):
        super().__init__(env_config)
        self.base_url = (
            os.environ.get("MYENV_SERVICE_URL")          # set per-client by run_fed (authoritative)
            or self.env_config.get("service_url")        # ad-hoc single-service fallback
            or "http://localhost:8080"
        ).rstrip("/")
        self.session_id = uuid4().hex
        self._client = None  # lazily-created httpx.AsyncClient

    async def reset(self, seed=0):
        c = self._c()
        await c.post("/create", json={"session_id": self.session_id})
        r = await c.post("/reset", json={"session_id": self.session_id, "seed": int(seed)})
        ...  # format the service obs into {"obs_str": ...}, return (obs, {})

    async def step(self, action_str):
        r = await self._c().post("/step", json={"session_id": self.session_id, "text": action_str})
        d = r.json()
        info = {"success": bool(d.get("success", False)), ...}
        return {"obs_str": ...}, float(d.get("reward", 0.0)), bool(d.get("done", False)), info

    async def close(self):
        ...  # POST /close, aclose the httpx client
```

**服务侧**是一个 FastAPI app，预热一**池** env 实例（这样 episode 不必付启动成本），并
服务于 borrow → reset → step\* → return 这条生命周期。client 期望的五个端点是
`/health`、`/create`、`/reset`、`/step`、`/close`。在**服务端**解析模型的 action 文本
（env 的 projection 函数就在那里）。pool 大小从一个环境变量读取，并且**必须 ≥ 生成 batch**：

| 端点 | 作用 |
|---|---|
| `GET /health` | 就绪探针；driver 轮询它直到池预热完成（并回显 partition 信息） |
| `POST /create` | 从池中借一个预热好的 env 给一个 session |
| `POST /reset` | 把那个 env 重置到一个 goal/seed；返回 obs + 可行 action |
| `POST /step` | 在服务端解析文本、step 该 env、返回 obs/reward/done/success |
| `POST /close` | 把该 env 还回池中 |

在 `server.py` 旁边加一个 `run_service.sh`，它激活正确的 conda env 并拉起 uvicorn。
然后在你的 `run_fed.py` 里，你需要一个类似于 `start_webshop_services` /
`start_alfworld_services` 的 launcher（它设置 `MYENV_PORT`、`MYENV_POOL_SIZE`、下面的
partition 环境变量桥接，并等待 `/health`），以及在 `run()` 里一个 `env_kind: myenv` 分支
来调用它。在 `run_client` 里设置 per-client 的 `MYENV_SERVICE_URL`，使 client *c* 与
`base_port + c` 上的服务通话。共享的**未扰动** validation 服务（一个跑在单独端口上的
full-env 服务，用于每 `test_freq` 轮给聚合后的全局模型打分）在
`start_val_service` / `eval_global` 中以同样的方式接线。

完整的 client/service 契约见 [`../envs/README.md`](../envs/README.md)，两个已实现的服务见
[`../envs/webshop/service/README.md`](../envs/webshop/service/README.md) /
[`../envs/alfworld/service/README.md`](../envs/alfworld/service/README.md)。

---

## 2. 添加一个异质性策略

### 在哪里

每一种构造都是一个 `fedagent/hetero/` 下自包含的、仅 numpy 的模块，暴露一个公开的
**`*_for_client(...)`** 函数。（这些函数是从 verl-agent-0.3.1 的 `partition_strategy.py`
**逐字**复制过来的，所以 per-client 分配与 baseline 是 bit-identical 的 —— 这是科学红线；
只有那层薄薄的 `*_for_client` 公开 API 是新的。）策略由 `partition_strategy` 选择，并通过
一个**环境变量桥接**穿进每个 env 服务。

### 两条轴

| 轴 | 经由谁进入 | 返回 | 示例 helper |
|---|---|---|---|
| **Task-level**（preference / coverage / hardness） | *prompt*（一个 client 抽哪些 goal） | 该 client 的**goal 索引**列表 | `preference_for_client`、`coverage_for_client`、`hardness_for_client` |
| **Env-level**（catalog_split / variants 2–5） | *transition kernel* `P_i`（catalog / search 动态） | 一个 catalog + goal-idx 对，或一个 `env_kwargs` dict | `catalog_split_for_client`、`bm25_variant_for_client`、`lookalike_injection_for_client`、`rank_wrapper_for_client` |

Task-level 策略让 env 保持未扰动（完整 catalog）且对 FedAvg 鲁棒；env-level 策略扰动
隐藏的动态，最坏情况下不鲁棒。taxonomy 以及旋钮命名的注意事项见
[`./heterogeneity.md`](./heterogeneity.md) 和 [`../hetero/README.md`](../hetero/README.md)
（例如 `size_std`/`success_std` 是 Beta *concentration* ξ/ξ′，而非标准差）。

### 契约

这些公开函数共享一个固定的形状（在 `client_id`、`client_num` 之后是 keyword-only 旋钮）。
一个**task-level** 函数返回该 client 的 goal 索引：

```python
# fedagent/hetero/webshop_catalog_split.py (env-level — returns a (catalog, idxs) pair)
def catalog_split_for_client(
    client_id: int,
    client_num: int,
    *,
    env_div: float = 0.7,
    keep_ratio: float = 0.7,
    min_goals_per_client: int = 100,
    holdout_file: Optional[str] = None,
    base_seed: int = 42,
    data_dir: Optional[str] = None,
) -> Tuple[List[str], List[int]]:        # (catalog_asins, client_goal_idxs)
    ...
```

每个策略都遵守的两条不变式：

- **在 `client_id` 上是确定性的。** 每个 client 进程独立运行该函数，并且必须就全局分配
  达成一致，这样 FedAvg 每一轮看到的都是*相同的*per-client 切片。现有代码硬编码
  `base_seed=42`，并把 per-client RNG seed 为 `np.random.RandomState(42 + client_id)`（或一个
  共享的 `default_rng(42)` 用 `client_id` 索引）。**不要**从 Python 内建的字符串
  `hash()` 来 seed（它每个解释器都加盐）。
- **保证下限。** 如果你的抽样不够，就补足到 `min_goals_per_client`。

### 步骤

1. **写 `fedagent/hetero/myhet.py`**，暴露 `myhet_for_client(client_id, client_num, *,
   <knobs>, min_goals_per_client=100, env_goals=...)`。按 goal *内容*（category/size/hardness）
   选择的 task-level 函数把 env 真实的 goal 列表作为 `env_goals` 接收，并返回指向它的绝对
   索引；env-level 函数返回一个 catalog/idx 对或一个 `env_kwargs` dict 以合并进 env 构造器。

2. **在服务 `server.py` 里接好环境变量桥接。** driver 把旋钮作为环境变量传入；服务
   在 `PARTITION_STRATEGY` 上 dispatch 并调用你的函数。当前的 WebShop 桥接
   （`fedagent/envs/webshop/service/server.py`）读取：

   | 环境变量 | 含义 |
   |---|---|
   | `PARTITION_STRATEGY` | dispatch key（你的新 key） |
   | `CLIENT_ID` / `CLIENT_NUM` | 这个 client 的 id 和 cohort 大小 |
   | `MIN_GOALS_PER_CLIENT` | per-client 下限 |
   | `OMEGA` | preference Dirichlet 扩散（ω） |
   | `SIZE_STD` / `SUCCESS_STD` | coverage / hardness Beta concentration（ξ / ξ′） |
   | `ENV_DIV` / `KEEP_RATIO` | catalog-split 强度 / distractor 密度 |
   | `VARIANT_N` | 池中 env-variant 臂的数量（bm25/lookalike/rank） |
   | `TRAJECTORIES_FILE` | hardness：`task_id → success` 标签文件 |

   加一个 `elif` 分支：

   ```python
   # fedagent/envs/webshop/service/server.py
   elif PARTITION_STRATEGY == "myhet":
       from fedagent.hetero.myhet import myhet_for_client
       # task-level (content-dependent) idxs are DEFERRED to _lifespan, where the warmed
       # env's real server.goals exist (see _compute_task_partition). Order-independent
       # strategies (a contiguous range / env_kwargs) can be computed here at import time.
       _DEFERRED_TASK_PARTITION = "myhet"   # then add a branch in _compute_task_partition()
   ```

   注意这个**deferral 微妙之处**：一个内容相关的 task partition 必须寻址 env *实际的*、
   seed-42-shuffle 过的 `server.goals`，而它们只在池被预热之后才存在 —— 所以这些是在
   `_compute_task_partition()`（从 `_lifespan` 调用）里计算的，而非在 import 时。Catalog/variant
   策略（顺序无关的 range 或 `env_kwargs`）在 import 时计算。未扰动的 **validation** 服务
   总是以清空的 `PARTITION_STRATEGY` 启动，所以分歧只能归因于扰动本身。

3. **从 `run_fed.py` 转发旋钮。** 把任何新旋钮作为一个 key 加进 `fedagent/fed/run_fed.py`
   的 `DEFAULTS` dict，然后在服务 launcher（`start_webshop_services` /
   `start_alfworld_services`）里把它导出为一个环境变量，例如：

   ```python
   env.update({
       "PARTITION_STRATEGY": cfg.partition_strategy or "",
       "CLIENT_ID": str(c), "CLIENT_NUM": str(cfg.total_clients),
       "MYHET_KNOB": str(cfg.get("myhet_knob", <default>)),   # <- your new knob
       ...
   })
   ```

4. **从配置选择它。** 异质性完全通过联邦 YAML 选择（一个**扁平**的 schema = `DEFAULTS`
   dict —— 没有嵌套的 `federated:` 块）：

   ```yaml
   partition_strategy: "myhet"   # the dispatch key
   myhet_knob: 0.5               # your knob (also add it to DEFAULTS)
   min_goals_per_client: 100
   ```

---

## 3. 添加一个 RL 算法（GRPO / PPO 之外）

### 在哪里

RL 更新完全活在**原生 verl** 里，由 `algorithm.adv_estimator` 选择。联邦 overlay 从不触碰
loss；它只 (a) 选择 estimator、(b) 携带 client 写出的任何 checkpoint。

### 今天 GRPO 和 PPO 是怎么被选中的

联邦 YAML key `adv_estimator` 驱动 `run_fed.py` 的 `run_client` 里的一个分支：

```python
# fedagent/fed/run_fed.py
if str(cfg.get("adv_estimator", "grpo")).lower() == "gae":
    cmd += ["algorithm.adv_estimator=gae"]          # PPO: flips need_critic on
    if critic_model_path:
        cmd += [f"critic.model.path={critic_model_path}"]
```

- **GRPO**（默认，`adv_estimator: grpo`）：actor-only。组大小 *G* 来自 `rollout.n`
  （论文 *G = 8*，经由 `client_overrides` 里的 `actor_rollout_ref.rollout.n=8`）。client
  **只**写一个 actor checkpoint，driver 也只对 actor 做 FedAvg/merge —— GRPO 命令与已验证的
  baseline 是逐字节相同的。
- **PPO**（`adv_estimator: gae`）：加一个 **critic**（value 模型）。driver 检测 actor 旁边的
  critic 分片目录（`critic_dir_for`），对**两个**组件都做 FedAvg，各自 merge 到 HF，并把联邦
  value 模型携带向前（`critic.model.path` 每轮设置；第 1 轮的 critic = base 模型 —— backbone 上
  一个随机的 value head）。为了让这能工作，PPO 配置**必须**在 `client_overrides` 里包含
  `critic.checkpoint.save_contents=[model]`（以及 `...actor.checkpoint.save_contents=[model]`），
  这样 aggregator 才能找到 value 模型权重。见
  `fedagent/config/examples/webshop/scaled/ppo.yaml`。

### 联邦化一个新算法需要做什么

1. **在 verl 里添加 estimator**，使 `algorithm.adv_estimator: my_algo` 能解析（verl 上游自带
   PPO/GAE、GRPO 等；新加一个就在 verl 的 advantage 计算 dispatch 里并排添加）。FedAgent
   不 patch 这个。

2. **从联邦 YAML 选择它。** 如果你的算法是 actor-only 且不需要额外的 checkpoint，在
   `run_client` 里加一行分支（照搬 `gae` 分支）以传入 `algorithm.adv_estimator=my_algo`；
   通过 `client_overrides`（每一项都是施加到 per-client 子进程的字面 Hydra override）暴露
   它的超参数。

3. **留意 checkpoint 形状 —— 这是唯一的联邦层面顾虑。** aggregator（§4）操作于 **FSDP
   分片布局**（`checkpoints/global_step_<n>/<component>/model_world_size_*_rank_*.pt`），而非
   算法内部。一个 **actor-only** 算法不需要额外的东西。如果你的算法在 actor 之外增加了一个
   可训练组件（就像 PPO 增加了 critic），确保那个组件落在同样的
   `global_step_<n>/<component>/` 布局下、**带**
   `checkpoint.save_contents=[model]`，并用同一套机制对它 FedAvg/merge ——
   `run_fed.fedavg(..., kind="<component>")` 和 `merge_to_hf(..., kind="<component>")` 已经是
   组件无关的（它们对所给的任何分片目录做平均与 merge）。merger 从分片的
   `huggingface/config.json` 读取架构，而 actor 与 value 模型**都**序列化为
   `...ForCausalLM`（value 模型只是多带一个标量 value head），所以不需要对各组件做特殊
   处理。

---

## 4. 添加一个聚合规则（FedAvg / FedProx 之外）

聚合有**两**道接缝，因为 FedProx 不是一个服务端规则 —— 它是一个 client 侧的近端项，
服务端聚合仍保持为 FedAvg。

### 4a. 服务端规则 —— `aggregate_fedavg_fsdp.py`

### 在哪里

`tools/verl08_migration/aggregate_fedavg_fsdp.py` 是活的服务端 aggregator。driver 每轮
（每个组件）经由 `torchrun` 向它 shell out 一次：

```python
# fedagent/fed/run_fed.py :: fedavg()
cmd = [
    "torchrun", f"--nproc_per_node={ws}", str(AGGREGATOR),
    "--phase", "aggregate",
    "--client-actor-dirs", ",".join(str(a) for a in client_dirs),
    "--output-actor-dir", str(agg),
    "--global-step", "0",
]                       # + ["--weights", "0.5,0.5"] when cfg.weights is set
```

### 形状（及其原因）

verl 0.8 FSDP1 把 per-rank 分片存为 `torch` `ShardedTensor`，它**无法**单进程加载。所以
FedAvg 运行在一个**匹配 world-size 的进程组**下（`torchrun --nproc_per_node == 保存时的
world_size`）：每个 rank 从每个 client 加载*它自己的* rank 分片，就地（带权）平均**本地**的
tensor，再 `torch.save` 把这个 dict 写回。输出在字节结构上与一个 verl checkpoint 相同
（同样的 `ShardedTensor` 对象，只是本地的值变了），所以下一轮以 verl 自己的 FSDP wrap
原样加载它。

```python
# the averaging core (aggregate_fedavg_fsdp.py)
sds = [torch.load(c / rank_file, weights_only=False) for c in clients]
base = sds[0]
for k in base:
    acc = _get_local(base[k])              # writable local shard (ShardedTensor/DTensor/plain)
    acc.mul_(weights[0])
    for w, other in zip(weights[1:], sds[1:]):
        acc.add_(_get_local(other[k]), alpha=w)
torch.save(base, out / rank_file)          # same objects, averaged local values
```

### 新规则的契约

1. **添加一个 `--phase`（或一个兄弟平均例程）。** CLI 是
   `--phase {aggregate,verify}`、`--client-actor-dirs A,B`、`--output-actor-dir OUT`、
   `--weights`、`--global-step`。复用上面的 **load → average-local-shard → save**
   骨架；大多数规则（trimmed mean、median、FedAvgM、按 `|X_i|` 的 per-client 加权）只是
   在同一份 per-rank `sds` 列表上做一次不同的归约。保持输出结构不变：写
   `model_world_size_<ws>_rank_<rank>.pt`，拷贝 `fsdp_config.json` + `huggingface/`，并写
   `latest_checkpointed_iteration.txt` —— 那*正是*下一轮的 client 加载的格式。
2. **加权 hook 已经存在。** 如果你的规则是"加权的 FedAvg"，你不需要在 aggregator 里写
   新代码：计算权重并通过 `cfg.weights`（`--weights w0,w1,...`；它们必须和为 1）传过去。
   driver 把 `cfg.weights` 转发给每个组件。
3. **用内置的 `verify` 阶段验证。** `--phase verify` 重新加载写出的分片，并断言本地的值
   等于各 client 的（加权）均值（FedAvg 正确性），且它们能作为 `ShardedTensor` 往返（这样
   verl 才会加载它们）。在信任一个新规则之前，在一个真实的 round 上跑它。

### 4b. Client 侧规则 —— `sitecustomize` 的 FedProx hook

FedProx 在每次 optimizer step 之前给 actor 梯度加上 `mu * (w - w_t)`，其中 `w_t` 是轮起始
的全局模型。在 FedAgent 的 subprocess-per-round 设计里，每个 client-round 都是一个**全新
进程**，它加载聚合后的模型，所以 `w_t` 就是第一次 optimizer step 时的参数 —— 不需要任何
外部的 per-round reset。

这个 hook 的注入**不**用 Ray 的 `runtime_env` worker hook（那会破坏 verl 的 per-worker
`CUDA_VISIBLE_DEVICES`）。取而代之，repo 根目录的 `sitecustomize.py` 在 PYTHONPATH 上的
**每一个**进程中（driver 及其 Ray worker，因为 `run_fed` 设了
`PYTHONPATH=REPO_ROOT`）由 CPython 在解释器启动时自动 import，并以 `FEDPROX_MU` 为门控：

```python
# sitecustomize.py (repo root)
import importlib.util, os
mu = float(os.environ.get("FEDPROX_MU", "0") or "0")
if mu > 0:
    if importlib.util.find_spec("verl") is None:
        pass   # non-trainer env (e.g. a service conda env without verl) -> silent no-op
    else:
        from fedagent.fedprox import install_deferred_patch
        if not install_deferred_patch(mu):   # arms a meta-path hook; fail CLOSED
            raise RuntimeError("FedProx requested (FEDPROX_MU>0) but the patch could not be armed")
```

`install_deferred_patch` 装载一个 `sys.meta_path` finder，它在 verl **第一次** import
`verl/workers/engine/fsdp/transformer_impl.py` 时 monkeypatch `FSDPEngine.optimizer_step`
—— 这发生在 Ray worker 设好它的 per-rank `CUDA_VISIBLE_DEVICES` *之后*。（在这里于解释器
启动时急切地 import `FSDPEngine` 会在设备分配之前就拉入 torch/verl，并在多 GPU 时破坏
per-rank GPU 隔离，"Duplicate GPU detected"。）这个 patch 在第一次调用时快照 `w_t`，然后
在原始 step 之前给每个本地分片加上近端梯度（FSDP1 sharded view / FSDP2 DTensor —— 逐元素的
`grad.add_` 在每个分片上都正确）。driver 在 `fedprox_mu > 0` 时按 client 设置
`FEDPROX_MU`；普通的 FedAvg 让 `mu = 0`（一个 no-op，而且 `fedagent` 甚至从未被 import）。
Eval 总是剥掉 `FEDPROX_MU`。

要在同一模式上添加**另一个** client 侧规则：把 patch 写进 `fedagent/` 下的一个模块，给
`sitecustomize.py` 加一个环境变量门控，并让 `run_fed.py` 按 client 设置那个环境变量。让
patch 在 import 时保持 CUDA-free（import `FSDPEngine` 不得初始化 CUDA —— 它运行在 verl 分配
设备*之前*）。

---

## Smoke-test 配方

用进程内的 TinyGuess smoke（2 clients × 2 rounds，每 client 每 round 1 步 —— 几分钟内闭合
整个联邦循环，无服务）端到端验证任何扩展：

```bash
# inside the fedagent-verl08 conda env, on a GPU node
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml
```

它训练每个 client（`python -m fedagent.main_ppo_fed`），在一个匹配的进程组下对两个 client
的 FSDP 分片做 FedAvg，merge 回 HF，并从聚合后的模型重新进入第 2 轮。一次干净的 run 以
`FEDERATED LOOP CLOSED` 结束，并写出 `<output_dir>/federated_summary.json`（per-round 出处：
`started_from → aggregated_hf`）。CLI flag 覆盖 YAML：

```bash
python -m fedagent.fed.run_fed --config fedagent/config/examples/tinyguess_2cl_2rd.yaml \
    --rounds 3 --clients 4 --n-gpus 2 --output-dir /tmp/my_smoke
```

每个扩展点要检查什么：

| 你改了 | smoke 证明 | 然后练习 |
|---|---|---|
| **一个新的进程内 env** | 把 `env_spec` 指向你的 `config/envs/<env>.yaml` —— 它解析、rollout、并聚合 | 在真实数据上做完整 run |
| **一个 service-backed env** | （TinyGuess 测不了服务） | 设 `env_kind: <env>`，看 `/health` 起来，确认 episode 能 step |
| **一个异质性策略** | 设 `partition_strategy` + 旋钮；确认服务 `/health` 回显你的 partition 且 per-client 切片不同 | [`./reproducing.md`](./reproducing.md) 下的 het 臂 |
| **一个新算法** | 设 `adv_estimator`；对带 critic 的算法确认两个组件都 FedAvg/merge | 把 `examples/webshop/scaled/ppo.yaml` 作为 PPO 模板 |
| **一个聚合规则** | 在某真实 round 的分片上跑 `--phase verify` | 一次多轮 run，与 average-of-clients 做 diff |

run-mode/GPU 矩阵以及联邦 key 参考见 [`./running.md`](./running.md) 和
[`./configuration.md`](./configuration.md)；round 循环见 [`./architecture.md`](./architecture.md)；
per-component 的代码细节见 [`../fed/README.md`](../fed/README.md)、
[`../envs/README.md`](../envs/README.md) 和 [`../hetero/README.md`](../hetero/README.md)。
