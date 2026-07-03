# FedAgent agent-RL 子系统 vs NanoRollout — 深度对比与设计借鉴（2026-07-01）

> **本文是什么。** 对 FedAgent 的 agent-RL 子系统（[agent_rl_design.md](./agent_rl_design.md)）与
> NanoRollout（UCSD/cocoa-org，[本地克隆](../../others/NanoRollout/)，[博客](https://cocoa-org.notion.site/nanorollout)）
> 的逐机制代码级对比：架构边界、token 保真度、可靠性契约、加速效果、各自优劣、互相可借鉴什么，
> 以及**如果要为 WebShop/ALFWorld 一类文本环境设计一个独立（非联邦）的 Agent RL framework 应该怎么设计**。
> 两边均以实际代码为准（FedAgent @ `migrate/verl-0.8.0`；NanoRollout @ 本地克隆 2026-07-01；
> miles 侧 TITO 集成读自 [cocoa-org/miles/examples/nanorollout](https://github.com/cocoa-org/miles/tree/main/examples/nanorollout)）。

---

## 0. TL;DR

两个系统是**同一个设计模式（"环境执行与训练解耦成服务"）在两种环境重量级下的收敛解**，
边界画的位置相反，而且各自都画对了：

- **FedAgent = env-as-a-service（细边界，step 级）**：harness/agent loop 留在 trainer 侧、贴着
  tokenizer，每个 `env.step` 过一次 HTTP。因为文本环境每步毫秒级、科学要求 per-turn windowed
  prompt 和 token 级精确控制。
- **NanoRollout = rollout-as-a-service（粗边界，episode 级）**：整个 harness+env 循环在服务侧跑，
  模型反向暴露成 OpenAI 兼容 endpoint。因为 SWE/computer-use 环境是容器/VM 级、每步秒级，
  且要让 RL/蒸馏/评测三种负载复用同一套 harness。
- **加速上打的是同一场仗的两个战区**：都是"扩环境侧并发直到瓶颈转移回 GPU/模型侧"，都撞上了
  长尾（最慢 episode）这堵墙，都还没解决。FedAgent 的独有敌人是联邦带来的 ~140 次 cold-start
  （persistent trainer −62% 是最大单项收益）；NanoRollout 的独有敌人是重环境的供给与调度。
- **最值得互相借鉴的**：FedAgent ← episode 级 `/run` 门面（统一 eval/蒸馏/外部 trainer）、
  per-rollout 时间遥测规范、harness 显式分层；NanoRollout ← step 幂等/重试契约、replica
  sharding 的"进程数拆锁"诊断、种子契约与四级验证方法论。
- **若设计独立文本环境 Agent RL framework**：保留 step 级 env service 为 RL 主路径（token-native），
  在其上加一层 episode 级 `/run` 门面（message-native），harness 抽成独立可插拔层——
  **双层 API，各取两家之长**。详见 §7。

---

## 1. 两个系统各自在解决什么问题

| 维度 | FedAgent agent-RL | NanoRollout |
|---|---|---|
| 训练对象 | Qwen2.5-1.5B 级文本 agent | Qwen3 4B–32B 级 SWE / 终端 / computer-use / 统一 agent |
| 环境 | WebShop（15 turn）、ALFWorld（≤50 turn）：**进程内 Python env**，每步毫秒级 | SWE-Bench / Terminal-Bench / OSWorld / CocoaBench：**Docker/VM/云沙箱**，每步秒级、每 episode 分钟-小时级 |
| 科学约束 | 与论文严格等价（GRPO G=8、per-turn windowed prompt、3-seed 内复现结论） | 无单一论文忠实性约束；目标是规模与通用性 |
| 训练结构 | **联邦**：N clients × T rounds，FedAvg 聚合 → ~140 个 (client,round) 训练任务/论文跑 | 单次连续 RL run / 离线蒸馏 / 纯评测 |
| 负载种类 | RL + 每轮全局 eval | RL + 蒸馏（250K+ 轨迹）+ 评测（500 并发 worker） |
| 规模 | 单节点 1–4×H100；episode 并发 64–512 | 多节点；rollout worker 16–500；RL batch 至 4096 |
| trainer | verl 0.8（零 fork，扩展点注册） | miles / veRL / tunix（adapter 模式） |

这张表解释了后面所有分歧：**每步毫秒级 + windowed 忠实性 → 细边界；每步秒级 + 多 harness 复用 → 粗边界。**

## 2. 架构对比

### 2.1 边界画在哪、模型调用往哪个方向走

```
FedAgent（env-as-a-service，step 级细边界）:
  ┌ trainer 进程（verl 0.8）──────────────────────────────┐
  │ FSDP actor ↔ vLLM server engines（ZMQ 权重同步）       │      每 turn 一次 HTTP
  │ GymTextAgentLoop 协程 ×512（harness 在这里，token 原生）│ ────────────────────────► env service ×K replicas
  └────────────────────────────────────────────────────────┘      /create /reset /step /close
                                                                   （FastAPI + env pool + 分区注入）

NanoRollout（rollout-as-a-service，episode 级粗边界）:
  ┌ trainer（miles/veRL/tunix）────┐   每 episode 一次 POST /run    ┌ rollout server（FastAPI+Ray）─────────┐
  │ SGLang/vLLM engines + 优化器    │ ─────────────────────────────► │ harness（litellm 全历史循环）＋ env    │
  │ （miles: TITO proxy 捕获 token）│ ◄───────────────────────────── │ backend（docker/modal/enroot/aws…）    │
  └────────────────────────────────┘   OpenAI 兼容模型回调（反向）    └───────────────────────────────────────┘
```

两边有一个**镜像对称的巧合**：FedAgent 的 `_pick_replica`（[envs/base.py](../envs/base.py)）对 K 个
env service 做 round-robin；NanoRollout 的 veRL 集成对多个 rollout engine endpoint 做 round-robin——
同一个手法用在边界的两侧。

### 2.2 组件映射

| 职责 | FedAgent | NanoRollout |
|---|---|---|
| 任务/episode 抽象 | dataset row + `agent_name` env spec | `RunRequest{instance_id, task, agent, env_type, resources, extra_args}`（[core/models.py](../../others/NanoRollout/nanorollout/core/models.py)）|
| harness（prompt 策略/动作解析） | 分散在 agent loop + env client 模板（windowed `_memory`、legacy_prompts） | **独立一层**：`BaseAgent`（tools / system_prompt / `_step()`），litellm 全历史，15+ harness 注册表 |
| 环境接口 | `BaseTextEnv`：async `reset/step/close`，obs=`{obs_str}` | `ShellEnvironment`：`start/stop/execute/execute_tool`；`TaskAdapter.evaluate()` 算 reward |
| 环境执行体 | FastAPI 服务内 env pool（asyncio.Queue，进程内 Python env） | 可换后端：docker/enroot/modal/GCE + desktop 云商 ×7 + AIO 沙箱 |
| 并发控制 | 客户端**无界**协程；服务侧 pool 容量 + K replicas | 服务侧 `asyncio.Semaphore`（默认 256）+ Ray 资源 hints（cpu/gpu/mem）|
| 结果返回 | token ids + response_mask（训练就绪） | `RunResponse{reward, messages, exit_status, agent_metrics, tools}`（message 级） |
| reward | env service `step()` info 返回（+ invalid-action penalty 在 loop 侧） | 服务侧 `TaskAdapter.evaluate()`（跑测试/评分脚本） |
| trainer 集成 | verl 扩展点：`@register("gym_text")`、`agent_loop_manager_class`（2 行 verl patch） | `/run` adapter + （miles）TITO proxy + TIS |
| 编排 | `run_fed.py`：轮循环、三种生命周期、FedAvg、四种 eval 模式 | 无训练编排（trainer 自带）；`nro run/serve` CLI |

### 2.3 代码规模

| | FedAgent agent-RL 子系统 | NanoRollout |
|---|---|---|
| 编排核心 | run_fed.py + persistent_* ≈ **2,167 行**（含联邦特有逻辑） | core/ ≈ **924 行**（server 136 + scheduler 162 + runners 210 + models 49 + config 17 + local 349）——"900 行 core"的宣传基本属实 |
| agent loop / harness | agent_loops ≈ 471 行 | harness ≈ **17,671 行**（15+ 个 harness；claude-code 包装 957、cocoa controller 3,379…） |
| 环境层 | env 客户端+服务 ≈ 1,393 行（webshop 603 + alfworld 560 + base/registry 122 + 杂项） | envs ≈ **16,633 行** + adapters ≈ 4,802 行（容器/VM/云商后端 + benchmark 评分） |
| 合计 | ≈ **4,009 行**（25 文件） | ≈ **40k 行**（248 个 .py） |

**读法**：NanoRollout 的"轻"只轻在编排核心——它把复杂度推给了 harness/env/adapter 层（这是刻意的、
也是对的：加 benchmark = 加 adapter，不动 core）。FedAgent 整个子系统 4k 行，因为文本环境本身薄；
但其中 2.1k 行是联邦编排，真正的"rollout 框架"部分只有 ~1.9k 行。**两边的 core 都很小，
证明这个设计模式本身是廉价的——贵的从来是环境和 harness 生态。**

## 3. 关键机制逐项对比

### 3.1 Token 保真度与训练样本构造（最重要的分歧）

**FedAgent：token-native（token 在整条链路上不落地为文本）。**
[`GymTextAgentLoop`](../agent_loops/gym_text_agent_loop.py) 直接从 vLLM server 拿 `out.token_ids`，
response_mask 在循环里逐 turn 构造（模型 token=1 / obs token=0）；windowed 模式
（[windowed_agent_loop.py](../agent_loops/windowed_agent_loop.py)）一个 episode 产出 ~50 条
per-turn 训练行（batch 64 episodes → 实测 3184 行），`traj_uid` 广播用于 GRPO 分组，episode 回报
广播到每 turn。**不存在重新 tokenize，不存在 chat-template 对齐问题，rollout 与训练看到的是同一串
token。**代价：与 verl 0.8 内部深度耦合（windowed_manager 对 DataProto 的 `_windowed_slice/_union`
monkeypatch + LCM size-divisor——verl 升级时的脆弱点）。

**NanoRollout：message-native，token 恢复是 trainer 的作业。** `/run` 返回 `messages`（无类型
JSON dict 列表），core 完全不做 token 记账。miles 侧的解法是 **TITO（token-in-token-out）proxy**
（[tito_server.py](https://github.com/cocoa-org/miles/tree/main/examples/nanorollout)，233 行）：
每个 rollout worker 起一个本地 FastAPI 假 OpenAI endpoint，`api_key = tito-<instance>-<sample>`
充当会话键；每次 chat 调用增量 tokenize 新消息、转发 SGLang `/generate(return_logprob=true)`、
记录**精确的输出 token ids + logprobs**（还支持 MoE routed-experts 回放）；episode 结束
`finalize()` 拼出 tokens/loss_mask/rollout_log_probs 写回训练样本。loss mask 只给 assistant
内容 token（ChatML 包裹、user/tool 消息、畸形 tool-call turn 全部 mask 掉）；再用 **TIS
（截断重要性采样，clip 2.0）**修正 SGLang 采样与 Megatron 训练前向的失配 + Dr.GRPO 损失。

**TITO 的三个隐含假设（正是 FedAgent 不能走这条路的原因）：**
1. **消息历史 append-only**（`TaskState` 按 `pre_msg_length` 增量记录）→ **windowed 滑窗 prompt
   直接违反此假设**，每 turn 丢弃旧消息会打乱增量记账；
2. **ChatML 硬编码**（`<|im_end|>`、assistant 前缀 id 写死在 proxy 里）→ 换模型家族要改 proxy；
3. **一个 episode 一条训练样本**（prompt = 首个 assistant 之前，response = 之后全部）→ 没有
   per-turn 样本概念。

**两边有同一块伤疤**：miles 代码里留着 `H5-COMPARE` 设施——逐样本对比 TITO token 与重新 tokenize
的 token 并落盘 mismatch；agent 失败没打过模型调用时 fallback 到重 tokenize + 全零 logprobs。
这与 FedAgent 的等价性验证（`max|Δ|≈1e-5`）是同一类痛苦的产物：**"训练看到的 token 是否就是
rollout 采样的 token"在两种架构下都必须被显式验证，谁都逃不掉。**

| | FedAgent | NanoRollout(+miles) |
|---|---|---|
| token 来源 | vLLM server 直接返回 ids | TITO proxy 捕获 SGLang ids（或 fallback 重 tokenize） |
| loss mask | loop 内逐 turn 构造 | proxy finalize 时按消息角色重建 |
| windowed per-turn | **原生支持**（框架核心特性） | 架构上不可行（append-only 假设） |
| 采样-训练失配 | 无（同一 vLLM 权重同步链路；logprob 用 FSDP 重算） | TIS 修正 + logprob 监控 |
| 模型家族耦合 | 无（纯 token id） | ChatML/Qwen 硬编码在 proxy |
| tool calling | 无（文本动作 + projection） | 一等公民（fn-call 解析、畸形 turn mask） |

### 3.2 可靠性契约

| 机制 | FedAgent | NanoRollout |
|---|---|---|
| 会话 | sticky session（session_id → pool 内 env 实例，`/create` 借出 `/close` 归还） | 每 `/run` 新建环境（容器/VM），用完即毁 |
| 重试 | 客户端 8 次指数退避+抖动（仅传输层错误；HTTP 4xx/5xx **不**重试，真失步必须响） | **core 无重试**；调用方自理（miles 侧失败 → mock 消息 + reward 0 + `remove_sample`） |
| 幂等 | **`step_id` 幂等键**：服务端单槽重放缓存，重试不会双重施加 `step()` | **无幂等**；重试 = 重跑整个 episode |
| 超时 | `/create` 读超时禁用（合法等待 pool 空位）；其余按 endpoint 有界 | `task_timeout_s` → `ray.cancel` 优雅取消 → 10s 宽限 → 强杀；`SchedulerTimeoutError` |
| 失败语义 | episode 内 step 失败会重试到成功或大声失败 | episode 级 all-or-nothing：env 中途死 → 整条轨迹丢弃（`exit_status=error`） |
| 部分轨迹 | windowed 模式下截断 episode 的 per-turn 行**机械上仍可训练**（见 §7.5） | 无 partial rollout（博客明示 future work） |
| 产物 | 训练日志 + timing_s | **每次 run 全量落盘**（消息/评测报告，审计友好，但无 GC，磁盘无界增长） |

**结论**：FedAgent 的契约是为"一个 episode 内几百次 HTTP 交互必须精确一次"设计的（step 级边界的
必然要求）；NanoRollout 的契约是为"episode 是原子任务，失败就重跑"设计的（episode 级边界的自然
选择）。**边界粒度决定可靠性语义**——这句话在 §7 的设计里会反复用到。

### 3.3 并发与调度

- **FedAgent**：客户端**故意无界**（512 协程全部并发，env 延迟与 LLM 延迟互相隐藏），边界在服务侧
  ——env pool 容量 + **K 个 replica 进程**（K 个锁/GIL）。调度器不存在：vLLM 动态批处理就是调度器。
- **NanoRollout**：服务侧 `asyncio.Semaphore(concurrency)` 准入门 + **Ray** 把每个 rollout 发到
  worker（`num_cpus/num_gpus/memory_gb` hints → `@ray.remote` 选项；`max_calls=50`/worker 防内存
  蠕变）。
- **本质**：FedAgent 的瓶颈曾是**进程内锁**（`_TW_LOCK`/GIL），解法是多进程分片；NanoRollout 的
  瓶颈是**重环境的供给**，解法是资源感知的分布式调度。文本环境用 Ray 调度是杀鸡用牛刀（一次
  `env.step` 毫秒级，Ray 任务开销都比它大）；容器环境不用资源调度则会把节点打爆。**各自都选对了，
  但互换会都选错。**

### 3.4 依赖与资源隔离

- FedAgent：**依赖隔离**是 HTTP 边界的第一动机——WebShop 的 gym-0.24/pyserini/Java 与 ALFWorld 的
  textworld 各住各的 conda，trainer 环境永远干净。
- NanoRollout：解耦的是**资源**（GPU vs CPU/内存），不是依赖——`pyproject.toml` 把所有 benchmark
  的评分依赖（easyocr、librosa、playwright、boto3、swebench…60+ 项）装进**同一个包**；env 执行虽在
  容器里，宿主侧 harness/评分依赖仍是一锅烩。
- **互补而非高下**：容器天然解决执行依赖，所以他们不需要 per-env conda；但"评分依赖全进一个包"
  在长期维护上会疼。独立框架两个都要（§7）。

### 3.5 遥测

- FedAgent：**trainer 侧 per-stage** 分解（`timing_s`: gen / old_log_prob / ref / update_actor），
  这是整个加速方法论（"flat gen → env-bound"决策规则）的基础。
- NanoRollout：**episode 侧 per-rollout** 规范化指标（`AgentMetrics`: turns、tool_calls、
  `model_query_time_sum`、`env_execution_time_sum`、`eval_time`、`total_time`，miles 侧再聚合出
  model/env/eval 时间占比进 wandb）。
- **正交且都该有**：FedAgent 缺 per-episode 的 env/LLM 时间占比（现在靠曲线形状反推环境瓶颈）；
  NanoRollout 缺训练步分解（在 trainer 侧，不归它管）。§7 的框架把两个都收编为一等公民。

### 3.6 长尾——两边共同的下一堵墙

- FedAgent 实测：replica 分片后 gen 的残余地板 ~60s = **最慢单条 episode 的关键路径**
  （~50 turns ×（LLM 0.2–0.3s + env 86ms/K + HTTP）），K=4≈K=8 —— 再加 replica 无用。
- NanoRollout 实测：16→256 worker，P95 提速 10.8× 但 full completion 只 5.4×——**最慢几个任务
  定义了地板**，256→500 饱和。
- 两边给出的处方相同且都未实现：**partial rollout / 早停 / 异步化**。这是 §7 设计里必须预留的
  扩展点，也是 FedAgent windowed 模式的一个隐藏优势所在（per-turn 样本天然兼容部分轨迹）。

## 4. 加速效果对比

> **⚠️ 先声明不可比性。** 两边的绝对数字**完全不可直接比**：环境重量级差 2–3 个数量级
> （env.step 毫秒 vs 秒/分钟）、模型差一个数量级（1.5B vs 4–32B）、任务结构不同（联邦 ~140 个
> 短 job vs 单次连续 run）。**可比的是机制、相对收益和饱和形态。**

### 4.1 FedAgent（实测，4 级验证：机制/对照/组件/端到端）

| 杠杆 | 攻击项 | 实测收益 |
|---|---|---|
| #4 persistent / cross_round | cold-start（曾占 wall 的 **76–88%**） | 每轮 **−43%** / 跨轮 **−62%**（等价性 `max|Δ|≈1e-5`） |
| eval 模式（worker/parallel） | eval 摆放 | eval ≈ 移出关键路径（两环境排名互换但"解耦优于耦合"恒成立） |
| #3 client-parallel | 训练计算（次线性 FSDP） | 2×2 GPU 727s vs 串行 1116s（**−35%**，WebShop） |
| **replica 分片**（Tier-1） | rollout 的锁/GIL 串行 | ALFWorld gen 217.5→61.8s（K8）；4-GPU step 298→127.6s（**−57%**）；端到端 3509→2412s（**−31%**）；WebShop 仅 −12%（GPU-bound，镜像瓶颈） |
| 对照组的教训 | — | pool 8→64（K=1）gen 不变——**锁才是全部**；WebShop pool 16→64 反而 +14%（GIL 放大） |

### 4.2 NanoRollout（博客报告数字）

| 场景 | 数字 |
|---|---|
| SWE-Bench Verified 评测（500 任务，DeepSeek-V3.2 API） | worker 16→256：full **102→19 min（5.4×）**，P95 **97→9 min（10.8×）**；256→500 **饱和**（瓶颈转移到模型服务/长尾） |
| 跨 benchmark（TB2/CocoaBench/OSWorld） | 环境并发扩展 **1.9–3.3×** |
| RL rollout batch 512→4096（同 8 节点 rollout） | 轨迹 8× 但收集时间仅 **3.3×**（1800→6000s/update，长尾被大池子摊平）；到同一目标分数 wall-clock 更短（32% 目标：67.0h→56.7h） |

### 4.3 同一条物理规律的两种读法

1. **"扩环境侧并发直到瓶颈转移"两边都验证了**：FedAgent 的 K=4≈K=8 地板 ≙ NanoRollout 的
   256→500 饱和——扩到某个 K 后，限制变成最慢 episode / 模型服务，继续加环境侧资源无效。
   FedAgent 把这条规则提炼成了可移植的**决策程序**（两 GPU 数各跑 1-step `timing_s`：
   gen 平 → env-bound → 加 replicas；gen 随 GPU 缩 → GPU-bound → 加 GPU），NanoRollout
   只有事后曲线——**这个决策程序是 FedAgent 方法论上的真领先**。
2. **大 batch 摊平长尾**（NanoRollout 8× 轨迹只花 3.3× 时间）在 FedAgent 侧有对应物：无界协程下
   512 episodes 互相隐藏延迟。机制相同：并发池越大，调度气泡越小，直到长尾成为地板。
3. **cold-start 是联邦特有的敌人**：NanoRollout 的 RL 是单次连续 run，trainer/engine 只初始化一次，
   不存在 ×140 的 cold-start 问题——所以他们没有（也不需要）persistent-trainer 这个杠杆。反过来，
   任何"把一次 run 切成许多短 job"的训练结构（联邦、course-schedule、多 seed 扫描）都会立刻遇到
   FedAgent 已解决的问题。**看加速清单差异，本质是看训练结构差异。**

## 5. 各自优劣

### 5.1 FedAgent agent-RL

**强**
1. **Token 保真度免费**：token-native 全链路，无重 tokenize、无 chat-template 对齐、无 TIS 需求。
2. **windowed per-turn 训练**是独有能力（NanoRollout 架构性做不到），这正是论文忠实性所在。
3. **step 级可靠性契约成熟**：幂等 `step_id`、sticky session、阻塞 `/create`、退避重试——512 并发
   episode 压测存活的产物。
4. **加速有完整因果链**：每个杠杆都有机制/对照/组件/端到端四级证据 + 预注册预测（含两次被证伪的
   预测），可移植的决策规则。
5. 联邦语义（per-client service = 隐藏转移核 + 分区注入）在别处没有对应物。
6. 零 fork verl（一个 2 行 patch），依赖隔离干净。

**弱**
1. **verl 0.8 深耦合**：windowed_manager 的 DataProto monkeypatch、`agent_loop_manager_class`
   钩子——verl 升级即回归风险；换 trainer（如 miles）基本要重写 loop 层。
2. **无统一负载门面**：eval 复用训练栈（四种模式是编排技巧），蒸馏/轨迹导出没有产品化出口；
   外部系统无法"给一个任务拿一条轨迹"。
3. **harness 不成层**：prompt 策略散在 env client（`_memory`、legacy_prompts）与 agent loop 两处，
   加一种 prompt 风格（如 ReAct/few-shot）要改多处。
4. **每个环境手写一套服务**：webshop/alfworld 两个 server.py 是 ~400 行的近似孪生，第三个环境
   还要再抄一遍（模板化欠账）。
5. 无 tool-calling、无多模态（obs 约定预留了 `multi_modal_data` 但没走通）。
6. 编排是 1,624 行联邦专用脚本，"独立 RL 框架"的部分没有单独的可复用形态——**这正是你问的
   "独立 framework"要解决的**。

### 5.2 NanoRollout

**强**
1. **core 极小（<1k 行）而生态极大（15+ harness × 14+ env backend × 4 域 benchmark）**：
   复杂度全部推到插件层，加东西不动 core——扩展性设计的教科书。
2. **一个 `/run` 统一三种负载**（RL/蒸馏/评测），schema 规范（`RunRequest/RunResponse/AgentMetrics`）
   ——蒸馏 250K 轨迹与 18 分钟评完 SWE-Bench 是这个统一的直接红利。
3. **trainer 无关性真实成立**（miles/veRL/tunix 三家），trainer 只需暴露 OpenAI 兼容 endpoint。
4. harness 多样性有科学结论支撑（单 harness 训练跨 harness 掉 7–19 分）——把 harness 当数据维度。
5. 资源感知调度（Ray hints）+ 超时的优雅降级（graceful→10s→强杀）适配重环境现实。
6. 每 rollout 全量落盘 + 规范化 AgentMetrics，审计与调参友好。

**弱**
1. **token 保真度外包给 trainer**：TITO 只在 miles 落地（veRL/tunix 参考实现"coming soon"）；
   proxy 硬编码 ChatML、每调用重 tokenize 全前缀（代码里留着 TODO）、需要 TIS 打补丁；
   fallback 路径是重 tokenize + 零 logprobs——保真是**修补出来的**，不是构造出来的。
2. **core 无重试/无幂等/无部分轨迹**：episode 级 all-or-nothing，长任务超时即全丢；长尾问题
   自己的评测数据（P95 vs full 差 2×）就是证据。
3. **无种子契约**：`RunRequest` 没有 `seed` 字段，任务身份 = `instance_id`（SWE 实例固定）。
   对程序化生成 episode 的文本环境（WebShop goal 抽样、ALFWorld 游戏抽样）这是硬缺口，
   可复现性做不到 FedAgent 的水平（client-indexed seed、顺序无关）。
4. 全历史 append-only 是唯一上下文策略；窗口/摘要/压缩都推给单个 harness 自己看着办。
5. 宿主侧依赖一锅烩（60+ 直接依赖装一个包）；产物无 GC；messages 无 schema 校验。
6. 训练侧编排完全不管（这对他们是特性，对想要"一键跑 RL"的用户是空白）。

## 6. 互相可借鉴什么

### 6.1 FedAgent 值得从 NanoRollout 拿的（按性价比排序）

1. **episode 级 `/run` 门面（高优）**——在现有 agent loop 与 env service 之上加一个薄服务：
   `RunRequest{env, seed, harness, model_endpoint, sampling} → RunResponse{reward, messages,
   agent_metrics, (可选) tokens}`。收益：(a) 蒸馏/轨迹导出立刻可用（教师模型 endpoint 即插）；
   (b) eval 与训练栈解耦出第五种模式（外部 eval 服务）；(c) 为换 trainer（脱离 verl）预铺路。
   估计 300–500 行，因为循环与 env 客户端都是现成的。
2. **per-rollout AgentMetrics 遥测规范（高优，便宜）**——在 agent loop 里累计 model/env/HTTP
   时间与 turn 数，聚合进现有 metrics。现在判断"env-bound 还是 GPU-bound"靠两次整步探针；
   有了它一次训练 run 内就能连续读出比值，长尾（最慢 episode）也直接可见。
3. **harness 显式分层（中优）**——把 windowed/concat/legacy 模板从 env client 抽成
   `Harness{build_prompt(task, history, obs), parse_action(text), history_policy}` 注册表。
   立刻消掉 webshop/alfworld 客户端的模板重复，也为 prompt 多样性实验（他们的跨 harness 结论
   在文本环境同样可能成立）打开门。
4. **exit_status 归一化 + 截断样本策略**——`finished/max_turns/timeout/error` 规范化，配
   `filter_overlong / truncation_penalty` 两个 knob（miles 侧现成设计），替代现在隐式的
   "跑完算完"。
5. **服务模板化**——他们"加 benchmark = 加 adapter"的结构映射过来就是：把两个孪生 server.py
   合成一个 `service 模板 + EnvAdapter 插件`，第三个环境只写 adapter。
6. （远期，仅当环境变重时）资源 hints + 准入门——`ResourceManager` 一共 13 行，先抄进服务模板
   当可选项即可。

### 6.2 NanoRollout 缺的、FedAgent 已验证的（若与他们交流/发文的差异点）

1. **step 级幂等契约**（`step_id` 单槽重放）——他们一旦做 partial rollout/断点续跑就必须发明它。
2. **"K 个进程拆锁"诊断**——他们宿主侧 Python 评分/调度同样有 GIL；FedAgent 的对照实验
   （pool 变大无效、进程数才有效）是可直接引用的证据。
3. **种子契约与顺序无关性**（client-indexed seeding）——可复现性的地基。
4. **四级验证方法论 + 预注册预测**——他们的加速数字是单点报告，没有对照组结构。
5. **windowed per-turn 训练**与 per-turn 样本粒度——message-native 架构给不了的能力。
6. **eval 摆放研究**（worker/parallel/shared/inline 的排名及其随环境翻转的机制）。

## 7. 如果要设计独立的 Agent RL framework（WebShop/ALFWorld/文本环境类）

> 目标画像：**非联邦**、单机到少节点、1.5B–7B 模型、GRPO/PPO、多 turn 文本环境
> （WebShop/ALFWorld/ScienceWorld/TextCraft/BabyAI-text…）、windowed 与 concat 都要、
> 科学可复现优先、同时想要评测与轨迹蒸馏出口。

### 7.1 第一决策：边界画在哪 → **双层 API**

环境重量级决定边界粒度（§1 的结论），而文本环境全部落在"毫秒级 step"一侧，所以：

```
Layer A（RL 主路径，token-native，= FedAgent 的胜利成果）
  trainer 侧 asyncio agent loop ── step 级 HTTP ──► env service（pool + K replicas + 幂等 step）
  token ids/response_mask 直接产自推理引擎；windowed/concat 是 harness 插件

Layer B（统一负载门面，message-native，= NanoRollout 的胜利成果）
  POST /run{env, seed, harness, model_endpoint} ──► rollout worker（服务侧跑 Layer A 的同一循环）
                                                    ──► RunResponse{reward, messages, metrics}
  用途：评测、蒸馏轨迹工厂、外部 trainer 接入；可选 TITO 式捕获垫片给要 token 的外部 trainer
```

**关键点：B 建立在 A 之上而不是并列**——rollout worker 内部就是 A 的 agent loop + env client，
只是模型调用换成 OpenAI 兼容客户端。这样两层共享 harness、env 协议、可靠性契约与遥测，
不产生两套真相。RL 走 A（token 保真零成本），其他一切走 B。

### 7.2 分层与模块清单（LOC 按现有代码估计）

```
textrollout/                        # 名字随意
├── envs/        BaseTextEnv 协议 + registry + HTTP 客户端      ← 直接提升 fedagent/envs/* (~1.3k 行)
├── service/     单一 FastAPI 服务模板 + EnvAdapter 插件         ← 合并两个 server.py (~500 行模板 + 每环境 ~150 行 adapter)
│               （pool、per-session 锁、幂等重放、K-replica 启动器、/spec 清单缓存、可选准入门）
├── harness/     prompt 策略注册表：windowed / concat / react…   ← 从 env client + legacy_prompts 抽出 (~300 行新写)
├── loops/       token-native 循环内核 + verl 0.8 适配器          ← 提升 agent_loops/* (471 行)；内核与 verl 钩子分离
├── rollout/     Layer B：/run 门面 + RunRequest/Response(含 seed!) + AgentMetrics + TITO 垫片  (~500 行新写)
├── trainer/     单 run 驱动：persistent engine + 任务流 + eval 摆放（worker/parallel）          ← 从 run_fed 抽非联邦部分 (~400 行)
├── probes/      两 GPU 数 timing 探针、等价性校验（K=1 逐字节、max|Δ|）                        ← 从 tools/ 收编
└── configs/     环境 spec + 训练 recipe
```

约 3.5–4k 行，其中 ~2.5k 行从 fedagent 近乎原样提升。FedAgent 本体此后把该框架当依赖，
`run_fed.py` 只剩联邦编排（选客户端、FedAvg、轮循环）。

### 7.3 接口草案（把两家的正确答案定成协议）

```python
class BaseTextEnv(Protocol):                      # = 现有协议 + 显式种子契约
    async def reset(self, seed: int) -> tuple[Obs, Info]      # seed 必填，顺序无关
    async def step(self, action: str) -> tuple[Obs, float, bool, Info]   # info 必含 success
    # HTTP 侧不变：/create(阻塞) /reset /step(step_id 幂等) /close + /spec(能力清单, 可缓存)

class Harness(Protocol):                          # NanoRollout 的层次 × FedAgent 的语义
    def build_prompt(self, task, history, obs) -> list[Message]   # windowed=滑窗, concat=全量, …
    def parse_action(self, text) -> str
    sample_granularity: Literal["episode", "turn"]                # windowed ⇒ turn

class RunRequest(BaseModel):                      # Layer B，抄 NanoRollout 并补齐他们的缺口
    env: str; seed: int                           # ← 他们没有 seed，文本环境必须有
    harness: str = "windowed"
    model: ModelEndpoint                          # base_url/api_key 或 "internal"(走 A 层引擎)
    sampling: dict; max_turns: int; timeout_s: int
class RunResponse(BaseModel):
    reward: float; success: bool; exit_status: str        # finished/max_turns/timeout/error
    messages: list[Message]; agent_metrics: AgentMetrics  # turns/model_time/env_time/…
    tokens: TokenTrace | None                             # TITO 垫片开启时才有
```

### 7.4 设计规则清单（两边伤疤的蒸馏，直接写进框架文档）

1. **边界粒度由 env.step 延迟决定**：毫秒级 → step 级边界 + trainer 侧 loop；秒级以上 →
   episode 级边界 + 服务侧 harness。文本环境框架的默认是前者，但 Layer B 保住后者的出口。
2. **token 在链路上永不落地为文本再回来**（RL 主路径）；message-native 出口是门面不是主路径。
3. **可靠性语义跟随边界粒度**：step 级边界必须幂等（`step_id`）+ sticky + 阻塞 create；
   episode 级门面则要 exit_status 归一化 + 超时优雅降级（graceful→宽限→强杀，抄 Ray 那套语义）。
4. **并发无界在客户端、有界在服务端**；扩容手段是 **K 个进程**（拆锁/GIL），不是更大的 pool
   （对照实验已证伪 pool 扩容）。
5. **每个共享宿主资源带 per-job 唯一名**（Ray job id、rendezvous 端口、tmp socket、端口带）——
   cluster-per-job 结构的整类 bug。
6. **eval 只读 → 自由搬动**：worker（热引擎）/parallel（错开 GPU）默认可选；eval 永远不该
   出现在训练关键路径上。
7. **种子契约一等公民**：`seed` 进 RunRequest 与 reset()，client/round/episode 索引化，
   顺序无关是并行化的前提。
8. **遥测双面**：trainer 侧 per-stage `timing_s` + episode 侧 AgentMetrics；两 GPU 数探针
   （flat gen ⇒ env-bound ⇒ 加 replicas）做成内置命令。
9. **partial rollout 是预留扩展点而非补丁**：per-episode 死线 + 截断策略 knob
   （drop / keep+penalty / keep）；windowed 的 turn 粒度样本让部分轨迹天然可训练——
   这是本框架相对 message-native 架构的结构性优势，两边的长尾数据都证明这堵墙必然撞上。
10. **等价性验证内置**：任何加速 knob 必须有 K=1/off 的逐字节回退路径 + `max|Δ|` 检查脚本;
    加速声明按机制/对照/组件/端到端四级出证据。

### 7.5 明确不建的东西（同样重要）

- **不建容器/云后端与资源调度**（Ray、resource hints）——文本环境用不上，等真有秒级环境再把
  `service/` 后端换掉（接口已留缝）。
- **不建多 harness 生态**——windowed/concat/react 三个够起步；harness 是注册表，生态让它自然长。
- **不 fork trainer**——verl 适配器走扩展点；monkeypatch（windowed_manager 那两处）随 verl
  版本钉死并集中在一个文件里，升级时只审一处。
- **不做全历史之外的上下文魔法**（摘要/压缩）——那是 harness 插件的事，不进内核。

### 7.6 迁移路径

1. 先抽 `envs/ + service 模板`（纯搬运+合并孪生，风险最低，立刻回报第三个环境的接入成本）；
2. 抽 `loops/` 内核与 verl 钩子分离（windowed monkeypatch 集中化）；
3. 新写 `harness/`（把 `_memory`/legacy_prompts 收编）——此时 fedagent 改为依赖该框架，跑一次
   等价性校验（byte-identical 预期）；
4. 新写 `rollout/`（Layer B 门面）——蒸馏与外部评测出口上线；
5. `trainer/` 单 run 驱动收尾（persistent engine + eval 摆放），联邦逻辑留在 fedagent。

## 8. 结论

NanoRollout 验证了 FedAgent 架构直觉的普适性（环境与训练解耦、环境侧独立扩容、eval 白嫖并行），
也暴露了粗边界的代价（token 保真靠修补、无幂等、无部分轨迹、无种子契约）。FedAgent 在文本环境
这个重量级上的选择——step 级边界、token-native、幂等契约、四级验证——没有一条需要因为看到
NanoRollout 而推翻；真正该抄的是它的**组织学**：极小 core、插件化 harness/adapter、统一负载门面、
规范化遥测。两者合并的形态就是 §7 的双层 API 框架：**RL 走 token-native 的细边界，其余负载走
message-native 的粗边界，harness 独立成层，可靠性语义跟随边界粒度。**

## 出处

- FedAgent：[agent_rl_design.md](./agent_rl_design.md)、[acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md)、
  [acceleration_cross_env.md](./acceleration_cross_env.md)；代码 [agent_loops/](../agent_loops/)、[envs/](../envs/)、[fed/run_fed.py](../fed/run_fed.py)。
- NanoRollout：[本地克隆](../../others/NanoRollout/)（core/server.py、core/scheduler.py、core/models.py、
  harness/agents/swe/base.py、envs/shell_env/base.py）；[博客存档](../../others/NanoRollout/NanoRollout%20Scale%20digital%20agent%20rollouts%20without%20p%20a9e76ae9fc48826987f781378bfbf1e6.md)。
- miles TITO 集成：[cocoa-org/miles/examples/nanorollout](https://github.com/cocoa-org/miles/tree/main/examples/nanorollout)
  （generate_with_nanorollout.py、proxy/tito_server.py、proxy/tito_state.py、loss.py；2026-07-01 shallow clone 审读）。
- 数字出处：FedAgent 侧全部为 GPU 实测（见 tier1 报告 §5 与预测记分卡）；NanoRollout 侧为其博客
  自报数字（Figure 4/8/9 及正文表格），未独立复现。
