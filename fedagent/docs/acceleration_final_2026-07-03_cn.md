# FedAgent 加速 —— 最终定稿记录（2026-07-03）

> **本文自包含。** 不需要读任何其它文档就能回答四个问题：*最终加速方案是什么 → 快了多少 →
> 为什么快 → 是怎么一步步试出来的（包括所有失败的尝试）*。§6 给出各章节对应的详细文档，
> 供想看原始日志和逐实验细节的读者。
>
> 全文常量：Qwen2.5-1.5B 策略模型、GRPO 每 prompt G=8 采样、paper 的 windowed rollout、
> 单节点 4×H100（qgpu3021）、stock verl 0.8 + 薄 `fedagent/` overlay（不 fork）。
> "paper config" = `uniform/1.5B/main/grpo/<env>`：100 个客户端、每轮抽 2 个、每客户端轮
> 3 个 optimizer 步、70 轮，WebShop 与 ALFWorld 两个环境。

---

## 0. 三十秒背景（读懂后文的最小知识）

FedAgent 做**联邦**智能体训练：每一*轮*抽样若干客户端，各自在自己的环境切片上微调一份
模型副本，服务器再对权重做平均（**FedAvg**）。原始实现的一轮长这样：

```
第 r 轮：
  1. 启动环境服务          每客户端一支环境服务器舰队（WebShop 商品池 / ALFWorld 游戏集），
                          每轮都是全新进程
  2. 逐个客户端（串行）：
       拉起一个训练子进程：  建 Ray 集群 + FSDP 模型引擎 + vLLM 推理引擎、加载权重
                                                            ← 「冷启动」
       rollout + 3 个 PPO/GRPO 步   （多轮 episode 打客户端自己的服务）
       存 FSDP checkpoint 分片，整体销毁
  3. FedAvg 客户端 checkpoint      → 聚合 FSDP 分片
  4. model_merger                  → 分片转 HuggingFace 目录（「HF 导出」）
  5. 评估                          再拉一个子进程（又一次冷启动）在验证集上给聚合模型打分
  6. 停环境服务
最终：再一个冷子进程给最后的模型打分（「终评」）
```

两个结构性事实决定了下面的一切：

- **每轮的 GPU 计算量很小**（2 客户端 × 3 步的 1.5B 模型），所以一切*非*训练数学的东西——
  进程启动、引擎构建、环境目录遍历、格式转换、冷评估——在墙钟里占比巨大。
- **科学 bar 是等价而非"差不多"：** 每个优化必须让训练的*字节流*不变（相同 episode、相同
  batch、相同种子），以最终 checkpoint 对比 `max|Δ| ≤ 1e-4` 为判据。改变算法的东西一律
  出局（或显式隔离为选择性开关，§4.8）。

---

## 1. 最终方案

以下全部以**各自独立、默认关闭的配置旋钮**交付——一个旋钮都不开时，代码路径与原始实现
逐字节一致。采纳的两套配方：

```yaml
# ---------- ALFWorld paper 运行 ----------
use_persistent_trainer: true      # 每个 RUN 一个训练进程，而非每个客户端一个   (§4.1)
persistent_scope: cross_round     #   …并且跨轮存活                             (§4.1)
eval_mode: worker                 # 评估跑在这个进程的热引擎上                  (§4.2)
alfworld_replicas: 8              # 每客户端环境服务分片 ×8                     (§4.3)
alfworld_manifest_cache: true     # 缓存 8810 游戏的磁盘遍历                    (§4.6)
service_scope: run                # 环境服务舰队跨轮保温                        (§4.6)
final_eval_mode: worker           # 最终模型也在热引擎上打分                    (§4.6)
hf_export: final                  # 跳过逐轮 分片→HF 转换                       (§4.6)

# ---------- WebShop paper 运行 ----------
use_persistent_trainer: true
persistent_scope: cross_round
eval_mode: worker
service_scope: run
final_eval_mode: worker
hf_export: final
client_overrides:
  - +actor_rollout_ref.model.use_fused_kernels=True   # 融合 log-prob/熵 CUDA 核 (§4.4)
# 注意：WebShop 不开 replicas —— 真实配置下实测打平（§4.5）
```

**刻意不进配方的**（每一项都试过、测过，见 §4）：`parallel_clients` lanes（打平：−2%）、
WebShop replicas（paper config 打平）、`use_dynamic_bsz`（反而*更慢*：+8~11%）、轮内更深
的异步（已饱和：剩余的每个屏障都是数据依赖）、以及 verl 的 `one_step_off_policy`（能跑、
实得 −10%——但改变算法；作为显式离策略的 ADDITIONAL OPTION 保留，§4.8）。

## 2. 快了多少

三个参照点，全部在真实 paper config 上（70 轮、每 5 轮评估一次）：

| 栈 | ALFWorld 70 轮 | WebShop 70 轮 | 是什么 |
|---|---|---|---|
| A. 原始子进程栈 | ≈ 41 h（重构¹） | ≈ 32 h（重构¹） | §0 的流程图原样 |
| B. wiring 栈（persistent + worker 评估 + ALF replicas） | **29.3 h**（实测块²） | **20.4 h**（实测块²） | §4.1–4.3 之后的栈 |
| C. **最终配方** | **16.7 h** | **9.4 h** | B + §4.4–4.6 的旋钮 |

- **C vs B：−43%（ALFWorld）、−54%（WebShop）**——真实 paper-config 2 轮实测 + 下方公式投影。
- **C vs A：≈ ×2.5（ALFWorld）、≈ ×3.5（WebShop）**——整场战役的总账。

¹ A = B 加回 B 已经消掉的两笔实测成本：每次客户端拟合 ~310 s 的引擎冷启动（直接测得：
探针的子进程客户端从拉起到第一个 optimizer 步耗时 ~310 s）×2 客户端 ×70 轮 ≈ +12 h，
再加冷子进程评估。
² 投影公式，每一项都是同日 2 轮实测块：
`T(70) ≈ 一次性 + 70 × 稳态轮 + 14 × 评估 + 终评`，如 ALFWorld combo
= 791 + 70×762 + 14×389 + 389 s。2 轮运行本身：ALFWorld 3719 → 3202 s、WebShop
2802 → 2309 s（差值比投影小，因为一次性成本主导 2 轮短跑；真正可扩展的数字是
**稳态轮：762 vs 1125 s** 与 **402 vs 905 s**）。

C 相对 B 的收益从哪来（以 ALFWorld 每稳态轮计）：

| 消掉的成本 | 机制 | 证据 |
|---|---|---|
| 每轮 ~250 s 服务重热 | 舰队跨轮存活（`service_scope: run`） | A/B 臂：−16% |
| 每波 ~146 s 游戏目录遍历 | manifest 缓存 | A/B 臂：−18%，日志 24/24 HIT |
| 每轮 ~40–60 s 分片→HF 转换 | `hf_export: final` + 分片直载 | A/B 臂等价 8.8e-6 |
| 终评 578 → 389 s | 热引擎 eval-only 计划 | worker 日志："scored on the hot engine; no fit" |

## 3. 为什么答案长这样

剖析（§4.4）反复指向同一个结论：**在这个模型尺寸下，流水线在反复支付固定成本，而不是在
算数学**。四个瓶颈家族，按发现和消除的顺序：

1. **冷启动（最大头）。** 每次客户端拟合和每次评估都要建 Ray 集群、FSDP 引擎、vLLM 引擎、
   从磁盘加载权重——小规模运行里占墙钟 76–88%，1.5B 下每次 ~310 s。*解法：每个 run 一个
   持久化 trainer；客户端变成喂给它的计划文件；评估直接用它已经热的引擎。*（§4.1–4.2）
2. **环境管道。** ALFWorld 的环境服务让所有 agent 挤一把 TextWorld 解释器锁（86 ms ×
   每批数千步全串行）、每次启动都在 GPFS 上遍历 8810 个游戏目录（146 s × 24 服务 ×
   每轮）、舰队每轮拆了重建。*解法：×8 副本分片；manifest 缓存；run 级舰队。*（§4.3、§4.6）
3. **交接开销。** 每轮把 FSDP 分片转成 HF 目录只为下一轮能加载，最终模型还要再起一个冷
   进程打分。*解法：分片直接交接（worker 的 checkpoint manager 仅模型加载），HF 只在最后
   导出一次；最终模型在热引擎上打分。*（§4.6）
4. **GPU 本身（基本已经没问题）。** 步内唯一存活的杠杆是 WebShop 的融合 log-prob/熵核
   （步 −6.5%）。批形状技巧适得其反；把客户端拆到 GPU 半区并行是打平，因为拟合本身
   GPU-bound——2 客户端 × 2 卡 ≈ 顺序 4 卡。更深的异步会破坏 on-policy 等价 bar。
   （§4.4–4.5、§4.7–4.8）

一个贯穿性的安全性理由：采纳的旋钮没有一个碰到模型看到的东西。相同进程或相同输入 →
相同 episode 流 → 相同 batch；checkpoint 对比（§5）证实到测量噪声以下。

## 4. 历程——每一站，包括死胡同

### 4.1 「为什么没变快？」→ 持久化 trainer

迁移到 verl 0.8 正确但不快。插桩显示小规模联邦运行的 76–88% 是引擎冷启动。把 trainer
重造成**一个长驻进程**（客户端以计划文件到达；客户端之间从 base 模型重置权重/optimizer
——保持原实现"每轮全新 optimizer"的语义）在开发 rig 上给出：子进程 **909 s → 持久化
515 s（−43%）→ 跨轮持久化 342 s（−62%）**，full-loop checkpoint 等价 `max|Δ| = 1.13e-5`
——而且是*跨轮穿过 FedAvg 复合*的等价，不只是单客户端。PPO/critic（GAE）同样处理
（critic 引擎重建 + critic FedAvg），GPU 验证通过。

### 4.2 不冷启动的评估 → `eval_mode: worker`

造了四种评估模式并全部 GPU 验证（inline / parallel / shared / worker）。采纳的 worker
模式在持久化 trainer 自己的热引擎上给轮模型打分（跨模式权重等价 3.8e-6/7.6e-6）。路上
根因定位了一次真实崩溃：worker 内评估时 vLLM 还持着旧权重 → 修复是 FSDP→vLLM 权重同步，
不是绕过。windowed rollout（paper 的每轮新 prompt 模式）设为默认；评估节奏语义
（每轮 vs 任务内）钉死。

### 4.3 ALFWorld 的地板 → 副本分片（`alfworld_replicas`）

引擎热了 ALFWorld rollout 还是慢。诊断：所有并行 episode 挤过环境服务里**一把** TextWorld
解释器锁——每 env 步 86 ms、每批 ~3200 步、完全串行。把每客户端服务分成 8 个副本（游戏
划分、agent 路由）后 ALFWorld 步 **298 → 127.6 s（−57%）**、端到端 **−31%**。WebShop
探针提示 −12%；§4.5 在 paper 规模上否掉了它。这一站还发现了后来被守卫现场抓住的运维
规则：100 客户端 × K 副本时服务端口带是 `[base, base+100K)`——验证端口必须放在带外。

### 4.4 前沿研究——剖析、证伪、开清单

大石头搬完后，对真实运行做逐相位分解，找到每次短跑 ~800–1000 s 的*可寻址管道*（服务
预热、重热、merge、冷终评）——即 Tier-2 清单。同样重要的是用探针**证伪**了几个诱人的
GPU 侧想法，而不是凭感觉采纳：

- `use_dynamic_bsz`（token 均衡微批）：**两个环境都更慢**（ALFWorld 步 127.6 → 141.7 s，
  +11%；WebShop 82.2 → 88.9 s，+8%）。GPU 项本来就是 FLOP-bound，形状抖动还破坏
  kernel/CUDA-graph 复用。
- 融合 log-prob/熵核：WebShop 步 **−6.5%**（带宽型项收缩：old-log-prob −30%、ref −22%），
  ALFWorld 打平（+2%）——只为 WebShop 采纳，后经等价验证（1.116e-5）。
- 更多异步：轨迹级已经饱和（瓶颈是 episode 关键路径；剩余每个屏障都是数据依赖）。唯一
  剩下的异步形态——用上一步的 rollout 训练——构造上就是*离策略*：§4.8。

### 4.5 paper-config 现实校验（幸存者过滤器）

探针必要但双向说谎，所以一切在真实 paper config（截断 2 轮）上重新锚定：

- **史上第一次完整 ALFWorld paper-config 运行**：3719 s（r1 1766 + r2 1375 + 冷终评
  578），验证成功率 0.043 → 0.114——70 轮投影 ≈ 29 h，放得进 2 天配额。WebShop wiring：
  2802 s（490/764/905/643）。
- **冷探针对稳态悲观 ~3×**：热引擎稳态步 ~50 s（gen ~10 s），而冷的单步探针 82 s
  （gen 36 s）——跨步前缀缓存 + 暖引擎。探针算术系统性低估长跑收益。
- **WebShop replicas：−12% 的探针结论没有幸存**真实配置（打平）——从配方移除。
- 一处陈旧文档修正也在这站：真实 paper ALFWorld 几何是 windowed 2048/512/2560（不是已
  废弃的 16384/8192 concat 设计），所以此前所有探针其实一直就在 paper 几何上。

### 4.6 Tier-2：其余一切，做成默认关的旋钮

§4.4 的四个管道修复加上 lanes 和 one_step_off 成为六个独立旋钮（§1 配方框；启动时强制
组合门）。每个旋钮一对匹配 A/B——两臂只差这个旋钮、种子 42——对比第 2 轮聚合 actor
checkpoint：

| 旋钮 | 墙钟效果（A/B rig） | 等价 max\|Δ\| |
|---|---|---|
| 同配置重跑（对照组） | 白得 −6.5% | **9.293e-5 = 噪声底** |
| `alfworld_manifest_cache` | **−18%**（消掉 146 s/波 ×3 波遍历） | 9.199e-5 ✓ 贴底 |
| `service_scope: run` | **−16%**（无轮间重热） | 9.090e-5 ✓ 贴底 |
| `final_eval_mode: worker` | **−11%**（无冷终评） | 9.241e-5 ✓ 贴底 |
| 四个 Tier-2 全开 | **−24%**（次可加：有重叠） | 8.752e-5 ✓ 贴底 |
| `hf_export: final`（训练面 rig） | 该 rig 上 −37% | **8.825e-6** ✓ |
| `parallel_clients: 2`（训练面 rig） | 该 rig 上 −32%（⚠ 见 §4.7） | **1.144e-5** ✓（经 HF 导出对比） |

对照组那一行本身就是一个发现：两次*完全相同*的运行相差 9.293e-5（GPU 非确定性），所以
全部贴底或更低的旋钮臂**没有引入任何可与噪声区分的偏差**。这个对照组的存在还要感谢一个
被抓住的 bug：第一个 cache 臂静默失效——conda 环境里 editable verl-0.3.1 的 `.pth` 一直
在 `sys.path` 上遮蔽 vendored 的 ALFWorld 引擎（命名空间包解析竞争；用
`sys.path.insert(0, ...)` 修复）——一个独立的潜伏溯源 bug 被这套件冲了出来。

### 4.7 最优组合——以及 lanes 死在 paper 规模

胜出的旋钮组随后按 wiring 基线的同一方法实测（真实配置、2 轮）：**ALFWorld 3202 s vs
3719；WebShop 2309 s vs 2802**，稳态轮 **762 vs 1125 s**、**402 vs 905 s**，热终评
389 vs 578 / 326 vs 643 s——即 §2 的数字。lanes（2 客户端并发在 2+2 卡上）虽然在小 rig
上 −32%、更早的小模型探针 −35%，但在**两个环境都是打平**（−2%）：1.5B 拟合 GPU-bound，
两个 2 卡拟合 ≈ 一个 4 卡拟合跑两次，而 2 卡热终评又吐回了环境重叠的节省。与 WebShop
replicas 同一条幸存者教训：**小 rig 的并发收益不会迁移到 GPU-bound 的 paper config。**

### 4.8 `one_step_off`——保持可选的附加选项

verl 0.8 实验性的 `one_step_off_policy` 在专用 GPU 上生成第 t+1 批、同时训练第 t 批
（步墙钟 → `max(gen, train)`）。在 FedAgent 下把它立起来花了五层修复（hydra 主配置规则
→ 配置拆分；批量整除约束 → 2+2 卡切分；分离式布局断言 → `hybrid_engine: False`；上游的
资源块拷贝 → 在我们入口镜像；逐 token rollout log-probs → 两个 agent loop 现在都输出
`response_logprobs`，顺带解锁 verl 的 rollout-correction 算法族）。探针随即跑通：步
**116 → 65 → 72 s**——生成真的从 step 2–3 消失（`gen ≈ 0`；71/64 s 的 `generate_async`
完全藏在训练下面），对照串行 93.4 s 步：**稳态 −23~31%**。但 FedAgent 的 3 步客户端轮
每轮重付 116 s 灌管：**253 vs 280 s ≈ 实得 −10%**。这一点加上一步离策略（更新批采样自
上一步权重——*科学*变更而非工程变更），把它留在配方之外：`one_step_off: true` 可用、
仅子进程路径、需签核。

## 5. 测量纪律（这些数字凭什么可信）

1. **默认关的旋钮**：什么都不设时走旧字节路径——采纳一个杠杆是配置决定，永远不是代码迁移。
2. **匹配 A/B + checkpoint 对比**：每个旋钮对着只差该旋钮的臂测；等价 = 最终聚合 actor
   `max|Δ| ≤ 1e-4`（FSDP 分片 diff；分片布局不同时用 HF 导出 diff）。
3. **先立噪声底**：同配置重跑（9.293e-5）标定了这块硬件上"等价"二字的物理含义；旋钮偏差
   对照它评判，而不是对照零。
4. **paper-config 幸存者规则**：任何探针收益不在真实配置上幸存就不进配方——这个过滤器
   杀掉了 WebShop replicas（−12% → 0）和 lanes（−35% → −2%）。
5. **稳态优先于冷探针**：长跑结论来自真实 2 轮运行的稳态轮块（冷探针被证明悲观 ~3×），
   用显式公式投影。
6. **val 曲线不是判据**：140 局、近随机成功率的评估被采样噪声主导（等价权重在不同运行
   打出 0.114 vs 0.021）；权重对比才是科学门。

## 6. 出处——每章对应的详细文档

| 章节 | 文档 |
|---|---|
| 持久化 trainer + 评估模式 + 杠杆栈 | [acceleration_report.md](./acceleration_report.md) · 数字：[acceleration_results.md](./acceleration_results.md) · 计划：[acceleration.md](./acceleration.md) |
| ALFWorld vs WebShop 迁移原则 | [acceleration_cross_env.md](./acceleration_cross_env.md) |
| 副本分片深度验证 | [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md) |
| 前沿研究（分解 + 证伪） | [acceleration_frontier_2026-07-02.md](./acceleration_frontier_2026-07-02.md) |
| Tier-2 旋钮、噪声底、combo、one_step_off | [acceleration_tier2_2026-07-02.md](./acceleration_tier2_2026-07-02.md) |
| 运行日志（PM 条目） | `fedagent/EXPERIMENTS.md` |

实现在 `fedagent/fed/run_fed.py`（旋钮、编排）、`fedagent/fed/persistent_task_runner.py`
+ `persistent_patch.py`（持久化 trainer）、`fedagent/agent_loops/`（concat + windowed
loop、log-prob 管道）、`fedagent/envs/*/service/`（副本化环境服务、manifest 缓存）；A/B
配置与驱动在 `tools/verl08_migration/accel/`；原始运行日志在 gitignored 的 `runs/`。

English version: [acceleration_final_2026-07-03.md](./acceleration_final_2026-07-03.md)
