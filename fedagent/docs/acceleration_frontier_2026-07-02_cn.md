# 加速前沿研究（2026-07-02）—— 还剩什么、更多异步值不值、verl 0.8 还有什么可用

> **本文档回答的三个问题**,提出于 Tier-1 副本分片落地之后
> ([07-01 报告](./acceleration_tier1_report_2026-07-01_cn.md)):
> 1. **FedAgent 还能再加速吗?** 能 —— 量化后仍剩 ~1.6–2×,但位置移动了:前沿如今是
>    **轮间管道**(2412 s 整跑中 ~800–1000 s 可解决),而不是训练 step。step 内的 config 面
>    本轮已**测量至穷尽** —— 包括两个被证伪的候选。
> 2. **(更多)异步有用吗?** 轨迹级异步早已是主干且已*饱和*;剩余的安全异步 ≈ 0(剩下的每个
>    barrier 都是数据依赖)。下一层异步(相位级重叠 / one-step-off)**在 verl 0.8 里存在且
>    production-ready —— 但它是 off-policy 的**,即需要明确签核的科学性改动。
> 3. **verl 0.8 还有哪些特性可用?** 已穷尽式审计(§5):安全清单几乎收割完毕;两个未试过的
>    config 杠杆本轮已探测 —— **`use_dynamic_bsz` 被证伪(反而慢 +8–11 %);`use_fused_kernels` = WS −6.5 % 点缀(等价性已验证 §8)、ALF 持平** —— 只剩 offload
>    调优与签核层(one-step-off、rollout-logprob 复用)是仅存未探索的 verl 特性。
>
> 配套阅读: [acceleration_cn.md](./acceleration_cn.md) §9、[07-01 Tier-1 报告](./acceleration_tier1_report_2026-07-01_cn.md)、
> [agent_rl_design_cn.md](./agent_rl_design_cn.md)(异步模型 §4)、[acceleration_cross_env_cn.md](./acceleration_cross_env_cn.md)。
> 常量: 1.5B,GRPO G=8,windowed,batch 8×8;4×H100 qgpu3021;探针 = 1 step、单次运行(±5–10 %)。

---

## 1. 方法 —— 三个证据来源,然后单变量探针

1. **相位分解**: 按日志时间戳分解当前最优端到端 run(`worker_r8`,2412 s)—— 非 step 的
   ~1900 s 究竟去了哪里。
2. **有效 config 提取**: 来自 run 日志的 config dump —— 今天*实际生效*的性能 flag 是哪些
   (而不是默认值声称的那样)。
3. **verl 0.8 源码树审计**(`others/verl`)—— 每一个性能特性、其默认值、成熟度,
   以及是否触及训练算法。

随后,每个由此得出的假设都用**对照已测基线的单变量 A/B、且预测先行写下**的方式探测 ——
与 07-01 campaign 相同的纪律,而这一轮它真的起了作用:最大的那个预测错了(§4)。

## 2. Q1 —— 剩余时间在哪里

### 2.1 端到端预算(worker_r8 = 2412 s,已相位分解)

| 相位 | 实测 | 修复 | 可解决 |
|---|---|---|---|
| 4 个训练 step | 508 s | (Tier-1 之后;见 §2.2) | — |
| 服务 warm,round 1(24 个副本进程各自遍历 8810 个游戏的目录) | ~270 s | **manifest 缓存**(遍历一次,以后加载一个文件) | → 数秒 |
| **服务在 round 2 重复 warm** —— 同样的 16 个副本被原样重启 | ~200–250 s | **服务跨轮持久化**(均匀 shard 与轮无关) | ~全部 |
| **最终 eval 以冷子进程运行**(热 worker 在最后一个聚合模型被评分前就被拆除) | ~330 s | 拆除前在热引擎上多跑一次 `_validate` | ~250 s |
| 2 × FedAvg + HF merge | ~260 s | 直接 shard 加载(训练路径上跳过 `model_merger`) | ~150 s |
| 训练器冷启动 + r0/r1 热 eval + 拆除 | 其余 | 部分(拆除瘦身) | 一些 |

**2412 s 中有 ~800–1000 s 是可解决的管道** —— 没有一项触及训练数学。结合 §2.3,
worker 探针规模的整跑预计 **2412 → ~1300–1500 s**(相对 06-30 基线的累计:
3509 → ~1400 ≈ **2.5×**)。

### 2.2 step 内预算现已关闭(本轮探针)

Tier-1 之后的 step = gen(episode 关键路径)+ GPU 计算:

| 4×H100 每步 | gen | old_log_prob | ref | update_actor | step |
|---|---|---|---|---|---|
| ALFWorld `g4_r8`(基线) | 51.7 | 15.2 | 14.1 | 43.7 | **127.6** |
| + `use_dynamic_bsz` | 49.5 | 21.2 | 18.7 | 49.2 | **141.7(+11 %)❌** |
| + `use_fused_kernels`(triton) | 44.0¹ | 16.6 | 13.9 | 52.2 | **129.7(≈ 持平)** |
| WebShop `g4_p64r4`(基线) | 35.7 | 10.8 | 10.6 | 26.0 | **82.2** |
| + `use_dynamic_bsz` | 33.8 | 10.8 | 10.6 | 31.2 | **88.9(+8 %)❌** |
| + `use_fused_kernels`(triton) | 31.3¹ | **7.6(−30 %)** | **8.3(−22 %)** | 27.0 | **76.9(−6.5 %)** |

¹ gen 在 run 与 run 之间波动 ±15 %(ALF 44–52 s);fused kernels 不触及 rollout 路径 ——
gen 的差值请当噪声读,olp/ref 的差值才是(已确认生效的)fused 效果。数值等价性已于当日
验证(§8):全循环 off/on A/B → 最终聚合 actor 的 max|Δ| = **1.116e-5** ≤ 1e-4 的 bar。

- **gen 地板**(52–66 s ALF / ~34 s WS)= 最长 episode 的关键路径(~50 turns ×
  (LLM 0.2–0.3 s + env 86 ms/K + HTTP));实测过了 K=4–8 再加副本无用。
- **GPU 算力地板**: dyn-bsz 证伪(§4)表明,73 s(ALF)/ 47 s(WS)的 GPU 项在这个
  模型尺寸下已经是 FLOP/通信瓶颈 —— 不是调度瓶颈。没有任何 config 旋钮能撼动它;
  只有每 client 更多 GPU(#3 组合)或模型/算法层面的改动才行。

### 2.3 剩余杠杆清单(已排序)

| 杠杆 | 攻击的项 | 预计收益 | 成本 | 科学性 |
|---|---|---|---|---|
| 服务跨轮持久化 + manifest 缓存 | 管道 | ~400–500 s/run | run_fed 小改 + 服务 warm 缓存 | 安全 |
| 最终 eval 走热引擎 | 管道 | ~250 s/run | persistent-runner 小改 | 安全(eval 只读) |
| 直接 shard 加载(循环内跳过 HF merge) | 管道 | ~150 s/run(随轮数增长) | 中等 | 安全(精确的加载路径) |
| #3 × 副本(run_fed 里的 parallel-round launcher) | 训练段 | steps 的 ~−18 % | 中等 | 安全(FedAvg 与顺序无关、seed 按 client 索引) |
| 多节点 #3 | 轮 ∥ client | ~随节点数线性 | launcher + 算力分配 | 安全 |
| one-step-off(verl `experimental/one_step_off_policy`) | gen∥train 重叠 | step → max(gen, GPU) ≈ −35 % | GPU 划分 + config | **off-policy —— 需签核** |

## 3. Q2 —— 异步的裁定

这个系统里的异步有三层;审计 + 测量对每一层都给出了结论:

1. **轨迹级(已收割,已饱和)。** 64–512 个 episode 协程把 env 延迟与 LLM 延迟互相藏进
   彼此之下;vLLM 把一切动态合批;`agent.num_workers=8` 已是 verl 默认值,且瓶颈在 env、
   不在 worker。这一层正是 gen 恰好等于 episode 关键路径、不多不少的*原因* ——
   已经没有任何可再调度掉的东西。
2. **流水线级(安全余量 ≈ 0)。** 剩下的每一个同步都是**数据依赖**:FedAvg 需要该轮的
   全部 client(联邦语义);round r+1 需要 model_r。eval 是唯一可移动的相位,且已经移出
   关键路径(worker/parallel 模式)。服务 warm-up 重叠(杠杆 #2)在服务持久化(§2.1)之后即失去意义。
3. **相位级(存在,但要付科学代价)。** verl 0.8 自带
   `verl/experimental/one_step_off_policy` —— 在单独的 GPU 上生成第 t+1 个 batch,同时第 t 个
   batch 在训练;按上游说法 production-ready(他们的 DAPO-32B 案例:−40 %)。用在这里,
   step 会从 `gen + GPU` 变为 `max(gen, GPU)` ≈ 76 s(ALF)。**但它让 GRPO 变成 one-step
   off-policy** —— 在论文复现红线之外。`fully_async_policy` 是同一笔交换,只是更年轻。
   裁定:*在 on-policy 红线之内,异步已经做完;下一层异步是算法决策,不是工程决策。*

## 4. dyn-bsz 证伪(值得记录 —— 它关闭了一整类假设)

**假设**(来自有效 config 审计):rmpad 已开而 `use_dynamic_bsz` 关闭时,ALFWorld 把
~3200 个 windowed 行按每个 4 行(~2.2 k tokens)、共 ~200 个 micro-batch 来训练,而额度有
16 k —— "GPU 项是调度瓶颈;token 打包能把它砍掉 ~2×。"

**结果:** 两个 env 上都更慢,且在全部三个 GPU 组件上方向一致
(ALFWorld 上 olp +40 %、ref +32 %、update_actor +13 %;step +11 % / +8 %)。

**假设为什么错了:** 对账指向相反方向 —— 43.7 s / 200 个
micro-batch ≈ 每个 2.2 k token 的 micro-batch 218 ms,对开着梯度检查点的 1.5B
forward+backward 而言这已经是 FLOP 主导。打包到 16 k 一个 FLOP 都省不下,反而加上
Karmarkar-Karp 均衡、concat 拷贝,以及**破坏 torch.compile/CUDA-graph 复用的 shape 抖动**。
教训: *"micro-batch 没喂饱"必须拿每 batch 的毫秒数来检验,而不是拿 token 额度。*
这次证伪关闭了整个"config 层 GPU 算力杠杆"假设类 —— GPU 项是实打实的工作量。

## 5. Q3 —— verl 0.8 特性审计(三层)

**✅ 安全且有用 —— 本轮之后的状态**

| 特性 | key | 在本项目的状态 |
|---|---|---|
| 移除 padding(rmpad) | `model.use_remove_padding` | 已开 —— 早已收割 |
| 动态 token 打包 micro-batch | `actor.use_dynamic_bsz`(+ref/rollout) | **已探测,被证伪**(§4) |
| 融合 logprob/entropy kernel | `model.use_fused_kernels` + `model.fused_kernel_options.impl_backend=triton` | **已探测**: WS **−6.5 %**(olp −30 %、ref −22 % —— 带宽机制是真实的),ALF **持平**(+2 %)。**等价性已通过**(§8: max|Δ|=1.116e-5 ≤ 1e-4)→ **WebShop 上可采用**;ALFWorld 上跳过。 |
| offload 调优 | `param/optimizer/grad_offload` | 1-GPU ref 爆表(108 s)的候选;未测,"不再有 1-GPU client"之后优先级低 |
| seqlen 均衡 | `balance_batch` | 已开 |
| 前缀缓存 / chunked prefill / CUDA graph / sleep / dummy-load / `free_cache_engine` | rollout.* | 全部已开 |
| 权重同步 bucket | `update_weights_bucket_megabytes=2048` | 同步只有 0.3–0.9 s —— 不是瓶颈 |

**⚠️ 存在,但改变科学性(签核层)**
- `experimental/one_step_off_policy`(§3)—— 已知剩余最大的 step 级收益(−35 %),
  off-policy 一步。
- `rollout.calculate_log_probs` + `actor.use_rollout_log_probs`(+ verl 的重要性采样
  修正 helper)—— 跳过 olp 重算(~15 s),但把数值路径在 vLLM↔FSDP 之间对调。
- `over_sample_rate` —— 中止掉队的 episode:**让采样偏向短 episode**;不要用。
- LoRA / QAT-FP8 / MTP-speculative —— 分别改变训练数学、量化采样,或面向长回复
  (~100 token 的 turn 让投机解码毫无意义)。

**➖ 在此不适用:** fsdp2 迁移(1.5B 下收益边际)、`multi_turn.*` tool-calling 配置
(rollout 形态不同)、disaggregation/layered-summon(多节点大模型特性)。

## 6. 预测记分卡(本轮)

| 预测 | 实测 | 裁定 |
|---|---|---|
| dyn-bsz: ALF step 127.6 → 85–105 s | 141.7(+11 %) | ❌ **证伪** |
| dyn-bsz: WS step 82.2 → 60–75 s | 88.9(+8 %) | ❌ **证伪** |
| fused-kernels: olp+ref 收缩(带宽机制) | WS olp **−30 %**、ref **−22 %** → step −6.5 %;ALF 持平(+2 %) | ⚠ 机制确认,幅度 = 点缀 |
| fused-kernels: 数值等价(预期 ~1e-5) | 全循环 A/B 最终聚合 max|Δ| = **1.116e-5** | ✅ 等价 bar 通过(§8) |
| 管道 ≈ 端到端的 1/3 可解决 | §2.1 相位表(2412 中的 ~800–1000 s) | ✅ 已量化 |
| 安全异步余量 ≈ 0 | §3 依赖分析 | ✅(分析性) |

"GPU 项很容易撼动"连续两次落空 —— step 已处在它的 config 层地板上。07-01 报告的记分卡
纪律,正是这件事成为一个*发现*而非难堪的原因:每一次证伪都在消耗一次生产 run 之前
关闭了一个假设类。

## 7. 路线图现在的位置

```
已完成 : 冷启动(#4) · eval 摆放(modes) · env 串行化(replicas) · 并发(fixes)
本轮   : step 内 config 面已关闭(rmpad 已开; dyn-bsz 被证伪; fused = WS 点缀已等价/ALF 持平)
下一步 : Tier-2 管道(~800-1000 s/run, 全部安全)  →  #3×replicas launcher(steps −18 %)
       →  多节点 #3(campaign 吞吐)
签核层 : one-step-off(step −35 %, off-policy)  ·  rollout-logprob 复用(−15 s, 数值路径)
地板   : gen = episode 关键路径(52-66 s) + GPU 算力 = 真实 FLOPs(73 s ALF / 47 s WS)
```

## 8. 当日加注(2026-07-02 下午)—— 等价性 + paper 几何修正

上午的研究之后,由"真的*完整*测试了吗?"这一质询触发,当天下午落地了四个跟进:

1. **fused-kernels 等价性 —— 通过。** 在既有等价 rig 上做全循环 off/on A/B(TinyGuess,
   2 client × 2 round × 2 step,GRPO,seed 42,子进程两臂**只差**那两个 fused override;
   `accel/dev/fused_ab_{off,on}.yaml`):最终聚合 actor 的 **max|Δ| = 1.116e-5**(mean 2.2e-7)
   ≤ 1e-4 的 bar —— 与历史上 persistent-trainer(1.13e-5)、PPO(1.16e-5)两次 A/B 同量级。
   WS 的 −6.5 % 从"已测量"升级为**可采用**。
2. **paper 几何修正(一个 stale-doc 陷阱)。** ALFWorld 探针并不是"从 paper 8192 裁剪来的":
   真实的 paper 几何是**有界 windowed 模板** —— prompt 2048 / response cap **512** /
   max_model_len **2560**(`gen_paper_configs.py:148`)。`config/envs/alfworld.yaml` 与
   `reproducing.md` 里的 16384/8192 描述的是*已废弃*的 concat 设计(两处今日已修)。
   windowed 实测:response 均值 ~100 tok,512-cap 截断率 0.13 %。推论:−57 %/−49 % 的
   ALFWorld 结果**一直就在 paper 几何上**(探针只是 cap 放得更宽)。
3. **1×H100 @ 真 paper cap**(`accel/alfworld/alf_scale_g1_paper_r1.yaml`):step **514 s**
   (gen 212 / olp 52 / ref 105 / update_actor 139),对比旧 4096/6144 cap 的 534 s ——
   cap 的选择无关紧要,与 env-bound 模型的预测完全一致。(K=4 重复臂随分配回收未跑完。)
4. **端口带守卫 —— paper 规模下的首次实战拦截。** 首批*真* paper-config 启动(100 客户端 ×
   K 副本)在 21 s 内触发 Tier-1 冲撞守卫:per-client 服务端口带是
   `[base, base + total_clients×K)` —— ALFWorld K=8 时宽达 800 个端口 —— `*_val_port` 落在
   带内会在任何 GPU 工作之前被拒。部署规则:**把 `*_val_port` 放在 `base + total_clients×K`
   之后**(已修:ALF 43100、WS 10700)。ALFWorld 的 wiring run 当日落地
   (`accel/alfworld/paper_alf_wiring_r8.yaml` —— 真 ALFWorld paper config 的**首次**运行,
   携带已采用栈): **rc=0,总计 3719 s** —— round 1(24 副本 warm + 首评 + 2 客户端 × 3 步 +
   轮评 + FedAvg/merge)**1766 s**、round 2 **1375 s**、冷终评 **578 s**;val success
   0.0429 → **0.1143**(n=140);**70 轮投影 ≈ 27 h**(test_freq=5)≈ WebShop 的 2.2×。
   WebShop r4 复跑(`accel/webshop/paper_ws_wiring_r4.yaml`)撞上 walltime,排入下一个窗口。

## 9. 出处

- 探针: `accel/alfworld/alf_scale_g4_r8dyn.yaml`、`alf_scale_g4_r8fused.yaml`;
  `accel/webshop/ws_scale_g4_dyn.yaml`、`ws_scale_g4_fused.yaml`(单变量,对照 07-01
  基线 `g4_r8` / `g4_p64r4`)。日志: gitignored 的 `runs/{alf,ws}_{g4dyn,fused}.log`。
- 相位分解: `runs/alf_em/worker_r8.log` 的时间戳。
- fused 等价 A/B(§8): `runs/fused_ab/{off,on}.log`;对比 =
  `compare_fsdp_checkpoints.py --a <off actor> --b <on actor>`,作用于 `round_2/aggregated`(旗标必填)。
- 有效 config 提取: `runs/alf_scale/g4_r8.log` 的 config dump。
- config key 陷阱(代价是 3 次失败启动): fused 后端是 `model.fused_kernel_options.impl_backend`
  (dict 字段,普通 override —— 不带 `+`);`fused_kernels_backend` 是 `apply_monkey_patch`
  里的*函数参数*,不是 config key,而且这个 verl 的 `HFModelConfig` 会把它当 kwarg 拒收。
- verl 审计: `others/verl` @ 固定住的 0.8 树(`verl/workers/config/*.py`、
  `verl/experimental/one_step_off_policy/`、`verl/utils/seqlen_balancing.py`)。
- 英文版: [acceleration_frontier_2026-07-02.md](./acceleration_frontier_2026-07-02.md)
