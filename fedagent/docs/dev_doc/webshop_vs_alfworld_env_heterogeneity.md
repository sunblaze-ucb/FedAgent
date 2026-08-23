# WebShop vs ALFWorld:环境级异质性的系统对比

> 状态:定稿(GPU 冒烟与 cell 库两处数字随运行回填)。日期:2026-08-23。作者:canyu + Claude。
> 前置文档:[../heterogeneity.md](../heterogeneity.md)(WebShop 五变体构造参考)、
> [alfworld_env_heterogeneity.md](./alfworld_env_heterogeneity.md)(ALFWorld 四 arm 设计与验证)、
> [alfworld_query_env_decoupling.md](./alfworld_query_env_decoupling.md)(query×env 解耦 / cell 库)。
>
> 本文回答:**同一套"隐藏通道扰动"方法论在两个环境上分别长成了什么样、为什么长得不
> 一样、哪些差异是本质的、论文该如何并排呈现。**
> 生词/概念(绑定率、恒真面、cell、措辞归一化、五层验证等)先查
> [README.md](./README.md) 的概念手册。

## 0. TL;DR

两个环境实现了同一个科学契约——**保持可观测的 τ 分布,扰动不可观测的环境通道,
在共享的无扰动 val 上打分**——但通过完全不同的机制:

- **WebShop** 的 P 是一条**检索流水线**(content → encoding → matching →
  rendering),扰动 = 换流水线的某一级部件;天然全隐藏(agent 从看不到目录)。
- **ALFWorld** 的 P 是一个**符号转移系统**(PDDL domain + problem + grammar 三个
  字符串,逐字节稳定),扰动 = 对这三个字符串做按 client 确定性的锚点改写;但
  admissible commands 喂进 prompt,把大半扰动降为"经后继状态感知"的部分隐藏,
  **完全隐藏的只有动作 effect 与 goal 判定**。
- 由此,两侧的"最强攻击"同构而不同层:WebShop 的 Lookalike 在**内容**里注塞
  奖励陷阱,ALFWorld 的 goal_variant 直接改写**隐藏成功判定**;二者都制造
  π\* 的结构性发散,都预期 GRPO 崩、PPO 部分修复。
- ALFWorld 额外提供一个 WebShop 给不出的对照:**query×env cell 矩阵**
  (τ 构造性置零),反向补齐了 WebShop 那侧只能靠 `task_disjoint` 近似剥离的
  空缺。

---

## 1. 两个环境的 MDP 结构差异(一切设计分歧的根源)

| 维度 | WebShop | ALFWorld |
|---|---|---|
| **P 的载体** | Lucene/BM25 检索引擎 + 商品目录(运行时对象) | `game.tw-pddl` 内嵌的 `pddl_domain` + `pddl_problem`(数据文件字符串,全库字节稳定) |
| **R 的载体** | 引擎端 reward 函数(属性/选项/价格匹配),固定 | `PddlState.check_goal()` 对 `(:goal)` 块求值——**goal 即 R,且是数据的一部分** |
| **观测通道 O** | 结果页/商品页渲染 | `grammar` 模板渲染(feedback 规则)+ **admissible commands 列表** |
| **agent 看得到什么** | 只有查询结果页;**目录内容从不暴露** | 每回合喂 admissible(由 PDDL 前提推导)⇒ 动作可用性的变化**当回合可见** |
| **动作空间** | 自由文本 search + 点击 | 从 admissible 中选(服务端 projection 解析) |
| **每 episode 的加载** | 服务常驻,goal 换 index | textworld **每次 reset 重新 load 游戏文件** ⇒ wrapper 层每 episode 改写成为可能 |
| **任务实例** | goal = (asin, options, price) 元组,1000 商品目录共享 | game = (goal 块, scene/trial 布局) 焊在一个文件里;goal 类型量化、可移植 |

两条决定性推论:

1. **隐藏性天花板不同**。WebShop 的一切扰动天然全隐藏;ALFWorld 因 admissible
   泄漏,只有 effect(做了才知道)与 goal 判定(episode 末才知道)达到同级隐藏。
   ALFWorld 的 arm 分级必须按此校准,不能照抄 WebShop 的 stage-severity 叙事。
2. **注入点不同**。WebShop 在**运行时**换部件(`env_kwargs` / 服务参数);
   ALFWorld 在**数据层**改字符串(innermost wrapper + 临时文件,每 episode 生效)。
   前者的工程面是服务配置,后者的工程面是锚点改写 + 计数断言(fail-loud)。

---

## 2. 通道分解的对照(四级流水线 ↔ 符号系统)

WebShop 把 P 拆成四级;ALFWorld 的对应物如下(其中 reward 通道是 ALFWorld 独有的
第五行——WebShop 的 R 固定在引擎里,不可分 client 扰动;ALFWorld 的 R 就在数据里):

| 通道 | WebShop 变体 | ALFWorld 对应 arm | 隐藏性(ALFWorld 侧) |
|---|---|---|---|
| **content**(世界里有什么) | Catalog Split(`catalog_split`) | `scene_disjoint`(哪些 FloorPlan 存在) | 部分(经后继状态) |
| **encoding/rendering**(状态如何变文本) | Field-Subset(`bm25_field_subset`)/ Rank Wrapper(`rank_wrapper`) | `obs_variant`(grammar 反馈规则改写) | 部分(admissible 仍泄漏内容线索) |
| **matching/dynamics**(动作如何映射转移) | BM25 Reweighting(`bm25_reweight`) | `dyn_variant`(前提/效果改写) | effect 全隐藏;前提门控经 admissible 半可见 |
| **content×reward 交互** | Lookalike(`lookalike`) | `goal_variant`(隐藏成功判定) | **全隐藏至 episode 末** |
| **任务-环境解耦对照** | `task_disjoint`(同 goal 切片、全目录) | query×env cell 矩阵(同 query 集、异 scene) | —(对照组,不是攻击) |

注意一个**方向相反**的对照关系:WebShop 需要 `task_disjoint` 证明"catalog_split 的
效应不是 goal 切分的伪影"(从 env-arm 里剥出 task 成分);ALFWorld 的 cell 矩阵
证明"env 效应在 τ 逐字相同下依然存在"(把 task 成分构造性置零)。两者是同一个
混淆的两侧解法,论文里互为跨环境的鲁棒性证据。

---

## 3. arm 级对照(构造 + 实测)

### 3.1 content:Catalog Split ↔ scene_disjoint

| | Catalog Split | scene_disjoint |
|---|---|---|
| 发散数学 | `e=(1-div)·u+div·v_k`,按 ASIN 字符串 keying,top-`keep_ratio·D` | **同一公式**,按 scene id 经全局排序 keying,房型分层 top-(spc/4) |
| 保护底座 | per-client ~100 goal 的 target ASIN floor(可解性保障) | 任务型配额匹配全局边缘(τ 保障)+ 固定 100 游戏/client(数据量解耦) |
| div=0 floor | 目录最大重叠(共享 u 排序) | **20 client 字节级相同 shard**(实测 J=1.000) |
| div 扫描实测 | Jaccard 随 div 单调下降(paper 四点 0/0.3/0.7/1.0) | J(scenes) 1.000→0.195→0.080→0.043;J(games) 1.000→0.033;\|games\| 恒 100 |
| 已修的坑 | 运行时 target floor(SimServer shuffle 后再取 ASIN,2026-07-28) | 前身 `env_disjoint` 的两大缺陷:scene-J 恒 ~0.86(instance 级切分)+ 数据量混淆(并集 8.8%→82.7%) |
| τ 侧净度 | goal 切片不相交(需 task_disjoint 对照) | 语义级恒等;词汇级 2-模板噪声由归一化 wrapper 关闭(550 spec 全部收敛到 1 措辞) |

### 3.2 encoding/matching:BM25 变体 ↔ obs/dyn_variant

| | WebShop(Field-Subset / Reweight) | ALFWorld(obs / dyn) |
|---|---|---|
| 部件 | 索引字段子集 / (k1,b) 极端角点 | grammar 反馈规则 / PDDL 前提+效果 |
| 分配 | `RandomState(42+cid)`,`pool[rng.randint(N)]` | **逐字同款数学** |
| 实测强度 | Field-Subset N=4:J@10 0.39 / top-1 分歧 69.8%(effective-fields 勘误后);Reweight:0.62 / 63.9% | obs:动力学纯净 30/30(default 计划仍赢),观测面 30/30 生效;dyn:examine_gate 绑定 27/30=90%,+0.9 步 |
| 已知勘误 | lowercase `description`/`features` 从不挂载 ⇒ N=8 塌缩为 4 个不同索引(方案B 附录) | `(openable X)` 是**实例级**事实 ⇒ v_autoclose 最优路径绑定面仅 pick_two×实例-openable(Fridge/Safe 实测 +1 BIND;Drawer-304 恒真);类型级 25% 是上界 |
| N=2 语义 | 池前二 | 池前二 = [control, 最强全库变体];**cid 0/1 在 N=2 双双抽中 default**(冒烟须 N=4) |

### 3.3 最强攻击:Lookalike ↔ goal_variant

| | Lookalike Injection | goal_variant |
|---|---|---|
| 机制 | 注入合成商品:骗过 BM25 + 击穿一个 reward 子项(价格/颜色…),逼 agent 显式校验属性 | goal 块注入隐藏合取(`(checked ?r)` / `(not (opened ?r))`),逼 agent 多做一个显式动作;**指令文本一字不动** |
| π* 发散方式 | 不同 client 攻击不同属性 ⇒ 校验策略结构性不同 | 不同 client 隐藏要求不同 ⇒ 收尾协议结构性不同("放好即走 / 放好并 examine / 并关门 / 全都要") |
| 形式化归属 | P(内容)×R 交互;R 函数本身不变 | **直接扰动隐藏 R**(check_goal);文档如实标注这一形式化区别 |
| 实测 | 2 个 reward-validated 攻击(v_price/v_color),N=4 加 v_size/v_price_color | v_examined 绑定 30/30=100%(+1 步);v_closed 在 openable 面 8/8 绑定(+1 步);270 组合 0 不可解 |
| GRPO/PPO 叙事 | collapse 臂,PPO 部分拯救(critic 吸收隐藏方差) | 同一预期,配对同一张图(GPU 冒烟已通,规模曲线待跑) |

### 3.4 rendering:Rank Wrapper 的教训与 ALFWorld 的对策

Rank Wrapper 的 invert 臂在 `search_return_n=200` 下把 target 推到 ~19 页深、超出
15 步预算——**~33 个 client 实际不可解、近零奖励**,论文只能事后在附录披露。这是
"扰动强度未经可解性验证"的事故原型。ALFWorld 侧因此把 **planner 验证做成硬性
前置层**:每个 arm × 分层抽样游戏跑 fast-downward 出计划、逐步执行、断言 `won`、
Δsteps 有界(≤+1 实测,预算 50)——**结构上杜绝 invert 式事故重演**。这是两侧
方法论上最重要的一条单向改进。

### 3.5 对照组:task_disjoint ↔ query×env cell 矩阵

| | task_disjoint | cell 矩阵 |
|---|---|---|
| 回答的问题 | catalog_split 的效应是否目录扰动所致(而非 goal 切分) | env 效应在 τ 逐字相同时是否仍在 |
| τ 处理 | 同 catalog_split 的 goal 切片(不相交),目录还原为全量 | **所有 client 逐字同 query 集**(措辞归一化后) |
| 成本 | 零(复用切片数学) | 一次性离线生成(实测 89 min / 6 worker) |
| 状态 | 已实现、paper 值对齐 catalog_split | **库已生成**:8169 cell(6704 合成 usable 79.5% + 1465 天然),query 深度中位 15,难度保真 5.83 vs 5.8(§5.4);运行时分区 arm 待接线 |

---

## 4. 科学不变量的两侧实现

| 不变量 | WebShop 实现 | ALFWorld 实现 |
|---|---|---|
| seed-42 红线 | 分区体 verbatim,`RandomState(42[+cid])`,catalog u/v 按 ASIN keying | 新增体沿用同批常数与公式;scene u/v 按全局排序 scene keying;变体分配逐字同款 |
| 跨轮稳定 | 同 client 同变体/同目录 | 同 client 同变体/同 scene shard(纯函数于 client_id) |
| val 不扰动 | `WEBSHOP_SPLIT=val` 忽略 PARTITION_STRATEGY,全目录 Lucene | val 服务不导 het env var ⇒ uniform;kernel wrapper 仅 train 插入;`extra.gamefile` 保留原路径(seed==index 模式安全) |
| τ 不变 | 任务级扫描固定全目录;env 级固定 uniform goal 切片 + task_disjoint 对照 | 语义级恒等(pddl_params/goal 块与 scene 无关)+ 词汇归一化 + cells 构造性置零 |
| 可解性 | 事后发现 invert 臂不可解(教训) | **前置 planner 验证层**(270 组合 0 不可解 + 定向面验证) |
| fail-loud | 服务启动校验(target 可达性拒启 등) | 锚点计数断言;未知措辞拒改;数据版本漂移即 raise |

---

## 5. 实测数字总表(两侧并排)

### 5.1 发散度

| arm | 度量 | 值 |
|---|---|---|
| WebShop field_subset N=4 | 替换后 J@10 / top-1 分歧(300 真实 query 重放) | 0.39 / 69.8% |
| WebShop reweight N=4 | 同上 | 0.62 / 63.9% |
| ALFWorld scene_disjoint div=1.0 | pairwise J(games) / J(scenes)(C=20) | 0.033 / 0.043 |
| ALFWorld scene_disjoint div=0.0 | 同上(同质 floor) | 1.000 / 1.000(字节级相同) |
| ALFWorld env_disjoint(弃用) | J(scenes) 全 div | ~0.86(恒),并集覆盖 8.8%→82.7%(混淆) |

### 5.2 ALFWorld 变体绑定率与可解性(planner 层,n=30/变体)

| 变体 | 可解 | Δsteps | 绑定率 | 约束面 |
|---|---|---|---|---|
| obs ×3 | 30/30 | 0 | —(动力学纯净) | 全库 |
| dyn/examine_gate | 30/30 | +0.9 | 90% | 全库(源容器需开门的任务除外) |
| dyn/autoclose | 30/30 | +0/+1 | 面内验证通过 | pick_two × 实例-openable(~6% 全库上界) |
| goal/examined | 30/30 | +1.0 | **100%** | 全库 |
| goal/closed | 25/25(+5 identity) | +1 | 面内 8/8 | openable 目标(≤25%) |

### 5.3 联邦 GPU 冒烟(0.5B,单卡 4090,2 client × 2 轮,每 client 每轮 1 步)

- **goal_variant(variant_n=4,client0=v_closed / client1=v_default):FEDERATED
  LOOP CLOSED ✅**(2026-08-23)。per-client 变体不对称在服务日志实证
  (`variant=v_closed` / `v_default`,uniform 切片 [0:1776]/[1776:3553]);两轮
  各 2 client 训练(~1.9M token/step,307s/step)→ checkpoint 落盘 → FedAvg
  (ws=1)→ round 2 从聚合模型 + 暖服务复用 → 无扰动 val(140 游戏,
  `partition=uniform`、零 KERNEL 行)→ final eval 用 round-2 聚合模型;全程
  0 Traceback / 0 OOM / 0 端口冲突。val 成功率 r1=r2=0.0——0.5B × 每轮 1 步
  的机制冒烟里符合预期(ALFWorld 零样本≈0),判据是链路而非分数。熵随轮
  下降(1.491→1.467→1.395/1.430),聚合模型确在演化。
  > 首次尝试踩了一个通用陷阱(与 arm 无关):固定 `total_training_steps=4` 大于
  > 可推导步数(dataloader 1 batch/epoch)⇒ fit 提前结束、末步保存不触发、无
  > checkpoint 可聚合 ⇒ driver 以 base 模型收尾、worker 等 round-2 计划,双侧
  > 静默互等(全程零报错)。修法 = `total_training_steps: 0`(由 epoch 推导),
  > 已写入两个冒烟配置注释与 run-harness memory。
- **scene_disjoint(env_div=1.0,spc=8):FEDERATED LOOP CLOSED ✅**(2026-08-23)。
  两 client 的 scene 集**完全不相交**(c0=[7,13,203,217,305,321,419,421] /
  c1=[20,28,206,208,316,322,415,426],各 100 游戏、房型 2+2+2+2、任务配比一致);
  措辞归一化 wrapper 实机生效(160 个 `fedagent_taskcanon_*` 临时文件随 episode
  滚动);两轮训练 → FedAvg → 暖复用 → 无扰动 val → final eval 用 round-2 聚合
  模型;0 Traceback。val r1=r2=0.0(同上,机制冒烟预期内)。

### 5.4 cell 库(query×env,2026-08-23 全量生成完毕)

`data/env_heterogeneity/alfworld_cells_v1.json`(生成器
`tools/env_heterogeneity/gen_alfworld_cells.py`,trial 级筛 + canContain 判据 +
进程级 45s 超时,6 worker 89 min):

| 量 | 值 |
|---|---|
| 合成候选(静态筛后) | 8437 |
| **usable** | **6704 / 8437 = 79.5%**(与解耦文档 139-cell 抽样预测的 79% 精确吻合) |
| 拒绝分类 | won@reset 1445 (17.1%) / timeout 212 (2.5%) / no_plan 39 (0.5%) / error 37 (0.4%) |
| 总库(含 1465 natural) | **8169 cell**,天然深度的 5.6× |
| 每 query 深度 | 中位 **15**(天然 2);≥8 cell 的 query 381/550,≥16 的 232 |
| 难度保真 | 合成 plan_len 均值 **5.83** vs 天然 walkthrough 5.8 |
| look_at 合成 | 133 cell(trial 级筛回收;仍是最薄的任务型,矩阵构造须单独配额) |

---

## 6. 论文呈现建议

1. **主图(env-het 跨环境泛化)**:WebShop 侧 catalog_split div 扫描 + 变体 N 扫描
   已有;ALFWorld 侧以 `scene_disjoint {0.0, 1.0}` + `goal_variant {2,4}` 为最小
   补充集——前者对齐 catalog_split(content),后者对齐 lookalike(最强攻击),
   每个 arm 天然有 GRPO/PPO 配对位。**两侧的全部 cell 已入统一配置家族**
   `config/{paper,paper_accelerated}/env_heterogeneity/{grpo,ppo}/{webshop,alfworld}/<arm>/`
   (2026-08-23 重组;含 WebShop 此前缺失的 `task_disjoint` 消融四点扫描),
   逐 cell 映射见生成的家族 README。
2. **隐藏性谱系图**:两环境的 arm 按"可观测→部分→全隐藏"排布;ALFWorld 的
   admissible 泄漏作为环境属性明说,反而强化论点——*同一方法论在观测通道更
   透明的环境里依然找得到全隐藏扰动位(effect 与 goal 判定)*。
3. **clean-room 对照**:cell 矩阵作为"τ 严格置零仍现 env 效应"的证据放附录或
   分析节;与 WebShop 的 task_disjoint 互为镜像,合起来把"env-het 效应不是 τ
   泄漏伪影"钉死。
4. **诚实披露清单**(两侧对称):WebShop 的 effective-fields 塌缩与 invert 臂
   不可解;ALFWorld 的 admissible 泄漏、openable 实例级稀释、词汇归一化的存在。
5. **方法论一句话**:*在检索型环境里扰动流水线部件,在符号型环境里扰动符号系统
   本身;两者共享同一分配数学、同一 val 契约、同一 fail-loud 工程纪律——这是
   env-het 结论跨环境成立的机制性保障。*
