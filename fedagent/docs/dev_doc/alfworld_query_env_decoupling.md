# ALFWorld:query 与 env 的解耦(一个 query 对应多个环境)

> 状态:**已实现——cell 库 v1 全量生成完毕(2026-08-23)**。
> `tools/env_heterogeneity/gen_alfworld_cells.py` →
> `data/env_heterogeneity/alfworld_cells_v1.json`:trial 级筛(§4.2 修订判据含
> canContain)8437 候选 → **6704 usable(79.5%,§5 的 79% 预测精确复现)**;
> 总库 8169 cell(含 1465 天然),query 深度中位 **15**(天然 2、≥8 cell 者
> 381/550);合成 plan_len 均值 **5.83** vs 天然 5.8(难度保真);look_at 合成
> 133 个(trial 级筛回收,§5 结论 4 比原判乐观);拒绝分类 won@reset 17.1% /
> timeout 2.5% / no_plan 0.5% / error 0.4%(error 为个别 query×host 的
> fast-downward 'val1' 翻译器病态,系统性可忽略)。运行时分区 arm(§7 步 6)
> 尚未接线。评审修订(2026-08-23):勘误 §3.3(a)、判据补 §4.2(5)、trial 粒度、
> manifest 交互、task 行替换显式化。
> 日期:2026-08-22/23。作者:canyu + Claude。
> 姊妹文档:[alfworld_env_heterogeneity.md](./alfworld_env_heterogeneity.md)(已实现的四 arm
> 设计)、[../heterogeneity.md](../heterogeneity.md)(WebShop 五变体的构造参考)。
>
> 本文回答一个独立问题:**ALFWorld 的 query(任务指令)和 env(场景)能不能拆开,
> 让同一个 query 在不同 client 上跑在不同环境里?** 答案是能,且成本可控。这条路
> 给出的是姊妹文档四 arm **都做不到**的东西——一个 τ 在构造上被严格置零的
> pure-environment 对照组。

## 0. TL;DR

| 问题 | 结论 | 证据 |
|---|---|---|
| query 和 env 在 ALFWorld 里绑定吗? | **文件层面绑定,数据结构层面可分离** | `(:goal ...)` 是类型量化的,不引用任何 scene 实例(§2) |
| 天然有多少"同 query 多 env"样本? | 550 spec 中 311 个跨 ≥2 scene,中位数 2,最多 22 —— **深度不够** | §4.1 |
| 能合成吗? | 能。类型兼容矩阵 550 query × 108 scene = **8929 个 cell**(天然 1465,6.1×) | §4.2 |
| 合成的能跑吗? | 能。移植后 `PddlEnv` 正常加载,fast-downward 出计划,逐步执行可赢 | §4.3 |
| 产出率多少? | **79% 可用**(139 cell 抽样);14% 开局即赢须丢弃 | §5 |
| 难度会失真吗? | **不会**。合成 cell 计划长度 mean 6.0,天然游戏 5.8 | §5 |
| 值得做吗? | 值得。这是唯一能把 τ **在构造上**置零的 env-het arm(§6) | — |

**同时更正姊妹文档 §1.2 的一处科学性错误**(§3):同 spec 跨 scene 的 goal 文本
**并非逐字相同**,80% 的 spec 带 2 种措辞变体。这对 `scene_disjoint` 是一个真实
(虽然二阶)的 τ 泄漏,修法三行(§3.3;已实现)。

---

## 0.5 直观版:cell 到底是什么(先读这个)

**一个 cell = "同一句任务指令,换一个世界跑"。**

ALFWorld 的一个游戏文件(`game.tw-pddl`)其实是三样东西拼在一起:

1. **goal 块**(赢的条件)——**类型量化**,只说"存在两本 BookType 的东西在某个
   DeskType 的容器里",**不点名任何具体房间里的实例**;
2. **场景布局**(init 事实)——这个房间有什么家具、什么物体、摆在哪。这才是
   "环境"本体;
3. **指令文本**(grammar 的 task 行)——agent 读到的
   "Your task is to: put two book in desk."。

因为 goal 块不引用场景实例、指令文本又是独立字符串,所以"任务"和"世界"虽然焊在
一个文件里,**数据结构上是可拆的**:把 "put two book in desk" 的 goal 块 + 指令
搬进任何一个"有两本书和一张桌子"的房间,`check_goal()` 依然良定义。这次搬运的
产物就是一个 **synthetic cell**;query 呆在自己原生房间里的原始游戏则是
**natural cell**。

**为什么要造它**:所有现有 env-het arm 里,"client 环境不同"都**顺带**导致
"client 任务实例不同"(scene_disjoint 只对齐任务型配比;WebShop catalog_split
各拿各的 goal 切片,只好再造 task_disjoint 对照)。审稿人永远可以问:*你的效应
是不是任务划分造成的?* cell 矩阵把这个混淆**构造性堵死**——所有 client 拿到
**逐字相同**的任务集合,只有房间不同,τ 的影响恒等于零,而不是"期望上抵消"。

**为什么天然素材不够**:550 个 query 天然跨 ≥2 房间的只有 311 个,中位数 **2**
——100 个 client 的"同 query 异 env"根本铺不开;合成后中位深度 15(§5)。

**质量怎么保证**:两层关卡——静态兼容筛(物体数量/容器类型/电器/逐房间的
canContain 表,按 **trial** 粒度)+ **规划器硬验证**(真实加载、fast-downward
求计划、逐步执行、必须赢;开局即赢/无解/超时的一律丢弃)。这一层是 WebShop
rank_wrapper invert 臂"静默不可解"事故的教训。

**将来怎么用**:固定一组 query,所有 client 都训同一组;client *i* 的 query *q*
跑在哪个房间由 catalog_split 同款的 `e=(1-div)·u+div·v_i` 在 q 的 cell 池内
排序决定。env_div=0 ⇒ 所有 client 数据完全相同(最干净的同质 floor);
env_div=1 ⇒ 各拿各的世界;任何 div 下每 client 的任务集/数量/配比/难度分布
**逐条相同**。名词表与更多背景见 [README.md](./README.md) §2.5–2.6。

---

## 1. 问题

姊妹文档的四个 arm 都遵循同一个契约:**保持 τ 的分布不变,扰动 P**。但"分布不变"
不等于"逐条相同"——`scene_disjoint` 给不同 client 的是**不同的游戏**(因而是不同
的 query 实例),只是任务型配额对齐;WebShop 的 `catalog_split` 更是直接给每个
client 一个**不相交的 goal 切片**,以至于必须再造一个 `task_disjoint` 对照组,才能
把 goal 划分的影响从 catalog 扰动里剥出来([../heterogeneity.md](../heterogeneity.md))。

理想的对照是:**所有 client 拿到逐字相同的 query 集合,只有环境不同**。这样

- τ 的影响不是"在期望上被抵消",而是**在构造上恒等于零**;
- 不需要 ablation arm,任何 client 间差异都只能来自 P;
- 每 client 的 query 数量、任务型配比、难度分布可以**精确**对齐,而不是近似对齐。

ALFWorld 的数据布局(`<task_type>-<obj>-<mrecep>-<recep>-<scene>/<trial>/game.tw-pddl`)
看上去把 query 和 scene 焊死在一个文件里。本文说明它们其实是可分的。

---

## 2. 核心机制:goal 是类型量化的

`game.tw-pddl` 是纯 JSON,带 `pddl_domain` / `grammar` / `pddl_problem` / `solvable` /
`walkthrough` 五个键(姊妹文档 §2.2 已确认前三者的字节稳定性)。决定性的事实在
`pddl_problem` 的 `(:goal ...)` 块里——它**完全用类型谓词表达,不出现任何 scene 实例名**:

```lisp
(:goal (and (exists (?r - receptacle) (exists (?o1 - object)
    (and (objectType ?o1 BookType) (receptacleType ?r DeskType) (inReceptacle ?o1 ?r)
         (exists (?o2 - object)
             (and (not (= ?o1 ?o2)) (objectType ?o2 BookType)
                  (receptacleType ?r DeskType) (inReceptacle ?o2 ?r))))))))
```

对比同一文件里的初始状态谓词,后者形如
`(inReceptacle book_bar__minus_01_dot_24_bar__plus_00_dot_74_... drawer_bar__...)`,
带完整实例名。**goal 不带**。

⇒ 一个 goal 块可以原样搬进**任何**含所需类型对象/容器的 scene,搬进去之后
`check_goal()` 依然良定义。这就是 query 与 env 可分的全部依据,不需要改 PDDL 语义、
不需要重跑 AI2-THOR、不需要任何生成期工具。

**同时注意**:`(:goal)` 与 grammar 里的 task 文本(`"Your task is to: ..."`)是**两个
互相独立的字符串**。前者决定"什么算赢",后者决定"agent 看到什么指令"。姊妹文档的
`goal_variant` 利用的正是这一点(改前者、不动后者);本文利用的是它的对偶——
**固定前者+后者,换 scene**。

---

## 3. 更正:goal 文本并非逐字相同

### 3.1 实测

姊妹文档 §1.2 称"同一 spec 跨 scene 的 goal 文本**逐字相同**"。全库实测(3553 个
train 游戏,按 spec 归并 grammar 的 `"task"` rhs):

| 量 | 值 |
|---|---|
| spec 数 | 550 |
| 每 spec 的**不同** task 字符串数(均值 / 中位 / 最大) | 1.80 / 2 / **2** |
| 带 >1 种措辞的 spec | **438 / 550 = 80%** |
| 全库不同 task 字符串总数 | 988 |

例:`pick_two_obj_and_place-Book-None-Desk` 的四个 scene 分别是

```
302 -> "put two book in desk."
309 -> "find two book and put them in desk."
310 -> "put two book in desk."
313 -> "find two book and put them in desk."
```

### 3.2 成因

[misc.py:86](../../envs/alfworld/engine/agent_system/environments/env_package/alfworld/alfworld/agents/utils/misc.py#L86)
的 `get_templated_task_desc`:

```python
template = random.choice(glib.gdict[goal_str]['templates'])   # 每类 goal 有 2 个模板
```

这是**生成期**的一次随机抽取,结果冻结进 `game.tw-pddl` 的 grammar 字符串。它与
scene **无关**,纯粹是生成 RNG 的噪声。

### 3.3 影响与修法

- 对 `goal_variant` / `obs_variant` / `dyn_variant`:**无新增影响**。
  > **勘误(2026-08-23 评审)**:原稿此处称"这三个 arm 给所有 client 的是同一批
  > 游戏"——**不对**。uniform 契约切片给各 client 的是**不相交的连续切片**
  > (冒烟实测 client0=[0:1776]、client1=[1776:3553]),task 文本在 client 间并不
  > 一致。结论仍成立,但正确理由是:variant arm 的游戏切分与**非异质 uniform 基线
  > 完全一致**,措辞噪声属于基线本身,不构成相对基线的新增混淆。
- 对 `scene_disjoint`:**有影响**。不同 client 拿到不同 scene 的游戏,于是同一个
  spec 在 client A 是 `put two book in desk.`、在 client B 是
  `find two book and put them in desk.`——一个词汇级的 τ 差异漏进了本应纯 env 的
  arm。量级是二阶的(语义完全相同、只有两种措辞),但它**违反了 arm 的形式化契约**,
  论文里"τ 逐字不变"的表述在词汇层不成立。
- 对本文的 query×env 矩阵:**必须处理**,否则"同一个 query"名不副实。

**修法(已实现,2026-08-23)**:实际量级不是"三行"——`scene_disjoint` 原本**不走**
kernel wrapper(只有三个 `*_variant` 策略插),需要 (a) 在
`alfworld_kernel_variants.py` 加 `canonicalize_task_text`(按 gamefile 路径解析
task type 选模板对,变体措辞 → `templates[0]` 规范措辞,未知措辞 fail-loud)与
`make_task_normalizer_wrapper()`(与 kernel wrapper 共用临时文件机制),(b)
`alfred_tw_env.init_env` 在 `scene_disjoint` + train 时默认插入(val 恒 uniform,
不受影响)。全库测试:归一化后 **550 spec 全部收敛到每 spec 恰 1 种措辞**
(`tests/test_alfworld_kernel_variants.py`)。姊妹文档 §1.2 已改为"语义级不变 +
词汇级由归一化保证"。

---

## 4. 三个层次的可行性

### 4.1 天然深度:不够

按 spec 归并 3553 个 train 游戏:

| 量 | 值 |
|---|---|
| spec 数 / scene 数 | 550 / 108(厨/卧/客/卫各 27) |
| 每 query 的天然 scene 数(均值 / 中位) | 2.66 / 2 |
| 跨 ≥2 / ≥4 / ≥8 / ≥16 个 scene 的 query 数 | 311 / 122 / 29 / **4** |
| (query, scene) cell 总数 | 1465 |

也就是说,**天然素材只够给约 2 个 client 配同一个 query**。100-client 的联邦跑不起来。

### 4.2 合成:矩阵撑开 6.1 倍

对 550 query × 108 scene 逐格做**类型兼容筛**。判据(全部可从 `pddl_problem` 静态
读出,无需加载环境):

1. scene 含 `object_target` 类型的实例 ≥ n(`pick_two_obj_and_place` 需 n=2,其余 n=1);
2. scene 含 `parent_target` 类型的 receptacle ≥ 1(`look_at_obj_in_light` 的
   parent 是 DeskLamp,按 object 或 receptacle 任一命中);
3. 若 spec 有 `mrecep_target`,scene 需含该类型;
4. clean / heat / cool 三类需 scene 内分别存在 SinkBasin / Microwave / Fridge。

> **评审补充(2026-08-23,两条实测修正)**:
>
> (a) **判据缺第 5 条:`canContain`**。它是**实例局部**的 init 事实,不是全局类型
> 表(随机 12 个 problem 的 canContain 对数在 34–130 之间;厨房游戏 vs 卧室游戏的
> 集合 Jaccard 仅 0.01)。移植 "put X in Y" 须要求 host init 含
> `(canContain <recep>Type <obj>Type)`(look_at 除外)。不加这条,部分 no-plan 是
> 系统性、可预筛的。
>
> (b) **"scene" 不是良定义的兼容性单元——库存是 trial 级的**。同一 FloorPlan 的
> 不同 trial 物体摆放不同(实测 scene 313 各 trial 的 book 数在 1–4 之间;§4.3 的
> "host 313 只有 1 本书"反例正是选中了 1-book trial,而 313 另有 4-book trial)。
> ⇒ 兼容筛须按 (query, **host trial**) 做,每 (query, scene) 以确定性规则解析 host
> trial(目标物计数最大者,平局按路径字典序);"8929" 是依赖代表-trial 选择的
> 估计值,实际容量以生成器产出为准。

结果:

| 量 | 天然 | 兼容筛后 |
|---|---|---|
| 每 query 可用 scene(均值 / 中位) | 2.66 / 2 | **16.23 / 15** |
| ≥2 / ≥4 / ≥8 / ≥16 / ≥32 个 scene 的 query 数 | 311 / 122 / 29 / 4 / 0 | **532 / 472 / 400 / 266 / 16** |
| (query, scene) cell 总数 | 1465 | **8929**(6.1×) |

### 4.3 端到端移植实验(已跑通)

取 `pick_two_obj_and_place-Book-None-Desk` 的 goal 块,移植进 4 个**非原生但兼容**的
卧室 scene,用 `PddlEnv(EnvInfos(won, admissible_commands, policy_commands)).load(dict)`
直接加载(textworld 的 `PddlEnv.load` 接受 Mapping,见
`textworld/envs/pddl/pddl.py:51`;注意 `GenericEnvironment.load` 不接受 dict,走服务
路径时仍需姊妹文档 §2.1(4) 的临时文件方案):

| host scene | 该 scene 的原生 spec | won@reset | admissible | fast-downward 计划 | 耗时 |
|---|---|---|---|---|---|
| 304 | look_at_obj_in_light-AlarmClock | False | 12 | **5 步** | 0.3s |
| 316 | 同上 | False | 10 | 4 步 | 0.2s |
| 318 | 同上 | False | 25 | 8 步 | 1.2s |
| 320 | 同上 | False | 12 | 8 步 | 0.3s |

反例同样重要:host **313** 只有 1 本 book(goal 需要 2 本),planner 返回 `None`。
⇒ 类型兼容筛里的**计数**判据(§4.2 第 1 条)不可省,且兼容筛之后仍需 planner 兜底。

---

## 5. 大样本验证:139 个合成 cell

方法:从 550 个 spec 随机抽 150 个(seed 7),每个配一个**随机的非原生兼容 scene**
(11 个 spec 无非原生兼容 scene,故实得 139 cell);移植 goal → `load` → `reset` →
`replan`。环境 `verl-agent-alfworld`。

```
solvable = 110 (79%) | won@reset = 20 (14%) | no-plan = 4 (3%) | planner 超时 = 5 (4%)
plan length: mean 6.0  median 6.0  p90 8  max 9
hist: {4:30, 5:10, 6:31, 7:14, 8:18, 9:7}
by task type: pick_two 33 / pick_and_place 28 / clean 27 / cool 13 / heat 8 / look_at 1
总耗时 1001s(139 cell,单核)
```

四个结论:

1. **产出率 79%** ⇒ 8929 个候选 cell 里约 **7000 个可用**,是天然 1465 的近 5 倍。
   100 client × 每 client ~100 query 的规模绰绰有余。
2. **难度不失真**。合成 cell 的计划长度 mean 6.0 / median 6.0;天然同 spec 游戏的
   `walkthrough` 长度 mean 5.8。⇒ 移植**不会**把 hardness 轴偷偷混进 env 轴。
   p90=8、max=9,相对 `Limit(50)` 的步数预算有充足余量。
3. **`won@reset` 14% 是真实的**——光靠类型兼容筛不够,planner 验证是**硬性**步骤。
4. **`look_at_obj_in_light` 基本合成不出来**(110 个里仅 1 个)。DeskLamp 只出现在
   少数卧室 scene,而那些恰好就是它的原生 scene。⇒ 该任务型只能继续用天然 cell,
   构造矩阵时须单独处理,否则六类任务的配比会跨 client 失衡。

---

## 6. 对 FedAgent 的意义

### 6.1 构造:query × client 矩阵

固定一组 query `Q`(比如 200 个),对每个 `q ∈ Q` 从其可用 cell 池里给每个 client
分一个 scene:client *i* 训练在 `{(q, scene_{i,q}) : q ∈ Q}` 上。此时

| 通道 | 状态 |
|---|---|
| τ(prompt 里的 goal 文本) | **所有 client 逐字相同**(经 §3.3 归一化后) |
| query 集合 / 数量 / 任务型配比 | **所有 client 完全相同** |
| 难度分布(planner 步数) | **可精确对齐**(cell 库自带 plan_len 标签) |
| P(FloorPlan、容器清单、物体摆放) | **完全不同** |

发散强度用与 `catalog_split` 同构的 `e = (1-env_div)·u + env_div·v_k` 在**每个 query 的
cell 池内**排序取 top-1 即可,`env_div=0` 是完美同质 floor(所有 client 同一个 cell),
`env_div=1` 各排各的。因为每个 query 恒定贡献 1 个 cell,**`env_div` 不影响每 client
的数据量**——这正好修掉了 `env_disjoint` 的缺陷 (b)(见姊妹文档 §2.7)。

> **floor 语义注意(评审补充)**:`env_div=0` 意味着所有 client 的训练数据
> **完全相同**(全复制),这与主实验 uniform 基线(各 client 不相交切片)是**不同
> 的基线语义**——作为"τ 与数据划分双重置零"的 clean-room floor 它是对的,但论文里
> 与 uniform 基线并排画曲线时须说明这一点。

### 6.2 与姊妹文档四 arm 的关系:互补,不重复

| | `scene_disjoint`(已实现) | 本文的 query×env 矩阵 |
|---|---|---|
| 分配单位 | scene(按房型分层 top-k)+ 任务型配额子采样 | (query, scene) cell |
| τ 对齐方式 | 任务型**边缘配额**对齐(近似) | query **逐条**相同(精确) |
| 是否需要合成 | 否,只用天然游戏 | 是,需离线生成 cell 库 |
| 联合分布 τ×scene | 与全局不同(姊妹文档 §3.1 的"知识局限") | 按构造对齐 |
| 成本 | 零(纯分区) | 一次性 ~小时级离线生成 |

⇒ `scene_disjoint` 是**廉价的主力 arm**,本文的矩阵是**昂贵但无可辩驳的对照组**。
论文里的定位建议:用 query×env 矩阵作为"τ 严格置零"的 clean-room 实验,证明
env-het 效应不是 τ 泄漏的伪影;再用 `scene_disjoint` 在全量数据上跑规模曲线。

---

## 7. 实现路径(建议离线生成 cell 库,不走运行时注入)

1. **兼容性筛**(纯字符串,秒级):正则抽 `(objectType ?x TType)` /
   `(receptacleType ?x TType)` 计数 → §4.2 四条判据 → 8929 候选。
2. **planner 验证**(可并行):移植 goal → `load` → `reset` → 丢弃 `won@reset` 与
   无解 → 记录 `len(policy_commands)`。
3. **归一化 grammar 的 task rhs**(§3.3),使"同一个 query"字面成立。
4. **落盘 cell 库**:`data/env_heterogeneity/alfworld_cells_v1.json`,每条
   `{query_spec, scene, host_gamefile, plan_len, task_type, natural}`,并给出
   sha256 —— 与 `data/alfworld_games/*.json` 的 manifest 同风格(pin 住集合与顺序)。
   **cell 游戏的构造规则须显式三条**(评审补充):goal 块 ← query donor;task rhs
   ← query 的**规范化**文本(**替换 host 的 task 行**,出现次数断言 fail-loud——
   漏掉这步 agent 读到的是 host 原生指令而 `check_goal` 判的是 query,灾难性错配);
   其余(init/grammar 反馈规则/domain)← host。
5. **落盘 per-client 游戏目录**(仅运行时 arm 需要):每个 cell 写出一个完整的
   `game.tw-pddl`。磁盘代价:~64KB/游戏 × 7000 ≈ 450MB,可接受。
   > **评审修正**:原稿称"运行时零改动(AlfredTWEnv 照常 walk/manifest)"——
   > 与第 6 步自相矛盾,且**漏了 authoritative manifest 交互**:manifest pin 的是
   > 集合,盘上多出的 cell 游戏会被**无声忽略**(这正是 manifest 的设计目的)。
   > cell arm 必须自带 manifest/game-list 接线(为 cell 树生成自己的 manifest 并经
   > `alfworld_game_list_env` / `ALFWORLD_GAME_MANIFEST` 指向它,或独立
   > `ALFWORLD_DATA` 树 + 显式禁用),否则加载器根本看不到 cell 库。
6. **分区函数 + 桥接**:`_cell_matrix_partition_alfworld` 走
   `partition_dataset`;`run_fed.py` / `server.py` 增
   `ALFWORLD_CELL_FILE`,与姊妹文档 §4.3 的 env var 桥同风格。val 服务不动。

**顺带的白捡收益**:`plan_len` 就是**免费的 hardness 标签**,不需要像现在
`data/hardness/` 那样用训练好的 checkpoint 跑 rollout 打标(而且按 WebShop 侧的审计,
那套标签本身被验证过不携带难度信号)。cell 库天然可以驱动一个"零成本 hardness arm"。

---

## 8. 风险与坑

1. **planner 的 SIGALRM 打不断**(踩过)。给每个 cell 加 `signal.alarm(20)` **无效**:
   handler 只能在 Python 字节码边界抛异常,而 fast-downward 是 C 扩展里的长调用。
   实测有单个 cell 卡了约 650s。⇒ 生成器**必须**用 `multiprocessing.Process` +
   `terminate()` 或 `subprocess` + `timeout=` 在**进程层面**砍。否则那 4% 的病态实例
   会把整条流水线卡死。
   *(2026-08-23:该机制已在 `verify_alfworld_kernel_variants.py` 落地——fork 子进
   程 + `Pipe.poll(timeout)` + `terminate()`,`--solve-timeout` 默认 120s;生成器
   直接复用同一 `_solve`。)*
2. **成本**:单核 7.2s/cell(被少数病态 cell 拉高),全量 8929 cell ≈ 18 core-hours;
   配进程级超时(建议 30s 上限)+ 16 路并行,墙钟约 1 小时。
3. **部分预满足**:`won@reset` 只能滤掉"开局就赢"的;还存在"开局已完成一半"的
   (例:scene 316 的 pick_two 只需搬 1 本书,另一本已在 desk 上)。⇒ 需按**任务型
   分别**设 `plan_len` 下限,否则难度分布跨 client 漂移,env 轴又混进 hardness 轴。
   本次抽样没有按任务型拆 plan_len,这一项**待补测**。
4. **`look_at_obj_in_light` 合成深度近乎为零**(§5 结论 4),须单独处理。
5. **`walkthrough` 与 `solvable` 必须重算**:原文件里的对新 goal 无效。直接用
   planner 的 `policy_commands` 覆写 `walkthrough`,`solvable` 置为验证结果。
6. **别走运行时 wrapper 注入**:`AlfredInfos.load` 把 `args[0]` 当 gamefile 存,
   `ALFWORLD_SEED_IS_INDEX` 的索引模式依赖 `extra.gamefile`。姊妹文档 §2.1 用临时
   文件绕过了这个问题,但对本 arm 而言离线落盘更简单、更可复现。
7. **数据版本漂移**:cell 库 pin 的是 host game 的路径 + goal 块内容。上游 ALFWorld
   数据版本变更会让锚点失配 ⇒ 生成器与加载器都要 fail-loud(与姊妹文档 §8.6 同策)。

---

## 9. 复现

本文所有数字来自以下测量,均可重跑:

| 数字 | 环境 | 方法 |
|---|---|---|
| §3.1 措辞变体统计 | 系统 python3 | 遍历 manifest 3553 游戏,正则抽 grammar 的 `"task"` rhs,按 spec 归并去重 |
| §4.1 天然深度 / §4.2 兼容矩阵 | 系统 python3 | 从每个 scene 的 `pddl_problem` 抽 `objectType`/`receptacleType` 计数,对 550 spec 逐一套 §4.2 四条判据 |
| §4.3 移植实验 / §5 大样本 | `verl-agent-alfworld` | 平衡括号提取 `(:goal ...)` → 替换 host 的对应块 → `PddlEnv.load(dict)` → `reset()` → 读 `policy_commands` |
| §2.7(姊妹文档)分区度量 | `fedagent-verl08` | 直接 import `partition_strategy.py`(stub 已不需要:FedAgent 侧的 matplotlib 硬 import 于 2026-08-22 改为 guarded-optional,与 AccelAgent 侧一致) |

manifest:`data/alfworld_games/train.json`(3553 游戏,sha256 `1489458706...`)。
游戏数据根:`$ALFWORLD_DATA`(默认 `~/.cache/alfworld`)。

**尚未落库的脚本**:§4.2 / §5 的测量目前是一次性脚本。建议实现时把它们固化为
`tools/env_heterogeneity/gen_alfworld_cells.py`(生成 + 验证 + 落盘 + `--check`
复现校验),与 `gen_holdout_alfworld.py` 同风格。
