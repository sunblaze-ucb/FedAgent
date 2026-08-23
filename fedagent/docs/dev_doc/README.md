# dev_doc 总入口:ALFWorld 环境异质性工作簿

> 日期:2026-08-22/23。作者:canyu + Claude。
> 这里是四份开发文档的**索引 + 概念手册**。文档正文假设读者熟悉项目上下文;
> 本 README 反过来,把每个概念从零讲清,并标注它在哪份文档哪一节展开、
> 对应哪些实测数字。生词先查这里。

## 0. 四份文档与阅读顺序

| 顺序 | 文档 | 一句话 | 状态 |
|---|---|---|---|
| 1 | [alfworld_env_heterogeneity.md](./alfworld_env_heterogeneity.md) | ALFWorld 四个环境异质性 arm(scene_disjoint / obs_variant / dyn_variant / goal_variant)的设计、实现与五层验证 | **已实现 + GPU 冒烟通过** |
| 2 | [alfworld_query_env_decoupling.md](./alfworld_query_env_decoupling.md) | query(任务)与 env(场景)可以拆开——cell 库:τ 构造性置零的对照组 | **cell 库 v1 已生成;运行时 arm 待接线** |
| 3 | [webshop_vs_alfworld_env_heterogeneity.md](./webshop_vs_alfworld_env_heterogeneity.md) | 两个环境的环境异质性体系逐项对比 + 论文呈现建议 | 定稿 |
| 4 | [../heterogeneity.md](../heterogeneity.md)(repo 正式文档) | WebShop 五变体的构造参考(本工作的方法论母本) | 定稿 |

---

## 1. 科学框架:为什么要做"环境异质性"

**Input-Dynamics Asymmetry(论文核心主张)**:联邦 agent RL 对**任务级**异质性
鲁棒(任务描述 τ 写在 prompt 里,策略看得见,聚合模型可以按 τ 条件化),对
**环境级**异质性最坏情形不鲁棒(转移核 P 藏在环境里,策略只能通过"做了动作之后
世界怎么变"间接感知)。要**测量**而非**假设**这个不对称,就需要一套能"只动 P、
不动 τ"的受控扰动构造——这就是环境异质性 arm 的全部使命。

**四个通道**(每个 arm 扰动其中一个):

| 符号 | 名称 | ALFWorld 里的载体 | agent 能看见吗 |
|---|---|---|---|
| τ | 任务描述 | grammar 里的 "Your task is to: ..." 一行 | ✅ 在 prompt 里 |
| P | 转移核(动作→世界怎么变) | `pddl_domain` 的动作前提/效果 + `pddl_problem` 的场景布局 | ❌ 只能经后继状态感知 |
| R | 奖励/成功判定 | `pddl_problem` 的 `(:goal)` 块(`check_goal()` 对它求值) | ❌ episode 结束才知道 |
| O | 观测核(世界→文本) | grammar 的反馈模板 + admissible commands 列表 | 部分(见"admissible 泄漏") |

**admissible 泄漏**:ALFWorld 每回合把"当前可执行动作列表"喂进 prompt,而这个
列表由 PDDL 前提**推导**而来——所以凡是改动作前提的扰动,agent 当回合就能从
列表变化里看出来(≠完全隐藏)。完全隐藏的只有**动作效果**(做了才知道)和
**goal 判定**(结束才知道)。这决定了四个 arm 的隐藏性分级,详见对比文档 §1-2。

---

## 2. 概念手册(按出场频率排序)

### 2.1 arm / 变体(variant)/ 池(pool)/ N 截断

- **arm** = 一种 `partition_strategy` 取值,决定"client 之间怎么不同"。
- **变体** = arm 内部的一个具体扰动配置。例如 `goal_variant` 的池是
  `[v_default, v_examined, v_closed, v_examined_closed]`。
- **池序是设计过的**:第 0 位永远是 `v_default`(对照,字节不改),第 1 位是
  "最强且全库生效"的变体,这样 `variant_n=2`(N 截断取前二)天然是
  "对照 + 最强攻击"的最小对比组——与 WebShop lookalike 的 N=2 语义对齐。
- **分配数学**(每个 client 拿哪个变体):`RandomState(42 + client_id)` 抽
  `pool[randint(N)]`,与 WebShop 的 `_bm25_variant_partition_webshop` **逐字同款**。
  同一 client 跨轮永远同变体(FedAvg 可比性要求),可离线重算。
- **已知小陷阱**:2-client 冒烟时 client 0 和 1 在 N=2 下**恰好都抽中 v_default**
  (seed-42 硬币的巧合),所以示例配置用 N=4(c0=v_closed, c1=v_default)。

### 2.2 绑定(binding)与绑定率(binding rate)

这是验证一个变体"真的改变了什么"的核心度量。

- **定义**:把**默认核**(未扰动环境)下的最优计划,原封不动放到**变体核**下
  重放。如果它不再获胜,称该变体在这个游戏上**绑定**(binding)。
  绑定率 = 抽样游戏中绑定的比例。
- **为什么这样定义**:它精确模拟了联邦异质性伤害的机制——client 在自己的环境
  里学会一套策略,聚合后这套策略在别的 client 的环境里失效。"default 计划失效"
  就是最小可测的"策略失效"。
- **三种结果**:
  - **identity**:变体对这个游戏连字符串都没改(如 look_at 任务没有目标容器,
    `v_closed` 无处注入)——不计入绑定率分母;
  - **非绑定(nobind)**:字符串改了但行为无差(如给不可开合的桌子加"必须关门"
    条件,恒真)——计入分母、不计入分子,称为**稀释**;
  - **绑定(BIND)**:default 计划真的输了,且变体核下存在新的(通常 +1 步的)
    获胜计划。
- **实测**(env-het 文档 §7.2,n=30/变体):`v_examined` **100%** 绑定,
  `examine_gate` 90%,`v_closed`/`v_autoclose` 在通用样本 0%——但在各自的
  **适用面**上定向验证 8/8、2/3 绑定(见 2.3)。obs 三变体的判据反过来:
  default 计划**必须仍赢**(它们只改观测,动力学必须纯净),实测 30/30。

### 2.3 恒真面 / 稀释(dilution)/ 实例级 openable

- `v_closed`(goal 里加"目标容器最终要关着")和 `v_autoclose`(放完东西容器
  自动合上)只对**可开合**的容器有行为差;对桌子/台面这类容器,加的条件恒真。
- 关键实测发现:`(openable X)` 是**实例级** init 事实——同为 DrawerType 的两个
  抽屉,一个有 openable 事实、另一个没有(卧室 304 的目标抽屉全程不开门直接放)。
  所以"25% 的游戏目标容器类型可开合"只是**上界**,实际绑定面更小。
- 设计上的应对:这两个变体排在池的第 2/3 位(N=2 不含它们),角色是 N=4 时的
  多样性成分与组合臂(`v_examined_closed`/`v_gate_autoclose`)的组成部分;
  真正的主力是全库绑定的 `v_examined` / `examine_gate`。论文按此如实表述。

### 2.4 措辞归一化(task-text canonicalization)

- **问题**:ALFWorld 生成器给每类任务准备了 **2 个措辞模板**(如
  "put two {obj} in {recep}" vs "find two {obj} and put them in {recep}"),
  生成每个游戏时随机抽一个**冻结**进文件。全库 550 个任务规格中 438 个(80%)
  两种措辞都有,共 988 个不同指令字符串。
- **为什么是 τ 泄漏**:`scene_disjoint` 下不同 client 拿到不同 scene 的游戏,
  同一个任务在 client A 是一种措辞、在 client B 是另一种——词汇级的任务差异
  漏进了"纯环境"arm(语义完全相同,故称**二阶**泄漏)。
- **修复**:`canonicalize_task_text`(按游戏路径解析任务类型,选对应模板对,
  变体措辞→规范措辞 `templates[0]`,未知措辞直接 raise),包成 wrapper 在
  `scene_disjoint` + train 时默认插入。全库测试:归一化后 550 规格**每个恰好
  收敛到 1 种措辞**。三个 `*_variant` arm 不需要它(它们的游戏切分与非异质
  基线逐字一致,措辞噪声属于基线本身)。

### 2.5 query / spec / scene / trial / host——cell 库的词汇表

- **spec(= query)**:任务规格,即目录名去掉场景号:
  `pick_two_obj_and_place-Book-None-Desk`。全 train 库恰好 **550** 个。
  它决定 goal 块的语义与指令文本("put two book in desk")。
- **scene**:AI2-THOR 房间号(FloorPlan),108 个,按号段分四房型
  (1–30 厨房 / 201–230 客厅 / 301–330 卧室 / 401–430 卫生间,各 27)。
- **trial**:同一 (spec, scene) 下的一次具体生成(目录里的
  `trial_T2019...`)。**关键**:同一房间号的不同 trial 物体摆放**不同**
  (实测 scene 313 各 trial 的书数在 1–4 之间)——所以"库存"是 trial 级概念,
  兼容性筛必须按具体文件做。
- **host**:被移植进 goal 的那个具体游戏文件(提供整个世界)。
- **natural cell**:query 在它的原生 scene 里(游戏文件原样)。
- **synthetic cell**:query 的 goal 块 + 规范指令移植进**非原生**的 host。

### 2.6 cell 库(query×env 矩阵)——是什么、为什么、怎么造

**是什么**:一个 cell = "同一句任务指令,换一个世界跑"。cell 库 =
所有(经验证可解的)(query, host) 组合的清单,
`data/env_heterogeneity/alfworld_cells_v1.json`。

**为什么需要**:所有现有 env-het arm 里,"client 环境不同"都**顺带**导致
"client 任务实例不同"(scene_disjoint 只对齐任务型配比;WebShop catalog_split
干脆各拿各的 goal 切片,只好再造 task_disjoint 对照)。审稿人可以永远追问
"你的效应是不是任务划分造成的"。cell 库把这个混淆**构造性堵死**:所有 client
拿到**逐字相同**的任务集合(措辞归一化后),只有房间不同——τ 的影响不是
"期望上抵消",而是**恒等于零**。

**为什么可行**(核心机制):goal 块是**类型量化**的
(`exists ?o BookType, exists ?r DeskType, inReceptacle`),不引用任何房间实例名
——可以原样搬进任何"有两本书和一张桌子"的房间;而指令文本与 goal 块又是两个
独立字符串。所以 query 与 env 在数据结构层面可分。

**天然素材为什么不够**:550 个 query 天然跨 ≥2 scene 的只有 311 个,中位数
**2**——给 100 个 client 配"同 query 异 env"根本铺不开。

**怎么造**(`tools/env_heterogeneity/gen_alfworld_cells.py`,两层关卡):

1. **静态兼容筛**(纯字符串):host 里目标物体类型数量够(pick_two 要 2)、
   目标容器类型在场、clean/heat/cool 的水槽/微波炉/冰箱在场、以及
   `canContain(容器类型, 物体类型)` 事实在场——canContain 是**每个房间自己
   声明的局部表**(34–130 对不等,厨房 vs 卧室的集合 Jaccard 仅 0.01),
   不查这条会漏系统性不可解。每 (query, scene) 取目标物库存最多的 trial 当 host。
2. **规划器硬验证**(fork 子进程 + 45s 进程级超时):真实加载 → fast-downward
   求计划 → 逐步执行 → 必须赢。踢掉三类坏 cell:**won@reset**(host 布局开局
   就满足 goal,占 17.1%)、no-plan(0.5%)、超时(2.5%)。

**移植的三条规则**(漏掉第 2 条是灾难:agent 读 host 的旧指令、判的却是新 goal):
goal 块 ← query;指令行 ← query 的规范措辞(替换 host 的,出现次数断言);
其余一切 ← host;host 的 walkthrough/solvable 对新 goal 无效,由验证结果覆写。

**实测结果**:8437 候选 → **6704 usable(79.5%)**+ 1465 natural = **8169 cell**;
每 query 深度中位 **15**(天然 2),≥8 cell 的 query 381/550;合成 cell 计划长度
均值 **5.83** vs 天然 5.8——**难度不失真**(否则 env 轴混进 hardness 轴);
look_at 也合成出 133 个(trial 级筛的功劳,比解耦文档最初的悲观估计好)。
每个 cell 自带 `plan_len` 标签 = 一套免费的难度标签(现有 `data/hardness/`
标签经 WebShop 侧审计不携带难度信号,这是顺带的替代品)。

**将来怎么用**(`cell_matrix` arm,尚未接线):固定 200 个 query,所有 client
都训这 200 个;client *i* 的 query *q* 跑在哪个房间,由
`e=(1-env_div)·u+env_div·v_i` 在 q 的 cell 池内排序取 top-1。env_div=0 ⇒ 所有
client **完全相同的数据**(最干净的同质 floor,注意这与主实验 uniform 基线的
"不相交切片"是不同的基线语义);env_div=1 ⇒ 各拿各的房间。数据量与 τ 双双与
env_div 解耦。还差:分区函数、cell 游戏树落盘 + **自己的 manifest**(权威
manifest pin 集合,盘上多出的 cell 游戏会被无声忽略——必须显式接线)、
`ALFWORLD_CELL_FILE` env var 桥。

### 2.7 catalog_split 同构数学(所有"发散强度"旋钮的共同骨架)

```
u   ~ RandomState(42)                  # 全体共享的排序分
v_k ~ RandomState(42 + 1000*k)         # client k 私有的排序分
e_k = (1-env_div)*u + env_div*v_k      # 混合
选择 = 按 e_k 排序取 top-K
```

env_div=0 时人人按同一个 u 排序 ⇒ 选择完全一致(同质 floor);env_div=1 时各排
各的 v ⇒ 最大发散。**keying 是载荷细节**:u/v 必须按稳定身份(WebShop 按 ASIN
字符串、scene_disjoint 按全局排序的 scene 号)对齐,不能按池内下标——否则不同
client 的池内容不同,同一对象读到不同的 u,"共享 u"不变量就破了。

### 2.8 五层(+1)验证体系

| 层 | 验证什么 | 工具/环境 | 结果 |
|---|---|---|---|
| 0. 分区度量 | 发散度/尺寸/τ 配额(纯分区函数) | fedagent-verl08 | scene_disjoint:div=0 字节相同、\|games\| 恒 100、J(scenes) 1.0→0.043 |
| 1. 字符串单测 | 锚点改写、τ 红线、分配确定性、归一化收敛 | pytest, fedagent-verl08(无 textworld) | 20/20 |
| 2. planner 验证 | 可解性、Δsteps、绑定率、obs 纯净性 | verl-agent-alfworld(fast-downward) | 270 组合 0 不可解 |
| 3. 引擎直连 | wrapper 链位置、extra.gamefile 保原路径、goal 实改 | AlfredTWEnv 直连 | 全通 |
| 4. HTTP 服务 | env-var 桥、per-client 变体、episode 协议 | 真实 uvicorn 服务 | 全通 |
| 5. 联邦 GPU 冒烟 | 训练→checkpoint→FedAvg→暖复用→val→final | 0.5B 单卡 4090 | goal_variant + scene_disjoint 双双 LOOP CLOSED |

方法论出处:WebShop `rank_wrapper` 的 invert 臂当年**没验证可解性**就上线,
~33 个 client 实际不可解、只能事后附录披露——层 2 的存在就是为了让这类事故
在结构上不可能重演。

### 2.9 三条红线(所有 arm 共守)

1. **seed-42 / verbatim 红线**:既有分区函数一字不动;新增代码沿用同一批种子
   常数与公式;一切随机性 `RandomState(42[+f(client)])`,离线可重放。
2. **val 红线**:验证永远在共享的**无扰动** val 服务上打分(ALFWorld:
   `partition=uniform` 的 140 游戏 valid_seen;kernel wrapper 只在 train 插入;
   GPU 冒烟里实证了 val 服务日志零 KERNEL 行)。跨 arm 曲线因此可比。
3. **fail-loud 红线**:所有字符串改写用"精确锚点 + 出现次数断言",数据版本
   漂移时服务**拒绝启动**而不是静默跑错科学;未知措辞拒绝归一化。

### 2.10 工程机制备忘(实现者视角)

- **wrapper 注入点**:textworld 的 wrapper 列表按序 inner→outer 构建,我们的
  改写 wrapper 放**最内层**(列表第 0 位),外层 `AlfredInfos` 因此记录**原始**
  游戏路径(`extra.gamefile` / seed==index 模式不受影响),后端加载的才是改写副本。
- **临时文件方案**:`GenericEnvironment.load` 靠路径后缀猜后端,dict 传不下去
  ⇒ 改写后写进 wrapper 实例私有的 `*.tw-pddl` 临时文件再转发。每次 episode
  reset 都重新 load ⇒ 改写每 episode 生效,成本 ~60KB 字符串写,可忽略。
- **进程级超时**:fast-downward 是 C 扩展长调用,`signal.alarm` 打不断(handler
  只在 Python 字节码边界触发;实测有实例卡 650s)⇒ 唯一可靠的截杀是 fork 子进程
  + `Pipe.poll(timeout)` + `terminate()`。verify 工具与 cell 生成器都用这套。
- **replan 模板化上游 bug**:本 textworld 版本对带存在量词前提的动作
  (examineReceptacle)做 `plan_to_templated_actions` 时 IndexError——原生 goal
  从不需要 examine 所以从未暴露;一旦 `checked` 目标相关即触发。绕行:取
  fast-downward 的原始 grounded operator 序列,与 `_valid_actions`/
  `_valid_commands`(同索引)按"动作名 + 参数按序子集"匹配后执行。只影响验证
  工具,不影响训练路径(服务端从不请求 policy_commands)。
- **静默死锁陷阱**(联邦冒烟配置):固定 `total_training_steps` **大于**可推导
  步数(dataloader 每 epoch 1 个 batch ⇒ 步数=epochs)时,fit 在 dataloader 处
  提前结束、verl 的"到达总步数才保存"的末步 checkpoint 不触发 ⇒ FedAvg 无物可
  平均 ⇒ driver 以 base 模型收尾、worker 永远等下一轮计划,**双侧零报错互等**。
  修法:冒烟配置一律 `total_training_steps: 0`(由 epoch 推导)。诊断入口:
  `_xround/` 协调文件(done_N/go_N/plan)标明双方各自认为的阶段。
- **conda 环境分工**:`fedagent-verl08` = 训练器侧(有 numpy/verl,无
  textworld/matplotlib);`verl-agent-alfworld` = 环境服务侧(有 textworld/
  fast-downward/alfworld)。单测设计为前者可跑,planner 层工具须后者。

### 2.11 数据资产清单

| 文件 | 内容 | 生成/校验 |
|---|---|---|
| `data/alfworld_games/{train,eval_*}.json` | 权威游戏 manifest(pin 集合与顺序;train 3553) | `tools/gen_alfworld_manifest.py --check` |
| `data/env_heterogeneity/alfworld_cells_v1.json` | cell 库(8169 cell + 元信息 + 双 sha256) | `tools/env_heterogeneity/gen_alfworld_cells.py`(`--check` 复现校验) |
| `data/env_heterogeneity/holdout_alfworld_v1.json` | 8 个 OOD 保留 scene(scene_disjoint 可选) | `gen_holdout_alfworld.py --check` |
| `qwen05b_runs/configs/alfworld_{goal_variant,scene_disjoint}_smoke_0p5b.yaml` | 0.5B 单卡冒烟配置(已跑通) | — |
| `fedagent/config/examples/alfworld/2cl_{goal_variant,scene_disjoint}.yaml` | 4-GPU 1.5B 示例配置 | — |
| `fedagent/config/{paper,paper_accelerated}/env_heterogeneity/{grpo,ppo}/{webshop,alfworld}/<arm>/` | **正式 paper 配置家族**(2026-08-23 重组):每 arm 的 grpo/ = 全旋钮扫描、ppo/ = 最发散点;WebShop 6 arm(含新增 task_disjoint 消融)+ ALFWorld 4 arm,共 34 cell/树 | `python -m tools.gen_paper_configs [--accel]`;家族内自动生成 README.md 列全表 |

### 2.12 代码触点地图

| 文件 | 角色 |
|---|---|
| `fedagent/envs/alfworld/engine/agent_system/environments/alfworld_kernel_variants.py` | 变体池、分配、三类锚点改写、措辞归一化、rewrite-wrapper 工厂 |
| `.../environments/partition_strategy.py` | `_scene_disjoint_partition_alfworld`(+verbatim 的旧函数们) |
| `.../env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py` | 策略调度、uniform 切片、wrapper 插入(init_env) |
| `fedagent/envs/alfworld/service/server.py` | env-var 桥(`_partition_kwargs`) |
| `fedagent/fed/run_fed.py` | 配置键→env var 导出;val 服务不导 het 变量(红线) |
| `tools/env_heterogeneity/verify_alfworld_kernel_variants.py` | 层-2 验证器(raw-operator 执行 + 进程超时) |
| `tools/env_heterogeneity/gen_alfworld_cells.py` | cell 库生成器 |
| `tests/test_alfworld_kernel_variants.py` | 层-1 单测(20 条) |
