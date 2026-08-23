# ALFWorld 环境级异质性(Environment Heterogeneity)设计与实现

> 状态:设计定稿 + 全量实现 + 五层验证全通(含 0.5B 联邦 GPU 冒烟,§7.4)。
> 日期:2026-08-22/23。作者:canyu + Claude。
> **生词/概念先查 [README.md](./README.md)**(dev_doc 总入口:阅读顺序 + 概念手册,
> 绑定率/恒真面/措辞归一化/cell 等术语在那里从零讲清)。
> 姊妹文档:[../heterogeneity.md](../heterogeneity.md)(WebShop 五变体的构造参考)、
> [alfworld_query_env_decoupling.md](./alfworld_query_env_decoupling.md)(cell 库)、
> [webshop_vs_alfworld_env_heterogeneity.md](./webshop_vs_alfworld_env_heterogeneity.md)
> (两环境对比)、[../bugfixes.md](../bugfixes.md)(env_disjoint 知识背景)。

## 0. TL;DR

WebShop 的环境级异质性把检索流水线拆成 content / encoding / matching / rendering 四级,
每级一个扰动变体。本文把同一套方法论迁移到 ALFWorld:利用 `game.tw-pddl` 是
**纯 JSON(pddl_domain / grammar / pddl_problem 三个字符串)** 且 textworld 的
`PddlEnv` 在每次 episode reset 时重新 `load()` 的事实,在 wrapper 层对三个字符串做
**按 client 确定性的字符串改写**,得到四个新 arm:

| arm(`partition_strategy`) | 扰动通道 | WebShop 对应物 | 对 policy 隐藏性 | 预期 regime |
|---|---|---|---|---|
| `scene_disjoint` | content:哪些场景/游戏存在 | Catalog Split | 部分(通过后继状态感知) | stable → degrade |
| `obs_variant` | rendering/encoding:状态如何渲染为文本 | Field-Subset / Rank Wrapper | 部分 | degrade |
| `dyn_variant` | matching/dynamics:动作前提与效果 | BM25 Reweighting | ✅(仅经后继状态) | degrade → collapse |
| `goal_variant` | 隐藏成功判定(reward 通道) | Lookalike Injection | ✅✅(episode 末才暴露) | **collapse** |

四个 arm 复用 WebShop 的全部科学不变量:seed-42 红线、按 `client_id` 确定性、
跨轮稳定、val 服务不扰动、goal 文本(τ)逐字不变。

---

## 1. 动机与形式化定位

### 1.1 Input-Dynamics Asymmetry 在 ALFWorld 上的落点

论文主张:联邦 agent RL 对**任务级**异质性(τ 在 prompt 里,可观测)鲁棒,对
**环境级**异质性(转移核 P 隐藏,只能经后继状态感知)最坏情形不鲁棒。目前
ALFWorld 只有任务级三 arm(preference/coverage/hardness),环境级为空
(heterogeneity.md:"*(none, WebShop-specific)*"),仅有一个未见报告的
`env_disjoint`。补齐 ALFWorld 环境级 arm 之后,论文的 env-het 结论才具备
**跨环境泛化性**(roadmap 第 7 行的完整版)。

### 1.2 ALFWorld 比 WebShop 更干净的一点

同一 spec(`<task_type>-<obj>-<recep1>-<recep2>`)跨 scene 的 goal **语义逐字相同**
(pddl_params 与 `(:goal)` 块与 scene 无关;已在 3553 个 train 游戏上抽验)。

> **更正(2026-08-23,经 [alfworld_query_env_decoupling.md](./alfworld_query_env_decoupling.md) §3
> 指出并独立复核确认)**:goal **文本**并非逐字相同——`get_templated_task_desc`
> 在生成期对每类 goal 的 **2 个措辞模板**做一次 `random.choice` 并冻结进 grammar
> (与 scene 无关的生成噪声):550 spec 中 438 个(80%)带两种措辞,全库 988 个
> 不同 task 字符串。对三个 `*_variant` arm 无新增影响(它们的 uniform 游戏切片与
> 非异质基线完全一致,措辞噪声属于基线本身);对 `scene_disjoint`,τ 不变性
> 成立于**语义/类型层**,词汇层存在二阶泄漏(同 spec 在不同 client 可能是
> "put two book in desk." vs "find two book and put them in desk.")——论文表述
> 按此校准。**词汇归一化已实现(2026-08-23)**:`canonicalize_task_text` +
> `make_task_normalizer_wrapper`(alfworld_kernel_variants.py),`init_env` 在
> scene_disjoint + train 时默认插入,val 不受影响;全库测试 550 spec 归一后
> 每 spec 恰 1 种措辞。词汇级 τ 泄漏就此关闭,scene_disjoint 的 τ 不变性
> 恢复为构造性保证。

### 1.3 一个必须诚实处理的形式化区别

`goal_variant` 改写 `(:goal ...)`,形式上扰动的是 **reward 函数 R(s)** 而非转移核
P。但可观测性论证完全同构:R 与 P 一样不出现在 prompt 中,agent 只能通过
episode 末的成功/失败信号感知;且 WebShop 的 Lookalike 也正是"通过 reward
子项交互制造 π* 结构性发散"。文档与代码注释统一表述为:
**hidden-kernel 三 arm(scene/obs/dyn)扰动 P 或观测核 O,goal_variant 扰动隐藏
R;四者同属'不可观测通道',与任务级 arm(可观测 τ)相对**。

---

## 2. 机制分析(实现依据,全部已验证)

### 2.1 textworld 加载链(决定注入点)

服务端 `AlfredTWEnv.init_env(batch_size=1)` →
`textworld.gym.register_games(..., wrappers=[alfred_demangler, AlfredInfos])` →
`TextworldBatchGymEnv`。关键事实(源码:conda env `verl-agent-alfworld` 的
textworld 包):

1. **wrapper 链按列表序 inner→outer 构建**(`_make_env`:
   `for wrapper in wrappers + [Filter]: env = wrapper(env)`)。完整链:
   `GenericEnvironment → Limit(50 步) → wrappers[0..n] → Filter(最外)`。
   把 kernel wrapper 放 **wrappers[0]**(最内)即可:外层 `AlfredInfos` 先记录
   **原始路径**(`extra.gamefile`、`SEED_IS_INDEX` 索引模式全都不受影响),再向内
   转发。
2. **batch_size=1 → SyncBatchEnv,同进程**(`asynchronous` 仅在 batch>1 生效),
   无 pickle/多进程问题。
3. **每次 episode reset 都重新 `load(下一个 gamefile 路径)`**
   (`TextworldBatchGymEnv.reset` → `batch_env.load(gamefiles)`),所以 wrapper 在
   load 时做改写 = 每个 episode 都生效,且成本可忽略(~60KB 字符串操作,远小于
   fast-downward grounding)。
4. **`GenericEnvironment.load` 用 `path.endswith` 猜后端,dict 传不下去**
   (`_guess_backend` 对 dict 抛 AttributeError)。⇒ wrapper 采用**临时文件**方案:
   改写后写入本实例专属的 `*.tw-pddl` 临时文件,再向内转发临时路径。每实例一个
   固定临时文件,每次 load 覆写;`PddlEnv.load` 对它 `json.load` 照常工作。
5. `env.seed(seed)` 只 reshuffle 顶层 gamefiles 迭代器,与 wrapper 无交互。

### 2.2 game.tw-pddl 的字节稳定性(决定锚点改写可行性)

在 3553 个 train 游戏中随机抽 120 个实测:

- `pddl_domain`:**单一 md5,全库字节一致**(alfred.pddl 内容嵌入每个游戏)。
- `grammar`:去掉 task 行(`"rhs": "Your task is to: ..."`)后**全库字节一致**。
- ⇒ 所有改写用**精确子串锚点 + 出现次数断言**(锚点缺失/数目不符即 raise,
  fail-loud,科学红线风格),不需要 PDDL parser。

### 2.3 各 task type 的 goal 结构(决定 goal_variant 注入点)

6 类 goal 已全部 dump(见 §4.4):5 类(pick_simple / clean / cool / heat /
pick_two)含 `(receptacleType ?r XType)` + `(inReceptacle ?o ?r)`,注入点即该
`and` 块;`look_at_obj_in_light` 无目标 receptacle,但第二个 exists 有
`(holds ?a ?o)`(持物),注入 `(checked ?o)`(examineObject 前提恰是 holds)。

### 2.4 关键谓词与动作(决定 dyn/goal 变体语义)

- `examineReceptacle`:前提仅"在场",效果 `(checked ?r)`,1 步,处处可用。
- `OpenObject` 效果**顺带设 `(checked ?r)`** ⇒ `v_examined` 对 openable 目标可能
  被开门动作顺带满足,其约束力主要落在 **非 openable** 目标(75%);
  `v_closed`(`(not (opened ?r))`)只约束 **openable** 目标(25%)。两变体恰好
  **互补覆盖** goal-receptacle 空间。
- `PutObject` 效果块可无条件追加 `(not (opened ?r))`:对非 openable receptacle,
  `opened` 恒假,该文字是无害恒真改写 ⇒ 不需要条件效果(`when`)。
- `CleanObject` 需要 `SinkBasinType` receptacle 在场 + `cleanable ?o`。
- 步数预算:`Limit(max_nb_steps_per_episode=50)`(config_tw.yaml dagger 段)。
  各变体最优路径增量 ≤ +2 步/目标,远在预算内(由 planner 验证兜底)。

### 2.5 适用面统计(全库 3553 游戏逐一解析)

| 量 | 值 | 含义 |
|---|---|---|
| 任务型分布 | pick 790 / pick_two 813 / clean 650 / cool 533 / heat 459 / look_at 308 | — |
| goal receptacle openable(`v_closed` 有约束力) | **890 / 3553 = 25.0%**(pick 112, clean 230, cool 161, heat 176, pick_two 211) | 其余游戏该变体为恒真(≈identity) |
| `v_examined` 有约束力 | 全部 6 类适用;主要绑定非 openable 目标(≈75%)+ look_at 持物 | 事实上的全库变体 |
| `v_clean`(pick_simple ∧ cleanable ∧ sink) | 仅 76 / 3553 = 2.1% | **太稀,不入默认池**,列为扩展 |
| scene 结构 | 108 scene(厨/卧/客/卫各 27),550 spec,311 个多 scene spec(2966 游戏) | scene_disjoint 的原料 |
| 每 scene 游戏数 | 均值 ~33(kitchen 1696/27≈63,living 477/27≈18) | 子采样可行性 |

### 2.6 admissible-commands 泄漏(设计红线)

[alfworld_env.py](../../envs/alfworld/alfworld_env.py) 每回合把
`admissible_commands` 写进 prompt,而 admissible 由 **PDDL 前提推导**、其字符串由
grammar 模板渲染。⇒ **凡是改动作模板措辞或让动作从 admissible 列表中显式增删的扰
动,当回合即被 agent 看见**,只能算部分隐藏(与 WebShop"搜索结果不同"同级:
经后继观测感知隐藏核,不违反 env-het 定义,但到不了完全隐藏)。完全隐藏的只有:
**动作效果**(做了才知道)与 **goal 判定**(episode 末才知道)。四 arm 的隐藏性
标注(§0 表)据此给出;论文叙事沿用 WebShop 的 stage-severity 排序逻辑。

### 2.7 既有 `env_disjoint` 为何不够(实测)

20 client 分区实测(fedagent-verl08,stub 掉 matplotlib):

| env_div | J(games) | J(scenes) | scenes/client | 每 client 游戏数 | 100-client 并集 |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 97 | 311 | 311 (8.8%) |
| 0.3 | 0.348 | 0.878 | 96 | 311 | — |
| 0.7 | 0.119 | 0.856 | 96 | 311 | 2467 (69.4%) |
| 1.0 | 0.081 | 0.869 | 97 | 312 | 2939 (82.7%) |

三个缺陷:(a) 它是 **instance-disjoint 而非 scene-disjoint**——scene Jaccard 恒
~0.86,每个 client 都见 ~96/108 个 scene,P 的边缘分布几乎相同;(b) **env_div 与
数据量混淆**(并集覆盖 8.8%→82.7% 随 div 变化;C=10 时每 client 429→709);
(c) 同 spec 跨 scene 的抽象策略不变(receptacle 类型集 Jaccard 0.81),π* 基本
不发散。`env_disjoint` **保持原样不动**(verbatim 红线),新 arm 另起炉灶。

---

## 3. 四个 arm 的设计

### 3.1 `scene_disjoint`(content 级,Catalog Split 的 ALFWorld 同构物)

**目标**:client 间 scene 集合可控发散,同时 (i) 每 client 游戏数**固定**
(解耦数据量),(ii) 任务型边缘分布**匹配全局**(τ 不变),(iii) `env_div=0`
是完美同质 floor。

**构造**(`_scene_disjoint_partition_alfworld`,partition_strategy.py 新增):

1. **房型分层的 scene top-k**。scene 按 FloorPlan 号分四房型
   (1–30 厨 / 201–230 客 / 301–330 卧 / 401–430 卫,各 27 个)。每房型内:
   - `u`:`RandomState(base_seed)` 按 sorted(scenes) 顺序为每 scene 抽共享分;
   - `v_k`:`RandomState(base_seed + 1000*client_id)` 同序抽 per-client 分
     (与 catalog_split 的 ASIN-keyed u/v 完全同构,按 scene 字符串排序 keying);
   - `e_k = (1-env_div)·u + env_div·v_k`,取 top-(spc/4)
     (`spc = scenes_per_client`,默认 8 ⇒ 每房型 2 个)。
2. **任务型配额子采样**。合并所选 scene 的全部游戏后,按全局任务型边缘
   `frac_t`(§2.5)给出配额 `quota_t = round(n_games·frac_t)`(最大余数法配平),
   每型内用共享 `RandomState(base_seed)` 洗牌序取前 quota_t 个。
   `n_games = min_games_per_client`(默认 100)。某型池不足时取尽并把缺口按
   剩余容量回填到其他型(打印 WARNING)。
3. `holdout_scenes` 与 env_disjoint 同语义(shipped
   `data/env_heterogeneity/holdout_alfworld_v1.json` 直接复用)。

**性质**:`env_div=0` ⇒ 所有 client 的 scene 集与游戏集**逐字节相同**(纯 u 排
序);`env_div=1` ⇒ 各排各的 v。|games| 恒定,τ 配额恒定。注意 27 scene/房型、
C 大时 scene 必然重叠("disjoint"指趋势而非字面);发散度用 pairwise Jaccard 报
告(实测数字见 §7)。

**知识局限(写进论文 appendix)**:scene 与任务型天然相关(heat/cool/clean 只在
厨房),分层 + 配额把τ 边缘钉住,但 client 内"任务型×scene"联合分布仍与全局不
同(其厨房任务全部来自它的 2 个厨房)。这是 scene 级 content 扰动的固有代价,
对照 WebShop catalog_split 的"per-client 目标底座 + distractor 发散"属同类近似。

### 3.2 `obs_variant`(rendering/encoding 级)

每 client 确定性分到一个 **grammar 改写**;catalog、games(uniform 契约切片)、
dynamics、goal 全部不动,只有**状态→文本的观测核 O** 不同。池
(`N=4`,`N=2` 取前二):

| key | 改写(锚点级,任务 rhs 永不触碰) | 语义 |
|---|---|---|
| `v_default` | 无 | 对照 |
| `v_terse_goto` | `GotoLocation.feedback` rhs 去掉 ` #examineReceptacle.feedback#` | 到达后不再自动播报台面/容器内容,须显式 `examine`(内容才可见);admissible 仍会泄漏 take 目标(§2.6),定级 degrade |
| `v_blind_intro` | `intro` rhs 去掉 `#look.feedback#\n\n`(保留 `#task#`!) | 开局无房间描述,须先 `look`;τ 文本原样保留 |
| `v_paraphrase` | 5 条反馈 rhs 改写措辞(arrive/open/close/pick/move,占位符 `{r.name}` 等原样) | 表层语言漂移,最弱 |

### 3.3 `dyn_variant`(dynamics 级,纯隐藏 P)

每 client 确定性分到一个 **pddl_domain 改写**(块级定位:先按平衡括号提取
`(:action X ...)` 块,再在块内做锚点替换+次数断言)。池(`N=4`):

| key | 改写 | 后果 |
|---|---|---|
| `v_default` | 无 | 对照 |
| `v_examine_gate` | `PickupObject` 前提 `(pickupable ?o)` 后插入 `(checked ?r)` | 未 examine/open 过的 receptacle 上拿不了东西;"take"要等 examine 后才进 admissible——学会"先看再拿"协议 |
| `v_autoclose` | `PutObject` 效果追加 `(not (opened ?r))` | 放完东西容器自动合上(反馈不提示!);pick_two 需重新开门,启发式"开着就一直开着"失效 |
| `v_gate_autoclose` | 两者叠加 | 最强,对应 lookalike 的组合攻击位 |

**绑定面(planner 实测修正)**:`v_examine_gate` 约 83% 游戏绑定(未绑定的是
"源容器本来就要开门"的任务——open 顺带设 `checked`,如 fridge 里的 apple);
`v_autoclose` 对**最优路径**的绑定面仅 **pick_two × openable 目标 ≈ 26% 的
pick_two、~6% 全库**(单次放置任务放完即达标,关门不影响 goal)——它的主要作用
是扰动**探索期**的可达状态分布(RL 策略的"开着就一直开着"启发失效),
论文表述按此校准,不宣称最优路径级的普遍绑定。

### 3.4 `goal_variant`(隐藏 R 级,Lookalike 同构物)

每 client 确定性分到一个 **pddl_problem `(:goal)` 改写**;goal **文本**(grammar
task 行)逐字不变,只有 `check_goal()` 用的谓词变了——agent 拿到同样的指令,
但对某些 client"把东西放到位"还不够,须多满足一个隐藏子条件。池(`N=4`,
`N=2` 取前二):

| key | 注入(在 goal 的 `(receptacleType ?r T)` 处追加) | 约束面 |
|---|---|---|
| `v_default` | 无 | 对照 |
| `v_examined` | `(checked ?r)`;look_at 改注 `(holds ?a ?o)` → `+ (checked ?o)` | 全 6 类;主要绑定非 openable 目标(75%) |
| `v_closed` | `(not (opened ?r))`(look_at 恒等) | openable 目标(25%),与 v_examined 互补 |
| `v_examined_closed` | 两者叠加 | 最强 |

**π* 结构性发散论证**(与 lookalike 平行):不同 client 的最优策略在**同一指令**
下分别是"放好即走 / 放好并 examine / 放好并关门 / 全都要"。单一聚合策略无法同时
最优;PPO 的 critic 能吸收 per-client 回报偏移,GRPO 的组内基线不能——给出与
WebShop 相同的 GRPO-vs-PPO 配对故事。

**已弃选项**:`v_clean`(隐藏"还要洗干净")语义最漂亮,但适用面仅 2.1%(§2.5),
只列为 N=8 扩展位,须配 per-game 适用性过滤 + planner 重验。

### 3.5 变体分配(全部 arm 共用)

WebShop `_bm25_variant_partition_webshop` 的逐字同款数学:

```python
rng = np.random.RandomState(base_seed + client_id)   # base_seed = 42 硬编码
chosen = pool[rng.randint(len(pool))]                # pool = POOLS[strategy][:N]
```

同 client 跨轮同变体(FedAvg 可比性),分配可离线重算(实验记录)。

> **N=2 的小陷阱(实测)**:`RandomState(42).randint(2)` 与
> `RandomState(43).randint(2)` **都是 0**——2-client 冒烟在 `variant_n=2` 下两个
> client 都拿到 `v_default`,没有不对称。示例配置因此用 `variant_n=4`
> (client0=`v_closed`,client1=`v_default`)。大规模联邦(≥20 client)下各变体
> 占比正常(20 client N=4 实测 7/6/4/3)。分配数学是 WebShop verbatim 红线,
> 不为此改动。

---

## 4. 实现地图(文件级)

### 4.1 新文件

| 文件 | 内容 |
|---|---|
| `fedagent/envs/alfworld/engine/agent_system/environments/alfworld_kernel_variants.py` | 变体池、`variant_for_client()`、三类改写函数(锚点+次数断言)、`rewrite_game_data()`、`make_kernel_wrapper()`(textworld 懒导入 + 临时文件转发) |
| `tests/test_alfworld_kernel_variants.py` | 纯字符串/分配/分区单测(fedagent-verl08 可跑,无 textworld 依赖) |
| `tools/env_heterogeneity/verify_alfworld_kernel_variants.py` | planner 级可解性验证 + 端到端执行(verl-agent-alfworld 环境) |
| `fedagent/config/examples/alfworld/2cl_goal_variant.yaml`、`2cl_scene_disjoint.yaml` | 冒烟配置 |

### 4.2 修改点(全部为追加式,不触碰 verbatim 分区体)

| 文件 | 改动 |
|---|---|
| `.../environments/partition_strategy.py` | 追加 `_scene_disjoint_partition_alfworld`;`partition_dataset` 增 `scene_disjoint` 分支;错误信息里的策略清单更新 |
| `.../alfworld/agents/environment/alfred_tw_env.py` | `_shard_game_files` 增两分支:`scene_disjoint`(走 partition_dataset)与三个 `*_variant`(游戏走 uniform 契约切片 + 计算 `self.kernel_variant`);`init_env` 在 train 且有 variant 时把 kernel wrapper 插到 wrappers[0] |
| `fedagent/envs/alfworld/service/server.py` | `_partition_kwargs` 增 `scene_disjoint`(env_div/scenes_per_client/holdout)与 variant 三策略(variant_n);新读 `VARIANT_N` / `ALFWORLD_SCENES_PER_CLIENT` / `ALFWORLD_HOLDOUT_FILE` |
| `fedagent/fed/run_fed.py` | ALFWorld service env 增导 `VARIANT_N`、`ALFWORLD_SCENES_PER_CLIENT`(镜像 WebShop 分支的 969 行);DEFAULTS 增 `alfworld_scenes_per_client`;注释清单更新。**val 服务分支不动**(不导 PARTITION_STRATEGY ⇒ 天然 uniform,科学红线) |
| `fedagent/docs/heterogeneity.md` | ALFWorld env-het 段落与 arm 映射表补四行,指向本文档 |

实现期间的三处顺带修复:(a) `alfred_tw_env.slice_games_for_client` 尾部的
`get_partition_info` 对 variant 策略映射为 `uniform`(variant_n 不是
partition_dataset 的 kwarg);(b) FedAgent 版 `partition_strategy.py` 的
matplotlib/seaborn 硬 import 改为 guarded-optional(AccelAgent 侧早已如此;
fedagent-verl08 无 mpl,trainer 侧工具与单测因此可导入);(c) 引擎/run_fed 的
策略清单错误信息与注释同步新增四策略。

### 4.3 环境变量桥(run_fed → service → engine)

| run_fed 键 | env var | 消费者 | 适用 arm |
|---|---|---|---|
| `partition_strategy` | `PARTITION_STRATEGY` | server → AlfredTWEnv | 全部 |
| `env_div` | `ENV_DIV` | scene_disjoint(沿用既有导出) | scene_disjoint |
| `alfworld_scenes_per_client`(新,默认 8) | `ALFWORLD_SCENES_PER_CLIENT` | scene_disjoint | scene_disjoint |
| `variant_n`(复用 WebShop 键,0=池默认) | `VARIANT_N` | obs/dyn/goal_variant | 变体三 arm |
| `holdout_file` | `ALFWORLD_HOLDOUT_FILE` | scene_disjoint(OOD 保留) | scene_disjoint |

### 4.4 goal 注入锚点备忘(全六类实测结构)

- 5 类:`(receptacleType ?r XType)`(pick_two 中出现两次,注入函数按次数断言后
  **每处都注**——同一 ?r 的重复合取,PDDL 语义幂等)。
- look_at:两个 exists;注入点为 `(holds ?a ?o)`(第二 exists,持物)。
- `(:goal` 块用平衡括号提取,改写只发生在块内,拼回原 problem。

---

## 5. 验证协议(四层,全部自动化)

1. **纯字符串单测**(`tests/test_alfworld_kernel_variants.py`,fedagent-verl08):
   锚点改写生效且次数精确;task rhs 逐字不变(τ 红线断言);v_default 恒等;
   `variant_for_client` 确定性(跨进程重算一致)与 N 截断语义;
   scene_disjoint:div=0 全 client 字节相同、|games| 恒定、任务型配额匹配、
   确定性、holdout 剔除。
2. **planner 可解性验证**(`tools/env_heterogeneity/verify_alfworld_kernel_variants.py`,
   verl-agent-alfworld):按任务型分层抽样游戏 × 全部变体,`PddlEnv.load(改写 dict)`
   → fast-downward 求解 → 逐步执行 → 断言 `won`;记录计划长度增量
   Δsteps(应 ≤ +2/目标)与不可解计数(应为 0;>0 即 fail);另测每变体的
   **绑定率**(default 计划在变体核下重放是否失效)与 obs 变体的动力学纯净性
   (default 计划必须仍赢)。
   > **上游缺陷绕行(实测发现)**:本 textworld 版本的
   > `replan()`→`plan_to_templated_actions` 对参数带存在量词前提的动作
   > (`examineReceptacle`)模板化时 IndexError——原生 ALFWorld goal 从不需要
   > examine,所以从未暴露;一旦 `checked` 目标相关即触发。验证工具因此改取
   > fast-downward 的**原始 grounded operator 序列**,与
   > `state["_valid_actions"]`/`_valid_commands`(同索引)按"动作名 + 参数按序
   > 子集"匹配后执行——这正是 `PddlEnv.step` 内部的命令解析路径。
   > 该缺陷只影响验证工具的取计划方式,**不影响训练路径**(服务端从不请求
   > policy_commands)。
3. **端到端服务冒烟**:以 `PARTITION_STRATEGY=goal_variant CLIENT_ID=… CLIENT_NUM=2`
   起真实 HTTP service,/create → /reset → 按 walkthrough+补偿动作 /step,验证
   (a) 改写后 reward 语义生效(default client 按原 walkthrough 赢,examined client
   原 walkthrough 不赢、补 examine 后赢);(b) `extra.gamefile` 仍是原始路径
   (index 模式不破)。
4. **分区度量**(§7 数表):各 arm 的 pairwise Jaccard / scene 覆盖 / τ 配额
   偏差,写回本文档。

**科学红线检查单**:val 服务不受任何新 env var 影响(不导出即 uniform);
`env_disjoint`/既有分区体零改动;所有新随机性 `RandomState(42[+f(client)])`;
变体分配可离线重放。

---

## 6. 运行方式

```yaml
# goal_variant(隐藏成功判定,lookalike 同构;GRPO vs PPO 对照的主力 arm)
env_kind: alfworld
partition_strategy: goal_variant
variant_n: 4              # 0 ⇒ 池默认(4);N=2 取 [v_default, v_examined]

# scene_disjoint(content 级;catalog_split 同构)
env_kind: alfworld
partition_strategy: scene_disjoint
env_div: 0.7              # 0.0 同质 floor → 1.0 最大发散
alfworld_scenes_per_client: 8   # 每 client scene 数(房型各 1/4)
# holdout_file: data/env_heterogeneity/holdout_alfworld_v1.json   # 可选 OOD 保留

# obs_variant / dyn_variant 同 goal_variant,仅换 partition_strategy
```

冒烟:`python -m fedagent.fed.run_fed --config fedagent/config/examples/alfworld/2cl_goal_variant.yaml`。

**正式 paper 配置家族(2026-08-23)**:四 arm 的全扫描 cell 已入
`config/{paper,paper_accelerated}/env_heterogeneity/{grpo,ppo}/alfworld/<arm>/`
(scene_disjoint 四点 div 扫描 + 三个变体 arm N∈{2,4},各配 PPO 最发散点;
paper 协议 total-100 × 2/round × 70 轮)。全家族映射见生成的
[env_heterogeneity/README.md](../../config/paper/env_heterogeneity/README.md)。

---

## 7. 实测数字(2026-08-22,实现当日)

### 7.1 scene_disjoint 分区度量(C=20,spc=8,n_games=100,全 3553 游戏池)

| env_div | J(games) | J(scenes) | τ 配额最大偏差 | \|games\| | div=0 全 client 字节相同 |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 0.003 | 100(恒定) | ✅ |
| 0.3 | 0.160 | 0.195 | 0.087 | 100(恒定) | — |
| 0.7 | 0.068 | 0.080 | 0.003 | 100(恒定) | — |
| 1.0 | 0.033 | 0.043 | 0.057 | 100(恒定) | — |

对照 `env_disjoint`(同 C=20):J(scenes) 恒 0.856–0.878、每 client ~96/108 个
scene、并集覆盖随 div 从 8.8%→82.7%(数据量混淆)。scene_disjoint 三个设计目标
(scene 级发散、尺寸解耦、τ 边缘保持)全部达成;div 曲线在 0.3 处已陡降是
top-k 对混合噪声敏感所致,论文 arm 用端点 {0.0, 1.0}。

### 7.2 变体可解性 + 绑定验证(planner 层,每任务型 5 游戏 × 9 变体 = **270 组合,0 不可解**)

> **绑定(binding)的精确定义**:把默认核下的最优计划原封不动放到变体核下重放,
> 若不再获胜则该变体在该游戏上"绑定"。它模拟的正是联邦异质性的伤害机制——在
> 自己环境里学会的策略,换到别的 client 的环境里失效。三态:identity(字符串都
> 没改,不计入分母)/ nobind(改了但行为无差,恒真面/稀释)/ BIND。obs 变体的
> 判据相反:default 计划**必须仍赢**(只许改观测,不许改动力学)。详见
> [README.md](./README.md) §2.2–2.3。

| 变体 | 可解 | Δsteps 均值/最大 | 绑定率 | 备注 |
|---|---|---|---|---|
| obs/v_terse_goto | 30/30 | 0/0 | —(default 计划 30/30 仍赢,动力学纯净 ✅) | goto 反馈 30/30 不再列容器内容(default 27/30 列出) |
| obs/v_blind_intro | 30/30 | 0/0 | — | 开局房间描述 30/30 移除,task 行 **30/30 保留**(τ 红线) |
| obs/v_paraphrase | 30/30 | 0/0 | — | — |
| dyn/v_examine_gate | 30/30 | 0.9/1 | **27/30 (90%)** | 未绑定例 = 源容器本要开门的任务(open 顺带 `checked`) |
| dyn/v_autoclose | 30/30 | 0/0 | 0/30(manifest 序样本无实例级 openable 目标面) | 见实例级 openable 发现;工具对此发 WARNING 而非 fail |
| dyn/v_gate_autoclose | 30/30 | 0.9/1 | 27/30 | 由 gate 提供绑定 |
| goal/v_examined | 30/30 | 1.0/1 | **30/30 (100%)** | 每目标 +1 步(examine) |
| goal/v_closed | 25/25(+5 identity) | 0/0 | 0/25(同上,样本偏差) | look_at=identity(按设计);适用面行为见定向验证 8/8 |
| goal/v_examined_closed | 30/30 | 1.0/1 | 30/30 | — |

**定向验证(openable 目标容器,Drawer/Cabinet/Fridge/Microwave 各 2 游戏)**:
`v_closed` / `v_examined_closed` **8/8 绑定、+1 步、全部可解**——确认适用面上
行为真实。`v_autoclose` 在单次放置上 +0/不绑定(放完才关,goal 已满足)。

**实例级 openable 发现(pick_two probe,3 游戏)**:`(openable X)` 是**实例级**
init 事实——同为 DrawerType 的实例可以没有 openable 事实(bedroom-304 的目标
drawer:init 无 `opened`、default 计划 6 步全程不开门直接放)。probe 实测:

| 游戏 | v_autoclose | v_gate_autoclose |
|---|---|---|
| pick_two-Book-**Drawer**-304(实例非 openable) | +0 nobind(恒真面) | +2 BIND(纯 gate:两个源容器各 examine 一次) |
| pick_two-Bread-**Fridge**-7 | **+1 BIND**(放 1 → 自动关 → 重开再放 2) | +2 BIND |
| pick_two-CD-**Safe**-317 | **+1 BIND** | +3 BIND |

结论:(a) 基于类型的"890/3553=25%"是**上界**,实例级适用面更小;(b) goal 的
`exists ?r` 允许规划器选非 openable 实例绕过 `v_closed`/`v_autoclose` 的合取
(仍可解,该游戏无行为差,计入稀释);(c) `v_autoclose` 在其真实面
(pick_two × 实例 openable)上**绑定验证通过、+1 步、全部可解**,但它是池中
最弱成员——角色为探索期动力学扰动 + 组合臂成分,`N=2` 截断本就不含它;
(d) `v_examined` 不受实例问题影响(checked 对任何实例都须显式动作达成),
是 goal 池主力,n=30 实测 100% 绑定。

### 7.3 集成与服务冒烟(全通过)

- **AlfredTWEnv 直连**:wrapper 链实测
  `Filter → AlfredInfos → AlfredDemangler → _AlfredKernelWrapper → Limit →
  GenericEnvironment → PddlEnv`(kernel 最内);`extra.gamefile` 保留原始路径
  (val 的 seed==index 模式安全);后端 `pddl_problem` 含注入合取;连续 episode
  经同一临时文件正常。
- **HTTP 服务(goal_variant, VARIANT_N=4, client 0/2)**:env-var 桥 →
  `variant=v_closed` 日志 → /create → /reset → /step 全通,reward 路径正常。
- **HTTP 服务(scene_disjoint, ENV_DIV=1.0, spc=8, holdout v1)**:
  `|game_files|=100 from 8 scenes`,holdout 8 scene 剔除生效,episode 的 scene
  落在 client shard 内,task_mix={look 9, pick 22, clean 18, cool 15, heat 13,
  pick_two 23} ≈ 全局边缘 {8.7, 22.2, 18.3, 15.0, 12.9, 22.9}%。
- **单测**:`tests/test_alfworld_kernel_variants.py` 20/20(fedagent-verl08,无
  textworld 依赖;含全库措辞归一化收敛测试);相邻套件(game_manifest /
  catalog_split_runtime)无回归。

### 7.4 联邦 GPU 冒烟(第 5 层,2026-08-23,0.5B 单卡 4090)

`goal_variant`(N=4,c0=v_closed/c1=v_default)与 `scene_disjoint`
(env_div=1.0,spc=8,scene 集完全不相交 + 措辞归一化实机生效)**双双
FEDERATED LOOP CLOSED**:2 client × 2 轮,训练 → checkpoint → FedAvg(ws=1)
→ round 2 从聚合模型 + 暖服务复用 → 无扰动 val(140 游戏 uniform,零 KERNEL
行)→ final eval 用 round-2 聚合;全程 0 Traceback。配置:
`qwen05b_runs/configs/alfworld_{goal_variant,scene_disjoint}_smoke_0p5b.yaml`;
结果:`qwen05b_runs/out/alfworld_*_smoke_0p5b/federated_summary.json`。
细节与首次尝试踩到的 `total_training_steps` 静默死锁陷阱见
[webshop_vs_alfworld_env_heterogeneity.md](./webshop_vs_alfworld_env_heterogeneity.md) §5.3。

---

## 8. 风险与已知边界

1. **admissible 泄漏**(§2.6):obs/dyn 变体的部分后果当回合可见,隐藏性弱于
   goal_variant;定级与叙事已按此校准,不宣称超出实际的隐藏性。
2. **v_closed/v_autoclose 的恒真面(实测校准)**:`v_closed` 仅对 openable 目标
   (890/3553 = 25%)有行为差(该面上 8/8 绑定验证通过);`v_autoclose` 的最优
   路径绑定面更窄(pick_two × openable ≈ 6% 全库),其价值在探索期扰动。稀释
   效应写入 appendix;必要时用 `ALFWORLD_TASK_TYPES` 过滤子池加浓(如 task_types
   = pick_two 单型池可把 autoclose 绑定面提到 ~26%)。
3. **临时文件方案**:每 env 实例一个固定 tmp 文件(进程退出遗留 ≤ pool_size 个,
   放 TMPDIR);若未来 batch_size>1 且 asynchronous,wrapper 实例共享问题与
   AlfredDemangler 相同(现有代码同边界,注释标明)。
4. **步数预算**:变体增量 ≤+2 步/目标,50 步预算安全;由 planner 验证层兜底,
   杜绝 rank_wrapper-invert 式"静默不可解 arm"事故重演。
5. **scene_disjoint 的 τ 近似**(§3.1 知识局限):边缘配额精确、联合分布近似;
   与 catalog_split 的近似程度同级,论文如实披露。
6. **grammar/domain 版本漂移**:锚点断言 fail-loud;若上游 ALFWorld 数据版本
   变更导致锚点失配,服务启动即 raise 而非静默跑错科学。
