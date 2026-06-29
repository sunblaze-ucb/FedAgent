# Heterogeneity（异质性）

FedAgent 的科学内核是一套**两级异质性套件**：一组 partition 构造，向联邦各 client
注入*受控、可测量*的统计差异，并被切分为两条结构上彼此独立的通道，从而让论文的核心
论断可以被**测量**而非假设 —— 联邦 agent RL 对**任务级**异质性是**鲁棒**的（任务描述符
就在 prompt 里，所以单一的聚合 policy 能据此 condition），但对**环境级**异质性是
**最坏情况下非鲁棒**的（transition kernel 是隐藏的，policy 只能通过后继状态间接感知它）。
这就是 **Input-Dynamics Asymmetry（输入-动力学不对称性）**。

一个 FedAgent client 在一个 task-augmented MDP 上训练：每个 episode 抽取一个任务描述符
`tau ~ D_tau`，并在某个 transition kernel `P` 下展开。跨 client 的异质性可以从**任一**通道
进入，而整套套件的组织方式都是为了让两者可分离：

- **任务描述符 `tau`** 通过 policy 的*输入通道*进入 —— 它字面上就是 prompt 的一部分。
  policy 能读到它，所以 `tau` 是**可观测**的，对任务异质 client 做 FedAvg 会收敛到一个在
  任务分布之*并集*上表现良好的 policy。这是**鲁棒**情形。
- **transition kernel `P`** 隐含在动力学之中。policy 永远看不到 `P`，只能通过后继状态间接
  感知它。`P` 是**不可观测**的，所以对 `P` 的扰动会让每个 client 的最优 policy `pi*_i`
  彼此拉开，破坏朴素聚合。这是**最坏情况下非鲁棒**的情形。

本指南既是构造参考（每条 arm 经过验证的 partition 数学，方便研究者重建它），也是
操作者视角（每条 arm 对应的确切 `run_fed` 旋钮与论文配置族）。模块布局与 env-var 桥接见
[`../hetero/README.md`](../hetero/README.md)；完整的 `run_fed` 字段参考见
[`./configuration.md`](./configuration.md)；配置到图表的映射见
[`./reproducing.md`](./reproducing.md)。

**目录**

1. [两个级别](#the-two-levels)
2. [任务级构造（可观测 `tau`）](#task-level-constructions-observable-tau)
3. [环境级构造（隐藏 `P`）](#environment-level-constructions-hidden-p)
4. [非对称鲁棒性谱（stable -> degrade -> collapse）](#the-asymmetric-robustness-spectrum-stable---degrade---collapse)
5. [Arm -> knob -> paper-config 映射](#arm---knob---paper-config-map)
6. [选择一条 arm](#selecting-an-arm)
7. [横切不变式](#cross-cutting-invariants)

---

## The two levels

| 级别 | 被扰动的通道 | 对 policy 可观测？ | WebShop arms | ALFWorld arms |
|---|---|---|---|---|
| **Task** | 在一个**共享、未扰动**环境上的目标分布 `D_tau` | **是**（目标在 prompt 里） | `preference`、`coverage`、`hardness`，外加 `task_disjoint` ablation | `preference`、`coverage`、`hardness` |
| **Environment** | **transition kernel `P` / catalog**（检索 pipeline） | **否**（仅通过后继状态） | `catalog_split` + 4 个检索-pipeline 变体（`bm25_field_subset`、`bm25_reweight`、`lookalike`、`rank_wrapper`） | *（无 —— WebShop 专属）* |

在整个**任务级**扫描中，transition kernel 保持固定（每个 client 都搜索**完整的
1000-product catalog**）；在整个**环境级**扫描中，任务划分保持**均匀**（100 goals/client，
无目标偏斜）。因此每条扫描中的发散都可单独归因于该级别。每条 arm —— 两个级别皆然 ——
都在**同一个共享的未扰动验证服务**（`WEBSHOP_SPLIT=val`，它忽略 `PARTITION_STRATEGY`）上
打分，正是这一点让跨 arm 曲线可直接比较。

`task_disjoint` 是隔离环境效应的干净 ablation：它复用 Catalog Split 的 goal-slice 数学，
为每个 client 服务与 `catalog_split` **相同**的不相交目标切片，但作用在**完整** catalog 上
（它设 `CATALOG_ASINS = None`）。因此，在匹配的 `env_div`/`keep_ratio` 下，`catalog_split`
与 `task_disjoint` 之间的任何发散都归因于隐藏的 catalog 扰动，而非目标划分。

全部构造代码位于 [`../hetero/`](../hetero/)，每条 arm 一个模块。每个模块都把它的 partition
主体从原始 verl-agent 的 `partition_strategy.py` 中**逐字**复制（这条*科学红线*：
`base_seed = 42` 硬编码、精确复制、不做任何意译），仅为 verl-0.8 的远程 env 服务添加一个薄薄的
`*_for_client(...)` 包装。共享的 Beta-sizing primitives
（`default_r` / `generate_client_sizes` / `assign_with_overlap`）位于
[`../hetero/_beta_sizing.py`](../hetero/_beta_sizing.py)。

---

## Task-level constructions (observable `tau`)

各 client 共享同一个环境元组（完整 catalog、固定 `P`），但其逐 client 的目标分布
`D_tau_i` 各不相同。三个在操作上可分离的子类型，各由**单一弥散旋钮**支配，从而可以在不
扰动另外两个轴的前提下移动其中一个。每个子类型都有 WebShop 与 ALFWorld 两套后端
（ALFWorld 从目标的 task-file 路径而非 product 字段推导其类别/难度）。

| 论文名 | 问题 | `run_fed` 旋钮 | 逐字 partition | 端点（近均匀 -> 极端） |
|---|---|---|---|---|
| **Preference** | *什么样的任务？* | `omega`（`OMEGA`） | `_preference_partition_generic` | `0.01` -> `0.99` |
| **Coverage** | *多少个任务？* | `size_std`（`SIZE_STD`），论文的 `xi` | `coverage_partition` | `256` -> `1` |
| **Hardness** | *任务有多难？* | `success_std`（`SUCCESS_STD`），论文的 `xi'`（+ `trajectories_file`） | `hardness_partition` | `256` -> `1` |

> **旋钮方向告诫。** `size_std`/`success_std` 命名上像标准差，但实际被作为 **Beta
> concentration** `dispersion_s`（`generate_client_sizes`）前传。它们*反向*设定 spread：
> **大**端点（`256`）近均匀；**小**端点（`1`）才是极端不平衡。`omega` 则按自然方向运行
>（`0.01` 近均匀，`0.99` 极端）。

### Preference — 目标类别上的 Dirichlet 偏斜

[`../hetero/webshop_task.py`](../hetero/webshop_task.py)，
`_preference_partition_generic`。每个 client 的类别混合从一个以全局类别边缘分布 `pi`
为中心的 Dirichlet 抽取，然后通过多项抽样得到逐类别的目标计数：

```python
omega = float(np.clip(omega, 1e-3, 1 - 1e-3))   # clipped into (1e-3, 1-1e-3)
# pi = smoothed global category marginal (raw counts + eps_smooth=0.01, renormalized)
alpha_vec = pi * ((1.0 - omega) / omega)        # concentration scales as (1-omega)/omega
q = rng.dirichlet(alpha_vec); q = q / q.sum()   # this client's per-category probs; E[q] = pi
counts = rng.multinomial(min_samples_per_client, q)   # per-category goal counts
```

`E[q] = pi` 精确成立，因此**全局**类别混合在期望意义下被保留，而**逐 client**混合则
各有差异。当 `omega -> 0` 时，concentration `alpha` 无界增长，每个 client 都收敛到 `pi`
（近 IID）；当 `omega -> 1` 时，`alpha -> 0`，每个 client 都坍缩到一个 one-hot 顶点
（一个 client ~ 一个类别）。逐 client 的 RNG 是 `RandomState(42 + client_id)`；在计算边缘
分布之前，`fashion` 类别被子采样到 20%（`fashion_sample_ratio`）；一个容量溢出再分配循环
加上一个补足 pass 保证每个 client 达到 `min_goals_per_client`。仅传 `tau` 的 legacy 配置会被
别名到 `omega`（两者都存在时 `omega` 胜出）；这个 `tau` 是 *Preference 旋钮的旧名*，与
MDP 任务描述符 `tau` 无关。

### Coverage — 带 overlap 的 Beta-弥散 pool 大小

[`../hetero/webshop_coverage.py`](../hetero/webshop_coverage.py)，
`coverage_partition`，构建于
[`../hetero/_beta_sizing.py`](../hetero/_beta_sizing.py) 之上。每个 client 的 **pool
size** 从一个 Beta 分布抽取；然后以受控的跨 client overlap 发放目标，使各 client pool 的
**并集**覆盖整个目标池。这改变的是每个 client 看到多少目标的*弥散*程度，而不改变逐
client 的均值或全局混合。sizing 带宽是固定的：

```python
center = 500; low = min_samples_per_client; high = 1000   # per-client size band
rng = np.random.default_rng(42)                            # fixed -> every client agrees
r = default_r(total_size, client_num, low, center, high)   # overlap coefficient r
target_sum = int(round(r * total_size))                    # total assignments to hand out
client_sizes = generate_client_sizes(C=client_num, low=low, center=center,
                                      high=high, dispersion_s=size_std,
                                      target_sum=target_sum, rng=rng)
client_sets, k = assign_with_overlap(total_size, client_sizes, r, rng)
```

在 `generate_client_sizes` 内部，Beta 由均值 `mu = (center - low)/(high - low)` 与
concentration `s = max(dispersion_s, 2e-3)` 重参数化，即 `Beta(mu*s, (1-mu)*s)`；样本被
重新缩放以命中 `target_sum`，裁剪到 `[low, high]`，并用 largest-remainder + 校正 pass 取整
以保证整数和精确。`default_r` 计算 `r = clip(C*center/N, C*low/N, C*high/N)` —— 即在池上
的平均逐 client 大小，被裁剪到可行带宽内。`assign_with_overlap` 把每个目标复制
`k ~ {floor(r), ceil(r)}` 次，并以正比于剩余容量的概率把副本发给不同的 client（外加一个
补差 top-up）。

> **注意（对照原始描述）：** `coverage_partition` 签名里的 `overlap_ratio=1.3` 参数是一个
> **未使用的 legacy 默认值** —— 真正生效的 overlap 是由 `default_r` 计算出的 `r`，*而非*
> 固定的 `1.3`。更大的 `size_std` -> 更紧的 Beta -> pool 大小几乎相等；`size_std=1` ->
> 重尾大小 -> 极端 coverage 不平衡。

### Hardness — 成功标签上的 Beta-偏斜 easy/hard 混合

[`../hetero/webshop_hardness.py`](../hetero/webshop_hardness.py)，
`hardness_partition`。目标先按从 `trajectories_file` 读取的**逐任务成功标签**
（`task_id -> success`，由一个参考 policy 在整个 catalog 上 rollout 产出）分桶；然后一个
Beta 设定每个 client 的 "success"（easy）目标计数，其固定配额的剩余部分用随机目标填充。
每个 client 的目标数保持不变 —— 只有*难度混合*在移动：

```python
rng = np.random.default_rng(42)
center = min_samples_per_client // 2          # center = half the per-client quota
low = 0; high = min_samples_per_client        # success-count band: 0 .. min_goals
r = default_r(total_size, client_num, low, center, high)
target_sum = int(round(r * total_size))
success_counts = generate_client_sizes(C=client_num, low=low, center=center,
                                        high=high, dispersion_s=success_std,
                                        target_sum=target_sum, rng=rng)
# bucket goals by label, then fill each client: success_counts[client_id] easy
# goals from high_success_data, remainder random, capped at max_samples_per_client
```

目标被分桶为 `high_success`（标签 `True`）与 `low_success`（标签 `False` 或未知）；client
从 easy 桶抽取 `success_counts[client_id]` 个，若不足则从 hard 桶补足，再用随机目标填满其
配额的其余部分。更大的 `success_std` -> 跨 client 的难度均匀；`success_std=1` -> 极端
（有些 client 几乎只看到可解目标，另一些几乎只看到困难目标）。

> **随附输入。** 原始的**训练后 checkpoint**标签随附于 `data/hardness/`（WebShop +
> ALFWorld，完整 train 池），所以 Hardness 配置开箱即跑；`hardness_for_client` 对错误路径
> 仍会抛 `FileNotFoundError`。标签依赖于参考 policy，所以请用一个**训练后**的 checkpoint
> **按 backbone 逐一**重新生成（base/zero-shot 模型会让 easy/hard 划分坍缩），经由
> `tools/verl08_migration/gen_hardness_trajectories.py`，它写出的标签所用的 key 正是
> partitioner 使用的**精确** `task_id` 公式：
> ```python
> # webshop_hardness.py: task_id derivation (per goal dict)
> options_hash = int(hashlib.md5(str(sorted(item['goal_options'].items())).encode()).hexdigest(), 16)
> task_id = f"{asin}_{abs(options_hash)}"        # human-goal fallback: md5(instruction_text)
> ```
> 该查找**仅**能对上 env 的真实目标字典（它们携带 `goal_options` / `instruction_text`）；
> 离线 `data_dir` 回退只能产出 asin-only 的 id，**无法**匹配一个 options-hash 标签文件。

按构造，每个轴都提供对其自身弥散的目标控制、对另外两个度量的期望不变性、在期望意义下对
全局混合的保留，以及联合可配置性 —— 因此三者可以组合，也可以一次只变一个。

---

## Environment-level constructions (hidden `P`)

各 client 共享（均匀的）任务划分，但其 **transition kernel `P_i`** 各不相同。WebShop 的
检索 pipeline 分解为**四个阶段**，五个环境变体跨这些阶段进行扰动：

1. **content** —— *catalog 里有什么*（搜索能返回的产品）；
2. **encoding** —— *一个产品如何变成被索引的文本*（哪些字段喂给 BM25）；
3. **matching** —— *一个 query 如何被打分*（BM25 排序函数）；
4. **rendering** —— *排好序的页面如何被呈现*给 agent。

| 变体（论文） | 阶段 | `run_fed` 策略 | 逐字构造器 | Config dir |
|---|---|---|---|---|
| **Catalog Split**（Variant 1） | content | `catalog_split` | `_distractor_disjoint_partition_webshop_v5` | `env_heterogeneity/catalog_split/` |
| **Field-Subset Index**（Variant 2） | encoding | `bm25_field_subset` | `_bm25_variant_partition_webshop`（`fields_only` pool） | `env_heterogeneity/field_subset_index/` |
| **BM25 Reweighting**（Variant 3） | matching | `bm25_reweight` | `_bm25_variant_partition_webshop`（default pool） | `env_heterogeneity/bm25_reweighting/` |
| **Lookalike Injection**（Variant 4） | content + matching | `lookalike` | `_lookalike_injection_partition_webshop` | `env_heterogeneity/lookalike_injection/` |
| **Rank Wrapper**（Variant 5） | rendering | `rank_wrapper` | `_rank_wrapper_partition_webshop` | `env_heterogeneity/rank_wrapper/` |

> **命名告诫。** Catalog-Split helper 的 `_v4`/`_v5` 后缀是**论文 Variant 1 的实现-修订号**，
> *不是*论文 Variant 4（Lookalike）或 Variant 5（Rank Wrapper）。当前的 `catalog_split` key
> 由 `_v5` 修订服务（逐 client ~100-goal 切片、逐 client target 下限）；被取代的 `_v4` 修订
>（`distractor_disjoint`）共享 `goals[500:]`，未被任何已报告结果使用。

Catalog Split 划分**目标集**并返回逐 client 的 `goal_idxs`；Variants 2-5 保持任务划分
**均匀**，只返回一个被合并进 `gym.make` 的 `env_kwargs` 片段。Variants 2-5 的逐 client 变体
分配按 `client_id` 确定（`RandomState(42 + client_id)`，`chosen = pool[rng.randint(N)]`），
所以同一个 client 跨 round 保持同一个变体 —— 这是 FedAvg 平均可比 policy 所必需的。

### Catalog Split（Variant 1，content）

[`../hetero/webshop_catalog_split.py`](../hetero/webshop_catalog_split.py)，
`_distractor_disjoint_partition_webshop_v5`。每个 client 得到一个受保护的逐 client **target**
ASIN 下限（这样每个 client 仍能完成自己的目标），外加一个逐 client 的 **distractor** 池，
其抽取方式使各 catalog 彼此发散。最优的 "search -> click -> buy" 行为不因*哪些*额外产品在场
而改变，所以 `pi*` 基本保持不变 —— 这是最温和的 env 扰动。发散数学，由 `env_div` 与
`keep_ratio` 旋钮控制：

```python
# one shared u and one per-client v per ASIN (keyed by ASIN string, not pool index)
asin_to_u = {a: RandomState(base_seed).random()              for a in all_asins}      # shared
asin_to_v = {a: RandomState(base_seed + 1000*client_id).random() for a in all_asins}  # per-client
u = u[distractor_pool]; v = v[distractor_pool]
e = (1.0 - env_div) * u + env_div * v        # env_div=0 -> identical ranking u across clients
n_keep = int(round(keep_ratio * D))          # D = |distractor_pool| ~ 920 (per-client)
include_distractors = [distractor_pool[i] for i in np.argsort(e)[:n_keep]]
catalog_asins = sorted(client_target_asins | set(include_distractors))
```

`env_div in [0,1]` 是 **catalog-发散强度**：当 `env_div=0` 时，每个 client 都用*共享*的 `u`
对 distractor 排序并保留同一个 top-`n_keep`（catalog 最大程度重叠，近同质的下限）；当
`env_div=1` 时，每个 client 用它*自己*的 `v` 排序，catalog 最大程度发散。`keep_ratio`
设定 **distractor 密度**（保留的 ~920-item 逐 client 池的比例）。按 ASIN-string 做 key 这个
细节是 load-bearing 的：每个 client 的 `distractor_pool` 内容不同，所以同一个 ASIN 必须在
各 client 间读到相同的 `u`。这个逐 client target 下限（对照 legacy 的全-target 下限）拓宽了
pairwise-Jaccard 范围、强化了 `env_div` 信号，同时让任务划分与主实验保持一致。

### Field-Subset Index（Variant 2，encoding）与 BM25 Reweighting（Variant 3，matching）

[`../hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py)，**一个**函数
`_bm25_variant_partition_webshop` 同时服务两者 —— 池由 `variant_pool` 选择
（`"fields_only"` -> Variant 2；default -> Variant 3）。每个 client 被确定性地分配一个
`{fields, k1, b}` 配置，穿入 `env_kwargs['bm25_in_memory_config']`；catalog、goals 与 reward
在各 client 间相同 —— 只有搜索 transition `T(s'|s,a)` 不同。

- **Field-Subset Index**（`BM25_VARIANTS_FIELDS_ONLY`）：所有变体共享 `k1=1.2, b=0.75`；
  只有**被索引的字段子集**不同，所以同一个 query 对产品的排序不同，agent 必须学会逐
  client 的 query 雕琢。`N=4` 扫描是 `full {name,Title,description,features,BulletPoints}`、
  `name {name,Title}`、`desc {description}`、`bullets {BulletPoints}`（条目 5-8 ——
  `features`、`name_bullets`、`desc_features`、`no_name` —— 将其扩展到 `N=8`）。
- **BM25 Reweighting**（`BM25_VARIANTS_DEFAULT`）：所有变体都索引**完整**字段集，但使用
  **极端 `(k1, b)` 角点**，重塑 TF 饱和与长度归一化。`N=4` 扫描是 `(1.2, 0.75)`（default）、
  `(1.2, 0.00)`、`(0.3, 0.75)`、`(5.0, 0.75)`（条目 5-8 —— `(0.1,0.75)`、`(1.2,1.00)`、
  `(2.0,0.50)`、`(0.3,0.00)` —— 将其扩展到 `N=8`）。

### Lookalike Injection（Variant 4，content + matching）

[`../hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py)，
`_lookalike_injection_partition_webshop`。最强的 content 攻击：每个 client 得到一组逐 client
的合成 **lookalike products**，附加到基础的 1000-product catalog
（`env_kwargs['extra_products']`），它们被调校得既能愚弄 BM25 排序、**又能**击败某一个特定的
reward 子项，从而迫使 agent 去检查某个特定属性（price、color、…）以过滤掉这些假货。因为
不同 client 攻击不同属性，它们的最优 policy 在*结构上*发散。default `N=2` 覆盖两个经
reward 验证的攻击（`v_price`、`v_color`）；`N=4` 增加 `v_size`、`v_price_color`。JSON 随附于
`data/env_heterogeneity/lookalike_data/`（路径相对 `PROJECT_ROOT` 解析，由 runner 导出）。

### Rank Wrapper（Variant 5，rendering）

[`../hetero/webshop_env_variants.py`](../hetero/webshop_env_variants.py)，
`_rank_wrapper_partition_webshop`。每个 client 的结果被一个不同的 **wrapper** 在同一个 BM25
base 之上后处理（`env_kwargs['search_engine_variant']`），打破任何 "trust the top position"
的启发式，同时保留 reward 梯度（target 仍可在候选集中触达）。`N=4` 池
（`SEARCH_ENGINE_VARIANTS_DEFAULT`）：`v_bm25_default`（control）、`v_shuffled_topk`
（shuffle top 50）、`v_inverted_topk`（reverse top-K）、`v_partial_random`（50% 的 query 返回
随机）。逐 client 的 `seed = base_seed + client_id` 让共享同一变体的 client 之间 shuffle/random
行为各异。

---

## The asymmetric-robustness spectrum (stable -> degrade -> collapse)

两个级别位于一条**鲁棒性谱**的两端，而在每个级别内，旋钮值把一条 arm 沿谱移动。这是论文
核心观察的可操作化：*任务级异质性鲁棒；环境级异质性最坏情况下非鲁棒。*

| 区域 | FedAvg 做了什么 | 它在哪里 |
|---|---|---|
| **stable** | 单一聚合 policy 几乎无损地跨 client 迁移 | **全部任务级 arm**（`tau` 可观测）；env 级的**近同质下限**（`catalog_split` 在 `env_div=0.0`） |
| **degrade** | 聚合仍有帮助，但全局 policy 损失精度；发散真实但可恢复 | 中等强度的 env 级 arm（`catalog_split` `env_div ~ 0.3-0.7`；`bm25_field_subset` / `bm25_reweight`） |
| **collapse** | 各 client 的最优 policy `pi*_i` 结构性发散；朴素 FedAvg 在 GRPO 下崩溃 | 满强度的 env 级**最坏情况攻击**（`catalog_split` `env_div=1.0`；`lookalike`、`rank_wrapper`） |

这条谱编码的关键事实：

- **任务级 arm 在其整条扫描中保持 stable。** 即便在极端端点（`omega=0.99`、`size_std=1`、
  `success_std=1`）目标描述符仍在 prompt 里，所以聚合 policy 读取每个 client 的 `tau` 并服务
  其并集。任务异质 arm **不会**到达 *collapse*。
- **env 级严重度按阶段与强度排序。** *content* 扰动（`catalog_split`）让 `pi*` 基本不变、
  优雅降级；它的 `env_div` 旋钮把它从近同质下限（`0.0`，*stable*）经 *degrade*（`0.3`、
  `0.7`）滑到 *collapse*（`1.0`）。*encoding*/*matching* 变体（`bm25_field_subset`、
  `bm25_reweight`）处于 *degrade*（真实但可恢复）。两个最强攻击 —— *content+matching*
  （`lookalike`）与 *rendering*（`rank_wrapper`）—— 在 GRPO 下到达 *collapse*。
- **GRPO -> PPO 救援。** 两个 collapse-区域的攻击（`lookalike`、`rank_wrapper`）会击破朴素
  GRPO 聚合，但*在 PPO 下被救回到 degrade 区域* —— critic 吸收了隐藏动力学的方差。这正是为何
  每个 env-het 配置都有一个 `*_ppo` 兄弟：成对的 run 产出 env-异质性图的 GRPO-vs-PPO 两半。
  `task_disjoint`/`catalog_split` 这一对是匹配对照，把任何 collapse 归因于**隐藏 catalog**，
  而非目标划分。

---

## Arm -> knob -> paper-config map

每条 arm 都由单一的 `run_fed` key `partition_strategy` 加上该策略的旋钮来选择；`run_fed.py`
把它们作为 env var 导出到每个逐 client 的远程 env 服务，后者再 dispatch 到匹配的
`*_for_client` 构造器。论文取值是 `config/paper/{task,env}_heterogeneity/` 下实际存在的端点。
完整字段表见 [`./configuration.md`](./configuration.md)。

| `partition_strategy` | 级别 | Env(s) | 旋钮（`run_fed`）-> env var | 论文取值 | 配置族 |
|---|---|---|---|---|---|
| `preference` | task | WebShop, ALFWorld | `omega` -> `OMEGA` | `{0.01, 0.99}` | `task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/...preference_omega-*` |
| `coverage` | task | WebShop, ALFWorld | `size_std` -> `SIZE_STD` | `{256, 1}` | `..._coverage_std-*` |
| `hardness` | task | WebShop, ALFWorld | `success_std` -> `SUCCESS_STD`（+ `trajectories_file` -> `TRAJECTORIES_FILE`） | `{256, 1}` | `..._hardness_success_std-*` |
| `task_disjoint` | task | WebShop | `env_div`, `keep_ratio` -> `ENV_DIV`, `KEEP_RATIO` | 匹配 `catalog_split` | 针对 `catalog_split` 的 env-effect ablation |
| `catalog_split` | environment | WebShop | `env_div`, `keep_ratio` -> `ENV_DIV`, `KEEP_RATIO` | `env_div in {0.0, 0.3, 0.7, 1.0}`，`keep_ratio 0.7` | `env_heterogeneity/catalog_split[_ppo]/...div-*_keep-0.7` |
| `bm25_field_subset` | environment | WebShop | `variant_n` -> `VARIANT_N` | `{4, 8}` | `env_heterogeneity/field_subset_index[_ppo]/...field_subset_index_N-*` |
| `bm25_reweight` | environment | WebShop | `variant_n` -> `VARIANT_N` | `{4, 8}` | `env_heterogeneity/bm25_reweighting[_ppo]/...bm25_reweighting_N-*` |
| `lookalike` | environment | WebShop | `variant_n` -> `VARIANT_N` | `{2, 4}` | `env_heterogeneity/lookalike_injection[_ppo]/...lookalike_injection_N-*` |
| `rank_wrapper` | environment | WebShop | `variant_n` -> `VARIANT_N` | `4` | `env_heterogeneity/rank_wrapper[_ppo]/...rank_wrapper_N-*` |

注意：

- env-变体的**配置目录名**（`field_subset_index`、`bm25_reweighting`、`lookalike_injection`）
  与 `partition_strategy` **取值**（`bm25_field_subset`、`bm25_reweight`、`lookalike`）不同。
  服务 dispatch 用的是策略取值。
- `variant_n` 是逐 client 池中的变体数（作为 `N` 传入）。取值 `0` 表示 "用构造器默认值"
  （bm25/rank 为 4，lookalike 为 2）；论文配置显式设定它。
- 多点扫描（`catalog_split` 4 点 `env_div`；bm25/field-subset `N in {4,8}`；lookalike
  `N in {2,4}`）只存在于 GRPO 目录中；每个 `*_ppo` 兄弟只持有单一配置（最发散的那个扫描点）
  用于 GRPO-vs-PPO 比较。
- **ALFWorld** 只拿到 env-无关的任务级子集（`preference` / `coverage` / `hardness`）；没有
  `env_heterogeneity/` 的 ALFWorld arm，因为 WebShop 变体专门扰动 WebShop 的检索 pipeline，
  不可迁移。

---

## Selecting an arm

在 `run_fed` YAML 里设定 `partition_strategy` 与该策略的旋钮，然后用
`python -m fedagent.fed.run_fed --config <yaml>` 启动。示例从论文配置树复制而来。

**Catalog Split**（env 级），在 `keep_ratio: 0.7` 下的四点 `env_div` 扫描：

```yaml
env_kind: webshop
search_return_n: 200          # env-het perturbs the catalog -> paper top-K
partition_strategy: catalog_split
env_div: 0.7                  # 0.0 stable floor -> 1.0 collapse
keep_ratio: 0.7
```

**Preference**（task 级），极端端点：

```yaml
env_kind: webshop
partition_strategy: preference
omega: 0.99                   # 0.01 = near-uniform, 0.99 = extreme
```

**Hardness**（task 级）需要一个逐 backbone 的 success-labels 文件：

```yaml
env_kind: webshop
partition_strategy: hardness
success_std: 1                # 256 = uniform difficulty, 1 = extreme
trajectories_file: data/hardness/qwen2.5-1.5b_webshop_trajectories.json
```

一条 **env-变体** arm 就是一个策略名加 `variant_n`：

```yaml
env_kind: webshop
search_return_n: 200                # env-het perturbs the catalog -> paper top-K
partition_strategy: bm25_reweight   # or bm25_field_subset / lookalike / rank_wrapper
variant_n: 4
```

env-变体 arm 设 `search_return_n: 200`：抬高 BM25 top-K 让渲染的结果页在激进的逐 client
过滤之后依然填满，从而 target 永不会被悄无声息地丢掉。任务级 arm 把它留在引擎默认值（50），
与非 het baselines 一致。

---

## Cross-cutting invariants

这些在**每一条** arm 上都成立，正是它们让联邦 run 可复现、让跨 arm 曲线可比较。

- **Seed-42 科学红线。** 每个 partition 主体都被逐字复制，`base_seed = 42` 硬编码。共享随机性
  使用 `RandomState(42)` / `default_rng(42)`；逐 client 随机性使用 `RandomState(42 + client_id)`
  （Catalog Split 的逐 client `v` 用 `42 + 1000*client_id`）。因此分配**按 `client_id` 确定**
  且跨 round 稳定。

- **未扰动验证。** 无论训练时的扰动是什么，聚合的全局模型都在共享的**未扰动** val 服务上打分
  （`WEBSHOP_SPLIT=val`，它忽略 `PARTITION_STRATEGY`；在完整 Lucene 索引 / 完整 1000-product
  catalog 上的留出 `goals[0:VAL_SIZE]`）。ALFWorld val 服务是相应的全-game-set 服务
  （`PARTITION_STRATEGY=uniform`）。这是每条 arm 被评判的唯一标尺。

- **依赖内容的任务划分被推迟到运行时。** 三个任务级 partitioner（`preference` / `coverage` /
  `hardness`）按**内容**（category / size / hardness）挑选目标，所以 WebShop 服务把它们推迟，
  在 env 池预热后才划分 env 的**真实、seed-42-shuffled `server.goals`**
  （`_compute_task_partition` 从 `env.server.goals`）—— 于是索引 *i* 处被服务的目标恰好携带
  partition 所选中的那个属性。环境级 arm 在 import 时计算是安全的：`catalog_split` /
  `task_disjoint` 用一个与顺序无关的连续索引区间，而 bm25/lookalike/rank 变体保持目标划分均匀、
  只返回一个 `env_kwargs` 片段。**不要**把被推迟的任务划分换成一个重建出的目标列表 —— 离线回退
  不是顺序忠实的，无法匹配一个真实的标签文件。

---

参见 [`../hetero/README.md`](../hetero/README.md) 了解模块布局与 config -> env-var ->
constructor 桥接，[`./configuration.md`](./configuration.md) 了解完整的 `run_fed` 字段参考，
以及 [`./reproducing.md`](./reproducing.md) 了解完整的配置到图表映射。
