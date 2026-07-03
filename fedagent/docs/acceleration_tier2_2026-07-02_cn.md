# Tier-2 可选加速旋钮（2026-07-02）——实现、验证与最优组合

> **本文是什么。** [前沿研究](./acceleration_frontier_2026-07-02.md)剩余的全部加速杠杆——四个
> Tier-2 管道修复、#3 并行轮 lanes、fused-kernels 采纳、verl 实验性 `one_step_off_policy`——
> 全部实现为**各自独立可选的配置旋钮，默认全关**（默认 = 与旧行为逐字节一致）。每个旋钮都做
> 等价门控（checkpoint `max|Δ| ≤ 1e-4` 对其关闭臂）+ 分相计时；随后每个环境的**最优组合**在
> **真实 paper config 截断 2 轮**上实测（与 WebShop 707/496/630 基线、ALFWorld 首跑 3719 s
> 相同的 wiring 方法学）。
>
> 常量：Qwen2.5-1.5B、GRPO G=8、windowed（paper 几何）、4×H100（qgpu3021）。
> 姊妹篇：[acceleration_frontier_2026-07-02.md](./acceleration_frontier_2026-07-02.md)（这些旋
> 钮回应的诊断）· [acceleration_tier1_report_2026-07-01.md](./acceleration_tier1_report_2026-07-01.md)
> （replica 分片）· [acceleration_report.md](./acceleration_report.md)（杠杆栈）。

---

## 1. 旋钮一览

| 旋钮（run_fed 键） | 默认 | 作用 | 机制 | 科学安全性 |
|---|---|---|---|---|
| `alfworld_manifest_cache` | `false` | 缓存 8810 游戏清单遍历 | 服务的 `collect_game_files` 遍历（GPFS 上 os.walk + 每游戏 2 次 JSON 读）对每个服务进程完全相同；第一个完成的遍历持久化**洗牌前**列表（原子写，`(data_path, task_types)` 键自校验）；洗牌/分片/上限仍在相同输入上**原生**运行 | 环境流逐字节一致；过期/异键缓存退化为完整遍历，绝不产生错误数据 |
| `alfworld_manifest_dir` | `<repo>/runs/alfworld_manifest` | 清单存放位置 | 每 split 一个文件；跨 run 持久（遍历全集群只需付一次） | — |
| `service_scope: run` | `round` | 按客户端的环境服务舰队**跨轮保温** | 注册表 + LRU（容量 `service_cache_clients`，默认 4）；复用舰队做健康检查、失败自动重启；uniform 分片服务与轮次无关 | 相同进程、相同分片 → 逐字节一致；淘汰 = 普通停服 |
| `final_eval_mode: worker` | `subprocess` | 在 worker 停机前用**热引擎**给 model_T 打分 | **eval-only 计划**（第 T+1 轮，`eval_only` 标志）复用 worker 的轮首评估路径：重置到最终聚合 → `_worker_validate(T)` → 跳过拟合 | 评估只读（跨模式权重等价 3.8e-6/7.6e-6）；任何失败回退子进程冷终评 |
| `hf_export: final` | `every_round` | 跳过逐轮 `model_merger` HF 往返 | FedAvg 仍写聚合 FSDP 分片；worker 从 BASE 模型重建各客户端引擎（一如既往全新 optimizer/scheduler），再经引擎自身的 FSDPCheckpointManager **仅模型加载聚合分片**；HF 只在最终模型产出 | 权重 == merge 路径张量（等价门控）；要求 persistent/cross_round +（评估关 ∨ `eval_mode=worker`） |
| `parallel_clients: 2` | `1` | #3 lanes：本轮客户端在不相交 GPU 半区**并发**训练 | 每 lane 一个长驻 worker（lane 独立 `CUDA_VISIBLE_DEVICES` / `RAY_TMPDIR` / 权重传输 socket / 服务 URL 文件——#3 并发修复）；lane 0 承担 worker-eval + 终评职责 | FedAvg 与顺序无关；每客户端种子由 (round, client) 导出；**FSDP world size 改变**（2 vs 4）→ 数值路径变化，等价门控 | 
| `one_step_off` | `false` | **Additional option**：verl 实验性 `one_step_off_policy` | 生成在专用 GPU 切分（`rollout.n_gpus_per_node`）上提前一步跑；步墙钟 → `max(gen, train)` | **一步离策略——在 paper 复现 bar 之外。** 仅子进程路径；需明确科学签核；见 §5 |

组合门（启动时强制）：`hf_export=final` ⇒ persistent +（评估关 ∨ worker）；
`final_eval_mode=worker` ⇒ cross_round + worker（否则警告 + 回退）；`parallel_clients>1` ⇒
cross_round +（评估关 ∨ worker）+ `P | n_gpus`；`one_step_off` ⇒ 仅子进程路径、不与 lanes 组合。

## 2. 逐旋钮验证（等价 + 分相计时）

等价 rig：仅差该旋钮的成对臂，seed 42，关清理；对比第 2 轮聚合 actor
（`compare_fsdp_checkpoints.py`，bar ≤ 1e-4）。ALFWorld 臂：2c × 2r × 1 step、r8、paper 上限、
48 游戏 val（worker 模式）。TinyGuess 臂（训练面旋钮）：既有 full-loop rig，cross_round 路径。

**先看可复现性底线。** 套件里含一个同配置重跑（修复前的 cache 臂缓存实际未生效——见 §6
sys.path 注记——使其成为纯 base 重复）：两次同配置运行之间 **max|Δ| = 9.293e-5**。这就是本
rig 的 GPU 非确定性底线，恰在 1e-4 bar 之下。每个旋钮臂都落在**该底线之下或持平**——旋钮
引入的偏差与重跑噪声不可区分。

| 旋钮（ALFWorld 臂，base 墙钟 2185 s） | 等价（max\|Δ\| vs base） | 墙钟 | 计时效果 |
|---|---|---|---|
| — 同配置重跑（噪声底） | 9.293e-5 | 2043 s | −6.5 % = 重跑噪声带 |
| `alfworld_manifest_cache`（暖清单） | **9.199e-5 ✓** | 1784 s | **−18 %**——146 s/波的遍历（tqdm 8810 游戏）对全部 24 个服务跳过；`scope=round` 下 ×3 波（r1+r2+val）≈ 那 −401 s |
| `service_scope: run` | **9.090e-5 ✓** | 1843 s | **−16 %**——r2 舰队重热消失 |
| `final_eval_mode: worker` | **9.241e-5 ✓** | 1941 s | **−11 %**——冷终评子进程换成热引擎打分（eval-only 计划实证："scored round 2 model on the hot engine; no fit"） |
| 四个 Tier-2 全开 | **8.752e-5 ✓** | 1669 s | **−24 %**——次可加：cache 与 scope 在 r2 波上重叠 |
| `hf_export: final`（TinyGuess rig） | **8.825e-6 ✓**（低于 bar 一个量级） | 271 s vs 427 s | 分片直载 ≡ HF-merge 路径；小 rig 上墙钟 −37 %（merger 占大头）（规模化机制在 `all` 臂实证：r1 HF 延迟、worker 分片加载） |
| `parallel_clients: 2`（TinyGuess rig） | **1.144e-5 ✓**（FSDP ws 2 vs 4；经第 2 轮 **HF 导出**对比——`compare_hf_models.py`，ws2-vs-ws4 分片集无法直接 diff） | 292 s vs 427 s | 小 rig 上 −32 % |
| fused（WebShop） | **1.116e-5 ✓**（同日验证，frontier §8） | — | 步 −6.5 % |

注意两个 rig 的底线相差 10×：ALFWorld 臂携带 ~9e-5 的纯重跑噪声（规模化热引擎 rollout），
而 TinyGuess 训练面臂可复现到 ~1e-5——这就是 lanes/hf_export 的 Δ 虽是真实数值路径变化
却小 10× 的原因。

清单缓存证据：首个舰队写入（8× WROTE train + 8× eval，相同内容上原子后写胜出），此后
24/24 HIT；`runs/alfworld_manifest/manifest_train.json`（3553 游戏）+
`manifest_eval_in_distribution.json`（140）。值得一提：遍历访问 8810 个目录只留 3553 个——
缓存同时省掉了 GPFS 上 24 路并发的 stat 风暴。

## 3. 最优组合——在真实 paper config（2 轮）上实测

按项目 bar，最终栈的测法与 wiring 基线完全相同：真实 `uniform/1.5B/main/grpo/<env>` 配置、
70→2 轮、相同种子/端口纪律。

**ALFWorld**（基线：wiring_r8 = **3719 s** = r1 1766 + r2 1375（含 ~250 s 冗余重热）+ 冷终评 578）：

| 栈 | 配置 | 总计 | 预热+boot+base eval | r1 拟合+评估 | r2 拟合+聚合+HF | 终评 | Δ vs wiring_r8 |
|---|---|---|---|---|---|---|---|
| combo A = r8 + Tier-2 ×4 | `paper_alf_combo.yaml` | **3202 s** | 791 | 1252 | **762** | 389（热） | **−517 s（−13.9 %）** |
| combo B = A + lanes | `paper_alf_combo_lanes.yaml` | 3136 s | 856 | 1150 | 660 | 458（热，2-GPU） | −583 s（−15.7 %）——**lanes 在 ALFWorld ≈ 打平**：env/gen 重叠带来每轮 −102 s，被更慢的 2-GPU 终评（+69）和双 worker boot（+65）吐回；1.5B 拟合是 GPU-bound，2 客户端 × 2 GPU ≈ 顺序 4-GPU |

2 轮总计低估了收益：一次性成本（服务+worker 冷启动、base eval）主导 2 轮运行。可扩展的
数字是**稳态轮：762 s** vs wiring 的 1125 s（1375 − 被 `service_scope: run` 消掉的 250 s 重热）
= **每轮 −32 %**；热终评 389 s vs 冷 578 s（−33 %）。paper 规模机制证据：32 次训练服务清单
HIT + 8 次 val HIT（全程无 tqdm 遍历）、第 2 轮起始模型以 FSDP 分片目录交接（无逐轮 HF
merge）、r2 舰队保温复用。Val 曲线（n=140）：0.0286 → 0.0214 → 0.0214——近随机成功率下
140 游戏评估被采样噪声主导（wiring 跑出 0.0429 → 0.1143，权重等价）；科学门是 §2 的
checkpoint 对比，不是小 n val 曲线。

**WebShop**（基线：同日 r4 wiring 重跑 = **2802 s**，分相 490/764/905/643）：

| 栈 | 配置 | 总计 | 预热+boot+base eval | r1 拟合+评估 | r2 拟合+聚合+HF | 终评 | Δ vs r4 wiring |
|---|---|---|---|---|---|---|---|
| combo A = r4 + fused + Tier-2（scope/feval/hf_export） | `paper_ws_combo.yaml` | **2309 s** | 769 | 802 | **402** | 326（热） | **−493 s（−17.6 %）** |
| combo B = A + lanes | `paper_ws_combo_lanes.yaml` | 2255 s | 761 | 782 | 362 | 337（热，2-GPU） | −547 s（−19.5 %）——**lanes 在 WebShop 也 ≈ 打平**（每轮 −20/−40 s，终评 +11 s）；小模型探针的 −35 % 没能在真实配置存活，与 replicas 同命 |

与 ALFWorld 同构：稳态轮塌缩到 **402 s**（vs wiring 的 905 s r2 块），热终评 326 s（vs 冷
643）；2 轮总计剩下的大头是一次性预热+boot+base eval 块（769 s），70 轮摊薄后消失。
Val 曲线 0.012 → 0.010 → 0.020（wiring：0.018 → 0.012 → 0.030）——同样的小 n 噪声注意事项。

**推荐配方**（最终）：

```yaml
# ALFWorld paper 运行（= paper_alf_combo.yaml 的旋钮）
use_persistent_trainer: true
persistent_scope: cross_round
eval_mode: worker
alfworld_manifest_cache: true
service_scope: run
final_eval_mode: worker
hf_export: final
# 不采纳：parallel_clients（实测打平：−2 %）、r8 之外的 replicas（Tier-1 结论）

# WebShop paper 运行（= paper_ws_combo.yaml 的旋钮）
use_persistent_trainer: true
persistent_scope: cross_round
eval_mode: worker
service_scope: run
final_eval_mode: worker
hf_export: final
client_overrides: "+actor_rollout_ref.model.use_fused_kernels=True ..."   # fused：步 −6.5 %，1.116e-5
# 不采纳：parallel_clients（打平：−2.3 %）、replicas（paper config 打平，Tier-1）
```

两个配方都排除：`one_step_off`（离策略——§5）、lanes（两环境实测打平：1.5B 拟合是
GPU-bound，2 个并发 2-GPU 客户端 ≈ 顺序 4-GPU，且 2-GPU 热终评吐回大部分重叠收益）、
额外 replicas（WebShop 打平；ALFWorld r8 已在基线里）。

## 4. 70 轮投影

公式（test_freq = 5）：`T(70) ≈ 一次性 + 70 × 稳态轮 + 14 × 到期评估 + 终评`，各项均取自
同日 2 轮 paper-config 实测（稳态轮 = r2 块——拟合 + FedAvg（wiring 情形另含 merge、其
`scope: round` 世界另含重热）；到期评估 = 单次评估墙钟；一次性 = 预热 + worker boot +
base eval）。

| 环境 | wiring 栈 | 最优 combo | Δ |
|---|---|---|---|
| ALFWorld | 791 + 70×1375 + 14×578 + 578 ≈ **29.3 h** | 791 + 70×762 + 14×389 + 389 ≈ **16.7 h** | **−43 %** |
| WebShop | 490 + 70×905 + 14×643 + 643 ≈ **20.4 h** | 769 + 70×402 + 14×326 + 326 ≈ **9.4 h** | **−54 %** |

（早前"ALFWorld ≈ 27 h"的估计用的是相同 wiring 数字 + 更粗的评估模型——一致。combo 的
`service_scope: run` 项在此偏保守：70 轮里每轮从 100 客户端抽 2 个，重复抽中会落进 LRU 保温
缓存进一步削减舰队启动；本投影把每一轮都按新舰队计价。）

## 5. `one_step_off`——Additional option（离策略；签核层级）

verl 0.8 自带 `experimental/one_step_off_policy`：rollout worker 驻留在专用 GPU 切分上，在
batch t 训练的同时生成 batch t+1——步墙钟 `gen + train → max(gen, train)`。
此处接线为 `one_step_off: true`（子进程客户端路径；入口 `fedagent.main_one_step_off`，配置
`fedagent_one_step_off.yaml` 把上游增量叠在 `fedagent_ppo_body` 上；GPU 切分经
`client_overrides` 提供，如 `trainer.n_gpus_per_node=3 rollout.n_gpus_per_node=1`）。

接线过程浮出一个值得记录的 hydra 约束：`hydra.searchpath` 只允许在**主配置**里覆盖，因此
第二个入口不能简单地把 `fedagent_ppo` 列进 defaults。`fedagent_ppo.yaml` 由此拆分为
`fedagent_ppo_body.yaml`（全部 FedAgent 叶子，无 hydra 块）+ 只加 searchpath 的薄主配置——
各入口叠 body、各自声明 searchpath。已 compose 验证：paper 路径解析出的叶子逐项不变。

**为何不进任何推荐组合：**
1. **它改变算法**——更新批次采样自上一步的权重（一步离策略）。学习曲线不应期望与 paper
   吻合；采纳是*科学*决定，不是工程决定。
2. **FedAgent 的轮结构截断收益**：每客户端轮只有 3 个 optimizer 步，流水线在每个轮边界
   排空，稳态 −35 % 永远无法完整兑现。

**探针现状——接线已剥四层，剩一个真实缺口。** 在 FedAgent 下立起这个实验性 trainer 依次
暴露：(1) hydra searchpath 规则（→ 上文 `fedagent_ppo_body` 拆分）；(2) 上游
`validate_config` 要求训练 world size 整除 real_train_batch_size（64）→ GPU 切分 **2+2**
而非 3+1；(3) `OneStepOffRayTrainer` 断言分离式布局 → `hybrid_engine: False`（+
`load_format: safetensors`、`layered_summon: True`），上游只在 shell 示例里设置；(4) 上游在
自己的 `main()`（而非 task runner）里把顶层 `rollout:` 资源块拷进
`actor_rollout_ref.rollout` → 已在 `fedagent.main_one_step_off` 镜像。四项均已入库修复。
剩下的缺口是真实管道而非配置：切分跑起来后，trainer 拒收批——
`bypass_mode=True requires rollout_log_probs in batch`。上游基础 agent loop 把 vLLM 服务器
的逐 token `response_logprobs` 拼进批（`agent_loop.py:944`）；我们的 windowed `gym_text`
loop 尚未收集/输出它。把 log-prob 捕获穿过 windowed 多轮流是拿到计时数字的前置——与探针
一并延后（这条管道同时会解锁主路径上 verl 的 rollout-correction 算法族，更值得从容做而
非今晚赶）。

## 6. 出处与顺带发现

**sys.path 遮蔽修复（独立潜伏 bug，被本套件抓出）。** 修复前的清单缓存臂缓存静默未生效，
这正是发现的途径：`verl-agent-alfworld` conda 环境带一个 *editable* verl-0.3.1 安装，其
`.pth` 文件在解释器启动时把 ORIGINAL verl-agent 仓库放上 `sys.path`；`agent_system` 是命名空
间包，服务*追加*的 vendored 引擎路径在解析竞争中永远落败——ALFWorld 服务一直在导入未
vendored 的引擎（两份拷贝相同掩盖了它，直到缓存补丁只落在 vendored 副本才暴露）。修复：
`fedagent/envs/alfworld/service/server.py` 现在 `sys.path.insert(0, _ENGINE)`。WebShop 不受影
响（`web_agent_site` 只存在于 vendored 引擎）。上文已记录的后果：失效臂变成了度量可复现
性底线的同配置重跑。

**冷探针悲观性（稳态修正）。** 同日 WS wiring 显示热引擎稳态步 ~50 s（gen ~10 s），而冷的
单步探针 ~82 s（gen ~36 s）——跨步前缀缓存 + 暖引擎让稳态比探针算术便宜约 3×。frontier
文档里由探针导出的每步数字对长跑系统性偏悲观；上面的 combo 表（真实 2 轮运行）取而代之。

- 实现：`fedagent/fed/run_fed.py`（旋钮、lanes、热终评、分片加载门控、服务注册表）、
  `fedagent/fed/persistent_task_runner.py`（eval-only 计划、分片感知重置）、
  `fedagent/fed/persistent_patch.py`（`reload_client_model(..., shard_dir)`）、
  `fedagent/envs/alfworld/engine/.../alfred_tw_env.py`（清单缓存，外科式）、
  `fedagent/envs/alfworld/service/server.py`（sys.path 修复）、`fedagent/main_one_step_off.py`、
  `fedagent/config/fedagent_one_step_off.yaml` + `fedagent_ppo_body.yaml` 拆分（§5）。
- A/B 配置：`accel/dev/t2_*.yaml`、`accel/alfworld/alf_t2_*.yaml`；combo：
  `accel/alfworld/paper_alf_combo*.yaml`、`accel/webshop/paper_ws_combo*.yaml`；
  探针：`accel/dev/oso_probe.yaml`。驱动器：`accel/run_t2_stack.sh`（接在 WS r4 wiring 后）、
  `accel/run_oso_probe.sh`（修复后重跑）。工具：`compare_fsdp_checkpoints.py`、
  `compare_hf_models.py`（新——与 world size 无关的 HF diff，用于 lanes 臂）。
  日志在 gitignored 的 `runs/t2_*`、`runs/paper_*/combo*`、`runs/oso/`。
- English version: [acceleration_tier2_2026-07-02.md](./acceleration_tier2_2026-07-02.md)
