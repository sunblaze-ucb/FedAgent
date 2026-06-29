# FedAgent 文档

面向用户的 **FedAgent verl-0.8 overlay** 文档 —— 面向 LLM agent 的联邦强化
学习，作为原生 verl 0.8 之上的一层薄薄的 overlay 构建。请先看
[`../README.md`](../README.md) 中的 package 总览，然后：

| 文档 | 阅读它以了解 |
|---|---|
| [architecture.md](./architecture.md) | overlay 如何工作：联邦轮次循环、框架内 hooks、远程 env 服务、FedProx、eval。 |
| [installation.md](./installation.md) | 三个 conda 环境（trainer + WebShop + ALFWorld 服务）、数据与模型。 |
| [running.md](./running.md) | 运行 `run_fed.py`：运行模式、GPU、baselines、FedProx、validation、可运行的示例。 |
| [configuration.md](./configuration.md) | 配置文件解码器与完整的 federated-runner key 参考。 |
| [features.md](./features.md) | 每项能力 → 其 config key → 其源文件（一张导航图）。 |
| [heterogeneity.md](./heterogeneity.md) | 两级（task vs environment）异质套件，附每个 arm 的构造数学。 |
| [reproducing.md](./reproducing.md) | 论文的 176-config 矩阵映射到运行命令；3-seed 复现；baselines。 |
| [acceleration_report.md](./acceleration_report.md) | **加速与验证 —— 完整走读**：每个 lever 与 feature 的深入讲解（持久化 trainer、eval 模式、client-parallel #3、等价性）、各项调查 + 修正，以及如何运行。配套：[acceleration_results.md](./acceleration_results.md)（数字一览）· [acceleration.md](./acceleration.md)（最初的分析与计划）。 |
| [extending.md](./extending.md) | 扩展点：新增 dataset/env、异质策略、RL 算法或聚合规则。 |
| [migration.md](./migration.md) · [migration_report.md](./migration_report.md) | **迁移** —— `migration.md` 是浓缩的保真记录（相对 verl-agent-0.3.1 fork 改了什么 + 科学攸关的对齐项）；`migration_report.md` 是**完整的工程走读**（路线决策、依赖鏖战，以及 checkpoint / agent-loop / env-service / windowed 的深入剖析）。 |

## 按组件分的参考

每个 `fedagent/` 子包都有自己的 README，含代码级细节：

- [`../fed/`](../fed/README.md) —— 联邦轮次循环 + metrics logger
- [`../agent_loops/`](../agent_loops/README.md) —— 多轮 agent rollout（`GymTextAgentLoop`）
- [`../envs/`](../envs/README.md) —— `BaseTextEnv` 契约 + registry；TinyGuess / WebShop / ALFWorld clients
- [`../hetero/`](../hetero/README.md) —— 各项异质构造
- [`../envs/webshop/service/`](../envs/webshop/service/README.md) · [`../envs/alfworld/service/`](../envs/alfworld/service/README.md) —— 远程 env 服务
- [`../data/`](../data/README.md) —— `AgenticDataset`（verl `custom_cls`）
- [`../config/`](../config/README.md) —— configs + 论文矩阵
- [`../EXPERIMENTS.md`](../EXPERIMENTS.md) —— 正在进行的实验日志

## 范围

这些文档描述的是 **verl-0.8 overlay**（活跃的线上系统，位于 `fedagent/` 下）。仓库的
顶层 [`README.md`](../../README.md) 是这套系统的着陆页；*最初的*
verl-agent-0.3.1 工件作为历史参考归档在 [`legacy/`](../../legacy/README.md) 下。
两者的关系见 [migration.md](./migration.md)。
