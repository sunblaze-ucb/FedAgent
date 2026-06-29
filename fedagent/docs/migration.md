# Migration & fidelity record

This release re-implements FedAgent on **stock verl 0.8** as a thin overlay (`fedagent/`),
replacing the original **verl-agent 0.3.1 fork**. The bar is **scientific equivalence** —
reproducing the paper's conclusions within seed noise, not bit-identical outputs. This
document records what changed, what was kept identical, the fidelity fixes applied, and the
verification status. The running experiment log is [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

## What changed (and why)

| Aspect | Original (verl-agent 0.3.1 fork) | This overlay (stock verl 0.8) | Why |
|---|---|---|---|
| verl | forked; federated logic woven into the trainer | stock, imported as a library; **no fork** | track upstream, no fork maintenance |
| Control plane | `core/custom_fed_server.py` + a regex-rewritten base bash script | [`fed/run_fed.py`](../fed/README.md) — subprocess-per-(client,round) | clean, verl-agnostic |
| Env execution | in-process verl-agent env managers | **remote HTTP env services**, one per client | conda dependency isolation |
| Hooks | patched inside the vendored tree | verl extension points (`custom_cls`, agent-loop registry, Hydra `searchpath`) | stock trainer untouched |
| Config schema | nested `verl:/federated:/data_preprocess:` | flat keys → `run_fed.py` | matches the lean overlay |
| Checkpoints | `model_world_size_1` single-rank | FSDP shards → `aggregate_fedavg_fsdp.py` → `verl.model_merger` | verl 0.8 native FSDP |
| FedProx | in-trainer | `sitecustomize.py`, gated on `FEDPROX_MU` | avoids clobbering verl's per-worker GPU assignment |
| Algorithm / heterogeneity / protocol | GRPO G=8 / PPO; two-level het; N=100/M=2/E=3/T=70 | **identical** | scientific equivalence |

## Environment fidelity: the engines are reused, not reimplemented

The WebShop and ALFWorld remote services **`sys.path`-inject and import the original engines**
from the vendored `fedagent/envs/<name>/engine/` (via `importlib`) — the **same code the original FedAgent ran**.
The MDP is therefore unchanged:

- **WebShop** — `WebAgentTextEnv` / `SimServer` / `engine.py` / `goal.py` and the
  `webshop_projection` action parser are loaded verbatim. The graded reward `get_reward`, the
  `{0,10}` sparse training reward (won iff `done and score==1.0`), action validity, the
  catalog files, the **seed-42** goal shuffle, and the `val=goals[0:500]/train=goals[500:]`
  split are all the same. The heterogeneity math (catalog-split, preference, coverage,
  hardness, bm25/lookalike/rank) is a **verbatim copy** of `partition_strategy.py`.
- **ALFWorld** — `AlfredTWEnv` / TextWorld, the `alfworld_projection` parser, the game
  loader, the `10 × won` reward, the 6 task types, and the `uniform/preference/coverage/
  hardness/env_disjoint` partition set are all reused unchanged.

What differs is the **wrapping/driving** (HTTP service + verl 0.8's native multi-turn
agent-loop instead of the fork's in-process rollout) — equivalent information to the policy,
not a change to the environment.

## Science-critical alignments

These were verified during migration audits and fixed where they diverged (see
`../EXPERIMENTS.md` for the per-item record; codes B1–B-G2 there):

- **Algorithm** — GRPO with group size **G = 8** (`adv_estimator=grpo`,
  `actor_rollout_ref.rollout.n=8`). Stock verl 0.8 multiplies `ppo_mini_batch_size` by
  `rollout.n` internally, so the original's "1 update / rollout-batch" is reproduced with
  `ppo_mini_batch_size=8` prompts (GRPO) / 64 (PPO) — **not** 64×8.
- **Trajectories/step = `train_batch_size × rollout.n`, for PPO as well as GRPO.** Confirmed
  in the verl-agent source (`agent_system/multi_turn_rollout/rollout_loop.py:448` targets
  `train_batch_size * rollout.n`; `:504` does `gen_batch.repeat(rollout.n)`), both
  **unconditional** — *not* gated on `adv_estimator`, and PPO uses the same `multi_turn_loop`.
  So the original ran **GRPO 8×8 = 64** and **PPO 64×8 = 512** trajectories/step; the new
  configs reproduce both exactly. ⚠️ **`rollout.n` must stay 8 for PPO** — dropping it to 1
  would give 64/step, *unfaithful* to the paper. (Reviewed false-alarm: the new PPO is **not**
  doing 8× extra rollout vs legacy — legacy already did 512/step.)
- **Sparse reward + invalid-action penalty** — `{0,10}` with a `0.1 × n_invalid` penalty
  (the penalty moved from the trainer actor to the agent-loop; same total per episode).
- **Task-heterogeneity partitions the real shuffled `server.goals` at runtime** (not an
  offline reconstruction) — so each client's shard matches the original.
- **Round-threaded data seed** — `FEDAGENT_BASE_SEED = base_seed + round*100 + client`, and
  the service draws goals with `random.Random(seed)` (a plain modulo collapsed the round term
  and made every client see the same goals every round).
- **Full E epochs/round** — `total_training_steps: 0` → `null` (a smoke step-cap must never
  leak into paper runs); `save_freq` saves the round's last step; `resume_mode=disable` (the
  federation owns "resume" at the round level).
- **Validation** — a shared unperturbed val service, `test_freq=5`, `val_before_train`,
  val temperature 0.4, on the paper's held-out splits.

## Fidelity fixes baked into the config generator

`tools/verl08_migration/gen_paper_configs.py` (which emits the 176-config paper tree)
applies three fixes surfaced by the WebShop/ALFWorld implementation audits:

1. **WebShop `search_return_n` (BM25 top-K).** The original raised it only on env-het arms
   (which perturb the catalog/search and need targets reachable) and left the **engine
   default 50** on the uniform / task-het / decentralized / baseline runs. The migration had
   hardcoded 200 everywhere, which makes the non-het baselines easier. Now: **200** for
   `env_heterogeneity/` arms, **50** elsewhere — matching the original baselines.
2. **ALFWorld `max_turns = 50`** (was 12). The original ran 50-turn episodes; a smaller cap
   can only lower ALFWorld success. Set in `config/envs/alfworld.yaml` + `alfworld_val.yaml`.
3. **ALFWorld context window**, sized for the **windowed** (per-turn, `history_length=2`)
   default rollout — which is what changed the context sizing. Each turn is one training
   sample whose prompt is the bounded windowed template (task + last-2 (obs,action) + current
   obs), not a growing transcript, so the old growing-transcript budgets
   (`max_model_len=16384`, `response_length=8192`) are gone. The ALFWorld `client_overrides`
   now use `rollout.max_model_len=2560`, `response_length=512` (prompt `2048` for the short
   room text); WebShop uses `rollout.max_model_len=4608`, `response_length=512` (prompt `4096`
   for the long product pages). `rollout.n` stays at G=8.

> Fixes #2/#3 are **GPU-VERIFY**: confirm no OOM / prompt truncation at 50 turns on the
> target hardware; raise `max_model_len` further if episodes truncate before `done`.

## Config tree

The paper configs (`fedagent/config/paper/`) mirror the original `config/` tree 1:1 in
structure and naming — `uniform/<Model>/<setting>/<algo>/`, `env_heterogeneity/`,
`task_heterogeneity/{grpo,ppo}/{env}/`, `decentralized/` — 176 configs total (see
[reproducing.md](./reproducing.md)). The one intentional deviation: **centralized/local
baselines use T=70 × E=3 (=210 epochs)** rather than the original's 1 round × 210 epochs,
because the verl-0.8 runner draws goal variety from **rounds** (the round-threaded seed), so a
single round would repeat the same goals. Same total epochs; correct goal coverage.

## Residual differences

**Benign plumbing (no MDP effect):** the multi-turn history is verl 0.8's native concat-chat
rather than the fork's re-rendered template (equivalent information); the invalid-action
penalty is applied in the agent-loop, not the trainer; goal sampling uses a different (still
reproducible) RNG, so per-seed trajectories are not bit-identical to 0.3.1.

**GPU-pending verification:** the ALFWorld 50-turn budget (#2/#3) needs an OOM/truncation
check on the target GPU. PPO (`gae`) critic federation, the ALFWorld service path, and the
decentralized ablations are config-parse + code-audited but not yet smoke-run end-to-end (the
GRPO WebShop federated path **is** GPU-verified end-to-end).

## Verification status

| Path | Status |
|---|---|
| TinyGuess (in-process) | GPU-verified end-to-end |
| **WebShop GRPO federated** | **GPU-verified: full 2-round loop** (train → FedAvg → merge → round 2 → eval) |
| WebShop PPO (gae critic federation) | config-parses + code-audited; not GPU-smoke-run |
| ALFWorld (service + max_turns=50) | config-parses + code-audited; **GPU-VERIFY** pending |
| Decentralized ablations | config-parses + code-audited; not GPU-smoke-run |

## See also

- [migration_report.md](./migration_report.md) — the **complete migration walkthrough**: the route decision,
  the environment-build saga, and the hard problems (checkpoint/agent-loop/env-service/windowed) *in depth*.
  *This* doc is the condensed fidelity record; that one is the full engineering account.
- [architecture.md](./architecture.md) — how the overlay is built
- [reproducing.md](./reproducing.md) — the paper config matrix
- [`../EXPERIMENTS.md`](../EXPERIMENTS.md) — the running experiment log + per-fix detail
