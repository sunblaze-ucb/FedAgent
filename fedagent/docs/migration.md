# Migration & fidelity record

This release re-implements FedAgent on **stock verl 0.8** as a thin overlay (`fedagent/`),
replacing the original **verl-agent 0.3.1 fork**. The bar is **scientific equivalence**,
reproducing the paper's conclusions within seed noise, not bit-identical outputs. This
document records what changed, what was kept identical, the fidelity fixes applied, and the
verification status. The running experiment log is [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

## What changed (and why)

| Aspect | Original (verl-agent 0.3.1 fork) | This overlay (stock verl 0.8) | Why |
|---|---|---|---|
| verl | forked; federated logic woven into the trainer | stock, imported as a library; **no fork** (one 2-line exception, see note) | track upstream, no fork maintenance |
| Control plane | `core/custom_fed_server.py` + a regex-rewritten base bash script | [`fed/run_fed.py`](../fed/README.md), subprocess-per-(client,round) | clean, verl-agnostic |
| Env execution | in-process verl-agent env managers | **remote HTTP env services**, one per client | conda dependency isolation |
| Hooks | patched inside the vendored tree | verl extension points (`custom_cls`, agent-loop registry, Hydra `searchpath`) | stock trainer untouched |
| Config schema | nested `verl:/federated:/data_preprocess:` | flat keys → `run_fed.py` | matches the lean overlay |
| Checkpoints | `model_world_size_1` single-rank | FSDP shards → `aggregate_fedavg_fsdp.py` → `verl.model_merger` | verl 0.8 native FSDP |
| FedProx | in-trainer | `sitecustomize.py`, gated on `FEDPROX_MU` | avoids clobbering verl's per-worker GPU assignment |
| Algorithm / heterogeneity / protocol | GRPO G=8 / PPO; two-level het; N=100/M=2/E=3/T=70 | **identical** | scientific equivalence |

> **The one verl exception.** "No fork" remains the principle, with a single deliberate 2-line patch
> (the FSDP→vLLM weight-transfer socket, captured under `tools/verl08_migration/patches/` so it stays
> reproducible without forking; see [archive acceleration.md §7.7](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration.md)). It hardens concurrent
> same-node verl jobs and is needed only for client-parallel / eval-parallel runs, still no maintained fork.

## Environment fidelity: the engines are reused, not reimplemented

The WebShop and ALFWorld remote services **`sys.path`-inject and import the original engines**
from the vendored `fedagent/envs/<name>/engine/` (via `importlib`), the **same code the original FedAgent ran**.
The MDP is therefore unchanged:

- **WebShop**: `WebAgentTextEnv` / `SimServer` / `engine.py` / `goal.py` and the
  `webshop_projection` action parser are loaded verbatim. The graded reward `get_reward`, the
  `{0,10}` sparse training reward (won iff `done and score==1.0`), action validity, the
  catalog files, the **seed-42** goal shuffle, and the `val=goals[0:500]/train=goals[500:]`
  split are all the same. The heterogeneity math (catalog-split, preference, coverage,
  bm25/lookalike/rank) is a **verbatim copy** of `partition_strategy.py`; the one exception is
  `hardness_partition`, whose shard assembly was corrected to the paper's `X_i = Y_i ∪ F_i`
  (its Beta sizing and seeds are unchanged, see [`bugfixes.md`](bugfixes.md)).
- **ALFWorld**: `AlfredTWEnv` / TextWorld, the `alfworld_projection` parser, the game
  loader, the `10 × won` reward, the 6 task types, and the `uniform/preference/coverage/
  hardness/env_disjoint` partition set are all reused unchanged.

What differs is the **wrapping/driving** (HTTP service + verl 0.8's native multi-turn
agent-loop instead of the fork's in-process rollout), equivalent information to the policy,
not a change to the environment.

## Science-critical alignments

These were verified during migration audits and fixed where they diverged (see
`../EXPERIMENTS.md` for the per-item record; codes B1–B-G2 there):

- **Algorithm**: GRPO with group size **G = 8** (`adv_estimator=grpo`,
  `actor_rollout_ref.rollout.n=8`). Stock verl 0.8 multiplies `ppo_mini_batch_size` by
  `rollout.n` internally; the fork multiplied by its `actor_rollout_ref.rollout.n`, which
  was **asserted `== 1`** (its main_ppo.py:168, grouping came from `env.rollout.n` and
  never scaled the minibatch). The original always stepped in **64-row global minibatches**,
  and (2026-07-20 correction) collected **64 trajectories/step for both algos** — GRPO
  8 prompts × 8, PPO 64 prompts **ungrouped** (next bullet). On verl 0.8 the 64-row
  minibatch is therefore `ppo_mini_batch_size=8` prompts × `rollout.n=8` for GRPO and
  `=64` prompts × `rollout.n=1` for PPO, plus an explicit `critic.ppo_mini_batch_size=64`
  (the base body's `8` is GRPO-sized). (History: PPO shipped with mini=64×n8, corrected to
  8×n8 on 2026-07-09, and to 64×n1 on 2026-07-20 when `env.rollout.n` turned out to be a
  dead key on the PPO path; see `../EXPERIMENTS.md` + [`bugfixes.md`](bugfixes.md).)
- **Trajectories/step = 64 for BOTH algos (2026-07-20 correction, [`bugfixes.md`](bugfixes.md)).**
  The fork's grouping knob is `env.rollout.n`, and on the PPO path it was a **dead key**: the
  fork default is `-1` = disable grouping (`ppo_trainer.yaml:293`), the PPO base scripts never
  set it ("PPO has no GRPO/GiGPO group dimension", `scripts/verl-agent/ppo/run_*.sh`), and the
  fed orchestrator (`core/fed/script_builder.py`) only rewrites keys already present in the
  script. Envs = `train_batch_size × group_n` (`env_manager.py:1101,1181`) and the prompt
  repeat is gated on `env.rollout.n > 0` (`rollout_loop.py:282`), so the executed original ran
  **GRPO 8×8 = 64** and **PPO 64×1 = 64** trajectories/step; the configs now reproduce both
  (`rollout.n=8` GRPO / `=1` PPO). An earlier note here concluded the opposite ("`rollout.n`
  must stay 8 for PPO") from the *formula* `train_batch_size × rollout.n` on the dynamic
  filter-groups path (`rollout_loop.py:414/:448`) — the formula is real but the executed
  *value* was 1; that verdict is withdrawn. Pre-fix PPO configs really did 8× the paper's
  rollout volume (512/step).
- **Sparse reward + invalid-action penalty**: `{0,10}` with a `0.1 × n_invalid` penalty
  (the penalty moved from the trainer actor to the agent-loop; same total per episode).
  Within the episode the placement is **per-turn**: −0.1 lands at the turns whose own action
  was invalid, broadcast on top of the base episode return (`windowed_agent_loop.py`), not a
  uniform per-episode subtraction.
- **GRPO advantage estimator = stock verl 0.8 `grpo`.** The paper's fork did **not**
  per-trajectory-dedup its groups: the fork's `seen_pairs` dedup is gated on a flag whose
  default disables it, so group mean/std run over **all per-turn samples**, exactly what
  stock `grpo` computes. No custom estimator is needed, and none is shipped:
  the windowed loop tags each sample with `traj_uid` as the hook for a possible future
  per-trajectory estimator (`grpo_traj`), but no such estimator is registered; the
  estimator is stock `grpo`.
- **Task-heterogeneity partitions the real shuffled `server.goals` at runtime** (not an
  offline reconstruction), so each client's shard matches the original.
- **Round-threaded data seed**: `FEDAGENT_BASE_SEED = base_seed + round*100 + client`, and
  the service draws goals with `random.Random(seed)` (a plain modulo collapsed the round term
  and made every client see the same goals every round).
- **Full E epochs/round**: `total_training_steps: 0` → `null` (a smoke step-cap must never
  leak into paper runs); `save_freq` saves the round's last step; `resume_mode=disable` (the
  federation owns "resume" at the round level).
- **Validation**: a shared unperturbed val service, `test_freq=5`, `val_before_train`,
  val temperature 0.4, on the paper's held-out splits.

## Fidelity fixes baked into the config generator

`tools/verl08_migration/gen_paper_configs.py` (which emits the 176-config paper tree)
applies three fixes surfaced by the WebShop/ALFWorld implementation audits:

1. **WebShop `search_return_n` (BM25 top-K).** The original raised it only on env-het arms
   (which perturb the catalog/search and need targets reachable) and left the **engine
   default 50** on the uniform / task-het / decentralized / baseline runs. The migration had
   hardcoded 200 everywhere, which makes the non-het baselines easier. Now: **200** for
   `env_heterogeneity/` arms, **50** elsewhere, matching the original baselines.
2. **ALFWorld `max_turns = 50`** (was 12). The original ran 50-turn episodes; a smaller cap
   can only lower ALFWorld success. Set in `config/envs/alfworld.yaml` + `alfworld_val.yaml`.
3. **ALFWorld context window**, sized for the **windowed** (per-turn, `history_length=2`)
   default rollout, which is what changed the context sizing. Each turn is one training
   sample whose prompt is the bounded windowed template (task + last-2 (obs,action) + current
   obs), not a growing transcript, so the old growing-transcript budgets
   (`max_model_len=16384`, `response_length=8192`) are gone. The ALFWorld `client_overrides`
   now use `rollout.max_model_len=2560`, `response_length=512` (prompt `2048` for the short
   room text); WebShop uses `rollout.max_model_len=4608`, `response_length=512` (prompt `4096`
   for the long product pages). GRPO keeps `rollout.n`=G=8; PPO is ungrouped
   (`rollout.n=1`, 2026-07-20 fix, [`bugfixes.md`](bugfixes.md)).

> Fixes #2/#3 are **GPU-VERIFY**: confirm no OOM / prompt truncation at 50 turns on the
> target hardware; raise `max_model_len` further if episodes truncate before `done`.

## Config tree

The paper configs (`fedagent/config/paper/`) mirror the original `config/` tree 1:1 in
structure and naming, `uniform/<Model>/<setting>/<algo>/`, `env_heterogeneity/`,
`task_heterogeneity/{grpo,ppo}/{env}/`, `decentralized/`: 176 configs total (see
[reproducing.md](./reproducing.md)). The one intentional deviation: **centralized/local
baselines use T=70 × E=3 (=210 epochs)** rather than the original's 1 round × 210 epochs,
because the verl-0.8 runner draws goal variety from **rounds** (the round-threaded seed), so a
single round would repeat the same goals. Same total epochs; correct goal coverage.

## Residual differences

**Benign plumbing (no MDP effect):** the multi-turn history is verl 0.8's native concat-chat
rather than the fork's re-rendered template (equivalent information); the invalid-action
penalty is applied in the agent-loop, not the trainer; goal sampling uses a different (still
reproducible) RNG, so per-seed trajectories are not bit-identical to 0.3.1. Round re-entry
passes through `verl.model_merger`, which casts to **bf16**: every round boundary truncates
the fp32-aggregated weights (within the equivalence bar; if drift ever shows over long
horizons, switch re-entry to a model-only `resume_from_path` load, which skips the HF merge).

**Baseline dynamics (the renamed rd-70_ep-3 centralized/local configs):** each round is a
fresh subprocess started from the merged HF weights (`save_contents=[model]`,
`resume_mode=disable`), so the T=70×E=3 baselines inherit the federated arms' per-round
semantics: Adam moments re-initialize every 3 epochs, the `use_kl_loss` reference re-anchors
to each round's starting model (the original 1×210 baselines kept one optimizer and a fixed
base-model KL anchor), and the E epochs within a round replay that round's goal draw (the
original re-drew goals every epoch: 210 draws vs 70 here). Compute- and dynamics-matched to
the federated arms, but the legacy paper baseline numbers are not exactly re-derivable on
this stack; use the paper-reproduce branch for that.

**Seed-replication axis:** the 3-seed arms vary `base_seed` (42/21/84, client selection +
per-round goal draws); the original varied `SHUFFLE_SEED` (a reshuffle of the train pool
under fixed selection). Both are valid seed-noise axes over a fixed val split; they are not
the same randomness source (documented in `gen_paper_configs.py`).

**GPU-pending verification:** PPO (`gae`) critic federation and the decentralized ablations
are config-parse + code-audited but not yet smoke-run end-to-end; the larger backbones
(Qwen2.5-3B/7B, Llama-3.2-3B) and the full 70-round budget have not been exercised on this
stack. The GRPO federated path **is** GPU-verified end-to-end on both envs at the real paper
configs (WebShop; ALFWorld 2026-07-02/03 incl. the 50-turn budget, no OOM/truncation, see
[acceleration_final_2026-07-03.md](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/acceleration_final_2026-07-03.md) on the migrate/verl-0.8.0 branch).

## Verification status

(as of 2026-07-09; the running record is [`../EXPERIMENTS.md`](../EXPERIMENTS.md))

| Path | Status |
|---|---|
| TinyGuess (in-process) | GPU-verified end-to-end |
| **WebShop GRPO federated** | **GPU-verified: full multi-round loop** (train → FedAvg → merge → next round → eval), incl. the real paper config |
| **ALFWorld GRPO federated** (service + max_turns=50) | **GPU-verified: 2-round run on the real paper config** (2026-07-02/03); 50-turn budget OK |
| WebShop PPO (gae critic federation) | config-parses + code-audited; not GPU-smoke-run, use the 2026-07-20 corrected configs (`rollout.n=1`, `ppo_mini_batch_size=64`, critic mini 64) |
| ALFWorld PPO | config-parses + code-audited; not GPU-smoke-run |
| Decentralized ablations | config-parses + code-audited; not GPU-smoke-run |
| Larger backbones (3B/7B/Llama) / 70-round budget | not exercised on this stack |

## See also

- [migration_report.md](https://github.com/sunblaze-ucb/FedAgent/tree/migrate/verl-0.8.0/fedagent/docs/migration_report.md) (migrate/verl-0.8.0 branch), the **complete migration walkthrough**: the route decision,
  the environment-build saga, and the hard problems (checkpoint/agent-loop/env-service/windowed) *in depth*.
  *This* doc is the condensed fidelity record; that one is the full engineering account.
- [architecture.md](./architecture.md), how the overlay is built
- [reproducing.md](./reproducing.md), the paper config matrix
- [`../EXPERIMENTS.md`](../EXPERIMENTS.md), the running experiment log + per-fix detail
