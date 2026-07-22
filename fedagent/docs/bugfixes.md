# Bug fixes

A running log of notable correctness / robustness fixes to the FedAgent verl-0.8 overlay, with enough
mechanism to understand *why* each was wrong and how it was fixed.

---

## 2026-07-22: Concat loop: inter-turn glue mis-slice on reasoning-model templates (Qwen3-family) → eaten turn boundary

- **Files:** `fedagent/agent_loops/gym_text_agent_loop.py`; new verl-free helper
  `fedagent/agent_loops/_concat_glue.py`; offline regression `tests/test_concat_glue.py`.
- **Severity:** latent. The paper-default rollout mode is WINDOWED (per-turn prompts — no
  inter-turn glue, so it never had this bug); only CONCAT-mode runs on templates whose
  generation prompt ends in a scaffold are affected. Qwen2.5 (no scaffold) is byte-identical
  pre/post fix.

### The bug

The concat loop builds one token sequence per episode: `prompt + Σ_t (action_t + glue_t)`,
where `glue_t` (assistant-close `<|im_end|>` + observation-as-user-turn + next generation
prompt) was recovered by a blind token slice `obs_tokens = new_ids[len(cur_ids):]` — assuming
the fresh full re-render `new_ids` extends `cur_ids` (prompt + raw sampled action ids) as a
token prefix. That holds on Qwen2.5 but is FALSE on Qwen3/Qwen3.5: the generation prompt ends
with a `<think>\n\n</think>\n\n` scaffold that is DROPPED when the turn is re-rendered as
completed history, so `cur_ids` is not a prefix of `new_ids` and the slice is offset by the
scaffold length — silently eating the `<|im_end|>\n<|im_start|>user` boundary that closes the
assistant turn (~4 tokens/turn on Qwen3.5-style templates). The observation then runs straight
out of the action with no turn boundary. The glue is masked (`response_mask=0`) so the loss
TARGETS were never wrong — the corruption is in the CONTEXT the policy and the trainer's
recomputed logprobs condition on (measured where investigated: ~0.039 nats/token on the next
action's logprobs).

### The fix

Recover the glue by a **string** diff of consecutive rendered prompts
(`apply_chat_template(tokenize=False)`), anchored at their divergence point (where the action
content begins), then token-encode the glue text — immune to any scaffold the template adds or
drops, and to BPE context-dependence (never a token diff). A leading turn-terminator already
emitted by the sampler (trained as part of the action) is deduped; if the action text cannot
be located, fall back to the legacy slice (never worse than before). The helper is
dependency-free so it is unit-tested offline: `tests/test_concat_glue.py` locks Qwen2.5
parity (anchored == legacy == correct), the Qwen3 boundary recovery, terminator dedup, and
the fallback path.

---

## 2026-07-22: Cross-round persistent trainer: `_reset_engine` leaks a full model+Adam per round → reserved-VRAM creep → OOM (PPO ~2× GRPO)

- **File:** `fedagent/fed/persistent_patch.py` (`_reset_engine`); root cause lives in stock verl
  (`others/verl/verl/workers/engine/fsdp/transformer_impl.py:576-578`) but is only *activated* by
  our reload pattern, so the fix is overlay-side.
- **Severity:** blocker for every `cross_round: true` run — the process OOMs after a
  headroom-dependent number of rounds (crash + lost round, not corrupted numbers; resume from the
  last aggregated round is clean). All `paper_accelerated` configs (PPO **and** GRPO, WebShop
  **and** ALFWorld) set `cross_round: true`, so the whole accelerated recipe tree was exposed.
- **Provenance:** a WebShop PPO cross-round run OOM'd at round 14; lowering
  `rollout.gpu_memory_utilization` bought 13 rounds and it OOM'd again at round 27. The telling
  signature: training-side PyTorch memory grew **monotonically ~0.6 GB/round** (40 GB @rd14 →
  48 GB @rd27) — a leak with a per-round clock, not a per-step one. The only per-round
  allocation event is `reload_client_model`/`reload_critic_model`.

### The bug

`_reset_engine` hot-swaps a client's weights by calling `eng.initialize()` on the live FSDP
engine. In verl, `initialize()` → `_build_model_optimizer()` ends in a bare rebind:

```python
self.module = module          # transformer_impl.py:576 — old module never freed
self.optimizer = optimizer    # :577 — old Adam (fp32 m+v, the bulk) never freed
self.lr_scheduler = lr_scheduler
```

Stock verl calls `initialize()` **once per process**, so the rebind is harmless there. Our
cross-round trainer calls it **every round**, which activates two problems:

1. **Transient 2× peak.** The new module+optimizer are fully built (lines 402–420) *before* the
   rebind drops the old ones — both generations coexist at the moment of assignment.
2. **Permanent reserved-memory creep (the killer).** After the rebind the old objects are
   Python-garbage, but the CUDA caching allocator keeps their blocks **reserved**. The
   allocate-new-then-free-old ordering plus FSDP flat-param allocation patterns fragment the
   cache, so each round's rebuild can't fully reuse the previous round's freed blocks and asks
   the GPU for fresh memory instead. Nothing ever calls `empty_cache()` in the worker, so
   reserved grows monotonically → ~0.6 GB/round (PPO) → OOM.

Two aggravators made it worse and hid it:

- **`checkpoint_manager` pins the old generation.** `initialize()` also rebuilds
  `FSDPCheckpointManager(model=…, optimizer=…, lr_scheduler=…)` (transformer_impl.py:193) — but
  the *old* manager, still referenced by the engine until the rebind, holds refs to the old
  module/optimizer. Any release scheme that doesn't drop it frees nothing.
- **The existing hygiene call was in the wrong process.** `persistent_task_runner._reset_for_client`
  ends with `torch.cuda.empty_cache()` — but that runs in the **driver**; the engines (and the
  leak) live in the **Ray FSDP worker processes**. It never touched the leaked memory.

### Why PPO leaks ~2× GRPO (and why the leak scales with the algo)

Per round, `reload_client_model` rebuilds the actor (module + Adam) and the ref — but the ref is
`forward_only`, so verl builds **no optimizer** for it (transformer_impl.py:567): the ref leaks
only a backbone. PPO (`adv_estimator=gae`) additionally calls `reload_critic_model` →
`_reset_engine` on the value engine: a **second full module + Adam** per round. Since Adam's
fp32 m/v is the dominant term, PPO's leak rate ≈ 2× GRPO's (~0.6 vs ~0.3 GB/round observed).

### Why 70-round GRPO runs "worked" (rounds-to-OOM arithmetic)

Rounds-to-OOM ≈ training-side headroom ÷ per-round leak. The two arms differ on **both** terms:

|  | GRPO | PPO |
|---|---|---|
| leak rate | ~0.3 GB/round | ~0.6 GB/round |
| resident state | actor + ref | + full critic (module, Adam, grads, activations) |
| `rollout.gpu_memory_utilization` (recipe) | 0.6 | 0.5 — lowered to squeeze the critic in at all |

PPO starts with far less headroom *and* burns it twice as fast → wall at ~rd27. GRPO's
rounds-to-OOM simply exceeded the configured 70 rounds — those runs **finished while leaking**
(their logs should show the same monotonic climb at ~half slope). A 210-round GRPO config
(`ep_per_round_change/`, `rd-210`) or a larger model would have hit the same wall. Mode matrix:
the default subprocess-per-client path and `persistent`-per-round (process exits each round)
never accumulate across rounds; **only `cross_round: true` exposes the leak**.

### The fix

`_reset_engine` now releases the old generation **before** rebuilding, inside the worker:

```python
for _attr in ("module", "optimizer", "lr_scheduler", "checkpoint_manager"):
    if getattr(eng, _attr, None) is not None:
        setattr(eng, _attr, None)      # incl. checkpoint_manager — it pins module/optimizer refs
gc.collect()                            # actually collect the FSDP module's reference cycles
torch.cuda.empty_cache()                # return reserved blocks -> rebuild reuses freed memory
eng.initialize()
```

Release-before-rebuild cures **both** problems: the transient peak returns to 1× (the rebuild
allocates into just-freed memory), and reserved no longer accretes (back to baseline each
round). `empty_cache()` is safe next to the co-resident vLLM engine — it only returns *unused*
cached blocks; vLLM's KV cache is live allocation. Cost: one allocator round-trip per client
reload (ms-scale, amortized over a whole client fit). The cross-round acceleration (the entire
point of lever #4) is preserved.

**Coverage:** one function covers the full matrix — `reload_client_model` (actor+ref; GRPO and
PPO) and `reload_critic_model` (critic; PPO) both call `_reset_engine`, and the patch is
env-agnostic (the engines live in the trainer workers; WebShop/ALFWorld only differ in the
separate env-service processes). Install path unchanged: `sitecustomize` arms the deferred
import hook under `FEDAGENT_PERSISTENT=1` for every persistent/cross-round launch.

### Verification

- Source-verified against the vendored verl 0.8.0.dev actually in use (`others/verl`): the bare
  rebind (576–578), the once-per-process assumption, `initialize()` rebuilding
  `checkpoint_manager` (193 — so nulling it pre-call is safe), and the ref's
  `forward_only`→no-optimizer branch (567 — the release loop's `None`-guard handles it).
- No dangling references that would defeat the release: `TrainingWorker`/`ActorRolloutRefWorker`
  hold only the engine (they call methods, never cache `module`); the FSDP→vLLM weight bridge
  fetches `get_per_tensor_param()` fresh at each sync (engine_workers.py:711), so the rollout
  side never pins the old module; `flops_counter` holds only `hf_config`.
- **Live validation pending:** the definitive check is a cross-round run whose per-round
  training-side memory is FLAT (vs the 0.6 GB/round ramp). A hot-patched variant (gc +
  empty_cache, but **without** the `checkpoint_manager` drop) is under monitor on the original
  failing run; note that variant may retain a residual leak via the manager's pinned refs — if
  its curve bends but doesn't flatten, that's the expected signature, and this version
  (manager included) is the one to sync.

**Scope audit — not exposed:** AccelAgent (`accelagent/train.py` → stock `run_ppo` →
`fit()` once; `initialize()` runs once per process, no reload path — the verl rebind is inert
there, as in stock verl). The FedAgent subprocess and per-round persistent paths (process
lifetime bounds the accumulation). Completed runs' *numbers* everywhere (the leak crashes; it
does not corrupt weights, rewards, or advantages).

---

## 2026-07-21: Retroactive record — ALFWorld labelling identity gaps + label-miss hardening (`8b104ca`, `0a41188`)

Shipped the same day as the audit entries below but documented only in the commit messages
until now. ALFWorld is genuinely immune to the two construction RNG races (local
`random.Random` game shuffle; single-threaded game collection; `_TW_LOCK` over all reset/step
engine touches), but the sweep left four smaller items, fixed in `8b104ca`/`0a41188`:

- **No episode identity ever reached the dump** (`fedagent/envs/alfworld/alfworld_env.py`):
  `/reset` returns the episode's `gamefile` but the client dropped it, so the windowed/concat
  loops' `goal_id`/`task_type` harvesting never fired for ALFWorld. Now derives the hardness
  task_id (`alfworld_{grandparent}_{parent}_game`, VERBATIM with `partition_strategy`'s
  ALFWorld keying) + task_type from the gamefile and surfaces both in step `info`.
- **Pooled `_make_env` raced textworld's process-global registry**
  (`fedagent/envs/alfworld/service/server.py`): `init_env` → `textworld.gym.register_games`
  bumps a global `registry` dict non-atomically, so `POOL_SIZE` concurrent constructions could
  crash at startup ('dict changed size during iteration'). Now serialized under `_TW_LOCK`
  (startup-only; transitions still run concurrently). Crash class, not a correctness class.
- **Silent label-miss floor-to-hard** (`partition_strategy.py`, BOTH the ALFWorld and the
  vendored webshop-shaped `hardness_partition`): games/goals whose task_id is absent from the
  trajectories file were silently bucketed "hard" — a keying/catalog mismatch could collapse
  the whole pool to hard with no signal. Now counted + WARNED (>50 % miss flags a wrong-split/
  stale-labels mismatch explicitly), mirroring `webshop_hardness.py`.
- (Feature, same commit: `ALFWORLD_SEED_IS_INDEX` bijective seed→game mode + the
  `gen_alfworld_hardness_trajectories.py` generator — full-pool labelling in exactly N
  episodes instead of ~N ln N.)

---

## 2026-07-21: Residual construct-time RNG race: price clauses still diverged across pooled envs (the shuffle-race's last surviving facet)

- **File:** vendored `web_agent_site/envs/web_agent_text_env.py` (fix `e2b5eef`).
- **Provenance:** found during the full-population trajectory forensic pass over the v2
  hardness-label rollouts: joining all 6,410 recorded prompts back to an offline sequential
  goal reconstruction left 675 rows whose instruction differed ONLY in the price clause
  (e.g. same product, "lower than 40.00" vs "50.00"), including different clauses for the
  same product within one chunk — impossible under a single deterministic construction.
  Reproduced at HEAD: 4 concurrently-built envs split into two price variants (1,375/6,910
  instructions differ) while goal ORDER stayed identical.

**Bug.** The goal-order fix (`afa440f`) serialized SimServer's whole seeded window under
`_RNG_CONSTRUCT_LOCK`, but `WebAgentTextEnv.__init__` ends with a `self.reset()` that draws
from the process-global `random` OUTSIDE the lock (`random.choices` session id; `random_idx`
goal pick when `session_int is None`). During pooled-concurrent construction, env_j's
construct-tail reset lands inside env_k's held window — between its `random.seed(42)` and
`generate_product_prices()` — shifting the ~93 ranged-price products' `random.uniform` draws.
Downstream, `get_synthetic_goals` re-seeds 42 before sampling price bands, so the SAMPLE
stream stays aligned but the band CONTENT (`price_range` is a function of the drawn price)
differs → those products' goals get different `price_upper`/instruction price text per env.
Goal (asin, options) order is untouched, so the era's first-64 **task-id** pool guard —
price-blind by construction — passed silently. (The full-length `(asin, instruction_text)`
guard added in `337166f` would now hard-fail such a mixed pool at startup; this fix removes
the race itself.)

**Fix.** `_RNG_CONSTRUCT_LOCK` upgraded to an `RLock` (SimServer re-acquires it inside) and
`WebAgentTextEnv.__init__` holds it from the seeding block through the trailing
`self.reset()`. Sequential construction is byte-identical to the historical order.

**Verification.** (1) 4 concurrently-built envs: instruction md5 == sequential build's md5,
all four; (2) (asin, goal_options) sequence == the pre-fix sequential oracle (order and
val/train membership untouched); (3) the affected product's clause back to the canonical
value across all envs.

**Impact on the v2 hardness labels (`d6f3c9b`).** 675/6,410 label-gen episodes were served a
price clause deviating from the canonical sequential construction (ranged-price products
only). Each episode was INTERNALLY consistent — the agent saw and was scored against the
same served clause — so the labels remain valid as measured difficulty; the deviation is a
small label-noise source on those goals (price is one of `denom` score terms), quantified in
the trajectory-analysis archive (`hardness_labelgen_std4_v2/`). Offline forensics must use
the PROMPT's price clause as scoring truth (not a reconstruction), and 53 purchases of
range-priced items straddling the cap are price-undecidable offline (flagged, label-neutral:
all have score < 1 regardless).

---

## 2026-07-21: Post-race audit hardening: latent members of the same bug class

- **File:** `fedagent/envs/webshop/service/server.py`, the vendored
  `web_agent_site/envs/web_agent_text_env.py`, `fedagent/hetero/webshop_hardness.py`,
  `fedagent/data/agentic_dataset.py`.
- **Provenance:** two systematic sweeps (global-RNG × concurrency; identity/index-mapping)
  over the overlay + vendored engines, prompted by the goal-shuffle race below. All items are
  the same species: an unchecked implicit assumption whose violation is SILENT.

Fixed in one pass:

1. **Shared goal-dict pollution** (engine `receive()`): the `assigned_instruction_text`
   override wrote through `session['goal']` — a REFERENCE into the shared `self.goals` list —
   silently corrupting the canonical goal for every later session on that env. Currently inert
   (nothing sets the hack in this stack); now overrides a per-session copy.
2. **Partition `start_idx` drift** (service): preference/coverage/hardness hardcoded
   `start_idx=500` while uniform honored `WEBSHOP_VAL_SIZE` — changing the val size would have
   shifted every shard index by the delta, silently. All four now receive `start_idx=VAL_SIZE`.
3. **`NUM_GOALS` never validated** (service lifespan): the `/reset` seed→goal modulo trusted
   the env var; now HARD-FAILS unless it equals `len(env.server.goals)`.
4. **Pool-order guard sampled only the first 64 goals**: upgraded to a FULL-length
   (asin, instruction) comparison.
5. **Silent label-miss flooring** (`webshop_hardness.py`): goals whose task_id is absent from
   the trajectories file were silently defaulted to "hard" — the exact camouflage that let the
   race-era labels look plausible. Now counted and WARNED (a large miss count means the labels
   do not match the catalog/keying).
6. **Seed-window aliasing** (`agentic_dataset.py`): the `si*1_000 + i` row-seed layout collides
   when a non-last spec has `n_envs > 1000` (two rows → one env seed). Now refused loudly.

Verified end-to-end: `hardness_for_client` over the real `env.server.goals` with the
regenerated labels matches **6,410/6,410** train goals (zero misses; high 1,115 + low 5,295).
Checked and clean: task-id derivation verbatim-identical at both sites; uniform partition
val_size plumbing; per-env single-session use at request time; ALFWorld flow (path-keyed
identity, single-threaded collection, global TW lock); original Ray-actor stack (per-process
RNG).

---

## 2026-07-21: WebShop pooled service: goal shuffle raced across concurrently-built envs → nondeterministic per-env goal order

- **File:** `fedagent/envs/webshop/engine/webshop/web_agent_site/envs/web_agent_text_env.py`
  (SimServer's seeded goal-generation/shuffle window; WebAgentTextEnv's global reseeding) and
  `fedagent/envs/webshop/service/server.py` (new pool-consistency hard guard).
- **Severity:** science-correctness, broad. Every pooled WebShop service start built its envs
  from concurrent threads, and each env ended up with a DIFFERENT, nondeterministic goal order.
  Everything that assumes one shared order silently broke: seed→goal determinism (`/reset`
  serves env_j's `goals[sess]` while `_GOAL_TASKIDS` and runtime partitions are computed from
  env_0's order), per-goal `goal_id` logging (measured: only **~4.6 %** of labelling rollouts
  carried the id of the goal actually served), and index-based goal partitions (a client's
  shard indices need not select the same goals on the env that serves them). AGGREGATE metrics
  (mean success over ~uniformly mixed goals) remain approximately valid; anything per-goal is not.
- **Provenance:** vendored WebShop `SimServer.__init__` seeds the process-global `random` at the
  top (`random.seed(42)`), spends ~20 s loading products / building the search engine /
  generating goals — all of which draw from that same global stream — then
  `random.shuffle(self.goals)` at the bottom; the service `_lifespan` constructs POOL_SIZE envs
  via `asyncio.to_thread(...)` concurrently, interleaving every thread's draws.

### The bug

Detected via the hardness relabelling run: 6,409 rollouts collapsed to 4,478 unique task_ids
whose duplicate groups mixed UNRELATED products (a men's-t-shirt, a women's-swimsuit and a
CD-player episode under one `asin`+options key) — impossible at the data level, since the
canonical train slice has 6,402 unique keys with only 8 genuine ×2 phrasing duplicates. A
ground-truth join of the dump's instructions against the canonical goal list showed the served
goals scattered over the whole 6,910-goal list (val slice included, with within-chunk repeats),
while each chunk's logged-id set was the DESIGNED disjoint window under a fresh per-restart
permutation — the signature of every env (and every service start) shuffling differently.

### Impact

Ranked by how much it distorts results:

1. **GRPO group advantage was cross-task contaminated (training-signal quality).** GRPO repeats
   one dataset row `rollout.n=8` times and normalizes rewards within the group — assuming all 8
   rollouts solve the SAME goal. Each rollout borrows its own pool env, and the same `sess` index
   is a different goal on every env, so with a 16–24-env pool essentially **every group compared
   scores across 8 unrelated goals** (P(all 8 on one env) ≈ (1/K)⁷ ≈ 0). The group mean stops
   being a per-task baseline and collapses toward the global success rate: the update degenerates
   to global-baseline REINFORCE, losing GRPO's variance reduction — a rollout that happened to
   draw a hard goal gets a large negative advantage regardless of policy quality. Unbiased but
   noisier: slower convergence per unit compute, and a systematic handicap for GRPO in any
   GRPO-vs-PPO comparison (PPO's ungrouped n=1 critic path has no group semantics and is
   untouched by this channel).
2. **Task-heterogeneity partitions were not realized.** Client shards are index sets over env₀'s
   order; serving used env_j's order — every client actually trained on a ~uniform mix of the
   whole pool, so hardness/coverage/preference arms were diluted toward homogeneous and any
   cross-arm difference is noise. (Env-heterogeneity arms perturb the catalog/search, not goal
   order — their manipulation stands.)
3. **The train/val holdout did not exist.** Each env's positions `[500:]` hold a random
   6,410-goal subset of all 6,910: ~93 % of canonical val goals were served during training
   (~7 % of the sampling mass), and ~93 % of "val" episodes were actually train goals. Val
   metrics ≈ train-distribution performance, not generalization.
4. **Small-shard curricula became infinite streams.** `min_goals_per_client=100`-style configs
   were designed as heavy repetition over a fixed small set; the race turned them into fresh
   ~random goals on every reset (and a new mix every service restart). Results from such configs
   do not reflect the designed protocol (and may be inflated by the accidental diversity).
5. **Per-goal attribution was scrambled** — only ~4.6 % of logged ids matched the served
   episode; difficulty labels and per-goal success tables built through the service are noise.
6. **Run-to-run reproducibility:** the served goal stream changed on every service start,
   inflating the effective noise floor of any cross-run comparison.

**Still valid:** aggregate success rates (the served mix is ~uniform, so means are ~unbiased),
learning-curve shapes, and same-broken-way A/B comparisons on aggregate metrics — plus the
original verl-agent-0.3.1 paper runs (Ray-actor processes; see the scope audit below).

### The fix

- The entire seeded construction window (`seed(42)` → load → `get_goals` → shuffle →
  `setstate`) now holds a module-level `_RNG_CONSTRUCT_LOCK`; `WebAgentTextEnv.__init__`'s own
  global reseeding takes the same lock. Goal generation itself consumes the global stream, so
  only serializing the FULL window reproduces the historical sequential order — and val/train
  membership (`goals[0:500]`) is pinned to that order, so byte-for-byte reproduction is a hard
  requirement, not cosmetics. Cost: pool construction is serialized (~26 s/env).
- `server.py` `_lifespan` now HARD-FAILS if any pool env's goal order diverges from env 0
  (first-64 task-id comparison), so this bug class can never pass silently again.

### Verification

- Sequential 200-key goal-order fingerprint byte-identical before/after the change; 8
  concurrently-built envs identical to each other AND to the historical sequential order.
- 32-goal windowed labelling smoke through the full service stack after the fix: every dump
  row's served instruction, intended goal index and logged `goal_id` agree.
- Full-pool relabelling after the fix yields the goal data's true key cardinality (~6,402
  unique task_ids), not 4,478.

**Recipe boundary:** any per-goal-attribution artifact produced through a pooled WebShop
service before this fix (hardness labels, per-goal success analyses, realized heterogeneity
shards) is suspect and should be regenerated or re-audited.

**Scope audit — not exposed:**
- **ALFWorld service:** structurally safe on four counts — the game list is collected ONCE,
  single-threaded, in the shared base env BEFORE the pool fan-out (every pooled `init_env`
  wraps that same list); federated sharding/partitions filter that single list at collection
  time; `/reset` holds the global `_TW_LOCK` across `env.seed(seed)+reset()`; and the served
  game's identity comes from the episode itself (`extra.gamefile`), never from an env-0 index
  table. The identical unsafe idiom in `alfred_tw_env.py` (global `random.seed`+`shuffle`,
  adjacent calls) was nonetheless hardened to a local `random.Random` — byte-identical order —
  as defense in depth.
- **Original verl-agent-0.3.1 stack (the paper checkpoints):** safe — WebShop envs live in
  separate Ray-actor PROCESSES (`env_package/webshop/envs.py`), each with a private global
  `random`, so every actor computes the same deterministic seed-42 order; the race requires
  threads sharing one interpreter.

---

## 2026-07-21: Hardness labelling rolled out in CONCAT mode against WINDOWED-trained references → labels collapse toward zero-shot

- **File:** `tools/gen_hardness_trajectories.py` (missing rollout-mode injection) and
  `fedagent/agent_loops/windowed_agent_loop.py` (missing `goal_id`/`task_type` tag propagation).
- **Severity:** science-correctness. A windowed-trained reference measured in concat mode succeeds
  at ≈ the zero-shot rate, so the generated task-difficulty labels are near-degenerate and the
  ξ′ (Hardness) arm's easy/hard split loses its signal. **Symptom:** the label file is well-formed
  but the easy rate sits at ~1–2 % instead of ~20–30 %, and the run looks like "the reference
  checkpoint is broken" when it isn't.
- **Provenance:** `run_fed` injects the rollout mode into every train/eval command it builds
  (`inject_rollout_mode` + `FEDAGENT_HISTORY_LENGTH`; DEFAULTS `rollout_mode: windowed`), but the
  label generator built its verl val command from scratch and injected neither — labelling
  silently fell back to the stock concat `gym_text` loop.

### The bug

Two independent halves:

1. **Rollout-mode mismatch (the collapse).** Paper checkpoints are trained AND evaled windowed
   (fresh per-turn prompt = task + last-2 (obs, action) window, response = ONE action, budgets
   4096/512/4608). The concat loop instead accumulates the whole history into one growing prompt —
   out-of-distribution for a windowed-trained policy. Measured on the same 128 train goals,
   greedy: **1.6 %** strict success under concat (≈ the ~1.4 % zero-shot rate) vs **22.7 %**
   windowed with paper budgets.
2. **`goal_id` tag loss.** The concat loop copies env info tags (`goal_id`, `task_type`) into
   per-sample `reward_extra_info`; the windowed loop didn't, so once half 1 was fixed the
   labelling aggregation died loudly with "dump has no goal_id fields".

### The fix

- The generator appends its `client_overrides`, then calls `inject_rollout_mode(cmd, cfg)` and
  merges `history_length_env(cfg)` into the run env — labels now roll out in the SAME mode as
  training/eval by construction. The regen docs (`data/hardness/README.md` "Regenerating", the
  generator docstring, the hardness smoke-config header) now require a paper-budget config for
  paper references.
- `WindowedGymTextAgentLoop` collects `goal_id`/`task_type` from step info and stamps them into
  every per-turn output's `reward_extra_info`, mirroring the concat loop.
- Supporting: `FEDAGENT_SEED_OFFSET` (`fedagent/data/agentic_dataset.py`, additive per-row seed
  shift) lets full-pool labelling run as disjoint, resumable chunks on shared GPUs.

### Verification

- A/B on the same 128 train goals and checkpoint (greedy): concat **1.6 %** vs windowed+paper
  budgets **22.7 %** strict success — the collapse and its cure. (Aggregate rates; valid
  independently of the goal-shuffle race above.)
- Every dump row now carries `goal_id`; the aggregation's "dump has no goal_id fields" guard no
  longer trips.
- NOTE: the first full-pool regeneration run with this fix (2026-07-21, 4,478 keys / 20.7 %)
  was itself INVALIDATED by the pooled-service goal-shuffle race (previous entry) — its per-goal
  attribution was scrambled — and was reverted and redone after that fix.

---

## 2026-07-20: PPO rollout grouping: the original's dead `env.rollout.n` resurrected as a live `rollout.n=8` → 8× rollout volume

- **File:** `tools/gen_paper_configs.py` (→ all 85+85 generated PPO configs in
  `config/paper/` + `config/paper_accelerated/`), the 4 hardness-rerun PPO configs in
  `fedagent/tools/verl08_migration/accel/{webshop,alfworld}/` (a machine-local, gitignored
  working area — see `.gitignore` — not part of the tracked tree), and stale doc claims
  (`docs/migration.md`, `docs/configuration.md`, review reports).
- **Severity:** science-correctness + cost. Every migrated PPO run collected **512
  trajectories/step (64 prompts × `rollout.n=8`)** where the executed original collected
  **64 (64 × 1, ungrouped)** — 8× the paper's rollout volume, ~8× per-step optimizer updates
  (row pool ≤7680 vs ≤960 on WebShop, minibatch 64 rows), 512 concurrent sessions against a
  single env service, and the observed "PPO ≈ 8× slower than GRPO". GRPO is unaffected.
- **Provenance:** the original fed yamls carry `verl: env: rollout: n: 8` for BOTH algos, but on
  the PPO path that key was **dead**; the migration translated it into a live
  `actor_rollout_ref.rollout.n=8`.

### The bug

The fork groups rollouts via `env.rollout.n` (verl's own `actor_rollout_ref.rollout.n` is
asserted `== 1`, fork main_ppo.py:168). On the PPO path the yaml's `env.rollout.n: 8` never
reached the executed command:

1. the fork default is `env.rollout.n: -1` = "disable env grouping"
   (`third_party/verl-agent/verl/trainer/config/ppo_trainer.yaml:293`);
2. the PPO base scripts set no `env.rollout.n` (`scripts/verl-agent/ppo/run_webshop.sh` header:
   "PPO has no GRPO/GiGPO group dimension"; same for `run_alfworld.sh`) — the GRPO scripts DO
   set it (`grpo/run_webshop.sh:82`);
3. the fed orchestrator only regex-rewrites keys already present in the base script and never
   handles `rollout.n` (`core/fed/script_builder.py`).

So the executed original PPO ran **ungrouped**: envs = `train_batch_size × group_n(=1)` = 64
(`agent_system/environments/env_manager.py:1101,1181`), and the rollout loop's prompt repeat is
gated on `env.rollout.n > 0` (`agent_system/multi_turn_rollout/rollout_loop.py:282`). The
earlier "`rollout.n` must stay 8 for PPO" audit note (migration.md, since corrected) verified
the *formula* `train_batch_size × rollout.n` (dynamic filter-groups path, `rollout_loop.py:414`)
but not the executed *value*.

### The fix

`gen_paper_configs.py` now emits, for PPO only: `rollout.n=1` (ungrouped, == original),
`actor.ppo_mini_batch_size=64` (prompts; × n=1 = the original 64-row minibatch — GRPO keeps
8 × 8), and an explicit `critic.ppo_mini_batch_size=64` (the base body's 8 is GRPO-sized).
Both trees regenerated (85 PPO configs each; all 91+91 GRPO configs byte-identical). The 4
accel hardness-rerun PPO configs got the same three-line fix. Post-fix, PPO == GRPO in
per-step trajectory budget (64), minibatch rows (64), and env-service concurrency (64); the
arms differ only in estimator + critic, and PPO round wall-clock should drop ~8× to
≈1.1–1.4× GRPO.

**Recipe boundary:** any PPO output produced before this fix (n=8 recipe) is a different
recipe — do not mix in figures or resume into post-fix runs. **Paper-text erratum flagged:**
`main.tex:1327` ("PPO uses the same group size") describes the never-executed config;
`main.tex:1320`'s "Mini-batch size 64" is what actually ran (and is now the literal config
value again).

### Verification

- Residual sweep: `grep -rl "adv_estimator: gae" --include="*.yaml" | xargs grep -l
  "rollout.n=8"` → empty over `config/paper*/` + the accel rerun configs (the historical
  `tools/verl08_migration/poc/gpu_verify/` snapshots — preserved on the `migrate/verl-0.8.0`
  branch — keep n=8 by design; they document what was validated then).
- Regeneration diff = exactly {mini 8→64, n 8→1, +critic mini 64} × 85 × 2 trees; GRPO 0 files
  changed (also proves the port bands didn't drift).
- Evidence chain re-verified in the `paper-reproduce-verl-agent` checkout: fork default,
  both PPO base scripts, `script_builder.py` (no generic verl-section injection; targeted
  `_sub` rewrites only), env build math, and the `n > 0` repeat gate.

---

## 2026-07-19: Hardness partition: success quota double-drawn from the *unsuccessful* pool → floored difficulty

- **File:** `fedagent/hetero/webshop_hardness.py` (`hardness_partition`), and the ALFWorld copies in
  `fedagent/envs/alfworld/engine/agent_system/environments/partition_strategy.py`
  (`hardness_partition`, `hardness_partition_alfworld`).
- **Severity:** science-correctness, the reported task-level **Hardness** arm did not realize the
  dispersion the paper's control law prescribes. **Symptom:** none at runtime (shards are the right
  size `L`); visible only by measuring the realized per-client success rate.
- **Provenance:** inherited verbatim from upstream verl-agent `partition_strategy.py`; surfaced by
  a paper-fidelity audit.

### The bug

`HardnessPartition` builds each client's shard as `X_i = Y_i ∪ F_i`: a Beta-sized success quota `Y_i`
from the easy (high-success) bucket, remainder of the fixed quota `L` filled from the hard bucket, so
the realized success fraction is `rho_i = |Y_i|/L`. The upstream body computed the step-2 shortfall as
`current_success_count - len([s for s in current_client_data if s in high_success_data])` *after* step 1
had already `.remove()`d its picks from `high_success_data`, so that comprehension is always `0` and
`remaining == current_success_count` every time. Step 2 therefore unconditionally drew an **extra**
`|Y_i|` items from the **low-success** pool (mislabeled "additional success"), and step 3 topped up from
a **mixed** pool. Net effect: every client's success rate is floored at the global rate `g` (easy
clients reproduce correctly; hard clients with `|Y_i|/L < 0.5` are pulled up toward `g`), shrinking
`Delta^2_hard` to ~25-59% of `C_h/(xi'+1)` and drifting the mean `rho` with `xi'` (0.50→0.59 on WebShop
`g=0.278`; 0.50→0.70 on ALFWorld `g=0.594`), breaking the paper's D1 control law and D3 mean-invariance
for this arm.

### The fix

The shard assembly now follows the paper's `HardnessPartition` Algorithm **literally**:

1. **`Y_i` via CoveragePartition on the success pool.** The Beta success COUNTS `{|Y_i|}` are still
   drawn by `generate_client_sizes` (unchanged), but the easy goals themselves are now placed by
   `assign_with_overlap(|Y|, {|Y_i|}, r_easy, rng)`: the *same* Beta-sizing + overlap primitive
   `coverage_partition` uses, with `r_easy = target_sum/|Y|`. This is exactly the Algorithm box's
   `Y_i <- CoveragePartition(Y, N, (kappa_min, kappa_avg, kappa_max), xi', r)`. Effect: each easy
   goal lands in ~`floor/ceil(r_easy)` clients (exact cross-client replica budget) and the union of
   `{Y_i}` **covers the entire easy pool**, where the prior independent per-client `rng.choice`
   orphaned ~`exp(-r_easy)` (≈6% WebShop / ≈9% ALFWorld) of the easy goals.
2. **`F_i` from the unsuccessful pool.** The remainder up to `L` is filled **strictly from
   `low_success_data`** by simple without-replacement sampling; this alone removes the step-2
   flooring bug (the broken shortfall test on the easy bucket is gone). Step 3 remains only as a
   safety top-up for the degenerate `|U| < L-|Y_i|` case (never triggers at scale, `|U| >> L`).

Seed 42 is shared, so every client recomputes the SAME global assignment and indexes its own slice
(mirrors `coverage_partition`). Beta sizing and `base_seed=42` are unchanged, so `{|Y_i|}` (hence
`rho_i`, `Delta^2_hard`, `rho_bar`) are set exactly as before; only WHICH easy goals each client
sees (and their controlled overlap) changes. Both WebShop (`webshop_hardness.py`) and the two
ALFWorld copies (`partition_strategy.py`) are updated identically.

### Verification

Calling the REAL `hardness_partition` (WebShop, `g=0.278`) and `hardness_partition_alfworld`
(ALFWorld, `g=0.594`) for all 100 clients on synthetic labels (`N=100`, `L=100`):

| env | `xi'` | `Delta^2_hard` | `C_h/(xi'+1)` | `rho_bar` | easy coverage | indep-draw ref |
|---|---|---|---|---|---|---|
| WebShop  | 1   | 0.14933 | 0.12500 | 0.5000 | **100.0%** | 94.3% |
| WebShop  | 256 | 0.00102 | 0.00097 | 0.5000 | **100.0%** | 94.2% |
| ALFWorld | 1   | 0.14933 | 0.12500 | 0.5000 | **100.0%** | ~94% |
| ALFWorld | 256 | 0.00102 | 0.00097 | 0.5000 | **100.0%** | ~94% |

`Delta^2_hard` is bit-identical to the count-only fix (the 0.149 vs 0.125 at `xi'=1` is the known
finite-`L` boundary overshoot, matching the paper's distribution figure); `rho_bar = 0.5000`
exactly (D3); and the easy pool is now **fully covered** (vs the ≈6–9% orphaned by an independent
draw). The paper's `HardnessPartition` (D1/D3) holds, and the code now matches the Algorithm box's
`Y_i <- CoveragePartition(...)` line literally (was `0.088`/`0.589` `rho_bar` under the original bug).

---

## 2026-07-11: Env-service pool: `/create` double-borrow race → pool drain → rollout hang

- **Services:** `fedagent/envs/webshop/service/server.py`, `fedagent/envs/alfworld/service/server.py`
- **Severity:** blocker, a full-batch run can hang indefinitely. **Symptom:** training silently
  stalls, with no crash or traceback.
- **Provenance:** surfaced by the release audit of the extracted standalone framework (AccelAgent),
  where the same inherited services were fixed (commit `ac08200`); backported here.

### What the code does

Each env instance is expensive (`gym.make` ~26 s; ALFWorld also spins up a JVM), so the service
pre-warms a small **pool** (`POOL_SIZE`, default 4) and every episode **borrows** one:

```
/create      → env = await _pool.get()   # borrow; parks (awaits) when the pool is empty
/reset,/step → use it
/close       → _pool.put_nowait(env)     # return it
```

Under the full-PPO storm two conditions hold at once (the service's own comments note both):

- **The pool is exhausted**: `train_batch_size × rollout.n` (hundreds of) episodes share 4 envs, so
  `await _pool.get()` blocks. This is intended backpressure.
- **Sockets reset mid-flight**: the HTTP boundary is overwhelmed, so the client **retries** on
  transport errors, reusing the **same `session_id`**.

### The bug: a check-then-borrow TOCTOU race

The original `/create`:

```python
async def create(r: Sid):
    if r.session_id in _sessions:      # (1) CHECK
        return {"ok": True}
    env = await _pool.get()            # (2) BORROW, suspends here while the pool is empty
    _sessions[r.session_id] = _Session(env)   # (3) INSERT
    return {"ok": True}
```

The trap is the `await` between the check (1) and the insert (3): during that suspension the
`session_id` is **not yet in `_sessions`**. Timeline under pool exhaustion (session `S`):

```
T0  client /create(S) #0   → coroutine A: S absent → parks in await _pool.get()   (S not inserted)
T1  A's socket resets       → client gets TransportError → retries
T2  client /create(S) #1   → coroutine B: S STILL absent → B ALSO parks in await _pool.get()
T3  pool returns two envs   → A borrows env_a, inserts _sessions[S] = env_a
                              B borrows env_b, inserts _sessions[S] = env_b   ← overwrites
```

`env_a` is now **orphaned**: nothing references it and there is no reaper, so the pool's effective
size permanently drops by one. Repeated over a long run the pool drains to zero, every `/create`
blocks forever, and **the whole training run hangs** (Layer A's `await env.reset()` has no timeout).

**The pivotal, non-obvious fact:** when the client's connection resets, server coroutine A is **not
cancelled**: Starlette/uvicorn only set a "disconnected" flag; the running ASGI task keeps sitting in
`await _pool.get()`. So a client retry is not a *replacement* of the in-flight request, it is an
*additional* concurrent same-session `/create`.

### Why the existing guard didn't catch it

The handler already had `if session_id in _sessions: return` plus a comment, *"a retried /create must
NOT borrow a 2nd env, that would orphan the 1st and slowly drain the pool."* The **failure mode named
in the comment is exactly right**; the guard simply does not close it. It covers only the
*lost-response-after-completion* case (the first `/create` finished and inserted, its ok response was
dropped, the retry finds `S` present and short-circuits). It does **not** cover the
*in-flight-during-exhaustion* case (the first `/create` is still parked, so `S` is absent and the retry
borrows again), which is exactly the regime that pool exhaustion creates. A classic check-then-act
TOCTOU: the check and the borrow+insert are not atomic because an `await` sits between them.

### The fix

Reserve the session **before** the blocking borrow, and make an overlapping caller **wait** instead of
borrow:

```python
async with _create_lock:                       # guards the O(1) bookkeeping only, never the borrow
    if r.session_id in _sessions:
        return {"ok": True}
    ev = _pending.get(r.session_id)
    first = ev is None
    if first:
        _pending[r.session_id] = ev = asyncio.Event()   # occupy the slot BEFORE the await
if not first:
    await ev.wait()                            # a concurrent create is borrowing; wait for it
    return {"ok": r.session_id in _sessions}
env = await _pool.get()                        # only the FIRST caller borrows, exactly once
async with _create_lock:
    _sessions[r.session_id] = _Session(env)
    _pending.pop(r.session_id, None)
ev.set()                                       # wake the waiters
```

- The reservation (`_pending[S]` under `_create_lock`) is taken **before** `await _pool.get()`, so an
  overlapping retry sees it and does not double-borrow.
- The retry **waits on the event** and returns ok only after the env truly exists, so it also never
  returns ok before the following `/reset` can succeed.
- `/close` now pops under `_create_lock` and **drains under `sess.lock`**, so a `/close` racing an
  in-flight `/step` cannot recycle an in-use env to a second session.
- `/reset` now holds `sess.lock` (previously it did not), so a retried `/reset` cannot re-run
  `env.reset()` after the episode already advanced.

All federated heterogeneity / partition code is **unchanged**: only the pool-borrow concurrency was
touched.

### Why it hid for so long

It is a load- and timing-dependent race that presents as a *hang* (no crash, no wrong numbers); it was
"defended" by a guard + comment that read as correct; and it triggers only under the precise full-batch
regime (exhaustion + frequent resets) that equivalence / correctness validation never exercises.
Completed runs' *results* are unaffected; the bug risks a hang, not corrupted weights or rewards.

### Verification

`py_compile` clean; the idempotent-borrow algorithm is stress-tested: same-session storm → exactly one
borrow; exhaustion + retry → the retry waits, single borrow; random create/step/close churn → the pool
count is conserved with zero leak; lock order `_create_lock → sess.lock` never nests (no deadlock).
