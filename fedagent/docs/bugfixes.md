# Bug fixes

A running log of notable correctness / robustness fixes to the FedAgent verl-0.8 overlay, with enough
mechanism to understand *why* each was wrong and how it was fixed.

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
