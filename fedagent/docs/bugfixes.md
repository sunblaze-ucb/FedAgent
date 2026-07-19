# Bug fixes

A running log of notable correctness / robustness fixes to the FedAgent verl-0.8 overlay, with enough
mechanism to understand *why* each was wrong and how it was fixed.

---

## 2026-07-19 — Hardness partition: success quota double-drawn from the *unsuccessful* pool → floored difficulty

- **File:** `fedagent/hetero/webshop_hardness.py` (`hardness_partition`), and the ALFWorld copies in
  `fedagent/envs/alfworld/engine/agent_system/environments/partition_strategy.py`
  (`hardness_partition`, `hardness_partition_alfworld`).
- **Severity:** science-correctness — the reported task-level **Hardness** arm did not realize the
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
`g=0.278`; 0.50→0.70 on ALFWorld `g=0.594`) — breaking the paper's D1 control law and D3 mean-invariance
for this arm.

### The fix

The shard assembly now follows the paper's `HardnessPartition` Algorithm **literally**:

1. **`Y_i` via CoveragePartition on the success pool.** The Beta success COUNTS `{|Y_i|}` are still
   drawn by `generate_client_sizes` (unchanged), but the easy goals themselves are now placed by
   `assign_with_overlap(|Y|, {|Y_i|}, r_easy, rng)` — the *same* Beta-sizing + overlap primitive
   `coverage_partition` uses — with `r_easy = target_sum/|Y|`. This is exactly the Algorithm box's
   `Y_i <- CoveragePartition(Y, N, (kappa_min, kappa_avg, kappa_max), xi', r)`. Effect: each easy
   goal lands in ~`floor/ceil(r_easy)` clients (exact cross-client replica budget) and the union of
   `{Y_i}` **covers the entire easy pool** — where the prior independent per-client `rng.choice`
   orphaned ~`exp(-r_easy)` (≈6% WebShop / ≈9% ALFWorld) of the easy goals.
2. **`F_i` from the unsuccessful pool.** The remainder up to `L` is filled **strictly from
   `low_success_data`** by simple without-replacement sampling — this alone removes the step-2
   flooring bug (the broken shortfall test on the easy bucket is gone). Step 3 remains only as a
   safety top-up for the degenerate `|U| < L-|Y_i|` case (never triggers at scale, `|U| >> L`).

Seed 42 is shared, so every client recomputes the SAME global assignment and indexes its own slice
(mirrors `coverage_partition`). Beta sizing and `base_seed=42` are unchanged, so `{|Y_i|}` — hence
`rho_i`, `Delta^2_hard`, `rho_bar` — are set exactly as before; only WHICH easy goals each client
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

## 2026-07-11 — Env-service pool: `/create` double-borrow race → pool drain → rollout hang

- **Services:** `fedagent/envs/webshop/service/server.py`, `fedagent/envs/alfworld/service/server.py`
- **Severity:** blocker — a full-batch run can hang indefinitely. **Symptom:** training silently
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

- **The pool is exhausted** — `train_batch_size × rollout.n` (hundreds of) episodes share 4 envs, so
  `await _pool.get()` blocks. This is intended backpressure.
- **Sockets reset mid-flight** — the HTTP boundary is overwhelmed, so the client **retries** on
  transport errors, reusing the **same `session_id`**.

### The bug — a check-then-borrow TOCTOU race

The original `/create`:

```python
async def create(r: Sid):
    if r.session_id in _sessions:      # (1) CHECK
        return {"ok": True}
    env = await _pool.get()            # (2) BORROW — suspends here while the pool is empty
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

`env_a` is now **orphaned** — nothing references it and there is no reaper — so the pool's effective
size permanently drops by one. Repeated over a long run the pool drains to zero, every `/create`
blocks forever, and **the whole training run hangs** (Layer A's `await env.reset()` has no timeout).

**The pivotal, non-obvious fact:** when the client's connection resets, server coroutine A is **not
cancelled** — Starlette/uvicorn only set a "disconnected" flag; the running ASGI task keeps sitting in
`await _pool.get()`. So a client retry is not a *replacement* of the in-flight request — it is an
*additional* concurrent same-session `/create`.

### Why the existing guard didn't catch it

The handler already had `if session_id in _sessions: return` plus a comment — *"a retried /create must
NOT borrow a 2nd env — that would orphan the 1st and slowly drain the pool."* The **failure mode named
in the comment is exactly right**; the guard simply does not close it. It covers only the
*lost-response-after-completion* case (the first `/create` finished and inserted, its ok response was
dropped, the retry finds `S` present and short-circuits). It does **not** cover the
*in-flight-during-exhaustion* case (the first `/create` is still parked, so `S` is absent and the retry
borrows again) — which is exactly the regime that pool exhaustion creates. A classic check-then-act
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
env = await _pool.get()                        # only the FIRST caller borrows — exactly once
async with _create_lock:
    _sessions[r.session_id] = _Session(env)
    _pending.pop(r.session_id, None)
ev.set()                                       # wake the waiters
```

- The reservation (`_pending[S]` under `_create_lock`) is taken **before** `await _pool.get()`, so an
  overlapping retry sees it and does not double-borrow.
- The retry **waits on the event** and returns ok only after the env truly exists — so it also never
  returns ok before the following `/reset` can succeed.
- `/close` now pops under `_create_lock` and **drains under `sess.lock`**, so a `/close` racing an
  in-flight `/step` cannot recycle an in-use env to a second session.
- `/reset` now holds `sess.lock` (previously it did not), so a retried `/reset` cannot re-run
  `env.reset()` after the episode already advanced.

All federated heterogeneity / partition code is **unchanged** — only the pool-borrow concurrency was
touched.

### Why it hid for so long

It is a load- and timing-dependent race that presents as a *hang* (no crash, no wrong numbers); it was
"defended" by a guard + comment that read as correct; and it triggers only under the precise full-batch
regime (exhaustion + frequent resets) that equivalence / correctness validation never exercises.
Completed runs' *results* are unaffected — the bug risks a hang, not corrupted weights or rewards.

### Verification

`py_compile` clean; the idempotent-borrow algorithm is stress-tested — same-session storm → exactly one
borrow; exhaustion + retry → the retry waits, single borrow; random create/step/close churn → the pool
count is conserved with zero leak; lock order `_create_lock → sess.lock` never nests (no deadlock).
