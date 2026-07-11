# Bug fixes

A running log of notable correctness / robustness fixes to the FedAgent verl-0.8 overlay, with enough
mechanism to understand *why* each was wrong and how it was fixed.

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
