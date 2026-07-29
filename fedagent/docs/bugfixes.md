# Bug fixes

A running log of notable correctness / robustness fixes to the FedAgent verl-0.8 overlay, with enough
mechanism to understand *why* each was wrong and how it was fixed.

> Deliberate changes that were **not** defects (default flips, protocol revisions, regenerated
> shipped assets, layout moves) live in [revision.md](./revision.md).

---

## 2026-07-29: cross_round weight-sync OOM — `fit()` exits with vLLM AWAKE, so the client-end eval's `update_weights` gathers FSDP `full_tensor()` against the full `gpu_memory_utilization` pool

- **Files:** `fedagent/fed/persistent_task_runner.py` (post-`fit()` re-sleep + corrected
  `_worker_validate` comment); `fedagent/fed/persistent_patch.py` (`_reset_engine` FedProx-anchor
  drop reordered — completes the 2026-07-22 release-before-rebuild fix below).
- **Severity:** blocker for every `cross_round: true` run with `client_end_eval` (all
  `paper_accelerated` configs) — the process OOMs at a headroom-dependent round (crash + lost
  round; resume from the last aggregated round is clean, numbers are NOT corrupted).
- **Found while:** running the FedProx side of the ENV-heterogeneity mitigation ablation
  (does the proximal term recover the rank_wrapper degradation vs FedAvg?) — the first
  production runs combining the accelerated `cross_round` recipe with FedProx. Two cells
  co-tenant on one 8×H100 node, Qwen2.5-1.5B-Instruct, WebShop 15-turn:
  - **GRPO, GPUs 0-3** (run `fed_webshop_grpo_envhet_rankwrapper_N4_fedprox_qwen1.5b`):
    `config/paper_accelerated/env_heterogeneity/rank_wrapper/fed_webshop_grpo_…_p-rank_wrapper_N-4.yaml`
    + `--fedprox-mu 0.01` (CLI override — no committed fedprox variant exists, and 0.01 is
    the fork's paper default, NOT the 0.1 the example configs show), gmu 0.6. The dying
    process had resumed at round 3; it completed rounds 3-10 and OOM'd in round 11
    (rounds 3-10 aggregated clean; `round_11` quarantined to `_stale_rounds/` by resume).
  - **PPO twin, GPUs 4-7** (run `fed_webshop_ppo_envhet_rankwrapper_N4_fedprox_qwen1.5b`):
    the `rank_wrapper_ppo/` twin + the same mu, gmu 0.5. Its process died in round 8 with
    the identical signature at the identical call site.
  The FedAvg twins had not run yet — which is why "both dead runs are FedProx arms" briefly
  looked causal (see How-it-was-found §6).
- **Crash signature:** `torch.OutOfMemoryError: Tried to allocate 446.00 MiB` inside
  `actor_rollout_ref_update_weights()` → FSDP `full_tensor()`; the GRPO log carries exactly
  34 `Executor is not sleeping` warnings before the crash (the PPO log shows the same
  warning class; its count was not separately tallied). 446.00 MiB is the bf16 cast of
  `embed_tokens.weight` (151936×1536×2 B = 445.125 MiB, rounded to the 2 MiB allocator
  granule) — the sync's FIRST tensor: the fp32 all-gather (892 MiB) succeeded, the cast
  did not.

### Mechanism

verl's own loop keeps a strict dance: `sleep_replicas()` right after generation
(`ray_trainer.py:1471`, level-2 sleep = weights AND kv unmapped) → train → `update_weights`
(`:1675`, resume weights → push → resume kv). So every verl-internal sync gathers with
~`gmu × VRAM` returned to the driver. But `fit()` **ends** on the `:1675` sync — it returns
with the rollout fully AWAKE. The client-end eval (`_worker_validate`, called immediately
after `fit()`) then runs `cm.update_weights(...)` assuming verl's documented precondition
"rollout should be in sleep mode" (`engine_workers.py:672`) — which nothing enforces. On an
awake engine the two `resume()` wakes are no-ops (vLLM `executor_base.py:192-194` warns
`Executor is not sleeping` and returns — nothing was released, so nothing resumes) and the
gather must fit in what the resident 47.5 GB (gmu 0.6) pool leaves over. The old
`_worker_validate` comment claimed the engine "is asleep here (… after the previous fit()'s
last sleep_replicas)" — false: fit's last sleep is at `:1471`, BEFORE its last sync.

The warning count is the fingerprint: 2 no-op wakes per awake sync × (fits-in-process + 1).
GRPO: resumed at round 3 → 8 completed rounds × 2 clients + 1 = 17 awake syncs = 34
warnings, exactly the logged count. The PPO twin died at the same call site after a similar
round count — the same formula, different only in headroom. The fingerprint also proves the
engines' sleep mode was ON and functioning: with `enable_sleep_mode=False`, `sleep()`
"succeeds" vacuously (`is_sleeping=True` with nothing unmapped) and `wake_up` never warns.

Two aggravators made the marginal budget tip earlier on the FedProx arms (but the failure
family predates FedProx — see the 2026-07-22 entry's pre-FedProx r14/r27 provenance):

1. The FedProx anchor `_fedprox_w_t` (a full fp32 param-shard clone, ~1.44 GB/GPU for 1.5B
   @ 4-way FULL_SHARD) was `del`-ed AFTER `initialize()` and AFTER the 07-22 fix's
   `empty_cache()` — i.e. it was left out of the release-before-rebuild set that fix
   introduced, so on FedProx runs the anchor stayed co-resident with every rebuild and its
   (mid-training, activation-interleaved) blocks were never returned.
2. Ordinary reserved-memory fragmentation growth across rounds (dynamic batches), which the
   awake-pool ceiling turns into a crash instead of a nuisance.

Eval integrity note: the server-side `wake_up` resets the prefix cache UNCONDITIONALLY after
`engine.wake_up` (`vllm_async_server.py:611-616`), so the historical awake syncs never served
stale KV — eval numbers from the crashed runs are trustworthy.

### How it was found

An elimination chain, every step checkable from the crash logs + source alone (no GPU repro):

1. **A static memory audit refuted "model too big".** Summing every resident consumer at the
   sync instant (vLLM pool 47.5 GiB at gmu 0.6; fp32 actor shard 1.44 + Adam 2.88; FedProx
   anchor 1.44; CUDA ctx ~1.5; transients: the 2 GiB `update_weights_bucket_megabytes`
   staging buffer + 892 MiB fp32 gather + 446 MiB bf16 cast) gives ≈60 of 79.11 GiB — ~19 GiB
   nominal spare. So the crash required a growth term or a spike term; static arithmetic
   could not produce it. (Same conclusion for PPO at gmu 0.5 with the critic engine added.)
2. **The failing size named the tensor.** 446.00 MiB is the bf16 `embed_tokens.weight`
   exactly (445.125 MiB + 2 MiB large-block rounding), and its fp32 gather (892 MiB) had
   already succeeded — so the sync died on its FIRST tensor. Whatever consumed the VRAM was
   fully resident BEFORE the sync began, ruling out mid-sync accumulation.
3. **The warning text located the state bug.** `Executor is not sleeping` has exactly one
   emitter: vLLM `wake_up()` on an already-awake executor (`executor_base.py:192-194`), a
   warn-and-return no-op — meaning nothing had been released in the first place. Those syncs
   ran against a never-slept pool: the spike term of step 1.
4. **The count matched a per-fit counter, not a leak threshold.** 34 = 2 × 17 = 2 ×
   (fits-in-process + 1) exactly, on a process whose fit count (8 rounds × 2 clients + 1) is
   known from the resume history. A fragmentation/leak threshold has no reason to land on
   that counter value; a deterministic once-per-fit event must. The PPO twin dying at the
   same call site at a similar round count — despite a different gmu and an extra critic
   engine — fits the same formula.
5. **The verl state machine confirmed the event.** `fit()` sleeps at `ray_trainer.py:1471`
   but ENDS on the `:1675` sync → always exits awake; the only out-of-fit sync is the
   client-end eval's. Its comment asserting "asleep here (after the previous fit()'s last
   sleep_replicas)" was checked against the verl source and found false.
6. **FedProx was excluded as root cause by prior provenance, not by contrast runs.** Both
   dead runs were FedProx arms, but the FedAvg twins had not run yet (no controlled
   contrast existed); the 2026-07-22 entry's pre-FedProx r14/r27 crash supplied the
   discriminating evidence. The step-3 fingerprint also killed the rival hypothesis that the
   PPO run had sleep mode disabled (via an exported `expandable_segments:True`): the warning
   can only be emitted when sleep genuinely works — with `enable_sleep_mode=False`, `sleep()`
   vacuously sets `is_sleeping=True` and `wake_up` never warns.

Debugging lesson: the load-bearing claim was a prose comment, and the crash site was one
layer away from the bug — the eval code didn't break, `fit()`'s exit state did. When a
precondition exists only in prose ("rollout should be in sleep mode",
`engine_workers.py:672`; our own "is asleep here"), verify it against the code that produces
the state, not the site that consumes it.

### Fix

1. **Producer-side re-sleep** (`persistent_task_runner.py`, run loop): `cm.sleep_replicas()`
   immediately after `trainer.fit()` returns. fit() always exits awake, so this is never a
   no-op (zero log noise, and `Executor is not sleeping` stays meaningful as a regression
   alarm); sleep→update is verl's own per-step `:1471`→`:1675` cycle, so numerics are
   untouched. Recovers the full pool (~47.5 GB at gmu 0.6) at the exact gather moment.
2. **Anchor drop reordered** (`persistent_patch.py` `_reset_engine`): `del _fedprox_w_t`
   moved ABOVE the null-out loop, upstream of `gc.collect()` + `empty_cache()`.
   Semantics-neutral: the wrapper re-snapshots lazily at the next client's first
   `optimizer_step` (`fedprox.py`: `getattr(self, "_fedprox_w_t", None) is None`); the ref
   engine never steps and the critic passes through unanchored, so `hasattr` is a no-op there.
3. The false `_worker_validate` comment corrected in place.

Do **not** mitigate by lowering `gpu_memory_utilization` (the 07-22 provenance shows it only
defers: r14 → r27) and do **not** enable `rollout.layered_summon` (a no-op for full finetune —
it only gates `collect_lora_params` — and it silently downgrades sleep to level 1).
Post-fix signatures: zero `Executor is not sleeping` lines; two `It took … to fall asleep`
lines per client; per-round training-side `memory_reserved` FLAT — the latter closes the
2026-07-22 entry's pending live validation.

---

## 2026-07-28: concat glue anchor silently no-ops on reasoning models — the history rewrite deletes the text the anchor searches for (back-port from the AccelAgent audit)

- **Files:** `fedagent/agent_loops/_concat_glue.py` (identical copy in both repos);
  +5 regression tests in `tests/test_concat_glue.py`.
- **Severity:** medium-latent — every shipped paper config pins Qwen2.5, whose template
  renders history verbatim, so NO executed run is affected; the defect fires on a
  reasoning-model override (Qwen3-family) in concat mode.

### Mechanism

Qwen3-family chat templates REWRITE a completed assistant turn when re-rendering it as
history: the Jinja deletes the whole `<think>…</think>` block. The glue anchor searched
`rt_next` for the RAW sampled action text, which therefore never occurs there — so on
every VALID action (the prompts mandate `<think>`) the anchor returned None and the loop
fell back to the legacy blind slice, the exact mis-slice the 2026-07-12 fix exists to
eliminate. Worse than the original bug: with a CoT longer than the observation,
`new_ids` is SHORTER than `cur_ids` and the fallback glue is EMPTY — the observation and
turn boundary vanish from the emitted training sample, all `response_mask==1`, silently.
The rollout context itself stays correct (`cur_ids = new_ids`), so the harm is a corrupt
emitted sample (train/inference context mismatch), not a wrong loss target.

### Fix

When the full text misses, re-anchor on the template-KEPT form: the `</think>`-strip
candidate first (the known Qwen3 rewrite; the kept part is a suffix of the raw action),
then each newline-delimited tail of the action (template rewrites drop a prefix at a
block boundary), with an 8-char floor below which the legacy fallback applies exactly as
before. Same-day sibling fix: the sitecustomize gates now print the root-cause traceback
inside their `except` blocks — `_die`'s own `print_exc()` always printed "NoneType:
None" because the exception context is gone by the time the gate calls it.

---

## 2026-07-28: `/close` failures leaked pooled envs — a swallowed TransportError never returned the server-side session (back-port from AccelAgent)

- **Files:** `fedagent/envs/webshop/webshop_env.py`, `fedagent/envs/alfworld/alfworld_env.py`
  (`close()` in both).
- **Severity:** low-medium — needs a close-storm transport failure to trigger, but the
  consequence is the documented pool-starvation class, and it is invisible when it happens.
- **Provenance:** found by the AccelAgent tree-diff cross-audit (the sibling framework had
  hardened its identical client; this tree had not). Reviewed alongside two more
  reverse-direction candidates — see the note below.

### Mechanism

Both remote-env clients closed with a BARE `self._client.post("/close", ...)` inside a
blanket `except Exception: pass`. At batch end every episode closes at once (a close-storm
on one uvicorn boundary); a `TransportError` there was silently swallowed, with two
consequences: the pooled server-side env was **never returned** (`/close` is what puts the
session's env back in the pool), and `aclose()` — inside the same `try` — was skipped, so
the client socket leaked too. With block-on-`/create` semantics, every leaked session
permanently shrinks the pool; enough of them and rollout hangs (the exact starvation class
the 2026-07-11 double-borrow fix closed on the create side).

### Fix

`/close` goes through the same `_post(..., retry=True)` helper as `/create`/`/reset`/`/step`,
wrapped in `try/finally` so `aclose()` is unconditional. The retry is safe by inspection of
both servers: `/close` pops the session with a `None` default and returns `ok` even for a
lost-response resend (idempotent).

### Reverse-port review (the other two AccelAgent-ahead spots)

- **`convert_product_file_format.py` output dirs** — APPLIED: `search_engine/resources*`
  is untracked (built locally), so on a fresh clone the vendored indexing script died with
  `FileNotFoundError`; it now `os.makedirs` each output dir first.
- **Lazy matplotlib/seaborn in the ALFWorld engine's `partition_strategy.py`** — REVIEWED,
  NOT APPLIED: only the ALFWorld *service* env imports that module, and
  `alfworld_requirements.txt` pins `matplotlib==3.10.5` + `seaborn==0.13.2`, so the hard
  import cannot fail on a correctly-installed service env; restructuring imports across a
  3,400-line vendored file for zero live benefit loses to the byte-preservation norm.
  (AccelAgent needed it because its slimmer service envs may omit plotting deps.)

---

## 2026-07-28: the port band never banded the real draw — the patched verl function is DEAD code in 0.8 (and the pid salt collapsed on small bands)

- **Files:** `fedagent/port_band.py` (patch target re-pointed; salt headroom), `sitecustomize.py`
  (unchanged arming path). Offline regression: `tests/test_port_band.py`.
- **Severity:** medium — the collision class the band exists to kill was only *half* covered;
  the log-signature retry (independent layer) still caught survivors.

### Mechanism

The 2026-07-23 fix pinned `WorkerHelper._get_free_port` (single_controller/base/worker.py) to
band probing. In verl 0.8 that function is **dead**: its only consumer,
`get_available_master_addr_port`, has no caller. The real FSDP `MASTER_PORT` draw is the
`get_master_addr_port` Ray task (single_controller/ray/base.py:90), reached through
`RayWorkerGroup._get_master_addr_port` — whose `master_port_range` parameter nothing in the
stock stack ever passes, so every draw stayed a bare `bind(("", 0))` in the ephemeral range.
Net effect: vLLM's half of the band worked (`VLLM_PORT` pid-salting), verl's half banded
*nothing* — verl master-port collisions remained possible exactly as before the "fix".

Second defect, same module: `assign_vllm_port` computed its salt space as
`max(1, half - 8)`. At `port_band_stride <= 18` (`half <= 9`) that collapses to 1, giving
every process the SAME starting port — the one-shared-start race the salt exists to prevent.
The shipped default (stride 100 → half 50) was unaffected.

### Fix

`enable_port_band` now patches `RayWorkerGroup._get_master_addr_port` to default
`master_port_range=[start, start + span//2]` when the caller passes none — verl's own remote
probing loop then scans the band's lower half natively on the target node. The deferred
import hook re-targets `verl.single_controller.ray.base` accordingly.
`WorkerHelper._get_free_port` stays rebound as belt-and-braces for any future caller. The
salt headroom becomes `min(8, half // 4)`: identical layout at the default stride, a real
spread for small bands.

---

## 2026-07-28: five sitecustomize gates were fail-OPEN — a failed arm printed one line and ran the forbidden configuration anyway

- **File:** `sitecustomize.py` (all 5 armed blocks: FedProx, persistent worker, critic-loss
  parity, fp32 merge, port band).
- **Severity:** high *if* an arm ever failed (none is known to have); the failure mode was
  "experiment silently runs the configuration the gate exists to refuse".

### Mechanism

Each block ended in a bare `raise` on arming failure, believing that fails closed. It does
not: CPython's `site.execsitecustomize` wraps the whole module in `except Exception` — it
prints ONE stderr line ("Error in sitecustomize…") and the interpreter **continues with exit
code 0**. A FedProx run whose patch failed to arm would train as plain FedAvg, a
`FEDAGENT_CRITIC_LOSS_MODE=legacy_exact` PPO client would train on the unnormalized stock
critic loss, a banded run would fall back to the ephemeral port lottery — all while every
launcher-level check saw a healthy process.

### Fix

`_die(msg)`: print the reason + traceback to stderr, then `os._exit(1)` — the only reliable
stop at interpreter startup. Every block wraps its import+arm in `try/except` and routes ANY
failure (import error included) through `_die`. The env-service no-op paths (verl absent)
are unchanged.

---

## 2026-07-28: `--phase verify` could not fail — PASS/FAIL was print-only and the structural check asserted nothing

- **File:** `fedagent/fed/aggregate_fedavg_fsdp.py`.
- **Severity:** low-medium — verify is an operator tool (run_fed does not call it per round),
  but its one job is to be trustworthy under scripting.

### Mechanism

`verify()` printed `[verify] PASS`/`FAIL` on rank 0 and returned; the process exited 0
either way, so `torchrun … --phase verify && next_step` gated nothing. The docstring also
promised a structural round-trip check ("they round-trip as ShardedTensors"), but the code
only *counted* ShardedTensors into a print — a save/load that silently degraded every shard
to a plain tensor (which verl's strict load would later refuse) still printed "round-trip OK".

### Fix

The structural check is real (every key must come back in the same wrapper class client 0
carries), the verdict is all-reduced so every rank agrees, and FAIL raises `SystemExit(1)`
after the final barrier — torchrun propagates the nonzero exit.

---

## 2026-07-28: config-validation raises leaked the val service — 13 error paths ran serviceless cleanup

- **File:** `fedagent/fed/run_fed.py` (val-service start moved below the validation block).
- **Severity:** low — leak only on misconfigured launches, but those are exactly the runs a
  user relaunches immediately, and the orphan holds the val port.

### Mechanism

`start_val_service` ran early in `main()`, ~150 lines and 13 config-validation `raise`s
before the round loop's `try:`/`finally: stop_services(...)`. Any validation failure
(bad `eval_mode` combo, missing critic path, …) propagated OUT of the service's lifetime
scope: the run died, the WebShop/ALFWorld val service kept running, and the relaunch —
typically seconds later, after fixing the YAML — found the val port occupied.

### Fix

The service starts immediately before the round-loop `try:` — after every validation raise —
so from the first moment a service exists, the `finally` that stops it is armed.

---

## 2026-07-28: concat mode penalized VALIDATION rows — the invalid-action penalty leaked into val `reward_score`

- **Files:** `fedagent/agent_loops/gym_text_agent_loop.py`,
  `fedagent/data/agentic_dataset.py` (`is_validation` row column),
  `fedagent/config/envs/{webshop_15_val,alfworld_val}.yaml` (`validate: true` marker).
  Offline regression: `tests/test_concat_val_penalty.py`.
- **Severity:** low — concat is the non-default mode, and the paper metrics
  (`traj_success`, `task_score`) ride `reward_extra_info`, which was never penalized; only
  concat-mode val `reward_mean` was shifted by `-0.1 × invalid actions`.

### Mechanism

The legacy stack applied `apply_invalid_action_penalty` in the TRAIN reward path only
(`val_reward_fn` = the unpenalized episode return), and the windowed loop preserves that via
the `validate` flag its manager passes natively. The concat loop runs under stock verl,
which passes the dataset row's columns — but **not** `trajectory["validate"]` — into
`AgentLoop.run()`, so it had no way to know it was scoring a val row and subtracted the
penalty everywhere.

### Fix

The val marker rides the row: a spec with `validate: true` (set in both shared val specs)
makes `AgenticDataset` emit `is_validation=True` on every row, which reaches `run()` as a
kwarg; val episodes skip the penalty. Deliberate scoping: client in-job validation reuses
the TRAIN spec (no marker) and keeps train-reward semantics; a custom val spec without the
marker keeps the old penalized behavior.

**Known concat limitation (documented, not changed):** the concat loop re-renders and
re-tokenizes the FULL chat history every turn (`_tokenize_chat(messages)` on the whole
list), an O(turns²) tokenization cost per episode. Harmless at WebShop/ALFWorld lengths
and confined to the non-default mode; noted here so nobody rediscovers it as a perf bug.

---

## 2026-07-28: ALFWorld `env_disjoint` knobs were never forwarded — every run silently used `env_div=0.7`, `fallback='skip'`

- **Files:** `fedagent/envs/alfworld/service/server.py` (ENV_DIV / ALFWORLD_FALLBACK read +
  forwarded), `fedagent/fed/run_fed.py` (service env vars + `alfworld_fallback` DEFAULTS).
- **Severity:** medium *for the arm* — an env_disjoint config setting `env_div: 0.3` trained
  at 0.7 with no error; the paper's executed runs used the default, so no shipped result is
  affected.

### Mechanism

The partition function (`_env_disjoint_partition_alfworld`) takes `env_div` and `fallback`,
and the engine passes `**partition_kwargs` straight through (`alfred_tw_env.py` even names
the viz file after both) — but the service's `_partition_kwargs()` returned `{}` for
env_disjoint ("takes no extra kwargs"), and run_fed never exported the env vars. The whole
plumbing existed except the middle segment.

### Fix

run_fed exports `ENV_DIV` (the shared `env_div` knob) + `ALFWORLD_FALLBACK` (new
`alfworld_fallback` knob, default `skip` = the code default) to the ALFWorld services; the
service forwards `{env_div, fallback}` for `env_disjoint`. The partition fn still validates
the fallback vocabulary.

---

## 2026-07-28: ALFWorld hardness label-miss warning was gated on `show_progress` — silent in every real run

- **File:** `fedagent/envs/alfworld/engine/agent_system/environments/partition_strategy.py`.
- **Severity:** low — the warning is diagnostic; the floor-to-hard behavior it reports was
  already correct and counted.

### Mechanism

`hardness_partition_alfworld`'s miss summary ("N games had NO task_id match … floored to
HARD") required `show_progress and client_id == 0`. `show_progress` defaults False and no
service sets it, so the guard built for the "labels don't match this pool" failure mode
could never speak in production — and even had it fired, each client's service logs to its
own file, so the `client_id == 0` gate would have hidden it from every other client's log.

### Fix

The warning is ungated (it fires on `n_label_miss` alone). Progress chat stays behind
`show_progress`.

---

## 2026-07-28: windowed divisor padding always cloned the FIRST rows — every step over-weighted episode 0's turns

- **File:** `fedagent/agent_loops/windowed_manager.py` (`_adjust_to_divisor`).
  Regression: `tests/test_windowed_union_tag.py`.
- **Severity:** low — a few duplicated rows per batch; but the duplication was *systematic*
  (same batch position every step), not noise.

### Mechanism

The per-turn batch is episode-major (episode 0's turns first). Padding to the trainer's
divisor duplicated rows `arange(to_add) % bs` — i.e. always the HEAD of the batch — so the
first episode's opening turns were over-weighted in every single optimizer epoch, a small
but persistent directional bias toward whatever goals sort first.

### Fix

Dup indices are evenly spaced over the batch (`linspace(0, bs-1, to_add)`): still
deterministic (no RNG, resume-safe), no positional bias. Executed paper runs used the head
slice; the reweight is a few rows of one batch, so curves remain comparable — noted in
[revision.md](./revision.md).

---

## 2026-07-28: windowed turn samples carried empty `metrics` — rollout timing reported zero

- **File:** `fedagent/agent_loops/windowed_agent_loop.py`.
- **Severity:** cosmetic/observability — training math never reads these fields.

### Mechanism

Each per-turn `AgentLoopOutput` was built with `metrics={}`, which pydantic fills with
`AgentLoopMetrics` defaults (`generate_sequences=0.0`) — so verl's per-request timing
aggregation reported zero generate time for every windowed rollout, while the concat loop
(which wraps its generate calls in `simple_timer`) reported real numbers.

### Fix

Each turn times its own generate call (`simple_timer("generate_sequences", …)`) and carries
that on its row; summed over an episode's rows this equals the concat loop's single summed
row, so the two modes' timing is comparable.

---

## 2026-07-28: ALFWorld per-task-type tool measured a different quantity than the paper tables — could not reproduce them even in principle

- **Files:** `tools/eval_alfworld_by_tasktype.py` (single-pass mode added, now the default),
  `fedagent/fed/eval_dumps.py` (`summarize_val_dump_by_tasktype`, `read_latest_rows`);
  offline regression `tests/test_tasktype_breakdown.py`.
- **Severity:** high for release reproducibility; training validation, curves, and the
  tables' All column were never affected.

### Mechanism

The paper's per-type columns are a **partition of ONE 64-trial val pass** — the 2026-07
audit verified all 240 cells sit on the 1/n_t grids of the pass composition
(Pick 14, Look 8, Clean 12, Heat 9, Cool 11, Pick2 10) and the weighted pool equals the
All column on 40/40 rows. The tool (born in the verl-0.8 migration; the original repo
had no such tool) instead ran **7 independent passes**: with the per-type filter the pool
becomes that type's WHOLE split (35/13/27/16/25/24 games) and the pass still runs 64
episodes, so games repeat via `gfs[seed % len(gfs)]` — every cell it produced was k/64 on
a repetition-inflated pool, and its "All" was an eighth independent pass. Values like the
tables' 14.3 (=2/14) or 22.2 (=2/9) are not multiples of 1/64, so the tool literally could
not output them.

### Fix

The agent loop has always tagged every val row with `task_type` (derived from the episode
gamefile in `alfworld_env.py`); nothing aggregated it. `--mode single` (new default) runs
ONE unfiltered pass and groups the dump rows by that tag — the exact table estimator, with
the pooling identity by construction. The 7-pass estimator survives as
`--mode per-type-passes` (higher per-type sample size, explicitly labeled as a different
quantity). Note: at HEAD the val-64 composition is (15,4,14,7,10,14) after the 2026-07-26
machine-independent ordering fix, so fresh runs' per-type denominators differ from the
paper's (14,8,12,9,11,10).

---

## 2026-07-28: catalog_split protected the products of the WRONG goals — target floor derived from the PRE-shuffle goal order

- **Files:** `fedagent/envs/webshop/service/server.py` (strategy branch + `_apply_deferred_catalog`
  in `_lifespan`), `fedagent/hetero/webshop_catalog_split.py` (`_assemble_catalog` factored out,
  new runtime API `catalog_from_target_asins`; `catalog_split_for_client` demoted to legacy
  reference); regression `tests/test_catalog_split_runtime.py` (5 tests, incl. a
  bug-vs-fix side-by-side on a shuffled goal order).
- **Severity:** high for the catalog_split arm (paper Variant 1); `task_disjoint` unaffected
  (full catalog). Inherited from verl-agent — the paper's Catalog-Split runs had it too
  (the 2026-07 audit measured ~29% of a client's served goals losing their target product
  from its catalog).

### Mechanism

`_generate_goal_asins_for_partition` mimics the goal list in **generation order**, but
SimServer **shuffles** `self.goals` inside its seed-42 construction window before serving.
The service sliced goal indices against the served (shuffled) order — correct, pure index
arithmetic — but then derived `client_target_asins = {goal_asins[i] for i in idxs}` from the
**pre-shuffle** list: the catalog's "target floor" protected the products of the wrong ~100
goals, so a client's actual targets could be dropped by the keep_ratio filter, making those
goals unwinnable at full reward. The safety check at the end
(`client_target_asins - set(catalog_asins)`) was tautological — the catalog is built as
`targets | distractors`, so it could never fire.

### Fix

Slice AND catalog are now both computed at runtime: the slice via `uniform_for_client`
(line-identical arithmetic), and the catalog in `_lifespan` from
`{env.server.goals[i]['asin'] for i in CLIENT_GOAL_IDXS}` via `catalog_from_target_asins`
(the verbatim steps 4–7, which are ASIN-keyed and hence derivation-agnostic). The filter is
retrofitted onto the warmed pool by swapping `all_products`/`product_item_dict` — equivalent
to the constructor filter, which SimServer consumes only at request time. A real end-to-end
tripwire replaces the tautology: every served goal in the slice must keep its product
reachable, or the service refuses to start.

---

## 2026-07-28: load_products guards — two latent lookalike traps (fail-fast, zero behavior change today)

- **Files:** `fedagent/envs/webshop/engine/webshop/web_agent_site/engine/engine.py`
  (`load_products` extra_products block; comment corrected — the append is BEFORE the
  `products[:num_products]` truncation, not after).
- **Severity:** latent only; neither combination is exercised by any shipped config.

1. `extra_products` + `num_products` ⇒ raise. The lookalike append precedes the
   `[:num_products]` cut, so a num_products cap would silently drop every lookalike.
2. `extra_products` + `human_goals` ⇒ raise. Range-priced lookalikes consume
   `generate_product_prices`' global `random.uniform` draws inside the seeded construction
   window (L_color: 79 draws, L_size: 91, price/price+color: 0). The synthetic-goal path
   survives because `get_synthetic_goals` re-seeds before drawing; `get_human_goals` does
   NOT — per-arm draw counts would desync goal price caps and the goal shuffle across
   federated clients (cross-client goal-index protocol breakage + baseline-val
   contamination).

---

## 2026-07-27: Hardness `F_i` fill was a pure function of the fill SIZE — clients with equal easy quotas trained on byte-identical hard goals

- **Files:** `fedagent/hetero/webshop_hardness.py` (step 2 of the partition body) and the
  same block ×2 in
  `fedagent/envs/alfworld/engine/agent_system/environments/partition_strategy.py`
  (`hardness_partition`, `hardness_partition_alfworld`); offline regression
  `tests/test_hardness_fill.py`.
- **Severity:** medium, hardness arms only. The paper's formal quantities are all
  UNAFFECTED — `rho_i = |Y_i|/L` is fixed upstream by the Beta sizing, so `Delta^2_hard`,
  `rho_bar`, `n_bar` and the (D1)–(D3) control/invariance laws are identical before and
  after. What was wrong is the **cross-client joint**: which hard goals fill the shards.

### Mechanism

Every client computes the partition locally from the shared seed
(`np.random.default_rng(42)`). Steps 1's consumers (`generate_client_sizes`,
`assign_with_overlap`) consume the stream identically in every invocation — that is the
invariant that makes local recomputation consistent. Step 2 then drew ONLY the invoking
client's fill:

```python
filler = rng.choice(low_success_data, size=L - |Y_i|, replace=False)
```

Identical stream state + identical pool ⇒ `F_i` is a pure function of the fill size
`L-|Y_i|`. Clients with equal easy quotas therefore trained on **byte-identical** hard
sets. At the paper scale (100 clients, L=100, 6410 goals) that is 73–97% of clients
across the ξ′ sweep, and the effect is *worst at the near-uniform end*: at ξ′=256 the
quotas concentrate on 16 distinct values, so ~5000 hard slots were filled by only **16
independent draws** (vs 48 at ξ′=1) — the near-uniform baseline saw the fewest distinct
hard goals of the whole sweep, an unmodeled diversity gradient riding along the ξ′ axis.

The paper's Algorithm HardnessPartition writes the fill as
`for i: F_i <- SampleWithoutReplacement(U, L-|Y_i|)` — in any faithful execution the
draws advance one stream, so equal-quota clients share only `E[nu^2/|U|]` (~0.5 goals),
not all ~51. Historical note: the ORIGINAL verl-agent body had this artifact on **both**
stages (its easy picks were the same client-independent `rng.choice`); the 2026-07-19 fix
moved the easy side onto the global `assign_with_overlap` machinery and left the hard
side as-was, so the overlay was already strictly closer to the pseudocode than the code
that produced the original numbers — this entry closes the remaining half.

### Fix

Step 2 now **replays the full fill loop from the shared stream**: every invocation draws
`F_0..F_{N-1}` in client order from the FULL unsuccessful pool (the pseudocode never
consumes `U`) and keeps only its own draw. All clients advance one global sequence — the
same single-execution semantics as the pseudocode — and the per-client sizes are derived
from `easy_sets` (identical in every invocation), never from the invoking client's
filtered pools, so the stream is consumed identically no matter who executes it.
Paper-scale check after the fix: **100/100 distinct fills at ξ′=256** (was 16), 81/100 at
ξ′=1 (the ~20 clients whose easy quota already fills L draw nothing — correctly).

> **Operational note:** the fix CHANGES every hardness shard. Do not `resume` a hardness
> run across this commit (rounds before/after would train on different partitions);
> finish in-flight runs on the old code or restart them fresh. Non-hardness arms are
> byte-identical.

### Verification

- `tests/test_hardness_fill.py` (4, offline; parametrized over BOTH shipped copies of the
  body): equal-quota clients get distinct fills with <50% overlap; all C fills pairwise
  distinct and their union covers >1.5× any single fill; the realized easy counts still
  match the replicated Beta-sizing + `assign_with_overlap` allocation exactly (marginal
  invariance); recomputation is deterministic.
- Paper-scale probe (6410 goals / 100 clients / L=100): distinct fills 16→100 (ξ′=256),
  48→81 (ξ′=1, rest empty-by-quota); equal-quota client counts unchanged (97 / 73) —
  confirming the quota marginals were untouched.

## 2026-07-27: nothing pinned WHICH ALFWorld games exist — the task set was a property of the machine's preprocessing state

- **Files:** new `fedagent/envs/alfworld/game_manifest.py` + `tools/gen_alfworld_manifest.py` +
  the shipped `data/alfworld_games/{train,eval_in_distribution,eval_out_of_distribution}.json`;
  consumed in `.../alfworld/agents/environment/alfred_tw_env.py` (`collect_game_files`) and
  bridged by `run_fed.alfworld_game_list_env`. Tests:
  `tests/test_alfworld_game_manifest.py`, `tests/test_alfworld_game_order.py`.
- **Severity:** reproducibility — the other half of the 2026-07-26 ordering fix below. That one
  made the ORDER a pure function of the collected set; this one pins the SET.

### The bug

`collect_game_files` keeps a trial when it has a `traj_data.json`, a `game.tw-pddl`, and that
file says `solvable: true`. **Neither the game file nor the flag ships with the raw ALFWorld
trajectories** — both are produced by ALFWorld's preprocessing step. So the collected set is a
function of how completely that step ran on the machine, and on the machine these manifests
were cut from, **47 of the 187 eligible `valid_seen` trials (25%) have no game file at all**.

Everything downstream is positional (the seeded shuffle, `slice_games_for_client`'s index
slices, `num_train_games`/`start_idx`, `games[seed]` under `ALFWORLD_SEED_IS_INDEX`), so a
machine that preprocessed further collects a larger list, every index shifts, and both the
client shards and the validation set become different games — with no error and no warning.
Sorting the walk (below) does not help: it guarantees the same order *given the same set*.

### The fix

A checked-in manifest per split — a sorted list of split-relative `game.tw-pddl` paths plus a
sha256 of that list — which `collect_game_files` reads **instead of** walking:

| | before | now |
|---|---|---|
| where the list comes from | `os.walk` of `$ALFWORLD_DATA` | `data/alfworld_games/<split>.json` |
| a listed game missing on disk | (not detectable) | **RuntimeError** naming the count and examples (`ALFWORLD_MANIFEST_STRICT=0` warns and drops instead — that renumbers, so triage only) |
| extra games on disk | silently included, shifting every index | ignored by construction |
| task-type subsets (the eval breakdown) | re-walked with a filter | filtered from the one manifest by path, same relative order |
| cost | walk + ~2 JSON reads per trial | one file read (also subsumes the optional `ALFWORLD_MANIFEST_CACHE` speed-up) |

The engine falls back to the historical walk when no manifest is found or
`ALFWORLD_GAME_MANIFEST=none`, so the vendored engine still works standalone.

### Verification

- The shipped manifests are **byte-identical to what the fixed walk produces** on the machine
  they were cut from: `tools/gen_alfworld_manifest.py --check` reports OK for all three splits
  (sha256 `ffa58e52` / `8b129cdf` / `14894587`). So this change is a no-op *here* and a
  guarantee elsewhere.
- Counts land where they should: 3553 train / 140 `valid_seen` / 134 `valid_unseen`. The 3553
  matches the game count the ALFWorld hardness labelling was built on, i.e. the shipped labels
  and the shipped manifest describe the same task set.
- `fedagent/envs/alfworld/__init__.py` now re-exports its httpx client **lazily** (PEP 562), so
  the manifest module stays importable under a bare interpreter — the generator has to run
  wherever the DATA is, which is not necessarily a machine with the trainer env installed.

---

## 2026-07-26: `eval_mode=worker` re-drew the validation set every round — the worker-eval dataset inherited the *training* client's `FEDAGENT_BASE_SEED`

- **Files:** `fedagent/fed/persistent_task_runner.py` (`unseeded_eval_data`, applied at the
  worker-eval dataset build); mechanism spans `fedagent/data/agentic_dataset.py:58,105` (row
  seeds) and `fedagent/envs/alfworld/service/server.py` (seed → game). Regression:
  `tests/test_worker_eval_val_seed.py`.
- **Severity:** silent measurement corruption of the paper's red line on ALFWorld. Training,
  aggregation and checkpoints are unaffected — only *which tasks the curve was measured on*.
  WebShop escaped by arithmetic (below), so no WebShop number changes.
- **Exposure:** `eval_mode: worker` **and** a per-round worker process (`persistent: true` +
  `cross_round: false`). All 176 `config/paper_accelerated/` configs set `cross_round: true`, so
  the repo tree never triggered it; the run-local configs that switched to `cross_round: false`
  to dodge the (since-fixed) cross-round VRAM leak did. Latent since `3b8aa2d` (2026-06-28), the
  commit that introduced `eval_mode=worker`; live only from ~2026-07-24.

### The bug

`AgenticDataset` seeds every episode row from a **process-global** env var:

```python
"seed": base_seed * 100_000 + si * 1_000 + e * n + i    # agentic_dataset.py:105
base_seed = int(os.environ.get("FEDAGENT_BASE_SEED", 0))
```

`run_fed` sets `FEDAGENT_BASE_SEED = base_seed + round*100 + client` per client-round — a
*training* knob: it is what gives each client a fresh draw of its shard every round. The
persistent runner sets the same var (`persistent_task_runner.py:64`, and again per client in
`_reset_for_client`) **before** building datasets, and the worker-eval **validation** dataloader
was built inside that window:

```python
os.environ["FEDAGENT_BASE_SEED"] = str(plan[0]["seed"])   # the training client's seed
...
wval = create_rl_dataset(self._worker_eval_spec, ...)     # <- val rows seeded from it
```

The orchestrator's subprocess eval path (`run_fed._build_eval`) never sets the var, so the two
eval modes scored *different task sets*, and `config/envs/*_val.yaml`'s documented contract —
"the same fixed set every round" — held only on the subprocess path.

With one worker process per round, `plan[0]["seed"]` changes every round, so the val rows'
seeds changed every round.

### Why only ALFWorld showed it

| | seed → task | consequence |
|---|---|---|
| WebShop | `sess = seed % VAL_SIZE`, `VAL_SIZE=500` | `100_000 = 200 × 500`, so `base*100_000 ≡ 0 (mod 500)` → `sess = i` → **always `goals[0:n_envs]`**, same order, every round. Immune by arithmetic, not by design. |
| ALFWorld | `RandomState(seed).shuffle(gamefiles)[0]` | no modular structure to absorb the offset → a fresh pseudo-random draw **with replacement** over the whole split per process. |

The WebShop immunity is fragile: any `WEBSHOP_VAL_SIZE` that does not divide `100_000`
(e.g. 300) would have shown the identical failure there.

### Evidence (from `run_artifacts/alfworld_ppo_hardness_std1`, 48 rounds)

- 48 rounds → **48 different game sets**; union = 140 = the whole `valid_seen` split.
- Per round: 64 episodes covering **52.2 unique games on average** (range 46–59) with ~12
  repeats — the signature of sampling *with replacement*: 140·(1−(1−1/140)^64) = 51.5 expected.
- Within a round, the aggregated point and the client circles were measured on **different**
  sets (~19 of 53 shared — exactly the 140·p² = 19.0 expected of two independent draws).
- The decisive test: `round_k/eval` is written by the round-**k+1** worker (which evals round k's
  starting model at its `i == 0`), while `round_k/client_*/eval` is written by the round-k
  worker. So they should match *across* rounds, not within one — and
  `round_k/eval == round_{k+1}/client_*/eval` holds **48/48**, byte-for-byte in order.
- Cross-run confirmation: `alfworld_hardness_std1` (GRPO, `cross_round: true`, one process for
  the whole run) shows **one** ordered list across all 70 rounds and all 140 client evals — and
  that list equals the PPO run's `round_0` set **and only round 0**, i.e. exactly where the two
  runs' base seeds coincide (both `42 + 1·100 + 14 = 156`).
- Noise decomposition on the PPO run: of the round-to-round variance in val success,
  **28% is task-composition** and 46% is binomial (n=64) — i.e. ~3/4 of the visible wiggle was
  measurement, not model. The GRPO (fixed-set) run's observed sd / binomial sd = 1.03; the PPO
  run's = 1.13.

### The fix

Build the val dataset with the var cleared, and restore it for training:

```python
@contextlib.contextmanager
def unseeded_eval_data():
    saved = os.environ.pop("FEDAGENT_BASE_SEED", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["FEDAGENT_BASE_SEED"] = saved
```

Val rows are then seeded `0..n_envs-1` — identical to the subprocess eval path, identical every
round, and identical across arms. Training seeding is untouched (`_reset_for_client` re-sets the
var per client regardless).

### What it means for existing curves

Pre-fix ALFWorld runs were each scored on *some* fixed-per-process set drawn from
`base = 42 + start_round·100 + first_client`; post-fix runs score on seeds `0..63`. Both are
fixed, but they are **different games**, so absolute numbers shift by a constant (≈5 pp at the
measured 3.4 pp composition sd). Within-run trends and cross-arm comparisons *among* pre-fix
runs remain valid — client selection is deterministic (`base_seed=42`), so same-shaped runs drew
the same val set (verified: 47/48 common rounds selected identical clients across the PPO and
GRPO runs).

---

## 2026-07-26: ALFWorld's game list came out in *filesystem* order — client shards and the validation set were machine-dependent

- **Files:** `fedagent/envs/alfworld/engine/.../agents/environment/alfred_tw_env.py`
  (`collect_game_files`). Structural guard: `tests/test_alfworld_game_order.py`.
- **Severity:** reproducibility. The same config, the same `base_seed`, the same dataset →
  a different partition and a different validation set on a different node. Not a FedAgent
  regression (stock ALFWorld behaviour), but FedAgent makes it load-bearing.

### The bug

`collect_game_files` walks the split with `os.walk`, filters to solvable games of the configured
task types, then applies a **seeded** shuffle (`random.Random(42).shuffle`). The shuffle is
deterministic — but its *input* is not: `os.walk` yields directory entries in filesystem order,
which differs between machines and between copies of the dataset. Everything downstream is
positional:

- the seeded shuffle itself,
- `slice_games_for_client` → `partition_dataset` (client index slices),
- `num_train_games` / `start_idx`/`end_idx` caps,
- `games[seed]` selection under `ALFWORLD_SEED_IS_INDEX`.

So the game→client and game→val-slot assignment was a function of the filesystem, not of the
data. Measured on the real `valid_seen` split (140 games): re-running the pipeline from three
different walk orders produced val sets sharing only **29 of 64** games.

A second, smaller instance in the same function: the capped-eval subsample used bare
`random.sample`, i.e. the **global** RNG, whose state at that point depends on whatever ran
earlier in the process — so a capped eval set was not reproducible even on one machine, and two
val-service replicas could serve different games.

### The fix

```python
self.game_files.sort()                 # canonical, content-only order — before the seeded shuffle
...
self.game_files = random.Random(42).sample(self.game_files, num_eval_games)   # was random.sample
```

The sort runs after the manifest-cache load as well, so a cache written before this fix is
re-canonicalized on load instead of resurrecting the old order (the cache stays a pure
speed-up, which is its contract). Verified: with the sort, the three walk orders above produce
**64/64** identical val sets.

### What it means for existing runs

Shards and val slots differ from pre-fix runs *on this machine* — but they were never
reproducible off it, so there is no ordering worth preserving. Curves from before and after are
measured on different games; see the comparability note in the entry above.

The canonical order is a function of the dataset's *paths* — and since the split root is a
common prefix of every entry and Python sorts by code point, that means the order is identical
on any machine **given the same set of games**. Pinning the set itself needed a second change:
the checked-in manifest, one entry up (2026-07-27).

---

## 2026-07-26: the windowed-expansion tag outlived its `union`, arming a silent `slice()` no-op

- **File:** `fedagent/agent_loops/windowed_manager.py` (`_windowed_union`). Regression:
  `tests/test_windowed_union_tag.py`.
- **Severity:** latent. Requires a batch in which *every* episode produced exactly one turn, so
  it never fired at ALFWorld's 50-turn / WebShop's 15-turn budgets. Fixed as a trap, not as an
  observed failure.

### The bug

Windowed mode tags the per-turn TRAIN expansion with
`meta_info["__windowed_expanded__"] = size_divisor`. The patched `DataProto.union` reads it to
*adopt* the longer per-turn batch (instead of letting verl's equal-size union truncate it), and
the patched `DataProto.slice` reads it to neutralize `fit()`'s
`combined_gen_output.slice(0, num_sampled_prompts)`.

The tag was popped **inside** the `len(self) != len(other)` branch. When the lengths match, the
stock union runs — and it *merges* `meta_info`, carrying the tag onto the merged **training**
batch. `_windowed_slice` then silently no-ops every later from-start shortening slice on it
(REMAX's baseline slice; any future trainer slicing).

### The fix

Pop whenever the tag is present, then decide the branch — bounding its lifetime to the one union
it exists for. The regression test pins the pre-fix behaviour explicitly
(`wm._orig_union(...).meta_info[TAG] == 4`), so a future re-nesting of the pop fails loudly.

---

## 2026-07-26: validation dumps were chosen lexicographically — `10.jsonl` sorted before `4.jsonl`

- **Files:** new `fedagent/fed/eval_dumps.py` (`dumps_by_step`, `summarize_val_dump`), imported
  by `fed/run_fed.py` and `tools/rebuild_summary.py`. Regression: `tests/test_eval_dumps.py`.
- **Severity:** latent. verl writes one dump per validated `global_step`
  (`<validation_data_dir>/<step>.jsonl`), and FedAgent's layout gives each (round, client) its
  own directory, so no directory has yet held two dumps. Any change to the eval cadence would
  have made `sorted(...)[-1]` read the **older** dump as "the latest".

### The fix

Sort numerically by stem (non-numeric names last, by name), and — since the same 20 lines had
been copy-pasted into the offline rebuild tool — move the reader into one dependency-free
module both import. The tool still runs under a bare interpreter (`run_fed` needs omegaconf),
which was the original reason for the copy.

---

## 2026-07-25: WebShop scored correctly-clicked options as MISSES — `get_reward` compared plain values against `(key, value)` tuples, and `normalize_color` rewrote only one side; 458 of 6725 option-bearing goals could not reach 1.0 no matter what the agent did

- **Files:** `fedagent/envs/webshop/engine/webshop/web_agent_site/engine/goal.py` (`get_reward`),
  `.../engine/normalize.py` (`normalize_color`); offline regression
  `tests/test_webshop_option_reward.py`. The long-form write-up (per-goal sweeps, the worked
  example, the ruled-out alternatives) lives outside the repo at
  `/home/canyu/fedagent/docs/webshop_option_reward_bug.md`; everything load-bearing is repeated
  here.
- **Severity:** silent correctness, affecting **both** the RL training reward and every hardness
  label set derived from this engine. Not a FedAgent regression — the defective code is stock
  `princeton-nlp/WebShop` (the vendored file carried no FedAgent modifications; single vendoring
  commit `a448628`, and the copies under `AccelAgent/` and `FedAgent-paper-reproduce/` are
  byte-identical at that line). Deliberately **not** propagated to those two: the reproduce tree
  must keep reproducing the paper, and AccelAgent is a separate project's call.

### How it was found

Not from a crash or a failing test — the engine never complained. It surfaced twice, from two
directions, as a *statistical* anomaly in the hardness label sets.

**1. A floor that did not move with capability.** The hardness arm labels each goal easy/hard by
whether a reference policy solves it. Three references were labelled: a zero-RL
`Qwen2.5-1.5B-Instruct` (2.6% easy), `grpo-hardness-std4` (17.4%), `grpo-hardness-std1` (63.1%) —
a ~24× spread in competence. A block of goals came back **hard in all three, at exactly 0/417**.
That signature is diagnostic: real difficulty is policy-dependent, so a set of goals that a 63%
policy fails at precisely the same rate as a 2.6% policy is describing the *environment*, not the
task. The first reading was "structural caps — no product on the shelf satisfies the full
specification", which is the natural guess and was **wrong**; constructing the oracle-best
purchase (buy the target ASIN, click every required option, at a qualifying price) showed the
shelf does carry a compliant product and the reward function is what miscounts.

**2. Independently, from the val side: stable maxima.** A separate pass over 211 correlated
evaluations (71 aggregate + 140 client checkpoints of one FL run) on the 64-goal val slice found
goals that never once reached 1.0 — and whose best-ever score was a *fixed fraction* (6/7, 5/6)
rather than a fluctuating one. Genuine difficulty produces variance; a constant `k/n` that is
always short by exactly one component is the signature of a component that cannot be satisfied.
Reading one such trajectory turn by turn closed it: val goal #4 wants `color: r.brown2060`,
`size: 6.5`, price < $70; the policy clicked `r.brown2060`, clicked `6.5`, bought at $33.99 — a
perfect play with no room for improvement — and scored 0.857.

**Localisation** was then a two-line diff of the arguments: one side plain values, the other side
`(key, value)` tuples, with a normalizer between them that silently tolerated both.

**Caveat, recorded because it cost a round of analysis.** The val-side pass initially attributed
**9** never-solved goals to this bug and derived a "measurable ceiling" of 85.9%. Only **5** are
this bug (ceiling 92.2%). The other 4 score a clean 1.000 under the oracle purchase — see
[Not this bug](#not-this-bug). "No policy ever solved it" is not evidence of an engine cap,
especially across checkpoints of a single run, which are highly correlated and share search
habits. Only the oracle sweep separates the two, which is precisely why that sweep is now a test.

### Mechanism

Two independently harmless defects that are fatal together.

**1. The shape error.** `get_reward` passed the two sides of the comparison in different shapes:

```python
r_option, num_option_matches = get_option_reward(
    list(options.values()),        # clicked -> ['r.brown2060', '6.5']
    goal['goal_options'].items()   # wanted  -> [('color', 'r.brown2060'), ('size', '6.5')]
    if isinstance(goal['goal_options'], dict) else goal['goal_options']
)
```

The `else` branch already passes a plain list, so the `dict` branch was simply the odd one out.

**2. The silence.** `get_option_reward` runs both sides through `normalize_color`, which tests
`norm_color in color_string`. Python's `in` is overloaded:

- `str` → **substring** test: `'brown' in 'r.brown2060'` → `True` → returns `'brown'`
- `tuple` → **membership** test: `'brown' in ('color', 'r.brown2060')` → `False`

A tuple therefore matched nothing in `COLOR_SET`, fell through the loop, and was returned
**unchanged** — no `TypeError`, no warning. Had the code been written `color_string.find(...)`,
the first call would have raised and this would have been caught in minutes.

Net effect: **the clicked side is normalized, the goal side is not**, and the two are then
compared with `fuzz.token_set_ratio(...) > 85`:

```
token_set_ratio('brown', "('color', 'r.brown2060')") = 21   ->  scored as a MISS
```

`normalize_color` is also applied to every option category, not just colours, and takes the
**first** `COLOR_SET` substring hit in list order — which is how the mangling gets creative:

| clicked value | → | why |
|---|---|---|
| `fashion black girl01` | `ash` | `'ash'` is index 3; `"f`**`ash`**`ion"` matches before `'black'` (index 8) |
| `rectangular` | `tan` | `"rec`**`tan`**`gular"` |
| `r.brown2060` | `brown` | substring |
| `16x24 inch` | `16x24 inch` | no colour substring — unchanged, so it still matched at 100 |

That last row is why the blast radius is 6.8% and not 100%: the two sides only diverge when
normalization actually *changes* the clicked value.

Every reward component must be complete for a WebShop success
(`total = (num_attr_matches + num_option_matches + r_price) / (len(attributes) +
len(goal_options) + 1)`, then `*= r_type`), so **one** missed option permanently caps the goal
below 1.0.

### Fix

`.items()` → `list(...values())`, making both arguments the same shape (2 lines with the
comment). Plus defense in depth: `normalize_color` now raises `TypeError` on a non-`str`
argument instead of returning it unchanged — the silence is what let the shape error survive,
so any future caller passing the wrong shape now fails at the first call rather than mis-scoring
6.8% of the pool.

No compatibility flag. A flag that silently changes reward semantics per run is the same class
of defect as the bug itself; older runs reproduce from the commit, not from an env var.

### Impact

Pool sweep with the engine's own functions (`items_shuffle_1000` catalog, synthetic goals,
SimServer's `random.seed(42)` + shuffle):

| | goals |
|---|---|
| total | 6910 |
| option-bearing | 6725 |
| **capped by the bug** | **458 (6.81% of option-bearing)** |
| …in the train pool `goals[500:]` | 420 |
| …in the held-out val pool `goals[0:500]` | 38 |
| …in the shipped 64-goal val slice `goals[0:64]` | **5** — indices 4, 13, 35, 46, 61 |

**It penalises capability.** Only a policy that already got the attributes, price, product type
and every *other* option right ever reaches the state the bug then denies. "Played perfectly yet
failed" across three reference policies:

| reference policy | affected train goals played perfectly yet failed | easy rate |
|---|---|---|
| `grpo-hardness-std1` | **352 / 417 = 84.4%** | 63.12% |
| `grpo-hardness-std4` | 82 / 417 = 19.7% | 17.40% |
| `Qwen2.5-1.5B-Instruct` (zero RL) | 12 / 417 = 3.5% | 2.58% |

So **any** policy comparison through this reward is biased *against* the better policy — which
is exactly the comparison federated aggregation lives on (aggregate vs client, round N vs N+1).

**It was the training reward, not just an eval metric.** `web_agent_text_env.py`'s `done`
handler calls this `get_reward`, so on those 420 train goals the policy was penalised for doing
the right thing for entire runs.

**It set the val ceiling.** On the shipped 64-goal slice the measurable maximum was
**59/64 = 92.2%**, not 100%. After the fix it is 64/64.

### Verification

1. **Unit.** `get_option_reward(['r.brown2060','6.5'], [('color','r.brown2060'),('size','6.5')])`
   returned `(0.5, 1)`; with values on both sides it returns `(1.0, 2)`.
2. **Oracle sweep, whole pool.** Buy the target product, click every required option, at a
   qualifying price. Post-fix: **6910/6910 goals score exactly 1.0**, 0 unconstructible
   (every goal's required option values are genuinely offered on its product page). Pre-fix the
   same sweep fails on 458. This is `test_webshop_option_reward.py` (~35 s, no model/GPU); it was
   run against a temporarily reverted engine to confirm it is not vacuous (3 passed → 3 failed).
3. **End-to-end, before the fix landed.** `grpo-hardness-std1` rerun over train chunk 0
   (goals 0–1023), everything else identical: **62.8% → 66.8% easy**; the 74 affected goals in
   that chunk went **0 easy → 51 easy**. Full-pool projection 63.1% → ~67.6%.
   Artifacts: `fedagent_trajectory/hardness_labelgen_webshop_instruct/analysis_backup/bugfix_rerun/`.
4. **Alternatives ruled out** (so the 458 are attributable to this bug alone). Across all 64 val
   goals: required option values offered on the product page 64/64; attributes satisfied by the
   target product 64/64; `r_type` = 1.00 for 64/64; price never binds (prices are keyed by ASIN
   only and `price_upper` is *sampled above* the target's own price, so it cannot cap the target).
   After the fix, **no** val goal has a ceiling below 1.0.

### What this invalidates

- **Hardness label files are stale.** They encode the buggy ceiling and pin those 420 train goals
  to "hard" for *every* policy — a policy-independent ~6.5% floor masquerading as difficulty.
  Regenerate with `tools.gen_hardness_trajectories`.
- **Reward curves from pre-fix runs are not comparable** to post-fix runs on the affected goals.
- **Absolute WebShop success rates reported before this fix are understated**, policy-dependently,
  by up to ~5 pp (train) / ~7.8 pp (the 64-goal val slice).
- **Comparability with published WebShop numbers is now explicitly broken**, since upstream and
  every downstream fork still carry the defect. Worth stating in any writeup; the setup already
  differs from the literature (15-turn budget, 64-goal val slice, own goal partition).

### Not this bug

An earlier reading attributed a wider set of never-solved val goals to "structural caps". Only
the 5 above are structurally capped. Others that a reference policy never solved are winnable —
their single satisfying product just sits deep in the BM25 ranking under the goal's canonical
`query` field (rank 33, or outside the top-50) while being rank 1–6 under the instruction text,
and 53 of the 64 val goals have **exactly one** satisfying product among the ~50–100 visible.
That is retrieval difficulty, not a broken denominator. Earlier notes calling the 417/420 train
goals "structural caps" ("no product on the shelf satisfies the full specification") are wrong in
the same way — the shelf does carry compliant products; the reward function miscounted.

---

## 2026-07-24: weight-transfer IPC namespace reused across rounds/retries — one crashed round left a stale `/tmp` socket that killed every later round of the run

- **Files:** `fedagent/fed/run_fed.py` (`_unique_ipc_env` applied in `stream` + `BgProc`;
  round added to the persistent job id; `verl_honors_job_id_override` preflight;
  `sweep_own_ipc_sockets`); offline regression `tests/test_ipc_namespace.py`.
- **Severity:** availability. Found from a field report: a WebShop PPO run died at round 48
  with a stale-IPC engine error, and the same round kept dying — until the whole run was
  restarted, which "fixed" it for reasons nobody could explain at the time.

### Mechanism

verl names the FSDP→vLLM weight-transfer ZMQ socket after the Ray job id
(`ipc:///tmp/rl-colocate-zmq-<job_id>-replica-<r>-rank-<lr>.sock`). A process that dies
hard leaves that FILE behind, so any later process computing the same path fails binding
it. The overlay already hands each launch a `VERL_RAY_JOB_ID`
(`tools/setup/patches/verl_weight_transfer_jobid.patch` makes verl honor it) — but
uniqueness rested on five format strings being collectively injective, and two were not:

1. **The per-round persistent worker.** `_persistent_cmd_env` used
   `f"{_RUN_TAG}-persist{lane}"` — **no round**. `cross_round=false` relaunches a NEW
   process every round, so every round of a run shared one socket path: round N's crash
   poisoned N+1, N+2, … forever. A *fresh run* draws a new `_RUN_TAG` uuid — exactly why
   restarting the run appeared to fix it while resuming into the same process did not.
   (`cross_round=true` launches once, so it was never affected.)
2. **The one-shot port-collision relaunches** (`stream_port_retry`,
   `_wait_launch_port_retry`) re-ran with the dead attempt's identical id — so a crash that
   left a socket behind was retried straight into that socket.

### Fix

`_unique_ipc_env` appends a process-global launch counter and is applied at the **two
places a verl process is born** (`stream`, `BgProc.__init__`). "No two verl processes share
a socket path" is now true *by construction* rather than by inspection of every job-id
string; the descriptive ids are kept for debuggability, and the round was added to the
persistent id anyway (defense in depth). No-op where no job id is set (aggregator, merger:
no vLLM, no socket).

Two supporting changes:

- **Preflight (anti-silence).** Unpatched verl ignores `VERL_RAY_JOB_ID` entirely and every
  isolated Ray cluster falls back to the same first job id `01000000` — all of a node's
  runs then share one path, *with no error saying so*. `verl_honors_job_id_override()`
  reads the installed source (`find_spec` on the top-level package locates without
  executing, keeping the driver torch-free) and a missing patch now prints a boxed
  `[warn]`. Returns `None` when the source can't be read — never a guess.
- **`/tmp` hygiene.** `sweep_own_ipc_sockets()` unlinks this run's sockets at the end,
  scoped strictly to its own `_RUN_TAG` so a concurrent run's LIVE sockets are never
  touched. They otherwise accumulate one-per-launched-process for the node's lifetime
  (the same field host also hit a full disk).

### Verification

- `tests/test_ipc_namespace.py` (5, offline): repeated `_unique_ipc_env` on the SAME env
  yields distinct ids and does not mutate the caller's dict; no-op without a job id; both
  launch primitives isolate (fake `Popen`, 3 launches → 3 namespaces); preflight
  True/False/None against synthetic stock vs patched sources; the sweep removes only the
  own-tag socket and leaves a foreign one.
- Preflight against the real installed verl in `fedagent-verl08` → `True` (patch applied).

## 2026-07-24: PPO warm start silently RESET the critic — `model_path` was the only seed entry, so a continuation run inherited the policy but opened with a randomly initialized value head

- **Files:** `fedagent/fed/run_fed.py` (`critic_model_path` DEFAULT + `--critic-path`;
  `hf_weight_keys` / `has_value_head` / `is_aggregated_actor` / `resolve_start_critic`;
  provenance log + `critic_init`/`start_critic` in the summary); offline regression
  `tests/test_critic_warm_start.py`.
- **Severity:** science, PPO-only, and **silent** — the run logged a critic path (which
  happened to be the *actor's*) and said nothing about the value head being random.
  Observed in the field: a WebShop PPO continuation seeded from `round_37/aggregated/hf`
  into a fresh `--output-dir`.

### Mechanism

Continuation was designed to run through `resume` (same `output_dir`), which restores
actor **and** critic and refuses to call a PPO round complete unless the merged critic
exists (`find_resume_round`). `model_path` means "the pretrained base for round 1".
Pointing `model_path` at a previous run's `round_<k>/aggregated/hf` to seed a NEW
`output_dir` is a warm start the design never covered:

```python
current_critic = base_model if is_ppo else None   # base_model == the seed ACTOR dir
```

verl then builds the critic as `...ForTokenClassification(num_labels=1)`
(`workers/engine/fsdp/transformer_impl.py:264`) over that actor checkpoint. The backbone
weights match and load; the `score.{weight,bias}` head does not exist there and is
**randomly initialized**. So the policy is warm and its value baseline is noise.

Consequences, in order of importance:

1. **GAE is uncalibrated for the opening rounds.** With `δ_t = r_t + γV(s_{t+1}) − V(s_t)`
   and a random `V`, advantages are at best baseline-free returns (in WebShop's sparse
   {0,10} that hands every token of a successful trajectory the same large positive
   advantage — no within-trajectory credit assignment) and at worst state-dependent noise
   that points some good actions the wrong way. PPO's clipping bounds step size, not
   direction, so the warm-started policy can regress before the critic catches up.
2. **No workaround existed.** `client_overrides` cannot supply `critic.model.path` —
   run_fed appends its own value *after* the overrides, so it always wins
   (`run_client`). And `trainer.critic_warmup` is not a fix here: each client is a fresh
   verl job per round, so `global_steps` restarts every round and the warmup would be
   re-paid every round, forever.
3. **The trained critic was on disk the whole time.** `round_<k>/aggregated/critic_hf` is
   written every PPO round and survives `cleanup_round_checkpoints` (which only deletes
   `checkpoints/` shard dirs). The asset existed; only the wiring was missing.

### Fix

`resolve_start_critic(cfg, base_model)` picks the first trained round's value model and
tags its provenance: **explicit `critic_model_path`/`--critic-path` > the aggregated
critic sitting beside an aggregated seed actor > the actor backbone (fresh head)**. The
middle rule makes the field case correct *by default*, so nobody has to know this trap
exists.

Detecting "is this a critic?" cannot use `config.json`: verl's `model_merger` writes
`architectures: [<Arch>ForCausalLM]` for the critic **and** the actor — verified against a
real merged pair, where the only difference is `score.weight [1, 1536]` + `score.bias [1]`
in the weights. `has_value_head` therefore reads tensor NAMES, and does it without
importing torch/safetensors (this driver is deliberately verl-free): 8-byte LE header
length + JSON header for a single file, `model.safetensors.index.json`'s `weight_map` when
sharded, and `None` ("undeterminable", never a guess) for legacy `.bin` or a corrupt
header.

Anti-silence guarantees, so the class cannot recur unnoticed:

- an explicit `--critic-path` **without** a value head is a hard `ValueError` (that is the
  exact mistake this option exists to prevent; omit the flag to reset deliberately);
- a warm start off an aggregated actor that still ends up with a fresh head prints a
  boxed `[warn]` block naming both paths, the consequence, and both remedies;
- a genuine fresh run from a pretrained base logs one line saying the head is fresh and
  that this is expected;
- `federated_summary.json` records `critic_init` (`resume` | `auto-sibling` | `explicit` |
  `fresh-value-head`) and `start_critic`, so a warm-started run's provenance is on the
  record for the paper.

GRPO is untouched (no critic exists), and a PPO run from a pretrained base resolves to
exactly its pre-fix value — no behavior change for fresh runs.

### Verification

- `tests/test_critic_warm_start.py` (6, offline): the fixture is checked to be a *real*
  safetensors container (guarded `importorskip`) so the parser tests aren't vacuous;
  head detection single-file/sharded/`v_head`; `.bin`/missing/corrupt → `None`;
  auto-sibling resolution; fallbacks for no-sibling and pretrained-base (no regression);
  explicit beats sibling; actor-as-critic raises; missing dir raises.
- Against the **real** merged pair from `runs/hardness_rerun/ppo_ws_std1/round_8`:
  actor → `False`, critic → `True`, and `resolve_start_critic` returns that round's
  `critic_hf` with mode `auto-sibling`.

## 2026-07-23: RESUME could FedAvg a dead attempt's checkpoint — a re-run round inherited the crashed attempt's partial artifacts, and `latest_actor_dir` picks the HIGHEST global_step

- **Files:** `fedagent/fed/run_fed.py` (`quarantine_stale_rounds` + its call right after
  the resume scan in `run()`); offline regression in `tests/test_launch_reliability.py`.
- **Severity:** latent correctness. Never observed corrupting a run — in the common case
  (crash + re-run under the SAME config) the re-run saves at the same step numbers and
  overwrites the dead attempt's files, so the scan picks the fresh checkpoint. The hole
  needs a step-count change between attempts to open (see below), which is exactly what
  happens when someone tweaks a config after a crash before relaunching.

### Mechanism

Round-level resume (`find_resume_round`) treats `round_k/aggregated/hf` as the completion
marker and re-runs everything above the last complete round. The re-run itself is clean —
clients pass `trainer.resume_mode=disable`, so verl never loads a partial checkpoint. But
the crashed attempt's `round_K/` dir stayed in place, and the post-training scan
(`latest_actor_dir`) returns the highest `global_step_N` that holds FSDP shards:

- Attempts with EQUAL step counts: same step numbers → dead attempt's files overwritten →
  correct result (why this never bit in practice).
- Attempt 2 with FEWER steps (config tweaked between attempts: `total_training_steps`
  lowered, `epochs_per_round`/`epoch_resample` changed, dataset shrunk): the dead
  attempt's higher-step checkpoint SURVIVES next to the fresh lower-step one, wins the
  scan, and its PRE-CRASH weights get FedAvg'd into the round — silently, since the shards
  are complete and well-formed. Same exposure for the stale critic dir (PPO), stale eval
  dumps, and stale `training.log`-derived metrics.

### Fix

`quarantine_stale_rounds(cfg, start_round - 1)` runs right after the resume scan: every
`round_j` dir with `j > last_complete` is RENAMED to `_stale_rounds/round_j.N` (N bumps on
repeated crashes; rename not delete — crash forensics; the log flags them safe to delete).
Every re-run round therefore starts from an empty dir, killing the whole class (stale
checkpoints, partial shard sets, stale eval dumps/metrics) rather than the one symptom.
Safety notes: `round_0` (base-eval dir) is never archived; consumers all address exact
`round_<k>` paths (verified: no `round_*` glob in the package), so archived dirs cannot
leak back into curves; `--fresh` into a populated `output_dir` now archives ALL prior
rounds aside instead of interleaving two runs' artifacts in one tree. This also hardens
the port-collision recovery path (entry below), which is precisely a die → resume →
re-run-the-round cycle.

## 2026-07-23: port-band regression (same-day fix): ONE static VLLM_PORT per job made every concurrent vLLM replica race the same start → deterministic EADDRINUSE at engine init

- **Files:** `fedagent/port_band.py` (`assign_vllm_port` per-process pid-salted assignment;
  `probe_in_band` pid-salted start + rotating cursor); `fedagent/fed/run_fed.py`
  (`_port_band_env` no longer injects a static `VLLM_PORT`); `tests/test_port_band.py`
  (+2 tests, 6 total). Regression introduced by `4a85d12` earlier the same day.
- **Severity:** availability — with `port_band_base` on (the default) and `n_gpus > TP`,
  round-1 engine init fails deterministically. Found in the field: a 4-GPU FedAgent run
  died at round-1 vLLM init with EADDRINUSE on 26050 (slot-0 band midpoint), immediately
  after pulling `4a85d12`.

### How it was found / mechanism

The banding design gave each launched JOB one band and exported one static
`VLLM_PORT = band midpoint` into its env. What it missed: verl's rollout starts **one
vLLM replica per GPU** (at `TP < world`, e.g. the paper's 4×TP1 shape = 4 replicas),
concurrently, all inheriting that same env — and vLLM's `VLLM_PORT` probing
(`_get_open_port`: bind-test upward from the start, **close the probe socket**, hand the
number to the engine which binds later) means simultaneous probers starting from the
SAME value all "win" the same port before any of them binds it. The original ephemeral
lottery collided occasionally; the shared-start band collided **every time** there was
more than one replica. A second, latent in-process flavor: `probe_in_band` (verl's
picker) scanned first-free from the band start every call, so two draws before the first
consumer bound its port returned the same number twice.

The field workaround (`port_band_base: 0`) was sound: it disables banding but keeps
`stream_port_retry`, which is independent — the original occasional collision stays
self-healing.

### Fix

- `assign_vllm_port()`: each PROCESS (sitecustomize runs in the driver, every Ray
  worker, and spawned engine cores) computes its own `VLLM_PORT = band_upper_half +
  (pid*7919) % (half-8)`, **overriding** the inherited value — concurrent replicas now
  probe upward from spread-out starts; vLLM's native probing handles the rest.
- `probe_in_band()`: scan starts at a pid-salted offset (cross-process spread) and a
  per-process rotating cursor advances past every returned port (consecutive draws
  differ even while earlier ports are probed-but-unbound).
- `_port_band_env` exports only `FEDAGENT_PORT_BAND`; no static `VLLM_PORT`.

### Verification

`tests/test_port_band.py` (6): salted start honored, squatter skipped, consecutive
draws differ WITHOUT binding, exhausted band fails closed, `assign_vllm_port` lands in
the upper half + overrides a stale static value + distinct pids get distinct starts.
Field re-enable: pull, drop the `port_band_base: 0` override, relaunch.

### Hardening + residual-risk accounting (added later the same day, after a "is it REALLY solved" pass)

Two more gaps closed:

- **Strict half partition.** verl's rebound picker scanned the WHOLE band, so a verl
  draw could claim-but-not-yet-bind a port in vLLM's upper half (the PG master binds
  its TCPStore late) while a vLLM replica probed and bound it first → late-binder
  EADDRINUSE. verl now draws from the lower half only; the pickers can no longer cross.
- **Case-insensitive retry signatures.** torch prints "Address already in use" but
  Ray/grpc surfaces the same errno lowercase — the cased match would have skipped the
  retry exactly when Ray's own (unpinnable) internal ports collided. vLLM's benign
  probe line stays excluded (it never contains "address").

Residuals that CANNOT be zeroed at this layer (accepted, with coverage notes):

1. **pid-salt collisions**: `(pid·7919) % 42` is a bijection on any 42-consecutive-pid
   window (7919 coprime to 42), so near-simultaneously spawned replicas always get
   distinct starts; two pids exactly a multiple of 42 apart still share one. Covered by
   `stream_port_retry` on the subprocess-client and single-persistent paths, and (round-2
   hardening below) by `_wait_launch_port_retry` on the cross-round/lane LAUNCH; a
   long-lived worker dying in a later round stays fail-fast (its relaunch cmd would bake
   the launch round's stale model_path) — round-level resume is the backstop there.
2. **The probe-close → real-bind TOCTOU** itself: only vLLM/torch handing over BOUND
   sockets could remove it. Inside a private per-slot band the only realistic squatter
   is another tenant explicitly binding 26000-29299 — pick `port_band_base` away from
   locally-used service ranges if that ever happens.
3. ~~The FedAvg aggregator's `torchrun --standalone` rendezvous port~~ — CLOSED by the
   round-2 hardening below (banded `--master_port` + one relaunch on a fresh port).
   Remaining unpinned pickers: **Ray internals** (gcs/raylet/worker ports — pinning them
   means threading port ranges through verl's internal `ray.init()`; not worth the new
   config surface at the observed ~1 collision/40 rounds with self-healing recovery).

### Round-2 hardening (same day, after a field RANDOM-ephemeral collision at ~round 43 of a 70-round run)

A run on the field machine lost ~10 min to a collision on port 32859 — ephemeral range,
i.e. a draw the band never covered (band disabled by the operator's `port_band_base: 0`
workaround, and/or one of the residual pickers above). Training self-healed exactly as
designed (outer babysit loop → `resume` → re-run round 43), but for 70-round unattended
runs the remaining exposure was worth shrinking:

- **FedAvg aggregator banded + retried** (`run_fed._agg_rdzv_args`, slot 32 — its own
  slot because an `eval_mode=parallel` async eval of round r can still be running when
  round r+1 aggregates): `torchrun --standalone` drew a random EPHEMERAL rendezvous port
  1-2× per round with NO retry — a collision failed the round and cost a full
  resume re-run. Now: band on → `--master_port=<probe_in_band(slot 32)>`; band off →
  `--standalone` kept. Either way the launch loop relaunches ONCE on a collision
  signature, rebuilding the cmd so the retry gets a FRESH port (banded: the probe cursor
  advanced; standalone: torchrun redraws). Band-exhausted falls back OPEN to
  `--standalone` (aggregation must not die because the band is busy; the trainer-side
  probe stays fail-closed).
- **Cross-round / lane workers: one-shot LAUNCH relaunch** (`_wait_launch_port_retry`,
  the `BgProc` twin of `stream_port_retry`): a port collision strikes at engine init,
  before any client trains, so relaunching the same cmd/env is always safe there. The
  relaunch re-registers the new proc in `xstate` first, so a failing retry still gets
  torn down by the run's `finally`. Failed log → `<log>.portfail`, stale
  `go_*`/`done_*`/`stop` signals cleared pre-relaunch.
- **RESUME quarantines incomplete rounds** — see the stale-round entry above this one:
  the collision-recovery path (die → resume → re-run round) is exactly the path that
  used to leave a dead attempt's partial artifacts in the re-run round's dir.

Offline regression: `tests/test_launch_reliability.py` (5) — banded slot-32 rendezvous
args + rotation across draws, fail-open on exhausted band / band off, quarantine
semantics (archive-above-last-complete, suffix on re-crash, `round_0` never archived),
real-`BgProc` relaunch on a signature death (forensics + signal hygiene asserted), and
re-raise without relaunch on a non-port death.

## 2026-07-23: vLLM/verl random-port collisions: per-round trainer rebuilds redraw ephemeral listen ports → occasional "Address already in use" kills the round on shared-netns hosts

> **Regression note (fixed same day):** the version of this fix shipped in `4a85d12`
> exported ONE static `VLLM_PORT` per job — with multiple rollout replicas (`n_gpus>TP`)
> they all raced the same start and died at engine init. See the entry directly above
> for the per-process pid-salted redesign; the description below is otherwise current.

- **Files:** new `fedagent/port_band.py` (band probing + verl `_get_free_port` rebind +
  deferred hook + collision-signature matcher); `sitecustomize.py` (5th arming block, gated
  on `FEDAGENT_PORT_BAND`); `fedagent/fed/run_fed.py` (`port_band_base`/`port_band_stride`
  DEFAULTS + `_port_band_env(cfg, slot)` injected into the subprocess-client, persistent
  (per-lane), and eval builders + `stream_port_retry` one-shot relaunch at the client and
  persistent launch sites); offline regression `tests/test_port_band.py`. Root cause lives
  in stock vLLM/verl port pickers; the fix is overlay-side (thin-overlay policy).
- **Severity:** robustness/infra, not science — a collision kills the round mid-run (resume
  recovers, manually). Probability scales with rounds × lanes × evals on a busy shared
  network namespace; observed in the wild on a devbox WebShop-PPO-hardness run (`r7-persist`
  died on a port in the ephemeral range).

### How it was found

A production run on another machine died at round 7 with the torch TCPStore signature
(`The server socket has failed to listen ... errno: 98 - Address already in use`) on port
54147 — squarely inside Linux's default ephemeral range (32768-60999). Mechanism verified
against the installed vLLM 0.11.0 and the pinned verl:

1. **vLLM** `_get_open_port()` (vllm/utils): with `VLLM_PORT` unset it does
   `bind(("", 0))` → takes the kernel-assigned EPHEMERAL number → **closes the probe
   socket** → the engine binds it later (distributed_init_method TCPStore / DP master /
   api server). Classic use-after-probe TOCTOU: anything in the shared netns can grab the
   number in the window. With `VLLM_PORT` SET, vllm instead probes UPWARD from that value
   until a bind succeeds — collision-tolerant by design.
2. **verl** `WorkerHelper._get_free_port` (`single_controller/base/worker.py:59`) is the
   same bare `bind(("", 0))` draw, feeding the FSDP process-group `MASTER_PORT`
   (`single_controller/ray/base.py:637`) — and has NO env knob upstream. So pinning vLLM
   alone would NOT have cured the class; both pickers had to move.
3. **FedAgent amplifies the lottery**: subprocess mode and the per-round persistent worker
   rebuild the trainer EVERY round → every round × lane × eval redraws ports → hundreds of
   draws per 70-round run; "occasional" becomes "expected a few times per run".

### Impact

- A collision aborts engine init → the client/persistent step dies → the run stops until
  someone reruns (round-level resume then continues — the observed run had already done
  `resumed_from_round=4` once). No numbers are corrupted; pure availability/babysitting
  cost. Unrelated to the ZMQ IPC weight-transfer collision patched earlier
  (`tools/setup/patches/`): same "concurrent verl on one node" family, different socket.

### Fix (two layers; both needed for an operational cure)

1. **Private port bands, outside the ephemeral range.** run_fed gives every launched
   trainer/eval process `FEDAGENT_PORT_BAND="<start>:<stride>"` with
   `start = port_band_base + slot*stride` (default 26000 + slot×100; slots: persistent
   lane l, subprocess client = position in the round, global eval 30, circle eval 31,
   FedAvg aggregator rendezvous 32 — unique among CONCURRENT processes;
   max 29300 < 32768). sitecustomize rebinds verl's
   `_get_free_port` to probe INSIDE the band (fail-closed if the band is exhausted — a
   silent fallback to the lottery would resurrect the bug unobserved), and `VLLM_PORT` is
   set to the band midpoint so vLLM's native upward probing works the upper half. The
   kernel never hands out <32768 as outgoing source ports, so the only possible squatter
   in a band is our own previous process's stale listener — which probing skips.
2. **One-shot relaunch on the collision signature.** The literal TOCTOU window
   (probe-close → real-bind) cannot be zeroed at the overlay level (that needs vLLM/torch
   to hand over BOUND sockets), and Ray keeps its own internal pickers — so
   `stream_port_retry` relaunches the client/persistent step ONCE iff the dead log's tail
   matches `Address already in use` / `EADDRINUSE` / `DistNetworkError` (deliberately NOT
   vllm's benign "Port X is already in use, trying port X+1" probe line). The failed log
   is preserved as `<log>.portfail`; the retry rewrites the canonical path so metrics
   parsing and checkpoint scans are unaffected. Cross-round/lane workers get the same
   one-shot treatment at LAUNCH only (`_wait_launch_port_retry`, round-2 hardening in the
   entry above); a long-lived worker's death in a later round is what round-level resume
   is for.

`port_band_base: 0` restores stock random ports (A/B or if 26000-29999 is contested on
some host).

### Verification

- `tests/test_port_band.py` (6, offline): salted probing returns the expected start and
  skips a live squatter; consecutive draws rotate; exhausted band fails closed; env
  parsing; pid-salted `VLLM_PORT` assignment overrides an inherited static value; the
  signature matcher fires on the real torch error text (and Ray's lowercase variant) and
  NOT on vllm's benign probe line. Plus `tests/test_launch_reliability.py` (5) for the
  round-2 hardening layer (aggregator rendezvous / launch relaunch / quarantine).
- Real-verl rebind exercised in `fedagent-verl08`: `WorkerHelper._get_free_port()` draw
  lands in-band (26000), idempotent re-arm refused.

## 2026-07-23: bf16 merge truncation: verl's model_merger quantized the fp32-aggregated weights to bf16 at every round boundary (the fork kept fp32 end-to-end)

- **Files:** new `fedagent/merge_fp32.py` (source-guarded rebind of the two truncating merger
  methods + deferred import hook); `sitecustomize.py` (4th arming block, gated on
  `FEDAGENT_MERGE_FP32=1`); `fedagent/fed/run_fed.py` (`merge_fp32` DEFAULT **on** +
  `merge_to_hf(fp32=...)` for the two AGGREGATED call sites only); offline regression
  `tests/test_merge_fp32.py`. Root cause lives in stock verl
  (`others/verl/verl/model_merger/{fsdp_model_merger,base_model_merger}.py`) but the fix is
  overlay-side (thin-overlay policy: never fork verl).
- **Severity:** science-fidelity, BOTH algos, BOTH envs, every config whose round loop goes
  through the HF merge — subprocess mode, persistent + `hf_export: every_round` (the
  accelerated-recipe default; `reload_client_model` loads the merged HF dir), and resume.
  `hf_export: final`'s direct shard reload never hit it. Register ID `fedavg-aggregation-4`
  (the last unfixed *medium migration-bug* in `review/review_docs_2/03_findings_register.md`).
- **Status note:** bounded — ALFWorld trained to ~63% through this truncation, so it is not a
  smoking gun for any observed regression; it is a systematic per-round precision loss the
  fork's path simply did not have.

### How it was found

2026-07-23 triage of the two review registers against HEAD ("这里还有什么bug要改吗"): most
register items were already fixed or are paper-vs-code decisions; `fedavg-aggregation-4` was
the one remaining medium. Before fixing, the *entire* round-boundary chain was re-verified so
the fix would provably land end-to-end:

1. Client FSDP model shards are saved fp32 (FSDP master weights).
2. `fedagent/fed/aggregate_fedavg_fsdp.py` averages **in place on the loaded shards**
   (`acc.mul_(w0); acc.add_(other, alpha=w)`) — no dtype cast anywhere → the aggregated FSDP
   shards stay fp32. (So the aggregator is NOT the truncation point.)
3. `run_fed.merge_to_hf` shells out to `python -m verl.model_merger merge`; stock verl casts
   there: `fsdp_model_merger._load_and_merge_state_dicts` collects every shard as
   `tensor._local_tensor.bfloat16()` (`:169`, DTensor) / `tensor.bfloat16()` (`:181`, plain),
   and `base_model_merger.save_hf_model_and_tokenizer` builds the save skeleton with
   `torch_dtype=torch.bfloat16` (`:379`, which also stamps config.json). The pinned verl
   (7aed6b2) has **no dtype knob** on the merger CLI.
4. The NEXT round's load would preserve whatever the file has: verl's training load forces
   `torch_dtype=fp32` (`transformer_impl.py:236-238` "if it is training, we force torch_dtype
   to fp32") — i.e. today's fp32 masters start from bf16-quantized values, and an fp32 HF file
   would be consumed exactly. This is what makes the fix effective end-to-end.

### Impact (ranked)

1. **Per-round quantization of the aggregated update.** bf16 keeps ~8 bits of mantissa
   (relative ULP ≈ 0.4%). At actor lr 1e-6 and a few dozen optimizer steps per client-round,
   the per-round aggregated weight motion is comparable to the bf16 ULP at typical weight
   magnitudes — a fraction of each round's *averaged* update is rounded away, every round, for
   70 rounds. It is rounding (unbiased), not drift, which is why it is bounded — but the fork
   ran this path losslessly, so it is a pure fidelity deviation.
2. **FedAvg precision claim.** The aggregation itself IS fp32 (and `verify` checks it), but
   the *delivered* model each round was bf16 — the paper's "fp32 aggregation" guarantee only
   held up to the merge.
3. **NOT affected:** rollout/eval numerics (vLLM casts to bf16 by config either way), the
   within-round training dtype (bf16 mixed precision, same as fork), `hf_export: final` runs.

### Fix

`fedagent/merge_fp32.py` **recompiles the two stock methods from their own source** with the
casts flipped (`.bfloat16()` → `.float()` ×2; skeleton `torch.bfloat16` → `torch.float32` ×1)
and rebinds them. The rebind is **guarded by exact marker counts** — if a verl upgrade changes
either method, arming raises with a re-derive instruction instead of silently rebinding (same
upgrade-trap discipline as `ppo_critic_loss._assert_stock_value_loss`). Armed via
`FEDAGENT_MERGE_FP32=1` + the sitecustomize deferred hook in the merger *subprocess* only;
`run_fed` sets it for the two **aggregated** merges (`cfg.merge_fp32`, default on) and leaves
the client-end-eval merge bf16 (it only feeds the bf16 vLLM eval rollout). Disk cost: ~2× on
aggregated `hf/` only (1.5B: 3.1→6.2 GB/round, pruned as usual); `merge_fp32: false` restores
stock behavior for A/B.

### Verification

- `tests/test_merge_fp32.py` (3): the rebind flips dtypes and preserves fp32 bit-exactly on a
  value bf16 cannot represent; marker-count drift fails closed ("re-derive"); a
  vendored-source contract test reads verl's REAL files (via `find_spec`, no import) and
  asserts the exact counts the runtime guard expects — a verl bump breaks CI, not round 1.
- Real-verl rebind exercised in the `fedagent-verl08` env: applies, idempotent, module
  globals (DTensor / init_empty_weights) resolve from the recompiled functions.

## 2026-07-23: FedProx anchored the PPO critic too — verl 0.8 puts actor AND critic on the same FSDPEngine class the patch wrapped (fork anchored dp_actor only)

- **Files:** `fedagent/fedprox.py` (`_make_optimizer_step` factory + value-model pass-through
  + docstring); `fedagent/fed/run_fed.py` (`fedprox_mu` DEFAULTS comment documenting the
  paper-equivalent ablation knob, fork default mu=0.01 — closes register `fedprox-7` too);
  new `tests/test_fedprox_actor_only.py`. Register ID `fedprox-6`.
- **Severity:** dormant landmine — **no paper run uses FedProx** (mu=0 everywhere, no FedProx
  table row), so nothing shipped was affected. Any future PPO+FedProx ablation would have
  silently gained a critic regularizer the fork's recipe never contained.

### How it was found

Register `fedprox-6` claimed the divergence; verified rather than trusted: the fork's FedProx
commit (`8c6a000`) patches `dp_actor.update_policy` and never touches `dp_critic`, while our
overlay wraps `FSDPEngine.optimizer_step` at CLASS level — and in verl 0.8's unified-engine
design the PPO critic worker builds the *same* `FSDPEngine` class (the old `dp_critic.py` was
deleted in the 0.8 refactor). The original docstring even shows the blind spot: it argued
"GRPO has no critic and the ref never steps" — true, but PPO's critic both exists and steps.
The clean discriminator came from verl itself: `EngineRegistry` registers engines by
`model_config.model_type` — `"language_model"` (actor) vs `"value_model"` (critic)
(`transformer_impl.py:921` / `:1297`) — so no parameter-name sniffing is needed.

### Impact

- With `fedprox_mu > 0` on a `gae` config, every critic optimizer step would have added
  `mu * (w_c - w_c^t)` pulling the value head toward the round-start critic — a *different
  algorithm* from the fork's FedProx (actor-only proximal term), biasing the ablation it was
  meant to run. GRPO+FedProx was and is unaffected (no critic engine exists).

### Fix

`_make_optimizer_step(orig, mu)` passes engines with `model_config.model_type ==
"value_model"` straight through (no snapshot, no proximal grad); absent `model_config` keeps
the old behavior (back-compat for single-engine paths). Startup log now says "actor-only".

### Verification

`tests/test_fedprox_actor_only.py` (3, offline): a drifted `language_model` engine receives
exactly `mu*(w-w_t)` gradient; a `value_model` engine receives zero AND takes no `w_t`
snapshot; an engine without `model_config` is anchored (back-compat).

## 2026-07-23: `search_return_n` defaulted to 200 — the ENV-het value — so any config that didn't pin it ran non-het WebShop with a 4× BM25 pool; 9 shipped examples were doing exactly that

- **Files:** `fedagent/fed/run_fed.py` (DEFAULTS 200→**50** + the two `cfg.get(...)`
  fallbacks in the train/val service builders); 11 env-het example configs now pin
  `search_return_n: 200` explicitly. Register ID `webshop-env-12`.
- **Severity:** config-hygiene with one real in-repo casualty class: all **192 generated
  paper/paper_accelerated configs pin the key explicitly** (uniform 50 / env-het 200) and
  were never affected — no paper run is implicated. But 9 shipped **non-het examples**
  (`homog_long`, `probe_signal`, `scaled/{homog,centralized,ppo,task,pref,coverage,hardness}`)
  carried no pin and silently ran at 200.

### How it was found

The register filed this as a "footgun for hand-written configs". During the 2026-07-23 fix
triage the actual blast radius was measured: `grep -rL search_return_n` over `config/`
showed every generated config pinned (so the flip is zero-risk for paper runs) — and that the
footgun had **already fired inside the repo**: the 9 non-het examples above inherited the 200
default. The paper scopes 200 to ENV-heterogeneity only (`main.tex:1347`); the fork exported
`SEARCH_RETURN_N` for env-het runs and every non-het baseline trained at the engine default
50.

### Impact

- `SEARCH_RETURN_N` is the BM25 retrieval pool the search page paginates over — it changes
  the "Total results: N" text, the reachable pagination depth, and which products an agent
  can reach at all. A non-het run at 200 is a *different environment* from the 0915 baselines
  (50): its numbers are not comparable to any baseline curve. Direction of harm: examples and
  any hand-written config only; generated paper configs were immune.
- The env-het arms NEED 200 (catalog filtering drops targets out of a 50-pool), which is why
  the old default chose 200 — but a default must serve the *unmarked* case, and the unmarked
  case is non-het.

### Fix

Default flipped to 50 (the engine/original non-het value); the 11 env-het examples that
relied on the 200 default (`2cl_catalog_split`, `envhet_long`, `fedprox_test`,
`scaled/{catalog,bm25field,bm25reweight,lookalike,ppo_lookalike,rank,envhet_fedprox,local}`)
now pin `search_return_n: 200` next to their `partition_strategy`. Non-het examples need no
edit — the new default IS their faithful value. Generated configs: no change (all pinned).

### Verification

`grep -h search_return_n config/paper*/**` → uniform 50 ×176, env-het 200 ×16 (unchanged);
examples: env-het all pin 200, non-het carry no key and now resolve to 50.

## 2026-07-22: PPO critic loss scale: stock verl 0.8's value_loss skips the engine's global normalization (+ a 0.5 the fork never had) -> critic gradient = 0.5 x M = 2x the paper fork at the paper recipe

- **Files:** new `fedagent/ppo_critic_loss.py` (value_loss parity wrapper + deferred import
  hook); `sitecustomize.py` (arming, gated on `FEDAGENT_CRITIC_LOSS_MODE`);
  `fedagent/fed/run_fed.py` (`critic_loss_mode` DEFAULT + gae-only env injection in both
  client builders); offline regression `tests/test_ppo_critic_loss.py`. Root cause lives in
  stock verl (`others/verl/verl/workers/utils/losses.py`, `trainer/ppo/core_algos.py`) but the
  fix is overlay-side (thin-overlay policy: never fork verl).
- **Severity:** science-fidelity, PPO/gae-only (GRPO builds no critic), BOTH envs, every PPO
  config in both trees. Actor is unaffected.
- **Status note:** the *practical* performance impact at the paper recipe is bounded by two
  dampeners (see Impact), so this is a fidelity/invariance fix, not a smoking-gun for any
  observed regression.

### How it was found

Follow-up to the 2026-07-22 "does PPO have remaining problems?" deep-check. A windowed-PPO
semantics diff (GAE horizon, whitening, masks, critic knobs, federation) came back IDENTICAL
on everything EXCEPT the critic loss chain, which was then hand-verified line by line:

1. verl 0.8's FSDP engine backwards EVERY micro-batch with no 1/M division
   (`transformer_impl.forward_backward_batch:654-660`) -- by design: it pre-computes the
   GLOBAL normalizers and injects them into the batch (`batch_num_tokens` DP-all-reduced +
   `dp_size`, `transformer_impl.py:620-627`; `global_batch_size` = global mini ROWS from
   `ray_trainer._update_critic:1339-1351`), expecting the loss fn to divide by them.
2. The actor's `ppo_loss` DOES consume them (`losses.py:65-68` -> `**config.global_batch_info`
   in every `agg_loss` call) -> actor loss is M/DP-invariant.
3. The critic's `value_loss` does NOT (`losses.py:167-173` calls `compute_value_loss` bare;
   its `agg_loss` token-mean then defaults to the LOCAL micro mask sum) -> per-micro local
   token-mean, summed over M micros.
4. verl 0.8's `compute_value_loss` multiplies by 0.5 (`core_algos.py:2121`); the paper fork's
   has NO 0.5 (`fork core_algos.py:542`) and instead divides each micro loss by the
   gradient-accumulation count (`fork dp_critic.py:243: loss = vf_loss / self.gradient_accumulation`).

**Upstream confirmation (added 2026-07-23):** this is verl's OWN bug, not a migration artifact.
The vendored `others/verl` is byte-clean upstream `verl-project/verl` @ `7aed6b2` (2026-06-01;
the only local patch is the 2-line weight-transfer jobid change, different files), the 0.8
engine refactor REMOVED the old `verl/workers/critic/dp_critic.py` (whose
`/gradient_accumulation` was correct), and upstream fixed the regression themselves a month
after our pin: `2eb020a` "[algo] fix: normalize critic value loss over the global mini-batch,
not per micro-batch" (#6957, 2026-07-07) -- the fix passes dp_size/batch_num_tokens/
global_batch_size into the critic's agg_loss "as the actor's ppo_loss does", i.e. the same
mechanism as this overlay's `global_token_paper_coef` mode, but KEEPS the 0.5 (upstream
convention). So even a verl upgrade past #6957 does not give paper parity -- `legacy_exact`
remains necessary -- and it changes what the wrapper receives: rescaling the POST-fix
value_loss would double-correct. `enable_critic_loss_parity` therefore carries a
source-version guard (`_assert_stock_value_loss`) that refuses loudly on a post-#6957
value_loss (locked by `tests/test_ppo_critic_loss.py::test_source_guard_rejects_post_6957_value_loss`).

An independent counter-audit (`bugfix/ppo_critic_normalization_and_recipe_fidelity_audit.md`
in the parent working area) reproduced the same chain and CORRECTED three earlier framings,
all accepted after verification: (a) "effective critic lr 2e-5" is WRONG -- AdamW is
approximately invariant to a constant gradient rescale (m/sqrt(v) cancels c) and grad-clip 1.0
was active on 87.7% (WebShop) / 93.1% (ALFWorld) of the 0915 production critic steps
(median pre-clip norms 12.5 / 29.1), where clip(2g)=clip(g) exactly; (b) therefore "halve the
critic lr" is NOT a valid workaround (it undercorrects when clip is inactive and
overcorrects when it binds); (c) the 0.5 itself is a standard PPO convention, not an upstream
bug -- the fidelity break is the missing normalization plus the coefficient DIFFERENCE vs the
fork.

### The bug (net effect)

Per optimizer step, the migrated critic's pre-clip gradient is `0.5 x M` times the fork's
objective, where `M = critic_mini_per_rank / critic_micro_per_gpu` (= 16/4 = **4** at the
paper recipe, so **2x**), and the scale silently drifts with `critic.ppo_micro_batch_size_per_gpu`,
DP world size, `rollout.n`, and dynamic batching -- the same recipe on a different GPU count
trains a different critic objective.

### Impact

1. **Fidelity/invariance (the real problem):** the critic objective is not the fork's and not
   config-invariant. Any PPO-vs-paper comparison carries an uncontrolled critic-scale factor.
2. **Bounded practical damage at the paper recipe:** with grad-clip 1.0 binding on ~90% of
   critic steps a pure 2x rescale is erased (identical post-clip gradient), and AdamW largely
   cancels a CONSTANT rescale on the rest. The residual effect concentrates in steps near the
   clip threshold and in the token-weighting difference -- real but small; do NOT attribute
   large curve gaps to this alone.
3. **PPO-only, env-symmetric:** cannot explain any WebShop-vs-ALFWorld asymmetry.

### The fix

`fedagent/ppo_critic_loss.py` wraps `verl.workers.utils.losses.value_loss` and rescales its
(0.5 x local-token-mean) per-micro output so the engine's Sigma-backward + FSDP DP-mean
reproduces an explicit, named contract (`critic_loss_mode` in run_fed DEFAULTS):

- **`legacy_exact` (DEFAULT):** `scale = 2 * dp_size * micro_rows / global_batch_size`
  == `2/M` for equal-size micros (paper configs divide evenly) -> per-micro loss
  `micro_token_mean / M`, i.e. the fork's `Sigma micro_mean / M`, coefficient 1.0. For
  strict old-run comparability.
- **`global_token_paper_coef`:** `scale = 2 * dp_size * micro_tokens / batch_num_tokens` ->
  a true global-token-mean (micro/DP-invariant even for unequal micros), coefficient 1.0.
  NOT byte-equivalent to the fork under unequal micro token counts -- a named alternative,
  not the default.
- **`upstream_standard`:** no patch (stock 0.5 x per-micro), for A/B forensics.

Armed exactly like FedProx: run_fed sets `FEDAGENT_CRITIC_LOSS_MODE` ONLY for gae clients
(both subprocess and persistent builders); the repo-root `sitecustomize` installs a deferred
import hook on `verl.workers.utils.losses` (so torch is never imported before Ray assigns
per-rank CUDA devices) and the rebind lands in every process BEFORE `ray_trainer`'s lazy
`from ... import value_loss` (init_workers, ray_trainer.py:879) builds the critic worker's
loss_fn partial. Fail-closed: a non-token-mean `loss_agg_mode` or missing engine metadata
raises instead of silently training on the stock scale; `[critic-loss] enabled: mode=...` is
printed for log verification, and `critic/vf_loss_scale` is emitted in the critic metrics.

### Verification

- `tests/test_ppo_critic_loss.py` (6 tests): `legacy_exact` == the fork objective
  (`(1/(dp*M)) Sigma micro_means`, no 0.5) across a 2-rank x 2-micro grid; micro-split
  invariance for equal-row micros; `global_token_paper_coef` == global token-mean under
  UNEQUAL micro token counts and DP splits; the stock/legacy ratio == 0.5*M == 2.0 at the
  paper-recipe shape (dp=4, M=4); fail-closed on non-token-mean mode and on missing engine
  metadata. Full suite 21 passed.
- Live check for the first PPO run: expect `critic/vf_loss_scale = 0.5` per micro at the
  paper recipe (2/M, M=4) and `[critic-loss] enabled: mode=legacy_exact` in every client log.

### Scope note -- the OPEN recipe-provenance decision this fix does NOT cover

The same counter-audit surfaced, and launch artifacts confirm, that the EXECUTED 0915
production recipe (the runs behind the paper numbers) differs from both the 2026-intended
config tree and the migrated tree: PPO ran `train 32 x n1` with mini 4 (WebShop) / 32
(ALFWorld) on 1 GPU; and (found in this session's follow-up) GRPO ran actor mini **16** rows
(WebShop) / **128** rows (ALFWorld) vs the tree's 64 -- i.e. production WebShop GRPO took ~4x
MORE optimizer steps per rollout batch than the migrated recipe while ALFWorld took ~2x FEWER,
an env-ASYMMETRIC delta aligned with the observed "WebShop drops, ALFWorld doesn't". The 0915
WebShop val set was `goals[0:128]` (val_data_size=128), not the migrated `goals[0:64]`.
Which contract to reproduce (executed-0915 vs intended-tree) is a science decision, tracked
outside this entry; do not label the current recipes "A/B-equivalent to the paper runs" until
it is made.

---

## 2026-07-22: Windowed rollout: generation saw up to max_ctx-1 prompt tokens but the emitted training sample was re-cut to prompt_length -> silent head-truncated training context

- **Files:** `fedagent/agent_loops/windowed_agent_loop.py` (generation prompt cap).
- **Severity:** low-frequency science-correctness, windowed mode (the paper default), both
  algos. ALFWorld is the exposed env (mismatch window (2048, 2559]); WebShop is structurally
  safe at paper budgets (the 13000-char NO_HIS guard caps the template at ~3250 tokens <
  prompt_length 4096).

### How it was found

Fresh-eyes residual sweep after the format_obs / epoch-resampling fixes: the loop capped the
GENERATION prompt at `max_ctx - 1` (= max_model_len - 1) but emitted
`prompt_ids[-prompt_length:]` as the training sample. The legacy stack tokenized ONCE at
`max_prompt_length` with `truncation=error` + `filter_overlong_prompts` and used that same
tensor for generation AND training (fork rollout_loop.py:115-137) -- no gen/train mismatch
was possible there.

### The bug

For a windowed prompt of length L with `prompt_length < L <= max_ctx-1`, the policy ACTED on
the full L tokens but the stored sample kept only the last `prompt_length` -- left-truncation
cuts the template HEAD first, i.e. the framing plus "Your task is to: {task}" line -- while
still carrying the full broadcast episode return and per-turn penalty. Training then
optimizes logprobs of an action under a context (task-less) the policy never actually saw.
Reachable on ALFWorld windowed (`prompt_length ~2048`, `max_ctx-1 = 2559`) for verbose scenes
with long admissible-action lists; rare, but each occurrence is a corrupted training row.

### The fix

Generation now caps the prompt at `min(prompt_length, max_ctx - 1)` -- the SAME budget the
emitted sample keeps -- restoring the legacy single-truncation semantics (gen context ==
train context, always). `prompt_length + response_length <= max_model_len` holds in every
shipped config (4096+512<=4608; 2048+512<=2560), so the server-ctx guard only binds in exotic
configs, where `min()` keeps it. The (now no-op) `[-prompt_length:]` cut on the emitted
sample stays as belt-and-braces.

### Verification

`py_compile` clean; full offline suite 21 passed (the loop itself needs a live vLLM server --
the invariant `len(gen prompt) <= prompt_length == len(emitted prompt)` is enforced by
construction after the one-line cap change). Live check: on the next ALFWorld windowed run,
grep any turn with prompt length > 2048 -- there should be none post-fix.

---

## 2026-07-22: WebShop observation fidelity: the overlay never applied the fed baseline's `format_obs` -> every turn's prompt (and history) diverged from the paper stack (WebShop-only)

- **Files:** `fedagent/envs/webshop/webshop_env.py` (new `_format_obs` + reset/step wiring);
  `fedagent/envs/legacy_prompts.py` (stale "raw obs" docstring); offline regression
  `tests/test_webshop_format_obs.py`.
- **Severity:** science-correctness / paper fidelity, WebShop-only, EVERY turn, train AND eval,
  ALL WebShop arms (uniform, task-het, env-het; GRPO and PPO; subprocess and accelerated).
  This is the migration audit's `webshop-env-1` (HIGH, CONFIRMED migration-bug), open since the
  audit shipped.
- **Provenance:** re-surfaced by the 2026-07-22 regression investigation into "WebShop GRPO
  uniform (accelerated recipe) trains well below the paper stack while the ALFWorld twin
  matches"; see "How it was found" below for the elimination chain that left this standing.

### How it was found

The symptom was a differential: qwen2.5-1.5B WebShop GRPO uniform on the accelerated recipe
ended far below the paper number, the ALFWorld twin did not -- and it persisted on latest
main, i.e. AFTER the goal-shuffle-race fixes. The investigation ran as an elimination over
everything that could be WebShop-only or WebShop-sensitive:

1. **Timeline first.** `git reflog` showed the run checkout sat at `b72941a` (07-20) until
   07-22 16:45 -- so every LOCAL webshop artifact predates `afa440f` and is race-era
   (explains those runs: GRPO groups compared 8 unrelated goals). But a re-run on
   latest-of-07-22 code still under-performed, so the race could not be the whole story.
2. **Magnitude calibration.** Original-stack uniform 1.5B GRPO: 0.617@r65 (paper 61.5 +/-
   3.9). Every migrated-stack WebShop 1.5B checkpoint on this machine peaked <= 0.31
   (checkpoint inventory), with `grpo_ws_std4` even regressing 0.297@r54 -> 0.172@r70. A
   2-3x plateau gap is a broken-training-signal signature, not noise.
3. **Eliminated with evidence:** training reward semantics (sparse {0,10} on BOTH stacks
   since `ea85c96` -- `git log -S` on the server's reward hunk); the GRPO grouping chain at
   HEAD (traced dataset row seed -> `repeat(interleave)` copies seed+uid unperturbed
   (ray_trainer.py:1439-1448,1497) -> `/reset` maps seed->goal as a pure function with the
   random `session_id` irrelevant (server.py:404-417) -> startup guard pins identical pool
   goal order -> stock grpo groups by uid: same-goal groups CONFIRMED); the acceleration
   layer itself (every knob parameter-level A/B'd <= the 9.3e-5 GPU-nondeterminism floor,
   WebShop GRPO actor 9.8e-6, fused kernels 1.116e-5); the val protocol (both stacks: fixed
   goals[0:64], 64 episodes, temp 0.4 -- the original fed config sets `val_batch_size: 64`,
   so `range(64)` = a permutation of the same fixed set); and an ALFWorld control audit
   (reward, prompts, parsing, mechanics, val: IDENTICAL down to file:line).
4. **What survived was the audit's own `webshop-env-1`.** Cross-checking the migration
   audit's findings register against HEAD showed the finding still open: `grep -rn
   format_obs fedagent/` returns ZERO hits. A counter-claim ("format_obs is a dead no-op,
   there is no divergence") was adjudicated and REFUTED: it had audited `env_manager`'s
   WebShop manager (whose `format_obs` matches `<instruction>` tags that never occur -- a
   genuine no-op), but the paper's fed entrypoint routes `main_ppo_fed.py:95,100 ->
   fed_make_envs -> fed_env_manager.WebshopEnvironmentManager`, whose `format_obs`
   (fed_env_manager.py:387-398) is real and applied at every reset/step. The same
   env_manager-vs-fed_env_manager trap that caused the original migration miss nearly
   survived the re-audit -- worth remembering: for baseline semantics, ALWAYS trace from the
   executed entrypoint, not from the class the code "looks like" it ports.
5. **The asymmetry check closed the loop:** `fed_env_manager.py` class listing shows ONLY
   `WebshopEnvironmentManager` defines `format_obs` (line 387); the ALFWorld manager has no
   formatting hook -- so this bug class CANNOT touch ALFWorld, matching the observed
   "WebShop drops, ALFWorld doesn't" exactly.

### The bug

The paper's federated runs do NOT consume the engine-raw observation. The fed path
(`main_ppo_fed.py:95,100` -> `fed_make_envs` -> `fed_env_manager.WebshopEnvironmentManager`)
post-processes EVERY reset/step observation with `format_obs`
(fed_env_manager.py:345,360-362,387-398): split on `` [SEP] ``, find the segment equal to the
episode's instruction, DROP everything up to and including it, and single-quote each remaining
part -- then stores the FORMATTED text as `pre_text_obs`, so the windowed history entries are
formatted too. Example (reset page):

```
raw:       WebShop [SEP] Instruction: [SEP] i need a gluten free vanilla cake mix ... [SEP] Search
baseline:  'Search'
```

The overlay client instead fed the raw page text into BOTH the current-obs slot and the
history memory (`webshop_env.py` pre-fix lines 145,168,170: `_pre_obs = raw`,
`memory.append({"text_obs": raw_pre_obs, ...})`, `current_obs=raw`).

Root cause of the miss: the overlay replicated `env_manager`'s WebShop manager, whose own
`format_obs` matches `<instruction>...</instruction>` tags that the text engine never emits --
an inert no-op -- and whose `extract_task` therefore falls back to a generic string. The FED
override (`fed_env_manager.py`, the class the paper runs actually executed) both extracts the
real task (`parts[2]` after asserting `parts[1] == "Instruction:"`) and applies the real
formatting. The audit's `agent-loop-rollout-3` / `webshop-env-8` findings verified the
TEMPLATE byte-identical -- the divergence is in the observation STRING fed into that template,
which is exactly what `webshop-env-1` flagged.

### Impact

Ranked by how much it distorts results:

1. **The training-input distribution is not the paper's (policy-learning quality, direction
   unproven but nonzero).** Every windowed prompt carried up to 4 copies of the instruction
   (task line + current obs + 2 history entries; typ. ~30-60 tokens each -> ~100-200 tokens of
   pure repetition per prompt) and lost the quote delimiters around every page part (asins,
   titles, prices, buttons -- ~10-80 parts on results/item pages) that the baseline policy
   conditions on when learning `click[<exact button text>]`. A 1.5B policy trained from the
   same base on this rendering is learning a DIFFERENT (noisier, more redundant) input
   encoding than the one that produced the paper's 61.5%; any WebShop-vs-paper comparison
   conflates algorithm fidelity with this prompt-format shift. ALFWorld is untouched, so the
   published webshop/alfworld asymmetry of the overlay's runs is partly attributable here.
2. **Baseline non-comparability is one-way but total.** Pre-fix WebShop numbers are internally
   consistent (the policy was trained AND evaled on raw rendering -- eval is not OOD for it),
   so pre-fix runs are valid experiments of a DIFFERENT recipe; they are NOT reproductions of
   the paper recipe, and pre/post-fix curves must never share a figure.
3. **The 13000-char NO_HIS fallback boundary moved.** `build_webshop_obs` drops to the
   no-history template above 13000 chars (legacy guard). Formatting shortens the obs by the
   duplicated instruction+prefix but lengthens it by 2 quote chars/part; borderline pages
   therefore flip templates at DIFFERENT points than the baseline -- occasional qualitative
   prompt changes (history silently dropped) at pages where the baseline kept it, and vice
   versa.
4. **Hardness labels measured a different difficulty.** Both label generations to date
   (including the post-race `d6f3c9b` regeneration) rolled the reference out under raw
   rendering; measured per-goal difficulty is difficulty-UNDER-RAW-RENDERING. Post-fix
   training rolls out formatted -- the labels are now a (mildly) mismatched difficulty
   ranking for the hardness arm until regenerated.
5. **Sub-turn plumbing was faithful all along** -- reward {0,10}, invalid-action penalty,
   projection, seeds, val protocol were verbatim (audit B-list) -- so this delta was invisible
   to every equivalence check that compared rewards/weights rather than prompt BYTES; the
   acceleration A/B equivalences (accel vs non-accel, both raw) remain valid.

**Still valid:** same-broken-way A/B comparisons where both sides used raw rendering (all
acceleration A/Bs, PPO-vs-GRPO on the overlay, task-het arm contrasts run entirely pre-fix);
ALFWorld everything; learning-curve SHAPES as qualitative evidence.

### The fix

`_format_obs(raw, task)` is a verbatim replica of the fed original: `parts.index(task)`,
quote-join of `parts[index+1:]`, raw-text fallback on a miss exactly like the original's bare
`except` -- plus one defensive extension, a stripped-comparison retry before the fallback, so
cosmetic whitespace drift between `_extract_task` and the page rendering cannot silently
disable the formatting (the failure mode that made the original no-op invisible). Applied at
reset AND step, BEFORE the text enters the prompt or the history memory, in both windowed and
concat modes (the fed manager formatted regardless of mode); the task is still extracted from
the RAW obs (formatting removes the instruction segment). `FEDAGENT_WEBSHOP_FORMAT_OBS=0`
restores the raw rendering solely to reproduce pre-fix runs for A/B forensics.

### Verification

- `tests/test_webshop_format_obs.py` locks `_extract_task`/`_format_obs` BYTE-FOR-BYTE against
  a verbatim re-implementation of the fed originals (extract_task assert included) over
  landing/results/item pages; plus the landing-page `'Search'` form, the prefix-drop +
  all-parts-quoted invariants, the raw fallback on instruction-free pages, and the
  stripped-match guard. 6 tests green.
- `py_compile` clean; the concat-glue suite (4 tests) unaffected.
- The formatting layer is client-side only: service, engine, reward, and seed->goal mapping
  are byte-untouched (no service restart needed beyond normal redeploy).

**Recipe boundary:** every WebShop artifact produced with the raw rendering -- ALL overlay
training runs to date and both hardness-label generations -- belongs to the pre-fix recipe.
Do not mix pre/post-fix curves; regenerate hardness labels with a reference rolled out under
the fixed rendering before relying on the hardness arm; re-run any WebShop-vs-paper
comparison from scratch on post-fix code.

---

## 2026-07-22: Per-epoch goal resampling: E local epochs replayed the SAME n_envs goals; the original drew fresh goals every epoch (24 -> 8 distinct per client-round at paper E=3)

- **Files:** `fedagent/data/agentic_dataset.py` (E-expansion + widened window guard),
  `fedagent/fed/run_fed.py` (`epoch_resample` DEFAULT + both client cmd builders -- subprocess
  AND persistent/accelerated); offline regression `tests/test_agentic_dataset_epochs.py`.
- **Severity:** science-correctness, both envs, both algos, all E>1 configs in BOTH config
  trees (`config/paper` + `config/paper_accelerated`). This is the migration audit's
  `data-dataset-seeds-6` (medium, REVISED: "E local epochs replay the SAME per-round goal
  batch; original & paper re-sample every epoch").
- **Provenance:** same 2026-07-22 regression investigation as the entry above. Mechanically
  env-agnostic, but the observed damage is WebShop-shaped (see Impact #2): ALFWorld's
  accelerated twin matched the paper regardless, WebShop's did not.

### How it was found

Second survivor of the same elimination chain (see the entry above for what was ruled out).
The audit register carried `data-dataset-seeds-6` as a REVISED medium ("E local epochs replay
the SAME per-round goal batch; original & paper re-sample every epoch"); the regression
investigation re-confirmed it live at HEAD by reading the three links of the chain in the
CURRENT code rather than trusting the register's snapshot:

1. `AgenticDataset` materializes rows ONCE per process with seeds `base*100_000 + si*1_000
   + i` -- nothing epoch-dependent (agentic_dataset.py).
2. run_fed threads `FEDAGENT_BASE_SEED = base + round*100 + client` -- per (round, client),
   NOT per epoch; its own comment claims parity with "the ORIGINAL fed sampler" but the
   original's freshness was per-RESET, i.e. per epoch (envs.py:677), not per round.
3. verl 0.8's fit loop is `for epoch in range(total_epochs): for batch in train_dataloader`
   (ray_trainer.py:1422) over a StatefulDataLoader with no `set_epoch` hook anywhere -- the
   dataset rows are provably identical across epochs.

The WebShop-vs-ALFWorld asymmetry initially argued AGAINST this bug mattering (it hits both
envs mechanically). What re-ranked it upward was the reward-density interaction surfaced by
the grouping audit: WebShop's strict perfect-match `{0,10}` reward makes all-zero GRPO groups
(zero gradient) the COMMON case early, so goal variety is precisely WebShop's scarcest
resource -- ALFWorld's mixed-outcome groups make the same 3x variety cut nearly free. An
env-symmetric mechanism with an env-asymmetric cost matched the symptom where every
env-symmetric-cost hypothesis had already been eliminated.

### The bug

The original fed sampler drew a FRESH goal batch at every local epoch: WebShop
`WebshopMultiProcessEnv.reset` -> `self._rng.choice(self.goal_idxs, size=env_num,
replace=False)` on a persistent, advancing `RandomState` (envs.py:677 -- each epoch's single
step calls `reset()` once, so E=3 epochs = 3 independent 8-of-shard draws, up to 24 distinct
goals per client-round); ALFWorld advanced its shuffled game iterator across resets, same
effect. The overlay's `AgenticDataset` materializes FIXED rows at construction (seed
`base*100_000 + si*1_000 + i`, agentic_dataset.py) with `FEDAGENT_BASE_SEED` threaded per
(round, client) but NOT per epoch (run_fed.py), and verl's dataloader re-iterates the SAME
rows every epoch (`for epoch in range(total_epochs): for batch in train_dataloader`,
ray_trainer.py:1422) -- so all E epochs replayed the same n_envs goals: 8 distinct, each
trained 3x, per client-round.

### Impact

Ranked by how much it distorts results:

1. **Per-round task exposure shrank 3x at paper settings.** GRPO: 8 distinct goals/round
   instead of ~24 (3 draws of 8 without replacement within a draw); PPO: 64 instead of ~192.
   Across a 70-round, M=2 run that is ~1,120 goal-draws instead of ~3,360 over the 6,410-goal
   train pool -- a client selected k times covers <= 8k of its 100-goal shard instead of
   <= 24k. The optimizer takes the SAME number of steps but sees a third of the task variety;
   steps 2-3 of every round partially memorize the round's 8 goals rather than sampling the
   shard, which is not the paper's sampling process (main.tex documents per-epoch
   re-sampling; the run_fed comment even says "re-draws goals ... every round (covering the
   shard over T rounds)" -- true per round, silently false per epoch).
2. **WebShop's GRPO bootstrap is disproportionately sensitive (the asymmetry channel).**
   WebShop's training reward is the strict perfect-match `{0,10}` (score==1.0 only) and the
   1.5B zero-shot success is ~0-3%, so early groups are usually ALL-ZERO -- and an all-zero
   group has zero within-group variance, hence ZERO GRPO gradient (`(score-mean)/(std+eps)`).
   The learning signal per round is roughly (number of distinct goals with at least one
   success in 8 tries); cutting distinct goals 24 -> 8 cuts the expected number of informative
   groups per round ~3x exactly in the regime where they are rarest. If the round's 8 goals
   happen to all be too hard for the current policy, the ENTIRE round (all 3 steps) yields ~no
   gradient -- the original would have re-rolled goals twice more within that round. ALFWorld
   (~10-30% early success, mixed groups almost surely) barely notices the same reduction --
   matching the observed "WebShop drops, ALFWorld doesn't".
3. **Curriculum/coverage semantics of small shards drifted.** `min_goals_per_client=100`-style
   arms were designed as repetition over a fixed 100-goal shard with fresh per-epoch draws;
   the replay made within-round repetition deterministic (same 8, 3x) and cross-round coverage
   slower, subtly changing the effective curriculum of every heterogeneity arm on both envs.
4. **No bias, no leakage.** The replayed goals are drawn from the correct shard with the
   correct round-threaded seed; val is untouched. This is a variance/coverage distortion, not
   a correctness break -- aggregate metrics remain unbiased for the recipe that ran, they are
   just not the paper's recipe.

**Still valid:** E=1 configs (byte-identical rows); the eval protocol and every val metric;
same-broken-way A/B comparisons (both sides replayed identically -- all acceleration A/Bs);
FedAvg/aggregation numerics.

### The fix

`epoch_resample` (run_fed DEFAULTS, **true**): both client builders (subprocess
`_train_client` and persistent `_persistent_cmd_env`) launch clients with
`trainer.total_epochs=1` + env `FEDAGENT_DATA_EPOCHS=E` +
`FEDAGENT_DATA_EPOCHS_FILE=<train spec>`, and `AgenticDataset` emits each spec's rows E times
with a DISTINCT per-epoch-slot seed:

```
seed = base*100_000 + si*1_000 + e*n_envs + i     # e = epoch slot, epoch-major order
```

Same optimizer-step count (E x n_envs / train_batch_size: GRPO 3x8/8=3 steps, PPO 3x64/64=3
steps; `total_training_steps = len(dl)*total_epochs` is invariant), fresh goals per epoch
slot -- the original sampler's semantics, now deterministic and resume-safe. Design points:

- `e=0` reproduces the historical layout exactly, so E=1/unset runs are BYTE-identical
  (no behavior change outside E>1 federated clients).
- The `_FILE` guard is load-bearing in the persistent (accelerated) path: the worker builds
  its worker-eval dataloader (`FEDAGENT_WORKER_EVAL` = the VAL spec) in the SAME process, and
  the val set must stay exactly n_envs episodes; expansion is confined to the file it names.
  (The client's own `data.val_files` equals the train spec in both builders; that dataset is
  never iterated -- `val_before_train=false`, `test_freq` pinned -- so its expansion is inert.)
- The cross-spec seed-window guard now checks the EXPANDED row count
  (`n_envs * E > 1000` with later specs present refuses loudly, was `n_envs > 1000`).
- Goal identity within a group is untouched: the GRPO group is still verl's `rollout.n`
  repeats of ONE row (one seed -> one goal), so the same-goal-per-group invariant verified in
  the shuffle-race entry holds verbatim.
- `epoch_resample: false` reproduces the pre-fix replay behavior for old-run forensics.

### Verification

- `tests/test_agentic_dataset_epochs.py` (5 tests green) locks: the expanded layout (E=3,
  n_envs=4 -> 12 rows, `e*4+i` stride, epoch-major, all seeds distinct), the E=1/unset no-op
  (byte-identical seeds), the `_FILE` guard (a sibling val spec stays at n_envs), and the
  widened window guard (400 x 3 > 1000 -> ValueError).
- Helper smoke on real DEFAULTS: E=3 default -> (resample on, trainer_epochs 1);
  `epoch_resample: false` -> (off, 3); E=1 -> (off, 1).
- grep-verified: the two client builders are the ONLY producers of `trainer.total_epochs`, so
  eval/final-eval/aggregation invocations are untouched; `FEDAGENT_DATA_EPOCHS` is consumed
  only by `AgenticDataset`.
- Step-count math: `len(dataloader)` grows E-fold while `total_epochs` drops to 1 -- LR
  schedule (constant), save cadence (`save_freq=100000` -> round-last step), and
  `total_training_steps=null` handling are all invariant.

**Recipe boundary:** for E>1 runs the served goal stream changes vs pre-fix main -- that is
the point (it restores the original sampler's semantics). Pre-fix curves are a different
recipe; do not resume a pre-fix run into post-fix code mid-figure. Old behavior remains one
switch away (`epoch_resample: false`) for exact reproduction of past runs.

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
  (manager included) is the one to sync. **2026-07-29 update:** two production FedProx runs
  (GRPO r11 / PPO r8) showed the curve was NOT yet flat — this fix's own `empty_cache()` ran
  while the FedProx anchor was still live (left out of the release-before-rebuild set), and
  the crash trigger was a separate awake-engine sync ceiling; both closed in the 2026-07-29
  entry above, which inherits this pending flat-memory validation.

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
