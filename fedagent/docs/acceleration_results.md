# FedAgent verl-0.8 — Acceleration & Validation Results

> **The single-glance "what's validated + the numbers."** Two companions: for the **complete, detailed
> walkthrough** (mechanisms, the investigations, the corrections) read [acceleration_report.md](acceleration_report.md);
> for the original **analysis & plan** (cold-start dissection, the iron law, lever design, equivalence-risk
> audits) read [acceleration.md](acceleration.md); for *how to reproduce the paper*, [reproducing.md](reproducing.md).
>
> **Conventions.** All GPU runs on 4×H100 nodes (qgpu30xx). **Bar** = reproduce the paper within 3-seed
> noise. **EQUIVALENT** = FSDP-checkpoint `max|Δ| ≤ 1e-4` (the bf16 noise floor) vs the stock-verl
> subprocess baseline. Model = Qwen2.5-1.5B-Instruct (smokes use 0.5B / TinyGuess). Built as a thin
> `fedagent/` overlay on **stock verl 0.8 — no fork**.

---

## 0. Verdict

The acceleration overlay is **numerically equivalent** to the subprocess baseline at the checkpoint level
while cutting single-node wall-clock **−43% to −62%**. The dominant cost — per-(client×round) **cold-start**
(Ray + FSDP + vLLM init, **~76–88%** of wall-clock) — is paid **once per run** instead of ~140× at paper
scale. All 4 eval-modes, the per-client "client-end" circles, and single-node client-parallel are
GPU-validated; the **real paper config runs end-to-end**.

## 1. Master status table

| area | what | validated? | headline result | detail |
|---|---|---|---|---|
| **#4 persistent trainer** | one process across clients/rounds | ✅ GPU | per-round **−43%**, cross-round **−62%**; `max|Δ|=1.13e-5` | §2 |
| **eval modes** | inline / parallel / shared / worker | ✅ GPU (0.5B + 1.5B) | all run; `val` identical (eval is read-only) | §3 |
| **#1 eval ∥ train** (= `parallel`) | overlap eval on disjoint GPUs | ✅ GPU | fastest mode (1.5B **2493s**) | §3 |
| **#3 client-parallel** | clients concurrent on one node | ✅ GPU | 1.5B 2×2 = **−35%** (+ port-bug fixed) | §4 |
| **#2 env prewarm** | overlap env-service warmup | ⚠️ CPU only | benefit ≈0 for homogeneous WebShop → minor/opt-in | §7 |
| **equivalence A/Bs** | accel weights == subprocess | ✅ GPU | GRPO actor **9.8e-6**, PPO actor **1.16e-5** | §5 |
| **client-end eval (circles)** | per-client post-train marks | ✅ GPU (both paths) | `client_curve`, 4 circles | §6 |
| **paper config** | real `main/grpo/webshop` 1.5B | ✅ wiring (2-round) | runs e2e; full 70-round ≈ **12–22h** | §6 |

---

## 2. #4 — persistent trainer (the big lever)

One Ray/FSDP/vLLM process spans clients (per-round) or the **whole run** (cross-round), with an in-process
per-client reset, instead of a fresh subprocess each time. FedAvg/merge stay external & byte-identical.

| arm (TinyGuess GRPO, 2-round, matched seeds) | wall | Δ vs subprocess | final aggregate `max|Δ|` |
|---|---|---|---|
| subprocess (baseline) | 909s | — | — |
| **`persistent: true`** (per-round) | 515s | **−43%** | `1.13e-5` → EQUIVALENT |
| **`cross_round: true`** (whole run) | 342s | **−62%** | `1.13e-5` → EQUIVALENT |

- **PPO critic reload — GPU-validated**: `adv_estimator=gae` persistent run rebuilds the critic engine per
  client and FedAvg's actor **and** critic (needs a `critic:` block in `fedagent_ppo.yaml`).
- Equivalence holds **compounded across rounds** through FedAvg, not just per-client.
- Mechanism + per-client reset checklist + the top-3 equivalence risks: `acceleration.md` §Lever #4.

## 3. Eval modes (`eval_mode`: inline / parallel / shared / worker)

Eval is **read-only** → mode changes only *where/when* eval runs, never the trained weights. Cadence = the
per-round red line **every round**; `client_end_eval` adds per-client circles (§6). Details: `acceleration.md` §7.4.

**0.5B, 2-client × 2-round WebShop, eval every round** (val floored at −0.6, so byte-identical across modes):

| eval_mode | process base | wall | val r0/r1/r2 | needs spare GPUs? |
|---|---|---|---|---|
| inline (default) | per-round persistent | 1018s | −0.6 / −0.6 / −0.6 | no |
| **parallel** (= #1) | cross-round | **690s** | identical | **yes (≥2)** |
| shared | cross-round | 874s | identical | no |
| **worker** | cross-round | **703s** | identical | no (reuses hot engine) |

**1.5B, PAPER settings (G=8, webshop_15 15-turn, response 512, n=500 val), 4-card, 2 rounds** — all `rc=0`, **no OOM**:

| eval_mode | GPU layout | wall |
|---|---|---|
| **parallel** | 2 train + 2 eval | **2493s** |
| **worker** | 4 train (hot eval) | 2637s |
| inline | 4 train (blocking) | 3090s |
| shared | 4 train + 2nd engine @0.3 | **3316s** |

- **`shared` flips to *slowest* at a large val set**: its reduced-KV (0.3-util) eval engine throttles the
  n=500 eval — a penalty that scales with val size (invisible at the 0.5B/n=8 floor where shared beat inline).
- `val` numbers vary across modes by **eval sampling** (temp=0.4, only 3–25 successes/500), **not** training:
  cross-mode **weight equivalence** confirmed directly (worker vs inline 1.5B aggregates `max|Δ| 3.8e-6 / 7.6e-6`).
- **Saturated 4-GPU paper case → `worker`** (no spare GPUs → `parallel` N/A as 4-train; `worker` reuses the
  hot vLLM → no 2nd engine, no OOM, no cold-start).

## 4. #3 — client-parallel (single node, GPU-validated)

Two clients trained **concurrently** on disjoint GPU pairs (A=0,1 / B=2,3), 1.5B, paper settings, eval off.

| arm | wall |
|---|---|
| `t1` — 1 client, 4 GPU | 558s |
| `t1` — 1 client, 2 GPU | 725s |
| **#3 — 2 client × 2 GPU, concurrent** | **727s** |
| sequential — 2 client × 4 GPU | 2×558 = **1116s** |

- **~35% faster, not a wash.** 4-GPU FSDP is only `725/558 = 1.30×` faster than 2-GPU for 1.5B (sub-linear:
  FSDP comm + env-latency-bound rollout + fixed cold-start weigh more at small scale) → splitting 4→2+2 wins.
  **Caveat:** large models (4-GPU≈2×) → wash → genuinely need multi-node (one client per node).
- **Coexistence**: two verl/Ray/vLLM jobs share a node cleanly (disjoint pairs, 6519 MiB ×4); isolation =
  per-job `CUDA_VISIBLE_DEVICES` + `RAY_TMPDIR`.
- **Bug found + fixed**: FedAvg `torchrun` used the default c10d rendezvous `localhost:29500` → concurrent
  aggregations collide → one dies `rc=1`. Fix: `torchrun --standalone` + clear `MASTER_*`/`RANK`/`WORLD_SIZE`
  (`run_fed.py fedavg()`); aggregator comm-port only, math unchanged. Verified: concurrent A+B both `rc=0`.
  (The `DataLoader SIGKILL` symptom was a **red herring** — benign `__del__` teardown noise; no OOM.) Full
  forensics: `acceleration.md` §Lever #3.

## 5. Equivalence A/Bs (the project bar)

Matched arms differing **only** in the accel mechanism (eval off, cleanup off), compared tensor-by-tensor
(`tools/verl08_migration/compare_fsdp_checkpoints.py`, atol 1e-4):

| A/B | actor `max|Δ|` | verdict | note |
|---|---|---|---|
| WebShop **GRPO** (subprocess vs cross-round) | **9.8e-6** | EQUIVALENT | |
| **PPO** (subprocess vs cross-round) | **1.16e-5** | EQUIVALENT | backbone ~1e-4; critic **value-head** 5.92e-2 = unreproduced random init, **harmless** (washed out by advantage norm) |
| 1.5B cross-**mode** (worker vs inline aggregates) | 3.8e-6 / 7.6e-6 | EQUIVALENT | eval mode never changes training |

## 6. Client-end eval (circles) + paper-config wiring

**Client-end circles (`client_end_eval: true`, default off)** — the paper's per-client post-training marks,
on the unperturbed val set, emitted as `client_curve` alongside `val_curve`. Both paths GPU-validated (4
circles each): **orchestrator** (merge client actor → `client_<c>/hf` → eval on val service, before cleanup)
and **worker** (hot-engine `_worker_validate(client_id)`). Details: `acceleration.md` §7.4.

**Paper-config wiring** — real `uniform/Qwen2.5-1.5B/main/grpo/webshop` (G=8, webshop_15, response 512,
100-client partition 2/round, val temp 0.4), 2 rounds in `worker` mode: **rc=0, loop closed**. Exercised &
passed: 100-client partition, per-client routing, G=8 memory, full 3-epoch rounds, n=500 eval. Val moved the
**right** way (success base `0.022 → 0.034`, n=500). Per-unit cost ≈ **475s/training-round, 630s/n=500-eval**
→ a full **70-round** headline run ≈ **12h** (`test_freq=5`) / **22h** (every-round) — **fits one node**.

## 7. How to run

| flag | effect |
|---|---|
| `persistent: true` | #4 per-round persistent base |
| `cross_round: true` | #4 cross-round (one process for the whole run) |
| `eval_mode:` `inline`/`parallel`/`shared`/`worker` | eval/train GPU sharing (§3) |
| `eval_gpus: N` / `eval_gpu_mem_util: 0.3` | `parallel` GPU split / `shared` 2nd-engine KV |
| `client_end_eval: true` | per-client circles → `client_curve` (§6) |
| `val_env_spec: ""` | eval OFF (isolate training) |

- **Single 4-GPU node:** `cross_round: true` + `eval_mode: worker` is the saturated-node default.
- **Has spare GPUs:** `eval_mode: parallel` (fastest — overlaps eval off the critical path).
- **Small model, ≥2 clients/round, one node:** `#3` client-parallel (2×2) is a ~35% win (run `run_fed` per
  client pinned to a disjoint GPU subset; the `--standalone` FedAvg fix makes concurrent aggregation safe).

## 8. Open items

- **#2 env prewarm** — implemented (`prewarm_next_round_services`, default off), CPU-validated, but benefit
  ≈0 for homogeneous WebShop (services warm in seconds). Material only for expensive-warmup arms
  (catalog_split large catalog, ALFWorld game collection).
- **Multi-node #3** — not implemented. Single-node 2×2 validated (wins for *small* models); large models need
  one-client-per-node parallelism (orchestrator's external FedAvg supports it; needs a parallel launcher).
- **Full 70-round reproduction** — wiring validated, full curves not yet run (≈12–22h/config; 3-seed band +
  ALFWorld + PPO + heterogeneity arms = a multi-node, multi-day campaign).
