# `envs/harbor/`: Harbor tasks as FedAgent episodes

[Harbor](https://github.com/harbor-framework/harbor) packages an agent task as an instruction, a
sandbox and a verifier, and runs it on a pluggable sandbox backend. `HarborEnv` makes any
directory of Harbor tasks a FedAgent environment: **one episode is one Harbor trial**. The
sandbox lifecycle, the task's timeouts and its verifier all go through Harbor's own trial path;
FedAgent keeps what it already owns, the turn loop and the policy.

Unlike WebShop and ALFWorld this env has **no service**: Harbor installs into the trainer env
(both need Python ≥ 3.12) and the env runs in-process, like TinyGuess. It also has **no
heterogeneity arms**: every client draws from the same task pool, so use it with
`partition_strategy: ""`.

## How one episode runs

```
agent loop (trainer)                HarborEnv                         Harbor Trial
────────────────────                ─────────                         ────────────
env.reset(seed)        ─────────►   pick_task(seed)
                                    Trial.create + trial.run()  ───►  start sandbox
                                                                      BridgeAgent.run(instruction)
                       ◄─────────   instruction (+ working dir) ◄───  ("ready", instruction)
generate a turn
env.step(text)         ─────────►   parse <action>…</action>
                                    (command, timeout)          ───►  environment.exec(command)
                       ◄─────────   "[exit code 0]\n…"          ◄───  ("result", ExecResult)
        … up to max_turns …
env.step("…submit…")   ─────────►   None                        ───►  BridgeAgent.run() returns
                                                                      task verifier -> reward
                       ◄─────────   reward, done=True           ◄───  TrialResult, sandbox torn down
```

Harbor expects the *agent* to own the turn loop; FedAgent's agent-loop has to own it, because it
generates every token on the policy being trained (that is where the exact token ids, log-probs
and response mask come from). [`bridge.py`](bridge.py) resolves this: `BridgeAgent` is a Harbor
external agent whose `run()` executes whatever command `HarborEnv.step()` sends and returns when
told the episode is over, which hands control back to Harbor for verification. This is the
"trainer owns the loop" pattern; no model-API capture proxy is involved, and Harbor's installed
agents (Claude Code, OpenHands, …) are out of scope here.

## The harness

One tool, bash. Each turn the model reasons in `<think> </think>` and gives one action in
`<action> </action>`: a shell command, or the word `submit`. The prompt tells it to submit as
soon as the task is done and that an identical consecutive command is not run again.

- **Parsing** ([`parse_action`](harbor_env.py)): the first `<action>` block counts; its case is
  preserved; a ``` fence around the command is tolerated. No block (or an empty one) is an
  **invalid action**: nothing runs, the turn is spent, and `info["is_action_valid"]` is False (the
  agent loops apply their usual invalid-action penalty on train rows).
- **Repeats** (`refuse_repeats`, on by default): a command identical to the previous one is
  not executed and counts as an invalid action. In a fresh shell per command a repeat does
  nothing new, but it is the cheapest way to use up a turn, and without this rule that is what
  the policy learns (see Verification).
- **Execution** ([`wrap_command`](harbor_env.py)): every command runs in a fresh
  non-interactive bash in the task's working directory. The runtime is bounded *inside* the
  sandbox (`exec_timeout_sec`) and the output is capped at the source; the observation is
  `[exit code N]` plus the output, clipped to `max_obs_chars` (head and tail kept).
- **Grading**: on `submit`, or on the env's last turn if the model never submits, Harbor runs
  the task's verifier. Its reward, scaled by `reward_scale` less `turn_cost` per turn used, is
  the episode reward (0 on every earlier turn) and `info["success"]` is
  `reward >= success_threshold` on the verifier's own scale. `info["task_score"]`
  carries the raw reward (meaningful for partial-credit verifiers) and `info["goal_id"]` the
  task name, so validation dumps can be broken down per task. An episode the agent loop
  abandons before either point is never graded and counts as a failure, as with the other envs;
  the concat loop does this when its context or response budget runs out mid-episode.

Both rollout modes work from one spec. In **concat** mode (`history_length: 0`) the observation
is the command output and the loop keeps the chat; in **windowed** mode the env renders the format
rules, the task and the last `history_length` (action, output) pairs every turn.
`FEDAGENT_HISTORY_LENGTH`, which `run_fed` sets from `rollout_mode` / `windowed_history_length`,
overrides the spec value, as for the other envs.

## Config (`config:` of the env-spec entry)

| key | default | meaning |
|---|---|---|
| `tasks_dir` | required | directory whose sub-directories are Harbor tasks (`<task>/task.toml`); relative paths resolve against the repo root |
| `task_slice` | all | `[start, end)` over the name-sorted task list, e.g. to carve a val split out of one directory |
| `sampling` | `shuffle` | seed → task map. `shuffle`: each (round, client) draws a fresh permutation, so its rows are distinct tasks. `sequential`: `tasks[seed % N]`, the fixed sweep a val spec wants |
| `max_turns` | `8` | the env's turn budget; **must equal the spec-level `max_turns`** (write `${..max_turns}`): the verifier runs on the env's last turn, so a larger value here ends every episode ungraded |
| `history_length` | `0` | windowed-mode history (see above) |
| `environment_type` | `docker` | Harbor sandbox backend (`docker`, `daytona`, `modal`, `e2b`, …); `environment_kwargs` is passed through to it |
| `exec_timeout_sec` | `30` | per-command limit |
| `max_obs_chars` | `2000` | per-observation clip |
| `agent_timeout_sec` | task's own | overrides the task's `[agent] timeout_sec`, the wall-clock budget of the whole agent phase; `timeout_multiplier` scales all of the task's timeouts instead |
| `reward_key` | `reward` | which entry of the verifier's rewards is the episode reward (a lone other key is accepted) |
| `success_threshold` | `1.0` | `success = reward >= threshold` |
| `reward_scale` | `1.0` | multiplies the verifier's reward into the episode reward; `success` and `task_score` stay on the verifier's scale. The shipped specs use `10.0`, the WebShop/ALFWorld `{0, 10}` convention the 0.1 invalid-action penalty is sized for |
| `turn_cost` | `0.0` | discount on a graded reward: `reward = score * (reward_scale - turn_cost * turns)`. Makes submitting a solved task pay; a failed episode pays 0 at any length. The train spec uses `0.2` |
| `refuse_repeats` | `true` | do not execute a command identical to the previous one; flag the turn as an invalid action |
| `start_retries` / `start_concurrency` | `2` / `8` | retries of a failed sandbox start; simultaneous sandbox starts per agent-loop worker |
| `fast_stop` | `true` | docker/podman only: stop the sandbox without the 10 s grace period ([`compose_fast_stop.yaml`](compose_fast_stop.yaml)) |
| `shared_network` | `true` | docker only: sandboxes of plain tasks join one pre-created network, `fedagent-harbor`, instead of one network each ([`compose_shared_network.yaml`](compose_shared_network.yaml)). A stock Docker daemon has address space for about 30 networks, fewer than one generation batch. The network is created with inter-container traffic disabled, so sandboxes still cannot reach each other. Tasks with their own `docker-compose.yaml` or a non-public network policy keep Harbor's per-trial network |
| `keep_trials` / `trials_dir` | `false` / `$FEDAGENT_HARBOR_TRIALS_DIR` or `<tmp>/fedagent_harbor_trials` | Harbor's per-trial directory (logs, verifier output) is deleted after the episode unless kept |

Single-step tasks only: a task with `[[steps]]` is refused at `reset`.

## Running it

```bash
pip install harbor                               # into the trainer env; needs a reachable Docker daemon
python tools/gen_harbor_shell_tasks.py           # the example suite -> data/harbor/shell_v1/{train,val}
python -m fedagent.fed.run_fed --config fedagent/config/examples/harbor/smoke.yaml --n-gpus 1
```

`env_kind: harbor` in the federated config selects the in-process path (the runner starts no
service for any `env_kind` other than `webshop` / `alfworld`). The example suite is 24 families
of short shell tasks (288 train + 144 val) that share one stock image and carry their inputs in
`environment/`, so a sandbox is up in about a second; it exists to exercise the env, the way
TinyGuess exercises the wiring. To train on another Harbor dataset, download it
(`harbor download <dataset>`) and point `tasks_dir` at it.

**Cost of the sandboxes.** With the example suite (prebuilt image, no build) a sandbox is up in
1-2 s and graded and torn down in about 1 s when run alone. Under a full batch the container
runtime is the bottleneck, not the GPU: 72 concurrent episodes spend about 35 s in sandbox
start + verify + teardown on an 8-core host. Budget CPU cores for the batch size you run.

**Host memory.** The env itself is light (an agent-loop worker with Harbor loaded is about
0.8 GB). The trainer is not: see the header of
[`examples/harbor/shell_1gpu.yaml`](../../config/examples/harbor/shell_1gpu.yaml) for what a
1.5B cross-round run holds and how to fit it on a 62 GB host.

**Tested scope.** Developed and verified against harbor 0.23.0 with the `docker` backend
(Docker 29, compose v5). Other `environment_type` values are handed to Harbor as is and have
not been exercised here; `fast_stop` applies to docker/podman and `shared_network` to docker only.

Harbor's own per-phase DEBUG logging is kept off the trainer's console
(`FEDAGENT_HARBOR_LOG_LEVEL`, default `WARNING`); with `keep_trials: true` each trial's
`trial.log` still has all of it.

Two failure modes are reported rather than scored silently: a sandbox that does not start is
retried and then raises, and an episode whose verifier produced no usable reward is scored 0
**with a warning**. A killed trainer can leave sandboxes behind; they are named
`fedagent-<id>__env-*`:

```bash
docker ps -aq --filter name=fedagent- | xargs -r docker rm -f
```

## Verification (2026-10-06)

One RTX PRO 6000 (96 GB), 8 CPU cores, 62 GB RAM, harbor 0.23.0, Docker 29; trainer stack as in
[`docs/installation.md`](../../docs/installation.md).

| check | result |
|---|---|
| Example suite through Harbor itself | `harbor run -a oracle`: reward 1.0 on all 288 train + 144 val tasks; `-a nop`: 0.0 on all 144 val tasks |
| Third-party tasks through `HarborEnv` | `hello-world@1.0` and four `terminal-bench-sample@2.0` tasks (`regex-log`, `log-summary-date-ranges`, `fix-code-vulnerability`, `polyglot-c-py`): the task's own `solution/solve.sh` as the one command scores 1.0, an untouched sandbox 0.0 |
| Federated smoke ([`examples/harbor/smoke.yaml`](../../config/examples/harbor/smoke.yaml)) | loop closed: 2 clients × 2 rounds of train → FedAvg → merge → val; no sandbox or trial directory left behind |
| Zero-shot success, 144 val tasks, val temperature 0.4, one pass each | windowed (history 5): Qwen2.5-0.5B 9.7% · 1.5B 31.2% · 3B 59.0%. concat: 1.5B 26.4% · 3B 62.5% |
| Signal run ([`examples/harbor/shell_1gpu.yaml`](../../config/examples/harbor/shell_1gpu.yaml)) | Qwen2.5-1.5B, 10 rounds × 2 clients × 3 local steps × 64 episodes: **val success 31.2% → 81.2%** on the 144 held-out tasks; loop closed in 113 min, no restart, no infra warning |

### The signal run, and why the harness has two shaping rules

Federated GRPO with the windowed rollout, `actor.optim.lr=3e-6` (3× the base config: a
10-round budget, not the paper's 70). Val success per round (144 tasks, temperature 0.4):

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped harness (%) | 31.2 | 48.6 | 50.7 | 52.8 | 65.3 | 60.4 | 63.9 | 69.4 | 68.1 | 70.1 | 81.2 |
| ... episodes ending in `submit` (of 144) | 29 | 43 | 49 | 53 | 52 | 88 | 67 | 90 | 123 | 119 | 122 |
| first harness (%) | 32.6 | 26.4 | 29.2 | 43.1 | 43.8 | 49.3 | 61.8 | | | | |
| ... episodes ending in `submit` (of 144) | 23 | 8 | 10 | 14 | 10 | 9 | 7 | | | | |

The first harness had neither `refuse_repeats` nor `turn_cost`, and its prompt did not ask for a
prompt `submit`. It learned (stopped after round 6, once the picture was clear), but not what was
intended:

- **It unlearned `submit` in one round.** The only signal `submit` got was the penalty for a
  premature one; a solved task paid the same with or without it.
- **It learned to repeat its last command until the turns ran out.** After a one-command task
  is solved, repeating that command is harmless, and windowed credit pays every turn of a
  successful episode. On two-step tasks the habit is fatal (`mkdir -p out` six times): the
  one-command families rose to 6/6 on val while `copy-file` went 5/6 → 0/6, `tail-lines` and
  `sort-numeric` 3/6 → 0/6.

With the shipped harness the same schedule ends at 18 of the 24 families at 6/6, including those
three; refused repeats in the val episodes fall from 214 to 51 and the mean episode length from
5.5 to 3.3 turns. `turn_cost` alone did not hold `submit`: a run with it and the prompt change
but without `refuse_repeats` went from 24 submits to 4 in its first round and was stopped there
(the cost acts only where a whole group succeeds, which is rare early on; the per-turn
invalid-action penalty lands on the repeat itself, from the first step). The rule has a price:
zero-shot, a refused repeat makes Qwen2.5-1.5B try *something else*, sometimes destructive, and
its untrained success was lower with the rule than without (31% vs 45%, one val pass each).

A single val pass is noisy: on an earlier 72-task val split, two passes over near-identical
models differed on 22 tasks (11 points of success rate), which is why the shipped split has 144.
Read differences of a few points between two rounds accordingly.

Tests: [`tests/test_harbor_env.py`](../../../tests/test_harbor_env.py) (no sandbox needed;
`FEDAGENT_HARBOR_DOCKER_TEST=1` adds one real trial) and
[`tests/test_gen_harbor_shell_tasks.py`](../../../tests/test_gen_harbor_shell_tasks.py) (every
task family: unsolved fails, the reference solution passes).
