# `agent_loops/`: multi-turn text rollout on verl 0.8's native agent-loop

FedAgent's rollout seam. Two `verl.experimental.agent_loop.AgentLoopBase`
subclasses drive a multi-turn text episode of the policy against one
[`BaseTextEnv`](../envs/base.py) instance per dataset row: **`GymTextAgentLoop`**
(concat, the whole trajectory as one verl `AgentLoopOutput`) and
**`WindowedGymTextAgentLoop`** (the paper's sliding per-turn window, one training
sample per turn, the **default**: `rollout_mode: windowed`, injected together with
`WindowedAgentLoopManager`). This is a thin overlay: it
plugs into verl's **stock** PPO/GRPO trainer and **native** async agent-loop
rollout; there is no trainer fork (see [`../README.md`](../README.md)).

```
fedagent/agent_loops/
├── __init__.py                 # package marker
├── gym_text_agent_loop.py      # GymTextAgentLoop (@register("gym_text"))
├── windowed_agent_loop.py      # WindowedGymTextAgentLoop (@register("gym_text_windowed")), default
└── windowed_manager.py         # WindowedAgentLoopManager/Worker (fans per-turn samples into the batch)
```

## What `GymTextAgentLoop` does

Defined in [`gym_text_agent_loop.py`](gym_text_agent_loop.py), decorated
`@register("gym_text")` so verl's `AgentLoopManager` can build it by name. Its one
public entry point is the async `run(self, sampling_params, **kwargs)` coroutine
(`@rollout_trace_op`), which verl calls once per dataset row. The episode is:

```
reset env -> ( build prompt -> server.generate -> decode -> env.step -> append obs )* -> AgentLoopOutput
```

The dataset adapter [`../data/agentic_dataset.py`](../data/agentic_dataset.py)
puts the per-row fields into `extra_info`, and verl forwards them as `**kwargs` to
`run`. The loop reads:

- **`env_name`** (default `"TinyGuess"`), looked up in
  [`../envs/registry.py`](../envs/registry.py) via `make_env(env_name, config)`
  (`TinyGuess` / `WebShop` / `ALFWorld`).
- **`config`** (default `{}`), env config dict passed to the env constructor.
- **`seed`** (default `0`), deterministic episode seed for `env.reset(seed=...)`.
- **`max_turns`** (default `6`), hard cap on generate→step iterations.

## How verl discovers it

verl selects the agent-loop registry file via
`actor_rollout_ref.rollout.agent.agent_loop_config_path`. FedAgent points that at
[`../config/agent.yaml`](../config/agent.yaml), which maps the `agent_name`
carried on each row to an `AgentLoopBase` `_target_`:

```yaml
# fedagent/config/agent.yaml
- name: gym_text
  _target_: fedagent.agent_loops.gym_text_agent_loop.GymTextAgentLoop
- name: gym_text_windowed
  _target_: fedagent.agent_loops.windowed_agent_loop.WindowedGymTextAgentLoop
```

The federated runner wires this for both training and the val pass, see
`cfg.agent_config_path` (= `fedagent/config/agent.yaml`) and the
`...agent.agent_loop_config_path={cfg.agent_config_path}` CLI override in
[`../fed/run_fed.py`](../fed/run_fed.py). `gym_text` is the default `agent_name`
in the dataset adapter, so rows need not set it explicitly.

## Turn-loop mechanics

1. **Prompt construction.** `system_prompt()` and `reset()` seed a chat history
   (`system` + first `user` turn). It is tokenized with `_tokenize_chat`, a
   **non-truncating** `apply_chat_template(..., add_generation_prompt=True)`. This
   intentionally bypasses the base loop's left-truncation to `prompt_length`,
   which would silently drop the system prompt + task and corrupt the
   observation-token delta on long episodes. `cur_ids` is the running concat
   prompt; bounds are reapplied only on return.
2. **Overflow guard.** Before each generation, if `len(cur_ids) >= max_ctx - 1`
   the episode stops cleanly (one token short). `max_ctx` is
   `rollout.max_model_len` (fallback `prompt_length + response_length`); verl's
   inference server *raises and aborts the whole batch* if a prompt reaches
   `max_model_len`, so this caps growth (a no-op for WebShop, relevant for
   ALFWorld).
3. **Generation.** `server_manager.generate(request_id, prompt_ids=cur_ids,
   sampling_params)` returns a `TokenOutput`; its `token_ids` are the action
   tokens. They are appended to `response_ids` / `cur_ids` with
   **`response_mask = 1`** (model-generated → trained on), under a
   `simple_timer("generate_sequences", metrics)`.
4. **Action + step.** The generated tokens are decoded
   (`skip_special_tokens=True`) into the text action, appended to the chat as the
   `assistant` turn, and applied via `obs, reward, done, info = await
   env.step(text)`. `reward` is accumulated into `env_rewards`; `success` is read
   from `info["success"]`; invalid actions (`info["is_action_valid"] is False`)
   are counted.
5. **Observation feedback.** Unless terminal, `obs["obs_str"]` is appended as the
   next `user` turn; re-tokenizing yields the new observation tokens
   (`new_ids[len(cur_ids):]`), appended with **`response_mask = 0`**
   (observation → masked out of the PPO/GRPO loss).
6. **Termination.** The loop ends on `max_turns`, `done`, `len(response_ids) >=
   response_length`, or the overflow guard. `env.close()` always runs in a
   `finally` (e.g. to return a pooled remote WebShop session).

So one `AgentLoopOutput` carries the full concat trajectory with a response mask
that is 1 on the agent's actions and 0 on environment text; outcome credit is
assigned over action tokens only. GRPO groups come from verl's `rollout.n`
repeats per row.

## Reward and per-sample tags

`reward_score` returned to verl is the **episode reward minus an
invalid-action penalty**:

```python
reward_score = sum(env_rewards) - invalid_penalty * n_invalid
```

`invalid_penalty` is `FEDAGENT_INVALID_ACTION_PENALTY_COEF` (default `0.1`; set
`0` to disable); stock verl 0.8 has no such hook, so the loop applies it here.

Per-sample fields are tagged in `extra_fields["reward_extra_info"]` so they land
in verl's validation JSONL dump:

- **`traj_success`**: `float(success)`, always present.
- **`goal_id` / `task_type`**: added only when the env surfaces them
  (string-valued; kept in the val dump but skipped by metric aggregation). Used
  by the WebShop hardness-labelling pass and the ALFWorld eval breakdown.
- **`trajectory`**: the whole episode, **eval rows only**, windowed mode only.

### The `trajectory` column

Eval collapses each episode to its **last turn** — verl's `_validate` does
`test_batch.union(test_output_gen_batch)` and requires 1:1, and `summarize_val_dump`
means over *rows*, so K rows per episode would weight long episodes more and silently
corrupt `success_rate`. The dump therefore only ever showed the terminal step. The rest
of the episode now rides the surviving row:

```json
{"n_turns": 12, "success": 1.0, "episode_return": 1.0, "task_score": 1.0,
 "goal_id": "B07…", "turns": [
   {"turn": 0, "obs": "<env's windowed template>", "prompt": "<what the model saw>",
    "action": "<what it emitted>", "reward": 0.0, "done": false, "valid": true,
    "prompt_truncated": false, "response_truncated": false}, …]}
```

`obs` is the env's raw windowed template; `prompt` is that same turn after the chat
template and left-truncation, i.e. verbatim what the model saw. In windowed mode the
response *is* the action (no obs tokens), so `action` covers both — `response_truncated`
flags the case where the stored training sample was cut to `response_length`.

Two implementation constraints, both locked by
[`../../tests/test_windowed_eval_trajectory.py`](../../tests/test_windowed_eval_trajectory.py):

- It is attached to `outputs[-1]` **only** — the row eval keeps. `agent_loop.py` reads
  the reward-extra key set from `reward_extra_infos[0]` and then indexes `info[key]` for
  every row, so a key present on only some rows is a `KeyError` inside a Ray worker.
  Train attaches nothing, so the key is absent there uniformly.
- It is a **dict, not a JSON string**. `np.array` over strings yields `<U<maxlen>` and
  pads every row out to the longest trajectory at 4 bytes/char; over dicts it yields
  `dtype=object`. It also serializes as a nested object rather than an escaped blob.

On by default. `FEDAGENT_EVAL_TRAJECTORY_DUMP=0` turns it off — worth doing for long runs
where the dumps are archived, since it is the dominant term in their size. Concat mode
needs nothing: its single sample's prompt is already the whole conversation.

The val pass in [`../fed/run_fed.py`](../fed/run_fed.py)
(`summarize_val_dump`) reads this dump and reports
`success_rate = mean(traj_success)` and `reward_mean = mean(score)` (`score` is
verl's name for `reward_score` in the dump), FedAgent's headline
`val/success_rate`.

## Config keys it reads

| Source | Key | Use |
|---|---|---|
| `rollout_config` | `prompt_length` | cap on returned `prompt_ids` |
| `rollout_config` | `response_length` | cap on `response_ids` / `response_mask`; per-turn termination |
| `rollout_config` | `max_model_len` | context ceiling for the overflow guard (fallback `prompt_length + response_length`) |
| row `extra_info` | `env_name`, `config`, `seed`, `max_turns` | which env, how to build/reset it, and the turn cap |
| env var | `FEDAGENT_INVALID_ACTION_PENALTY_COEF` | invalid-action penalty coefficient (default `0.1`) |
| env var | `FEDAGENT_EVAL_TRAJECTORY_DUMP` | `0` disables the eval `trajectory` column (default on) |
| env var | `VERL_LOGGING_LEVEL` | module log level (default `WARN`) |

## Adding an agent-loop

Subclass verl's `AgentLoopBase`, implement async `run(...) -> AgentLoopOutput`,
decorate with `@register("<name>")`, add a `{name, _target_}` entry to
[`../config/agent.yaml`](../config/agent.yaml), and set `agent_name: <name>` on
the dataset spec. The env it drives must satisfy the
[`BaseTextEnv`](../envs/base.py) contract (`system_prompt` / `reset` / `step` /
`close`, with `info["success"]`).
