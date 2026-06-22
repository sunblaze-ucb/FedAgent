# `envs/` — async multi-turn text environments for FedAgent

The environments a FedAgent agent-loop drives, one **instance per dataset row**. Every env
implements a single async contract (`BaseTextEnv`) and is selected by name through a small
registry. The smoke env (`TinyGuess`) runs fully in-process; the research envs (`WebShop`,
`ALFWorld`) are **thin HTTP clients** to per-client remote services that wrap the heavy real
environments (whose engines are vendored under `envs/<name>/engine/` and ship their own conflicting deps).

This package is a thin overlay on stock verl 0.8 — see [`../README.md`](../README.md) for the
project overview, and [`../agent_loops/`](../agent_loops/) for the loop that drives these envs.

## Layout

Shared contract + registry sit at the top; each non-trivial environment is its **own package**
(matching VAGEN's `envs/<name>/` convention), co-locating the trainer-side client with its
out-of-process backend:

| Path | Role |
| --- | --- |
| [`base.py`](base.py) | `BaseTextEnv` — the async `reset`/`step`/`system_prompt`/`close` contract every env implements |
| [`registry.py`](registry.py) | `ENV_REGISTRY` (`env_name` → class) + `make_env(env_name, env_config)` factory |
| [`tiny_guess.py`](tiny_guess.py) | `TinyGuessEnv` — dependency-free in-process smoke env (guess-the-number) |
| [`webshop/`](webshop/) | WebShop package: client [`webshop/webshop_env.py`](webshop/webshop_env.py) (`WebShopEnv`) + its [`webshop/service/`](webshop/service/) backend |
| [`alfworld/`](alfworld/) | ALFWorld package: client [`alfworld/alfworld_env.py`](alfworld/alfworld_env.py) (`AlfworldEnv`) + its [`alfworld/service/`](alfworld/service/) backend |

Inside each env package, `__init__.py` re-exports **only** the lightweight client (so importing
`fedagent.envs.webshop` never pulls the backend's conflicting deps); `service/` is the FastAPI
backend, imported only in its own conda env. See
[`webshop/service/README.md`](webshop/service/README.md).

## The `BaseTextEnv` contract

`base.py` defines the per-instance async interface the agent-loop `await`s. An observation is a
dict carrying at least `obs_str` (the text shown to the model); `info` should carry `success`
(bool) so the loop can record the episode outcome (FedAgent's headline metric is
`val/success_rate`).

```python
class BaseTextEnv(ABC):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None): ...

    async def system_prompt(self) -> Obs:                  # {"obs_str": ...}, shown once at episode start
    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict]:                  # -> (obs, info)
    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict]:    # -> (obs, reward, done, info)
    async def close(self) -> None:                          # release resources (override if needed)
```

`Obs = Dict[str, Any]`. The loop calls `system_prompt()` + `reset(seed=...)` once, then `step()`
each turn until `done`, then `close()`. See [`../agent_loops/gym_text_agent_loop.py`](../agent_loops/gym_text_agent_loop.py)
(`reset env → build prompt → generate → decode → env.step → append obs`).

## The registry

`registry.py` maps the `env_name` carried on each dataset row to its class:

```python
ENV_REGISTRY: Dict[str, Type[BaseTextEnv]] = {
    "TinyGuess": TinyGuessEnv,
    "WebShop":   WebShopEnv,    # HTTP client -> webshop/service
    "ALFWorld":  AlfworldEnv,   # HTTP client -> alfworld/service
}

def make_env(env_name, env_config=None) -> BaseTextEnv:   # KeyError on unknown name
```

The agent-loop calls `make_env(env_name, config)` per row. The `env_name` and `config` come from
an **env-spec YAML** in [`../config/envs/`](../config/envs/), where each entry names the env, the
number of episodes (`n_envs`), the turn budget (`max_turns`), the `agent_name`, and a `config`
dict passed straight to `__init__`:

```yaml
envs:
  - name: WebShop      # -> ENV_REGISTRY["WebShop"]
    n_envs: 8
    max_turns: 15
    agent_name: gym_text
    config:            # forwarded to WebShopEnv.__init__ as env_config
      timeout: 180.0
```

## `tiny_guess.py` — in-process smoke env

`TinyGuessEnv` is a dependency-free guess-the-number game used to validate the verl-0.8 wiring
end-to-end; it is **not** part of the research suite. A secret integer lives in `[lo, hi]`
(defaults `1..50`); `reset(seed)` derives a per-instance target as `lo + seed % span` (variety
across the dataset). Each turn the model replies `<answer>N</answer>`; the env answers `"higher"`
/ `"lower"`, or `"Correct!"` on a hit. **Reward** is `1.0` on the correct guess and `0.0`
otherwise; `done` once solved or `turn >= max_turns`; `info = {"success": solved, "turns": ...}`.

## `webshop/` & `alfworld/` — env client + remote service

Both env clients run in the trainer env (`fedagent-verl08`) and hold no environment state — the
real gym envs (and their conflicting deps: WebShop's Lucene/Java/pyserini, ALFWorld's
alfworld/textworld/torchvision pins) live behind FastAPI services in
[`webshop/service/`](webshop/service/) and [`alfworld/service/`](alfworld/service/).
**Action parsing happens server-side** (`webshop_projection` / `alfworld_projection`); the client
only ferries the model's text in and formats observations out using verl-agent's prompt content,
so the policy sees the same information as the 0.3.1 baseline.

### Service URL (env var, authoritative)

The federated runner sets the URL **per client** (each client talks to its own shard/Catalog-Split
service). The env-spec's `service_url` is only a fallback for ad-hoc single-service use:

```text
WebShop:   WEBSHOP_SERVICE_URL  > env_config["service_url"] > http://localhost:8080
ALFWorld:  ALFWORLD_SERVICE_URL > env_config["service_url"] > http://localhost:8081
```

Each instance mints a `session_id = uuid4().hex` and uses one `httpx.AsyncClient`
(`timeout` from `env_config["timeout"]`, defaults `120.0`).

### Request / response shape

Every method POSTs JSON keyed by `session_id`:

| Method | Request | Response keys read |
| --- | --- | --- |
| `system_prompt()` | *(local; returns the static `WEBSHOP_SYSTEM` / `ALFWORLD_SYSTEM` text)* | — |
| `reset(seed)` | `/create {session_id}` then `/reset {session_id, seed}` | `obs`; WebShop: `available_actions`, `goal_id`; ALFWorld: `admissible_commands` |
| `step(action_str)` | `/step {session_id, text}` | `obs`, `reward`, `done`, `success`, `is_action_valid`; actions as above (WebShop also `task_score`) |
| `close()` | `/close {session_id}` then closes the client | — |

`step` returns `({"obs_str": ...}, float(reward), bool(done), info)` with
`info = {"success": ..., "is_action_valid": ...}` (WebShop also forwards `goal_id` when the
service emits it, for the hardness-labelling pass). WebShop extracts the per-episode task from the
`reset` obs (`"... Instruction: [SEP] <task> [SEP] ..."`) and renders it in the first user turn;
both envs format admissible actions into the per-turn observation.

### Reward shape (SPARSE)

The clients pass `reward` through verbatim; the actual values are produced by the service:

- **WebShop** ([`webshop/service/server.py`](webshop/service/server.py) `/step`): the env's
  graded score in `[0,1]` is kept as `task_score` (diagnostic), but the per-step **`reward` is
  binary `{0, 10}`** — `10.0` iff `done and task_score == 1.0` (a perfect match), else `0.0`. This
  matches verl-agent's sparse training reward (`envs.py:32-40`).
- **ALFWorld** ([`alfworld/service/server.py`](alfworld/service/server.py) `/step`):
  **`reward = 10.0 * won`** (i.e. `{0, 10}`), `won` being the episode-success flag.

The **per-invalid-action penalty** is applied not by the env but by the agent-loop: it counts
steps where `info["is_action_valid"]` is false and computes the episode reward as
`sum(env_rewards) - coef * n_invalid`, with `coef = FEDAGENT_INVALID_ACTION_PENALTY_COEF`
(default `0.1`, `0` disables) — mirroring verl-agent's `apply_invalid_action_penalty`. The env's
sole job here is to surface `is_action_valid`. See
[`../agent_loops/gym_text_agent_loop.py`](../agent_loops/gym_text_agent_loop.py).

## See also

- [`../README.md`](../README.md) — project overview
- [`../agent_loops/`](../agent_loops/) — the loop that drives these envs
- [`webshop/service/`](webshop/service/) / [`alfworld/service/`](alfworld/service/) — the remote env service backends
- [`../config/envs/`](../config/envs/) — env-spec YAMLs that select an env by name
