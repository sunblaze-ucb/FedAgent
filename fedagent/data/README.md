# `data/` — the verl dataset adapter for FedAgent's agentic environments

This folder is the **dataset seam** between FedAgent's agentic environments and
[verl 0.8](../README.md)'s data pipeline. FedAgent is a *thin overlay* on stock
verl (it imports verl as a library; there is no trainer fork), so instead of
feeding verl a table of static prompts, we hand it a tiny custom `Dataset` whose
rows are **env specs**: each row tells the [agent-loop](../agent_loops/) which
environment to instantiate and which task instance to run.

- **`agentic_dataset.py`** — the `AgenticDataset` class (the `custom_cls`).
- **`__init__.py`** — package marker / one-line description.

---

## 1. What it is: a verl `custom_cls` dataset

verl 0.8 lets you swap its default dataset for your own via two config keys:

```yaml
data:
  custom_cls:
    path: fedagent/data/agentic_dataset.py   # this file
    name: AgenticDataset                     # the class below
```

verl imports the module at `path`, instantiates the class named `name` with
`(data_files, tokenizer, processor, config, **kwargs)`, and uses it as the
RL-dataloader dataset. The defining property of this seam (in verl's
async-rollout path): **every non-tensor column of a row is forwarded as a keyword
argument to `AgentLoop.run(**kwargs)`.** That is exactly how `env_name` / `seed` /
`config` / `max_turns` / `agent_name` reach the agent-loop.

The real class:

```python
# agentic_dataset.py
class AgenticDataset(Dataset):
    def __init__(self, data_files, tokenizer=None, processor=None, config=None, **kwargs): ...
```

[`../fed/run_fed.py`](../fed/run_fed.py) points verl at this file by passing
`data.custom_cls.path=<.../data/agentic_dataset.py>` on the `fedagent.main_ppo_fed`
command line (it does **not** import this module itself).

---

## 2. What a "row" is

The dataset does **not** emit tokenized text. It emits one row **per environment
instance**, and the agent-loop turns that row into a live episode. Concretely,
`__getitem__` returns this dict (built in `__init__`):

```python
{
    "env_name":   "WebShop",          # which env the agent-loop instantiates
    "seed":       4200000,            # per-instance seed (see §4)
    "config":     {"timeout": 180.0}, # env kwargs, passed straight to the env ctor
    "max_turns":  15,                 # episode turn budget
    "agent_name": "gym_text",         # which agent-loop to use (default: gym_text)
    "data_source": "webshop",         # name.lower(); verl's per-source bookkeeping
    "raw_prompt": [{"role": "user", "content": "<WebShop episode>"}],  # placeholder
    "ds_dummy":   torch.tensor([0]),  # single dummy tensor, for batch sizing only
}
```

Two non-obvious fields, both grounded in verl's stock contract:

- **`raw_prompt`** is a placeholder. verl's agent-loop postprocess stashes
  `kwargs["raw_prompt"]`, but our [`gym_text` loop](../agent_loops/) builds its own
  prompt from the live environment, so the content is never used as the model
  input.
- **`ds_dummy`** is the *only* tensor we emit. Stock verl's `_get_gen_batch` does
  not pop tensor keys before unioning the agent-loop's output back onto the batch,
  so emitting `input_ids` / `attention_mask` / `position_ids` here would collide
  with the ones the agent-loop generates. We therefore emit a single
  non-colliding `ds_dummy` purely so the batch has a defined length. (Forks that
  rewrite `_get_gen_batch` can emit dummy `input_ids`; we don't fork the trainer.)

GRPO grouping is handled **downstream** by verl's `rollout.n`: each row is
repeated `n` times, yielding one GRPO group per env instance. The dataset emits
one row per instance and nothing more.

---

## 3. How it reads the env-spec YAML

`data.train_files` / `data.val_files` point at an env-spec YAML under
[`../config/envs/`](../config/envs/) (e.g. `webshop_15.yaml`,
`tiny_guess.yaml`, `alfworld.yaml`). `_load_specs` loads the first path with
OmegaConf and reads its top-level **`envs:`** list. Each list entry is a spec; the
fields consumed (with the defaults applied if a key is absent) are:

| key          | default       | meaning                                              |
|--------------|---------------|------------------------------------------------------|
| `name`       | `"TinyGuess"` | env id → row `env_name` / `data_source`              |
| `n_envs`     | `64`          | number of rows (env instances) emitted for this spec |
| `max_turns`  | `6`           | per-episode turn budget                              |
| `agent_name` | `"gym_text"`  | agent-loop id (`DEFAULT_AGENT_LOOP`)                 |
| `config`     | `{}`          | env constructor kwargs, copied verbatim into the row |

Example (`../config/envs/webshop_15.yaml`):

```yaml
envs:
  - name: WebShop
    n_envs: 8
    max_turns: 15
    agent_name: gym_text
    config:
      timeout: 180.0
```

A spec with `n_envs: 8` produces **8 rows**, so `len(dataset) == 8`. Multiple
entries under `envs:` are concatenated. A given env-spec path that is missing /
unparsable / has no `envs:` now **raises** (`FileNotFoundError` / `ValueError`),
so a misconfigured run cannot silently train the TinyGuess toy objective. The
built-in `TinyGuess` spec (`n_envs: 64`, `max_turns: 6`, `config: {lo: 1, hi: 50}`)
is used **only** when `data_files` is empty/unset (the genuine "no env-spec" smoke
default).

---

## 4. Seeding: `FEDAGENT_BASE_SEED`

To make every client's env instances **distinct but reproducible**, the dataset
reads the `FEDAGENT_BASE_SEED` environment variable (default `0`):

```python
base_seed = int(os.environ.get("FEDAGENT_BASE_SEED", 0))
...
"seed": base_seed * 100_000 + si * 1_000 + i,   # si = spec index, i = instance index
```

[`../fed/run_fed.py`](../fed/run_fed.py) sets this per **(round, client)** before
launching each client's training subprocess:

```python
env["FEDAGENT_BASE_SEED"] = str(cfg.base_seed + round_num * 100 + client_id)
```

So the seed flows: `base_seed + round*100 + client_id` → `FEDAGENT_BASE_SEED` →
the dataset folds it into each row's `seed` (`base * 100_000 + spec*1_000 +
instance`, collision-free since `client_id < 100` and the stride leaves
`seed * 100_000 < 2**32`) → the agent-loop forwards `seed` to `env.reset(seed=...)`.
The `round*100` term re-draws each client's tasks every round (so a client covers
its data over `T` rounds rather than re-training on identical instances); the
`client_id` term keeps clients disjoint within a round.

---

## 5. The partition seam: where heterogeneity actually enters

`AgenticDataset` has a `_partition_specs` hook, but in this overlay it is the
**identity**:

```python
def _partition_specs(self, specs):
    return specs
```

This is deliberate. For the paper's two real benchmarks (**WebShop** and
**ALFWorld**), the client's data shard is **not** chosen here — it is selected
**server-side by the remote env service**:

- This dataset only decides *how many* episodes to run, *how long*, and *with what
  seed*. The same env-spec YAML is handed to every client.
- The shard itself (which catalog slice / which game subset / which goal
  distribution) is built inside the per-client env service. `run_fed.py` launches
  one service per client and passes the heterogeneity knobs as **env vars**:
  `PARTITION_STRATEGY`, `CLIENT_ID`, `CLIENT_NUM`, plus strategy-specific ones
  (`ENV_DIV`, `KEEP_RATIO`, `OMEGA`, `SIZE_STD`, `SUCCESS_STD`, `VARIANT_N`,
  `TRAJECTORIES_FILE`, `MIN_GOALS_PER_CLIENT`, …). The agent reaches its own
  client's service via `WEBSHOP_SERVICE_URL` / `ALFWORLD_SERVICE_URL`.
- The row `seed` is what *indexes into* that server-side shard: e.g. the WebShop
  service uses `random.Random(seed)` to pick a goal from the client's goal-index
  list, so the same `seed → goal` mapping is reproducible while the **shard
  membership** was fixed server-side by `PARTITION_STRATEGY` + `CLIENT_ID`.

In short: **the dataset is uniform; the heterogeneity lives in the env service.**
`_partition_specs` exists so a future in-process env could partition specs
client-side, but it is not on the WebShop/ALFWorld path. See
[`../envs/`](../envs/) for the env clients and the service wiring, and
[`../fed/`](../fed/) for the federated driver that sets all of the above.

---

### See also

- [`../README.md`](../README.md) — FedAgent overview (thin overlay on verl 0.8).
- [`../config/envs/`](../config/envs/) — the env-spec YAMLs consumed here.
- [`../fed/run_fed.py`](../fed/run_fed.py) — sets `data.custom_cls.path`,
  `data.train_files`, and `FEDAGENT_BASE_SEED` per round/client.
- [`../agent_loops/`](../agent_loops/) — consumes the row kwargs
  (`env_name` / `seed` / `config` / `max_turns`) in `AgentLoop.run`.
- [`../envs/`](../envs/) — the environment clients (WebShop / ALFWorld / TinyGuess).
