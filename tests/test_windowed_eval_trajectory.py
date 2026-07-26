"""Regression test for the eval full-trajectory capture (windowed rollout).

Eval collapses each episode to its last turn (windowed_manager.py keeps ``turn_outputs[-1]``
so verl's 1:1 ``_validate`` contract and the row-mean in ``summarize_val_dump`` stay intact),
so the val dump used to show only the TERMINAL step. The other turns now ride the surviving
row as a ``trajectory`` payload on ``reward_extra_info``.

Three things must hold, and each has bitten a comparable feature before:

1. **Every eval row carries the key, train rows carry none.** ``agent_loop.py`` takes the
   reward-extra key set from ``reward_extra_infos[0]`` and then does ``info[key]`` for every
   row -- a key on only *some* rows is a KeyError at rollout time, in a Ray worker.
2. **The payload is a dict, so numpy gives ``dtype=object``.** As a string, ``np.array``
   yields ``<U<maxlen>`` and pads EVERY row out to the longest trajectory at 4 bytes/char.
3. **It survives ``json.dumps(..., default=str)``** -- verbatim what ``_write_generations``
   does to build the JSONL line.

Capture is **opt-in** (``FEDAGENT_EVAL_TRAJECTORY_DUMP=1``), so most tests here set that flag;
the default-off path has its own tests, since "an unconfigured run must not silently pay for a
large dump" is the behaviour that flag exists to guarantee.

Runs the REAL ``run_episode_windowed`` against a stub tokenizer/server/env: no model, no
network, no GPU. Needs verl importable (the loop subclasses verl's AgentLoopBase); skips
cleanly otherwise.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np = pytest.importorskip("numpy")
pytest.importorskip("verl")

from fedagent.agent_loops.windowed_agent_loop import (  # noqa: E402
    WindowedGymTextAgentLoop, eval_trajectory_dump_enabled,
)
from fedagent.envs import registry  # noqa: E402

TASK = "put a cool apple in microwave"
ACTIONS = ["go to fridge 1", "open fridge 1", "take apple 1 from fridge 1"]


class StubEnv:
    """Three-turn episode; the third step ends it with reward 1.0 and success."""

    def __init__(self, env_config=None):
        self.t = 0

    async def reset(self, seed: int = 0):
        return {"obs_str": f"[task] {TASK}\n[obs 0] You are in the middle of a room."}, {}

    async def step(self, action_str: str):
        self.t += 1
        done = self.t >= 3
        info = {
            "success": done,
            "is_action_valid": self.t != 2,          # turn 1 is invalid, on purpose
            "task_score": 1.0 if done else None,
            "goal_id": "stub_goal_42",
        }
        return ({"obs_str": f"[task] {TASK}\n[obs {self.t}] result of {action_str!r}"},
                1.0 if done else 0.0, done, info)

    async def close(self):
        return None


class StubTokenizer:
    """Char-level; decode(ids) round-trips encode(). Enough for the loop's decode calls."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class StubOut:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.log_probs = None


class StubServerManager:
    """Returns the scripted action for each successive turn."""

    def __init__(self):
        self.n = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        text = ACTIONS[min(self.n, len(ACTIONS) - 1)]
        self.n += 1
        return StubOut([ord(c) for c in text])


class Loop(WindowedGymTextAgentLoop):
    """Bypass AgentLoopBase.__init__ (needs a full trainer config); set only what the
    windowed episode path reads. ``_tokenize_chat`` is stubbed so no event-loop executor or
    real chat template is involved -- the prompt is the obs with a marker, which is enough to
    assert that the recorded `prompt` is the rendered form and `obs` is the raw env string."""

    def __init__(self, prompt_length=4096, max_ctx=4608):
        self.prompt_length = prompt_length
        self.response_length = 512
        self._max_ctx = max_ctx
        self._invalid_penalty = 0.1
        self.tokenizer = StubTokenizer()
        self.server_manager = StubServerManager()

    async def _tokenize_chat(self, messages):
        rendered = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>"
        return [ord(c) for c in rendered]


@pytest.fixture(autouse=True)
def stub_env():
    registry.ENV_REGISTRY["StubEnv"] = StubEnv
    try:
        yield
    finally:
        registry.ENV_REGISTRY.pop("StubEnv", None)


@pytest.fixture(autouse=True)
def opt_in(monkeypatch):
    """Capture is OPT-IN, so the tests that exercise it must ask for it. The default-off
    behaviour is covered by its own tests, which clear this."""
    monkeypatch.setenv("FEDAGENT_EVAL_TRAJECTORY_DUMP", "1")


def run(validate, **kw):
    return asyncio.run(Loop(**kw).run_episode_windowed(
        {}, validate=validate, env_name="StubEnv", config={}, seed=0, max_turns=10))


def payloads(outs):
    return [o.extra_fields["reward_extra_info"].get("trajectory") for o in outs]


# --- 1. presence: exactly the row eval keeps, and nothing in train --------------------------

def test_only_the_surviving_eval_row_carries_the_payload():
    outs = run(validate=True)
    assert len(outs) == 3                              # one sample per turn, pre-collapse
    assert payloads(outs)[:-1] == [None, None]         # earlier turns: no copy held
    assert payloads(outs)[-1] is not None

    # windowed_manager's eval collapse, verbatim -> the kept row must have the key
    kept = outs[-1]
    assert "trajectory" in kept.extra_fields["reward_extra_info"]


def test_every_collapsed_eval_row_has_the_key():
    """The KeyError guard: simulate a worker batch of episodes and check the key set is
    uniform across rows, which is what agent_loop.py's `info[key]` for-every-row requires."""
    batch = [run(validate=True)[-1] for _ in range(4)]
    keysets = [set(o.extra_fields["reward_extra_info"]) for o in batch]
    assert all(k == keysets[0] for k in keysets)
    assert "trajectory" in keysets[0]


def test_train_attaches_nothing():
    outs = run(validate=False)
    assert payloads(outs) == [None, None, None]
    assert all("trajectory" not in o.extra_fields["reward_extra_info"] for o in outs)


def test_unset_means_off(monkeypatch):
    """The default: an unconfigured run pays nothing and its dumps keep the old shape."""
    monkeypatch.delenv("FEDAGENT_EVAL_TRAJECTORY_DUMP", raising=False)
    assert not eval_trajectory_dump_enabled()
    outs = run(validate=True)
    assert payloads(outs) == [None, None, None]
    assert all("trajectory" not in o.extra_fields["reward_extra_info"] for o in outs)


def test_env_flag_explicitly_off(monkeypatch):
    monkeypatch.setenv("FEDAGENT_EVAL_TRAJECTORY_DUMP", "0")
    assert not eval_trajectory_dump_enabled()
    assert payloads(run(validate=True)) == [None, None, None]


def test_flag_parses_both_spellings(monkeypatch):
    for on in ("1", "true", "TRUE", "yes", "on", " 1 "):
        monkeypatch.setenv("FEDAGENT_EVAL_TRAJECTORY_DUMP", on)
        assert eval_trajectory_dump_enabled(), on
    # anything else is off -- an unrecognised value must not silently enable a costly dump
    for off in ("0", "false", "no", "off", "", "maybe", "2"):
        monkeypatch.setenv("FEDAGENT_EVAL_TRAJECTORY_DUMP", off)
        assert not eval_trajectory_dump_enabled(), off


# --- 2. content: the whole episode, not just the terminal step ------------------------------

def test_payload_holds_every_turn_verbatim():
    traj = payloads(run(validate=True))[-1]
    assert traj["n_turns"] == 3
    assert traj["success"] == 1.0
    assert traj["episode_return"] == 1.0               # sparse: 0 + 0 + 1
    assert traj["task_score"] == 1.0
    assert traj["goal_id"] == "stub_goal_42"           # tags ride along: self-contained column

    turns = traj["turns"]
    assert [t["turn"] for t in turns] == [0, 1, 2]
    assert [t["action"] for t in turns] == ACTIONS
    assert [t["reward"] for t in turns] == [0.0, 0.0, 1.0]
    assert [t["done"] for t in turns] == [False, False, True]
    assert [t["valid"] for t in turns] == [True, False, True]

    # obs = raw env string for THAT turn (not the next one), prompt = the rendered form
    assert turns[0]["obs"].endswith("You are in the middle of a room.")
    assert turns[1]["obs"].endswith(f"result of {ACTIONS[0]!r}")
    for t in turns:
        assert t["prompt"] == f"<|im_start|>user\n{t['obs']}<|im_end|>"
        assert t["prompt_truncated"] is False and t["response_truncated"] is False


def test_truncation_flags_fire():
    traj = payloads(run(validate=True, prompt_length=8, max_ctx=9))[-1]
    assert all(t["prompt_truncated"] for t in traj["turns"])
    assert len(traj["turns"][0]["prompt"]) == 8        # the model saw only the tail


# --- 3. transport: numpy dtype + JSONL serialization ----------------------------------------

def test_numpy_keeps_it_as_object_not_padded_unicode():
    """A dict payload -> dtype=object (references). A string payload would give `<U<maxlen>`
    and pad every row to the longest trajectory at 4 bytes/char."""
    rows = [payloads(run(validate=True))[-1] for _ in range(3)]
    arr = np.array(rows)
    assert arr.dtype == object, f"expected object dtype, got {arr.dtype}"
    assert arr.shape == (3,), "dicts must not be broadcast into a 2-D array"

    as_strings = np.array([json.dumps(r) for r in rows])
    assert as_strings.dtype.kind == "U"                # the shape we are avoiding
    assert as_strings.nbytes > arr.nbytes


def test_row_serializes_the_way_write_generations_does():
    traj = payloads(run(validate=True))[-1]
    entry = {"input": "...", "output": "...", "score": 1.0, "trajectory": traj}
    line = json.dumps(entry, ensure_ascii=False, default=str)
    back = json.loads(line)
    assert back["trajectory"]["turns"][2]["action"] == ACTIONS[2]
    assert "\n" not in line                            # one JSONL row, not several
