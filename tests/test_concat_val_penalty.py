"""Regression test: the concat loop's invalid-action penalty must skip VALIDATION rows.

Stock verl never passes ``trajectory["validate"]`` into ``AgentLoop.run()`` -- the val marker
instead rides the dataset row: AgenticDataset copies the val spec's ``validate: true`` into an
``is_validation`` column, and verl forwards row columns as run() kwargs. Before the 2026-07-28
fix the concat loop penalized val rows too (the windowed loop, which gets ``validate``
natively from its manager, never did), so concat val reward_score silently differed from the
legacy stack's unpenalized validation. Three contracts:

1. TRAIN rows: ``reward_score = episode return - coef * #invalid actions`` (unchanged).
2. VAL rows (``is_validation=True``): ``reward_score = episode return``, no penalty.
3. AgenticDataset emits the column: True for ``validate: true`` specs, False otherwise
   (a val spec WITHOUT the marker keeps the old penalized behavior -- back-compat).

Runs the REAL ``GymTextAgentLoop.run`` against stub tokenizer/server/env: no model, no
network, no GPU. Needs verl importable (the loop subclasses verl's AgentLoopBase).
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("verl")
pytest.importorskip("torch")

from fedagent.agent_loops.gym_text_agent_loop import GymTextAgentLoop  # noqa: E402
from fedagent.data.agentic_dataset import AgenticDataset  # noqa: E402
from fedagent.envs import registry  # noqa: E402

ACTIONS = ["go to shelf 1", "buy now please", "click[buy now]"]


class StubEnv:
    """Three-turn episode: turn 2 is an INVALID action; turn 3 ends it with reward 1.0."""

    def __init__(self, env_config=None):
        self.t = 0

    async def system_prompt(self):
        return {"obs_str": "You are a shopping agent."}

    async def reset(self, seed: int = 0):
        return {"obs_str": "[task] buy a mug\n[obs 0] search page"}, {}

    async def step(self, action_str: str):
        self.t += 1
        done = self.t >= 3
        info = {"success": done, "is_action_valid": self.t != 2}
        return ({"obs_str": f"[obs {self.t}] result of {action_str!r}"},
                1.0 if done else 0.0, done, info)

    async def close(self):
        return None


class StubTokenizer:
    """Char-level: encode/decode round-trip; enough for decode(gen) and the glue's encode."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class StubOut:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.log_probs = None


class StubServerManager:
    async def generate(self, request_id, prompt_ids, sampling_params):
        # scripted action per successive turn (stateful counter on the instance)
        self.n = getattr(self, "n", 0)
        text = ACTIONS[min(self.n, len(ACTIONS) - 1)]
        self.n += 1
        return StubOut([ord(c) for c in text])


class Loop(GymTextAgentLoop):
    """Bypass AgentLoopBase.__init__ (needs a full trainer config); set only what run() reads.
    Chat "rendering" is a deterministic char-level fold of the messages, so the string-diff
    glue path exercises for real (its output is masked either way; only reward_score is
    asserted here)."""

    def __init__(self):
        self.prompt_length = 4096
        self.response_length = 512
        self._max_ctx = 8192
        self._invalid_penalty = 0.1
        self.tokenizer = StubTokenizer()
        self.server_manager = StubServerManager()

    @staticmethod
    def _render(messages):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<gen>"

    async def _tokenize_chat(self, messages):
        return [ord(c) for c in self._render(messages)]

    async def _render_chat_str(self, messages):
        return self._render(messages)


@pytest.fixture(autouse=True)
def stub_env():
    registry.ENV_REGISTRY["StubEnv"] = StubEnv
    try:
        yield
    finally:
        registry.ENV_REGISTRY.pop("StubEnv", None)


def run(**kw):
    return asyncio.run(Loop().run(
        {}, env_name="StubEnv", config={}, seed=0, max_turns=10, **kw))


def test_train_row_is_penalized():
    out = run()   # no marker at all == a train row
    assert out.reward_score == pytest.approx(1.0 - 0.1)


def test_val_row_is_not_penalized():
    out = run(is_validation=True)
    assert out.reward_score == pytest.approx(1.0)


def test_unmarked_val_spec_keeps_old_behavior():
    """Back-compat: an explicit False (a spec without `validate: true`) still penalizes."""
    out = run(is_validation=False)
    assert out.reward_score == pytest.approx(1.0 - 0.1)


def test_success_metric_is_penalty_independent():
    """The paper metric rides reward_extra_info.traj_success, not reward_score."""
    for kw in ({}, {"is_validation": True}):
        out = run(**kw)
        assert out.extra_fields["reward_extra_info"]["traj_success"] == 1.0


# --- 3. the dataset emits the column ---------------------------------------------------------

def _spec_yaml(tmp_path, body):
    p = tmp_path / "spec.yaml"
    p.write_text(body)
    return str(p)


def test_dataset_marks_validate_specs(tmp_path):
    path = _spec_yaml(tmp_path, (
        "envs:\n"
        "  - name: StubEnv\n"
        "    n_envs: 3\n"
        "    max_turns: 5\n"
        "    validate: true\n"
    ))
    ds = AgenticDataset(data_files=[path])
    assert len(ds) == 3
    assert all(row["is_validation"] is True for row in ds.items)


def test_dataset_defaults_to_train(tmp_path):
    path = _spec_yaml(tmp_path, (
        "envs:\n"
        "  - name: StubEnv\n"
        "    n_envs: 2\n"
        "    max_turns: 5\n"
    ))
    ds = AgenticDataset(data_files=[path])
    assert all(row["is_validation"] is False for row in ds.items)
