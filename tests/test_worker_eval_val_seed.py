"""The worker-eval val set must NOT inherit the training client's data seed (bugfix 2026-07-26).

``AgenticDataset`` seeds each episode row from the process-global ``FEDAGENT_BASE_SEED``
(agentic_dataset.py:58). ``PersistentFedTaskRunner`` sets that var to the CLIENT's training
seed (``base_seed + round*100 + client``) before it builds datasets, and eval_mode=worker built
its val dataloader inside that window -- so with a per-round worker process every round scored
a DIFFERENT val draw. On ALFWorld (seed -> ``RandomState(seed).shuffle(gamefiles)[0]``, a draw
with replacement over the 140-game valid_seen split) that produced 48 different game sets over
48 rounds in alfworld_ppo_hardness_std1, with the aggregated point and the client circles of
the same round landing on different sets. WebShop hid it: its val branch is ``seed % VAL_SIZE``
and 500 divides 100_000, so ``base*100_000 + i`` always lands on goals[0:n_envs].

These tests pin both halves: the seed->row contract in the dataset, and the runner clearing the
var around the val-dataset build while leaving training's seed intact.

The runner import pulls verl (installed in the fedagent-verl08 env); skipped when absent.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.data.agentic_dataset import AgenticDataset  # noqa: E402

VAL_SPEC = "envs:\n  - name: ALFWorld\n    n_envs: 8\n    max_turns: 50\n"


def _seeds(spec_path):
    return [it["seed"] for it in AgenticDataset(str(spec_path)).items]


@pytest.fixture
def val_spec(tmp_path):
    p = tmp_path / "val.yaml"
    p.write_text(VAL_SPEC)
    return p


def test_base_seed_shifts_every_row(val_spec, monkeypatch):
    """The mechanism itself: a different FEDAGENT_BASE_SEED == a different episode draw."""
    monkeypatch.delenv("FEDAGENT_BASE_SEED", raising=False)
    unseeded = _seeds(val_spec)
    assert unseeded == list(range(8))                 # the fixed val set the eval path documents
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "2065")  # base_seed 42 + round 20 + client 23
    assert _seeds(val_spec) == [206500000 + i for i in range(8)]
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "2164")  # the NEXT round's first client
    assert _seeds(val_spec) == [216400000 + i for i in range(8)]


def test_unseeded_eval_data_pins_the_val_draw(val_spec, monkeypatch):
    """Inside the context the val rows are seed-free; the client's training seed survives it."""
    ptr = pytest.importorskip("fedagent.fed.persistent_task_runner",
                              reason="needs verl (fedagent-verl08 env)")
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "2065")
    with ptr.unseeded_eval_data():
        assert "FEDAGENT_BASE_SEED" not in os.environ
        first = _seeds(val_spec)
    assert os.environ["FEDAGENT_BASE_SEED"] == "2065"   # training seeding restored

    monkeypatch.setenv("FEDAGENT_BASE_SEED", "2164")    # next round, next client
    with ptr.unseeded_eval_data():
        second = _seeds(val_spec)
    assert first == second == list(range(8))            # same val set every round


def test_unseeded_eval_data_restores_on_error(monkeypatch):
    ptr = pytest.importorskip("fedagent.fed.persistent_task_runner",
                              reason="needs verl (fedagent-verl08 env)")
    monkeypatch.setenv("FEDAGENT_BASE_SEED", "77")
    with pytest.raises(RuntimeError):
        with ptr.unseeded_eval_data():
            raise RuntimeError("boom")
    assert os.environ["FEDAGENT_BASE_SEED"] == "77"


def test_unseeded_eval_data_no_op_when_unset(monkeypatch):
    """The subprocess eval path never sets the var; the context must not invent one."""
    ptr = pytest.importorskip("fedagent.fed.persistent_task_runner",
                              reason="needs verl (fedagent-verl08 env)")
    monkeypatch.delenv("FEDAGENT_BASE_SEED", raising=False)
    with ptr.unseeded_eval_data():
        pass
    assert "FEDAGENT_BASE_SEED" not in os.environ


def test_alfworld_val_selection_defaults_to_seed_is_index():
    """WHICH games the fixed val set contains. Default (2026-07-26): seed == game index, so the
    spec's n_envs rows are games[0:n_envs], each exactly once. The legacy shuffle-draw covers
    only ~52 of 140 with ~12 repeats and is opt-in from here on."""
    rf = pytest.importorskip("fedagent.fed.run_fed", reason="needs omegaconf/verl")
    assert rf.DEFAULTS["alfworld_val_seed_is_index"] is True
    assert rf.alfworld_val_selection_env({}) == {"ALFWORLD_SEED_IS_INDEX": "1"}
    assert rf.alfworld_val_selection_env({"alfworld_val_seed_is_index": True}) == {
        "ALFWORLD_SEED_IS_INDEX": "1"}
    assert rf.alfworld_val_selection_env({"alfworld_val_seed_is_index": False}) == {}
