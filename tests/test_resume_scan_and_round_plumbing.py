"""Regression tests for the 2026-09-10 infrastructure fixes (docs/bugfixes.md, 2026-09-10 §1-§3):

* the resume scan is keyed on the disk's ``round_<k>/`` dirs, not on ``range(total_rounds)``;
* a resume whose ``total_rounds`` is below the highest round on disk is REFUSED (not archived);
* the persistent worker's launch env always carries the round number.

Offline: no GPU, no verl, no Ray -- run_fed is imported for its pure helpers only.
"""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedagent.fed import run_fed  # noqa: E402


def _hf(d: Path) -> None:
    """A dir that passes run_fed._valid_hf_dir (config.json + one weight file)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0" * 8)


def _cfg(out: Path, **kw):
    cfg = OmegaConf.create(dict(run_fed.DEFAULTS))
    cfg.output_dir = str(out)
    for k, v in kw.items():
        cfg[k] = v
    return cfg


def test_resume_scan_is_disk_keyed_not_total_rounds_bounded(tmp_path):
    _hf(tmp_path / "round_70" / "aggregated" / "hf")
    (tmp_path / "round_0" / "eval").mkdir(parents=True)   # base-model eval dir: never an anchor
    (tmp_path / "round_71").mkdir()                       # partial attempt (no hf): skipped
    (tmp_path / "_stale_rounds").mkdir()                  # a prior --fresh's archive: ignored
    cfg = _cfg(tmp_path, total_rounds=2)                  # the 2-round smoke on a 70-round dir
    assert run_fed._disk_rounds(cfg) == [71, 70]
    k, actor_hf, critic_hf = run_fed.find_resume_round(cfg, is_ppo=False)
    assert k == 70 and critic_hf is None
    assert actor_hf == tmp_path / "round_70" / "aggregated" / "hf"


def test_ppo_round_is_complete_only_with_its_critic(tmp_path):
    _hf(tmp_path / "round_3" / "aggregated" / "hf")                 # actor merged, critic not
    _hf(tmp_path / "round_2" / "aggregated" / "hf")
    _hf(tmp_path / "round_2" / "aggregated" / "critic_hf")
    cfg = _cfg(tmp_path, total_rounds=70)
    assert run_fed.find_resume_round(cfg, is_ppo=True)[0] == 2
    assert run_fed.find_resume_round(cfg, is_ppo=False)[0] == 3


def test_empty_or_missing_dir_starts_fresh(tmp_path):
    assert run_fed.find_resume_round(_cfg(tmp_path / "nope", total_rounds=5), False) == (0, None, None)
    assert run_fed._disk_rounds(_cfg(tmp_path / "nope")) == []


def test_resume_refuses_a_schedule_below_the_disk(tmp_path):
    _hf(tmp_path / "round_70" / "aggregated" / "hf")
    with pytest.raises(ValueError, match="RESUME REFUSED"):
        run_fed.refuse_resume_below_disk(_cfg(tmp_path, total_rounds=2))
    # a partial round above the schedule counts too: it is evidence of a longer run
    (tmp_path / "round_71").mkdir()
    with pytest.raises(ValueError, match="round_71"):
        run_fed.refuse_resume_below_disk(_cfg(tmp_path, total_rounds=70))
    assert run_fed.refuse_resume_below_disk(_cfg(tmp_path, total_rounds=71)) == 71
    assert run_fed.refuse_resume_below_disk(_cfg(tmp_path / "empty", total_rounds=2)) == 0


@pytest.mark.parametrize("round_num", [1, 2, 70])
def test_persistent_cmd_env_carries_the_round_number(tmp_path, round_num):
    cfg = _cfg(tmp_path, env_kind="tinyguess")
    plan = [{"out_dir": str(tmp_path / "c0")}]
    _cmd, env = run_fed._persistent_cmd_env(cfg, plan, tmp_path / "plan.json", "/base", None,
                                            round_num, {}, n_gpus=1, worker_eval=False)
    assert env["FEDAGENT_XROUND_START_ROUND"] == str(round_num)
    assert env["FEDAGENT_PERSISTENT"] == "1"
