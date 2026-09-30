"""TinyGuess must carry its rules in obs_str under WINDOWED rollout (no system turn there).

Regression: the windowed loop sends a single user message = obs_str, and TinyGuess used to
return only "Make your first guess as <answer>N</answer>." -- the policy never saw the range
or the higher/lower protocol, and the federated smoke scored 0.0 on every rollout.
"""
import asyncio

from fedagent.envs.tiny_guess import TinyGuessEnv


def _play(monkeypatch, history_length, guesses, seed=30):
    monkeypatch.setenv("FEDAGENT_HISTORY_LENGTH", str(history_length))

    async def run():
        env = TinyGuessEnv({})
        obs, _ = await env.reset(seed=seed)
        out = [obs["obs_str"]]
        for g in guesses:
            obs, reward, done, info = await env.step(f"<answer>{g}</answer>")
            out.append(obs["obs_str"])
        return out, reward, done, info

    return asyncio.run(run())


def test_windowed_prompt_carries_rules_and_history(monkeypatch):
    obs, _, _, _ = _play(monkeypatch, 2, [10, 40, 20])   # seed=30 -> target 31
    assert "A secret integer is in [1, 50]" in obs[0]
    assert obs[0].endswith("Make your first guess as <answer>N</answer>.")
    last = obs[-1]
    assert "A secret integer is in [1, 50]" in last
    assert "Your guess: 40 -> lower" in last and "Your guess: 20 -> higher" in last
    assert "Your guess: 10" not in last                   # window = last 2 pairs only
    assert "You have made 3 guess(es)" in last
    # the latest feedback is the last history line AND the current observation (see step())
    assert last.endswith("Your guess: 20 -> higher\nhigher")


def test_windowed_invalid_reply_is_recorded(monkeypatch):
    obs, reward, done, _ = _play(monkeypatch, 2, ["no guess here"])
    assert "Your guess: invalid -> Invalid response. Reply as <answer>N</answer>." in obs[-1]
    assert obs[-1].endswith("\nInvalid response. Reply as <answer>N</answer>.")
    assert reward == 0.0 and not done


def test_concat_prompt_unchanged(monkeypatch):
    obs, reward, done, info = _play(monkeypatch, 0, [10, 31])
    assert obs == ["Make your first guess as <answer>N</answer>.", "higher", "Correct!"]
    assert reward == 1.0 and done and info["success"]
