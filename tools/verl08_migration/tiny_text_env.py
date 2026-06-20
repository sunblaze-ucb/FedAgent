"""A tiny, self-contained, async multi-turn TEXT environment for the Phase 0(b) spike.

Mirrors the VAGEN GymImageEnv async contract (reset/step/system_prompt/close) but
text-only, with no external deps. Game = guess-the-number with higher/lower feedback.
Used to prove a custom verl-0.8 AgentLoop can drive a real multi-turn env end-to-end.
"""
import re
from typing import Any, Dict, Optional, Tuple

_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def parse_guess(text: str) -> Optional[int]:
    m = _ANS.search(text or "")
    if m:
        return int(m.group(1))
    nums = _INT.findall(text or "")
    return int(nums[-1]) if nums else None


class TinyGuessEnv:
    """Guess a secret integer in [lo, hi]; env replies higher/lower; reward 1.0 on hit."""

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        cfg = env_config or {}
        self.lo = int(cfg.get("lo", 1))
        self.hi = int(cfg.get("hi", 50))
        self.target = int(cfg.get("target", 25))
        self.max_turns = int(cfg.get("max_turns", 6))
        self.turn = 0
        self.solved = False

    async def system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": (
                f"You are playing guess-the-number. A secret integer is in "
                f"[{self.lo}, {self.hi}]. Each turn reply with EXACTLY one guess as "
                f"<answer>N</answer>. I will respond 'higher' (secret is larger) or "
                f"'lower' (secret is smaller). You have {self.max_turns} guesses."
            )
        }

    async def reset(self, seed: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.turn = 0
        self.solved = False
        # derive a per-instance target from the seed for variety across the dataset
        span = self.hi - self.lo + 1
        self.target = self.lo + (int(seed) % span)
        return {"obs_str": "Make your first guess as <answer>N</answer>."}, {}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.turn += 1
        g = parse_guess(action_str)
        if g is None:
            obs, reward = "Invalid response. Reply as <answer>N</answer>.", 0.0
        elif g == self.target:
            self.solved, obs, reward = True, "Correct!", 1.0
        elif g < self.target:
            obs, reward = "higher", 0.0
        else:
            obs, reward = "lower", 0.0
        done = self.solved or self.turn >= self.max_turns
        info = {"success": self.solved, "turns": self.turn}
        return {"obs_str": obs}, reward, done, info

    async def close(self):
        return None
