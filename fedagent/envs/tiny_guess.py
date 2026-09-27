"""TinyGuess — a tiny, dependency-free async multi-turn text env.

Game = guess-the-number with higher/lower feedback. It is NOT part of the research
suite; it exists to validate the verl-0.8 wiring end-to-end (the Phase 0(b) spike
env, now a first-class ``BaseTextEnv`` so the package can prove itself before the
real WebShop/ALFWorld ports land).
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fedagent.envs.base import BaseTextEnv, Obs

_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def parse_guess(text: str) -> Optional[int]:
    m = _ANS.search(text or "")
    if m:
        return int(m.group(1))
    nums = _INT.findall(text or "")
    return int(nums[-1]) if nums else None


class TinyGuessEnv(BaseTextEnv):
    """Guess a secret integer in [lo, hi]; env replies higher/lower; reward 1.0 on hit."""

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        cfg = self.env_config
        self.lo = int(cfg.get("lo", 1))
        self.hi = int(cfg.get("hi", 50))
        self.max_turns = int(cfg.get("max_turns", 6))
        self.target = int(cfg.get("target", (self.lo + self.hi) // 2))
        self.turn = 0
        self.solved = False
        # WINDOWED mode (history_length>0) sends ONE user message = obs_str and no system turn
        # (WindowedGymTextAgentLoop), so -- like the WebShop/ALFWorld clients -- the env must
        # build the full prompt itself (rules + recent guesses + current feedback); otherwise
        # the policy never sees the game rules. FEDAGENT_HISTORY_LENGTH (set by run_fed per
        # rollout_mode: windowed=2, concat=0) is AUTHORITATIVE; spec history_length is the fallback.
        self._history_length = int(os.environ.get("FEDAGENT_HISTORY_LENGTH")
                                   or cfg.get("history_length", 0))
        self._memory: List[Tuple[str, str]] = []   # [(guess, feedback)] for the windowed prompt

    def _rules(self) -> str:
        return (
            f"You are playing guess-the-number. A secret integer is in "
            f"[{self.lo}, {self.hi}]. Each turn reply with EXACTLY one guess as "
            f"<answer>N</answer>. I will respond 'higher' (secret is larger) or "
            f"'lower' (secret is smaller). You have {self.max_turns} guesses."
        )

    def _obs(self, current: str) -> str:
        """Concat mode: the per-turn body only (history is the growing chat). Windowed mode:
        rules + the last ``history_length`` (guess, feedback) pairs + ``current``."""
        if self._history_length <= 0:
            return current
        recent = self._memory[-self._history_length:]
        prior = ""
        if recent:
            pairs = "".join(f"\nYour guess: {g} -> {fb}" for g, fb in recent)
            prior = f"\nYou have made {self.turn} guess(es). Most recent:{pairs}"
        return f"{self._rules()}{prior}\n{current}"

    async def system_prompt(self) -> Obs:
        return {"obs_str": self._rules()}

    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        self.turn = 0
        self.solved = False
        self._memory = []
        # derive a per-instance target from the seed for variety across the dataset
        span = self.hi - self.lo + 1
        self.target = self.lo + (int(seed) % span)
        return {"obs_str": self._obs("Make your first guess as <answer>N</answer>.")}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
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
        self._memory.append(("invalid" if g is None else str(g), obs))
        # Windowed: the latest feedback is deliberately BOTH the last history line and the current
        # observation (ALFWorld's "current observation" slot). Measured against ending on a call to
        # action ("Make your next guess ...") instead: Qwen2.5-1.5B, 3 seeds x 64 episodes, solved
        # 82/192 vs 63/192 -- the bare feedback right before the reply is the cue the policy uses.
        return {"obs_str": self._obs(obs)}, reward, done, info
