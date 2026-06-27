"""ALFWorld env — thin async HTTP client to the ALFWorld remote service.

Runs in the trainer env (fedagent-verl08). The real ALFWorld/textworld env lives in the
verl-agent-alfworld env behind the ``service/`` backend (``fedagent.envs.alfworld.service``,
HTTP), because ALFWorld's deps (alfworld + textworld + gymnasium + torchvision pins)
conflict with verl 0.8.

Action parsing (``alfworld_projection``) happens server-side; this client ferries the
model's text in and formats observations out using verl-agent's ALFWorld prompt content
(``ALFWORLD_TEMPLATE_NO_HIS``) so the information the policy sees matches the 0.3.1
baseline (scientific-equivalence bar). The concat-chat ``GymTextAgentLoop`` supplies
multi-turn history as the literal chat, so per-turn observations carry only the current
observation + admissible actions (the no-history template is the per-turn body; verl-agent
reuses it every turn when history_length<=0).

Mirrors ``fedagent.envs.webshop.WebShopEnv`` exactly in structure.
"""
import asyncio
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from fedagent.envs.base import BaseTextEnv, Obs

# Format/reasoning instructions (env-level, no per-episode task) -> system message.
# This is the instruction tail of verl-agent's ALFWORLD_TEMPLATE_NO_HIS, lifted to the
# system turn (the per-turn body below carries the observation + admissible actions).
ALFWORLD_SYSTEM = (
    "You are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning "
    "process MUST be enclosed within <think> </think> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for "
    "current step and present it within <action> </action> tags."
)
# Per-turn body — the observation lines of ALFWORLD_TEMPLATE_NO_HIS, verbatim.
_OBS = (
    "Your current observation is: {obs}\n"
    "Your admissible actions of the current situation are: [{actions}]."
)


def _fmt_actions(cmds: List[str]) -> str:
    # mirrors verl-agent build_text_obs: "\n ".join(f"'{s}'" ...), 'help' already excluded
    # server-side. Quote each command and newline-join.
    return "\n ".join(f"'{s}'" for s in (cmds or []))


class AlfworldEnv(BaseTextEnv):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        # ALFWORLD_SERVICE_URL (env) is authoritative: the federated runner sets it PER
        # CLIENT so each client talks to its own game-shard service. The spec's
        # service_url is only a fallback for ad-hoc single-service use.
        self.base_url = (
            os.environ.get("ALFWORLD_SERVICE_URL")
            or self.env_config.get("service_url")
            or "http://localhost:8081"
        ).rstrip("/")
        self.timeout = float(self.env_config.get("timeout", 120.0))
        self.session_id = uuid4().hex
        self._client: Optional[httpx.AsyncClient] = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def _post(self, path: str, payload: dict, *, retry: bool = False,
                    block: bool = False, retries: int = 8, base: float = 0.3):
        """POST to the env service; raise on HTTP errors, and (only when ``retry``) retry transport errors.

        ``retry=True`` is used ONLY for the idempotent endpoints (/create, /reset): at the full PPO
        batch the rollout fires train_batch_size x rollout.n episodes at once, so they hit this
        client's pooled per-client service near-simultaneously and the HTTP boundary is overwhelmed
        (sockets reset mid-response -> httpx.ReadError). Bounded backoff + jitter spreads the retried
        requests across the pool. /step is NOT retried: it mutates env state, so replaying it after the
        server already applied the action would corrupt the trajectory -- it fails fast instead.
        raise_for_status() ensures a 4xx/5xx body (e.g. {"detail":"unknown session"}) is never
        silently parsed as an empty observation. (GRPO's smaller batch never trips the storm.)

        ``block=True`` (used for /create) disables the per-request read timeout: borrowing a pooled
        env legitimately blocks until one frees, and that wait scales with batch/pool, NOT with the
        180s timeout. ALFWorld is the stress case -- 512 episodes of mean ~30 (max 50) turns share a
        small env pool, so a waiting /create routinely exceeds 180s; a hard timeout there crashes the
        whole rollout. A blocking /create also removes the duplicate-create race: no timeout -> no
        retry-resend -> exactly one borrow per session, so the idempotency check can't be bypassed.
        """
        c = self._c()
        for attempt in range(retries + 1):
            try:
                # block: only the READ wait is unbounded (server-side _pool.get() can exceed the
                # 180s default); connect/write/pool stay bounded so a dead service still fails fast
                # instead of hanging the rollout forever.
                resp = (await c.post(path, json=payload, timeout=httpx.Timeout(self.timeout, read=None)) if block
                        else await c.post(path, json=payload))
                resp.raise_for_status()
                return resp
            except httpx.TransportError:
                if not retry or attempt >= retries:
                    raise
                await asyncio.sleep(min(base * (2 ** attempt), 4.0) + random.uniform(0.0, base))

    async def system_prompt(self) -> Obs:
        return {"obs_str": ALFWORLD_SYSTEM}

    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        await self._post("/create", {"session_id": self.session_id}, retry=True, block=True)
        r = await self._post("/reset", {"session_id": self.session_id, "seed": int(seed)}, retry=True)
        d = r.json()
        obs_str = _OBS.format(
            obs=d.get("obs", "") or "", actions=_fmt_actions(d.get("admissible_commands", []))
        )
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        r = await self._post("/step", {"session_id": self.session_id, "text": action_str})
        d = r.json()
        obs_str = _OBS.format(
            obs=d.get("obs", "") or "", actions=_fmt_actions(d.get("admissible_commands", []))
        )
        info = {
            "success": bool(d.get("success", False)),
            "is_action_valid": bool(d.get("is_action_valid", True)),
        }
        return {"obs_str": obs_str}, float(d.get("reward", 0.0)), bool(d.get("done", False)), info

    async def close(self) -> None:
        try:
            if self._client is not None:
                await self._client.post("/close", json={"session_id": self.session_id})
                await self._client.aclose()
        except Exception:
            pass
        self._client = None
