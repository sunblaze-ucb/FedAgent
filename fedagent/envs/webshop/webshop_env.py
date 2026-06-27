"""WebShop env — thin async HTTP client to the WebShop remote service.

Runs in the trainer env (fedagent-verl08). The real WebShop gym env + Lucene/Java live
in the verl-agent-webshop env behind the ``service/`` backend (``fedagent.envs.webshop.service``,
HTTP), because WebShop's deps (torch 2.6 / gym 0.24 / pyserini / numpy 1.26) hard-conflict
with verl 0.8.

Action parsing (``webshop_projection``) happens server-side; this client ferries the
model's text in and formats observations out using verl-agent's WebShop prompt content
(``WEBSHOP_TEMPLATE``) so the information the policy sees matches the 0.3.1 baseline
(scientific-equivalence bar). The concat-chat ``GymTextAgentLoop`` supplies multi-turn
history as the literal chat, so per-turn observations carry only the current page +
admissible actions (task is in the first turn / chat history).
"""
import asyncio
import os
import random
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import httpx

from fedagent.envs.base import BaseTextEnv, Obs

# Format instructions (env-level, no per-episode task) -> system message.
WEBSHOP_SYSTEM = (
    "You are an expert autonomous agent operating in the WebShop e-commerce environment. "
    "Each turn, first reason step-by-step about the current situation inside <think> </think> "
    "tags, then choose exactly one admissible action and present it inside <action> </action> tags."
)
_FIRST_OBS = (
    "Your task is to: {task}.\n"
    "Your current observation is: {obs}.\n"
    "Your admissible actions of the current situation are:\n[\n{actions}\n]."
)
_STEP_OBS = (
    "Your current observation is: {obs}.\n"
    "Your admissible actions of the current situation are:\n[\n{actions}\n]."
)


def _fmt_actions(avail: Dict[str, Any]) -> str:
    # mirrors verl-agent env_manager.format_avail_actions + its join
    actions = []
    if avail.get("has_search_bar", False):
        actions.append("search[<your query>]")
    for txt in avail.get("clickables", []):
        actions.append(f"click[{txt}]")
    return "\n".join(f"'{s}'," for s in actions)


def _extract_task(obs: str) -> str:
    # reset obs looks like: "WebShop [SEP] Instruction: [SEP] <task> [SEP] Search"
    if obs and "Instruction:" in obs:
        after = obs.split("Instruction:", 1)[1]
        parts = [p.strip() for p in after.split("[SEP]") if p.strip()]
        if parts:
            return parts[0]
    return (obs or "").strip()


class WebShopEnv(BaseTextEnv):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        # WEBSHOP_SERVICE_URL (env) is authoritative: the federated runner sets it
        # PER CLIENT so each client talks to its own Catalog-Split service. The spec's
        # service_url is only a fallback for ad-hoc single-service use.
        self.base_url = (
            os.environ.get("WEBSHOP_SERVICE_URL")
            or self.env_config.get("service_url")
            or "http://localhost:8080"
        ).rstrip("/")
        self.timeout = float(self.env_config.get("timeout", 120.0))
        self.session_id = uuid4().hex
        self._task = ""
        self._goal_id = None   # asin of the current goal (only when the service logs it)
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
        180s timeout. (train_batch_size*rollout.n=512 episodes share a fixed pool; with long-episode
        envs a waiting /create can exceed 180s -- a hard timeout there would crash the whole rollout.)
        A blocking /create also removes the duplicate-create race: no timeout -> no retry-resend ->
        exactly one borrow per session, so the idempotency check can't be bypassed by a re-sent create.
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
        return {"obs_str": WEBSHOP_SYSTEM}

    async def reset(self, seed: int = 0) -> Tuple[Obs, Dict[str, Any]]:
        await self._post("/create", {"session_id": self.session_id}, retry=True, block=True)
        r = await self._post("/reset", {"session_id": self.session_id, "seed": int(seed)}, retry=True)
        d = r.json()
        raw = d.get("obs", "") or ""
        self._task = _extract_task(raw)
        self._goal_id = d.get("goal_id")   # asin (hardness-labelling pass only); None normally
        obs_str = _FIRST_OBS.format(
            task=self._task, obs=raw, actions=_fmt_actions(d.get("available_actions", {}))
        )
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        r = await self._post("/step", {"session_id": self.session_id, "text": action_str})
        d = r.json()
        obs_str = _STEP_OBS.format(
            obs=d.get("obs", "") or "", actions=_fmt_actions(d.get("available_actions", {}))
        )
        info = {
            "success": bool(d.get("success", False)),
            "is_action_valid": bool(d.get("is_action_valid", True)),
        }
        if self._goal_id is not None:
            info["goal_id"] = self._goal_id   # carried for the hardness-labelling dump
        return {"obs_str": obs_str}, float(d.get("reward", 0.0)), bool(d.get("done", False)), info

    async def close(self) -> None:
        try:
            if self._client is not None:
                await self._client.post("/close", json={"session_id": self.session_id})
                await self._client.aclose()
        except Exception:
            pass
        self._client = None
