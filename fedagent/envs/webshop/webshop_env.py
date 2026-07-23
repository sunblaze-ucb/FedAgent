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

Observation rendering matches the FED baseline, not the plain env_manager: the paper's
federated runs (main_ppo_fed.py -> fed_make_envs -> fed_env_manager.WebshopEnvironmentManager)
post-process EVERY observation with ``format_obs`` -- drop the page prefix up to and including
the instruction segment, and single-quote each remaining `` [SEP] `` part -- before it reaches
the prompt OR the history memory. ``_format_obs`` below replicates that verbatim (audit finding
webshop-env-1: this client previously fed the engine-raw text, a per-turn train+eval prompt
divergence from the baseline; ALFWorld's fed manager has no such hook, so it was WebShop-only).
``FEDAGENT_WEBSHOP_FORMAT_OBS=0`` restores the raw rendering for A/B forensics only.
"""
import asyncio
import os
import random
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import httpx

from fedagent.envs.base import BaseTextEnv, Obs, resolve_service_url
from fedagent.envs.legacy_prompts import build_webshop_obs

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


def _format_obs(raw: str, task: str) -> str:
    """Verbatim replica of the fed baseline's ``WebshopEnvironmentManager.format_obs``
    (paper-reproduce-verl-agent third_party/verl-agent/agent_system/environments/
    fed_env_manager.py:387-398): split on `` [SEP] ``, locate the instruction segment,
    keep only what FOLLOWS it, and single-quote every remaining part. Applied to every
    reset/step observation before it enters the prompt or the history memory -- exactly
    where the original applied it. On any miss, fall back to the raw text (the original's
    bare ``except`` did the same); a stripped-comparison retry guards against cosmetic
    whitespace drift between ``_extract_task`` and the page rendering.
    """
    parts = raw.split(" [SEP] ")
    try:
        index = parts.index(task)
    except ValueError:
        index = next((k for k, p in enumerate(parts) if p.strip() == task), None)
        if index is None:
            return raw
    return " [SEP] ".join(f"'{p}'" for p in parts[index + 1:])


class WebShopEnv(BaseTextEnv):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(env_config)
        # Per-client routing: FEDAGENT_SERVICE_URL_FILE (persistent/cross-round, lever #4) wins;
        # else WEBSHOP_SERVICE_URL (subprocess path sets it per client); else spec service_url;
        # else default. See resolve_service_url for why the file beats process-env in persistent mode.
        self.base_url = resolve_service_url("WEBSHOP_SERVICE_URL", self.env_config,
                                            "http://localhost:8080")
        self.timeout = float(self.env_config.get("timeout", 120.0))
        self.session_id = uuid4().hex
        self._task = ""
        self._goal_id = None   # asin of the current goal (only when the service logs it)
        self._step_id = 0      # idempotency key for /step (incremented only after a success)
        # WINDOWED (faithful) mode: history_length>0 reproduces the paper's per-turn prompt
        # (task + last-N (obs, action) pairs + current obs). 0 (default) = concat mode (per-turn
        # body only; the GymTextAgentLoop supplies history as the growing chat).
        # FEDAGENT_HISTORY_LENGTH (set by run_fed per rollout_mode: windowed=2, concat=0) is
        # AUTHORITATIVE so ONE shared env spec drives both modes; spec history_length is the fallback.
        self._history_length = int(os.environ.get("FEDAGENT_HISTORY_LENGTH")
                                   or self.env_config.get("history_length", 0))
        # fed-baseline observation rendering (_format_obs). Default ON = faithful; the kill
        # switch exists only so a pre-fix run can be reproduced for A/B forensics.
        self._format_obs_on = os.environ.get("FEDAGENT_WEBSHOP_FORMAT_OBS", "1") != "0"
        self._memory: list = []     # [{"text_obs": <formatted obs before action>, "action": <projected action>}]
        self._pre_obs = ""          # formatted obs that led to the pending action (legacy fed pre_text_obs)
        self._client: Optional[httpx.AsyncClient] = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def _post(self, path: str, payload: dict, *, retry: bool = False,
                    block: bool = False, retries: int = 8, base: float = 0.3):
        """POST to the env service; raise on HTTP errors, and (only when ``retry``) retry transport errors.

        ``retry=True`` is used for ALL stateful endpoints (/create, /reset, /step): at the full PPO
        batch the rollout fires train_batch_size x rollout.n episodes at once, so they hit this
        client's pooled per-client service near-simultaneously and the HTTP boundary is overwhelmed
        (sockets reset mid-response -> httpx.ReadError). Bounded backoff + jitter spreads the retried
        requests across the pool. /step mutates env state, so a naive replay would corrupt the
        trajectory -- it is made retry-SAFE by an idempotency key (``step_id``): the server applies
        each id exactly once and replays the cached response for a re-sent id (see service/server.py).
        We therefore increment ``self._step_id`` only AFTER a success, so the in-flight id is the only
        one ever re-sent. raise_for_status() ensures a 4xx/5xx body (e.g. {"detail":"unknown session"}
        or a 409 step-ordering error) is never silently parsed as an empty observation, and -- being an
        HTTPStatusError, not a TransportError -- is NOT retried (a real desync surfaces loudly).

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
        self._step_id = 0   # fresh episode -> restart the /step idempotency counter (server does too)
        d = r.json()
        raw = d.get("obs", "") or ""
        self._task = _extract_task(raw)   # extract from RAW (formatting strips the instruction)
        self._goal_id = d.get("goal_id")   # asin (hardness-labelling pass only); None normally
        avail_str = _fmt_actions(d.get("available_actions", {}))
        obs_txt = _format_obs(raw, self._task) if self._format_obs_on else raw
        if self._history_length > 0:        # WINDOWED (faithful) mode: full legacy template
            self._memory = []
            self._pre_obs = obs_txt
            obs_str = build_webshop_obs(task=self._task, memory=self._memory, current_obs=obs_txt,
                                        available_str=avail_str, history_length=self._history_length,
                                        init=True)
        else:                               # concat mode: per-turn body only (fed rendering too)
            obs_str = _FIRST_OBS.format(task=self._task, obs=obs_txt, actions=avail_str)
        return {"obs_str": obs_str}, {}

    async def step(self, action_str: str) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        # retry=True is SAFE here: step_id makes the server apply/replay exactly once. Increment
        # only after the await returns (success) so a retried request always carries this same id.
        r = await self._post(
            "/step",
            {"session_id": self.session_id, "text": action_str, "step_id": self._step_id},
            retry=True,
        )
        self._step_id += 1
        d = r.json()
        raw = d.get("obs", "") or ""
        avail_str = _fmt_actions(d.get("available_actions", {}))
        obs_txt = _format_obs(raw, self._task) if self._format_obs_on else raw
        if self._history_length > 0:        # WINDOWED (faithful) mode
            # store (FORMATTED obs that led to this action, the PROJECTED action) — matches the
            # fed baseline's memory.store({text_obs: pre_text_obs, action: projection_f(text)}),
            # whose pre_text_obs was the format_obs output. The service parses the action
            # server-side and returns it as "action" (fallback: raw text).
            self._memory.append({"text_obs": self._pre_obs, "action": d.get("action", action_str)})
            self._pre_obs = obs_txt
            obs_str = build_webshop_obs(task=self._task, memory=self._memory, current_obs=obs_txt,
                                        available_str=avail_str, history_length=self._history_length,
                                        init=False)
        else:                               # concat mode (fed rendering too)
            obs_str = _STEP_OBS.format(obs=obs_txt, actions=avail_str)
        info = {
            "success": bool(d.get("success", False)),
            "is_action_valid": bool(d.get("is_action_valid", True)),
        }
        if d.get("task_score") is not None:
            # dense graded score in [0,1] -> the Task Score metric. The server always sends it
            # (service/server.py); it was previously dropped here, so info['task_score'] was never
            # set and the agent-loop task_score plumbing (-> dump -> figure) got None. Gated so
            # non-WebShop / older servers stay None (figure silently skipped, unchanged).
            info["task_score"] = float(d["task_score"])
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
