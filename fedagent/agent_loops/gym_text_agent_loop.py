"""GymTextAgentLoop — FedAgent's multi-turn text agent-loop for verl 0.8.

A verl ``AgentLoopBase`` subclass that drives ONE ``BaseTextEnv`` instance per dataset
row on verl's *native* async agent-loop seam:

    reset env -> (build prompt -> server.generate -> decode -> env.step -> append obs)* -> AgentLoopOutput

This is the verl-0.8 replacement for verl-agent's ``TrajectoryCollector.multi_turn_loop``.
It uses verl's STOCK trainer (no fork): the trajectory is returned as one
``AgentLoopOutput`` (concat multi-turn) with a response_mask that is 1 on
model-generated tokens and 0 on environment-observation tokens, so PPO/GRPO trains
only on the agent's actions. ``reward_score`` is the episode reward (GRPO groups are
formed by verl's ``rollout.n`` repeats of each row).

Fidelity note: per the agreed bar (SCIENTIFIC EQUIVALENCE), this uses verl 0.8's
native concat multi-turn rather than bit-reproducing the old per-turn/no-concat row
layout; outcome credit assignment over the action tokens is preserved.
"""
import logging
import os
from typing import Any, Dict, List
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from fedagent.envs.registry import make_env

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("gym_text")
class GymTextAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Hard context ceiling for the concat prompt (cur_ids) sent to the inference server.
        # verl's vllm_async_server.generate RAISES when len(prompt) >= max_model_len
        # (max_possible_tokens < 1), which aborts the WHOLE rollout batch -- not just the row.
        # On long multi-turn episodes (esp. ALFWorld) cur_ids can grow past it, so we stop the
        # episode cleanly one token short instead. Falls back to prompt+response when
        # max_model_len is unset (the server's own default budget).
        self._max_ctx = int(
            getattr(self.rollout_config, "max_model_len", None)
            or (self.prompt_length + self.response_length)
        )
        # Invalid-action reward shaping: the original sets use_invalid_action_penalty=True
        # (coef 0.1) in ALL 177 configs and stock verl 0.8 has no such hook, so we apply it
        # here -- subtract coef*(#invalid actions) from the episode reward, mirroring
        # verl-agent apply_invalid_action_penalty. Tunable via env var (0 disables).
        self._invalid_penalty = float(os.environ.get("FEDAGENT_INVALID_ACTION_PENALTY_COEF", "0.1"))

    async def _tokenize_chat(self, messages: List[Dict[str, Any]]) -> List[int]:
        """NON-truncating chat tokenization for the concat loop. The base
        ``apply_chat_template`` left-truncates to ``rollout.prompt_length`` (2048); in a growing
        multi-turn conversation that silently drops the system prompt + task and corrupts the
        obs-token delta (``new_ids[len(cur_ids):]``) from ~turn 4 on. cur_ids is instead bounded
        by the ``_max_ctx`` overflow guard so generation never exceeds the context window, and
        the returned prompt/response are capped to prompt_length/response_length on return."""
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        env_name = kwargs.get("env_name", "TinyGuess")
        env = make_env(env_name, kwargs.get("config", {}) or {})
        seed = int(kwargs.get("seed", 0))
        max_turns = int(kwargs.get("max_turns", 6))

        # Borrow + run the env entirely inside try/finally: reset() does /create+/reset for remote
        # envs, so if /reset (or tokenize) raises AFTER /create borrowed a pooled session, finally
        # still runs env.close() to return it. Otherwise that session leaks and -- with block-on-
        # /create -- the pool eventually starves permanently (no env ever comes back).
        try:
            sys_obs = await env.system_prompt()
            init_obs, _ = await env.reset(seed=seed)
            messages = [
                {"role": "system", "content": sys_obs["obs_str"]},
                {"role": "user", "content": init_obs["obs_str"]},
            ]

            metrics: Dict[str, Any] = {}
            prompt_ids = await self._tokenize_chat(messages)
            cur_ids = list(prompt_ids)
            response_ids: List[int] = []
            response_mask: List[int] = []
            env_rewards: List[float] = []
            info: Dict[str, Any] = {}   # last-step info; init so the overflow-guard early-break is safe
            success = False
            turns = 0
            n_invalid = 0   # invalid actions this episode (for the invalid-action penalty)

            for _ in range(max_turns):
                # Stop before the concat prompt would overflow the server's context window
                # (it raises if len(prompt) >= max_model_len). Need >=1 token to generate.
                # No-op for WebShop, whose response_length cap already bounds cur_ids.
                if len(cur_ids) >= self._max_ctx - 1:
                    break
                with simple_timer("generate_sequences", metrics):
                    out: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=cur_ids, sampling_params=sampling_params
                    )
                gen = out.token_ids
                response_ids += gen
                response_mask += [1] * len(gen)  # 1 = model-generated -> trained on
                cur_ids = cur_ids + gen
                turns += 1

                text = self.tokenizer.decode(gen, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": text})
                obs, reward, done, info = await env.step(text)
                env_rewards.append(float(reward))
                success = bool(info.get("success", False))
                if not info.get("is_action_valid", True):
                    n_invalid += 1

                if done or len(response_ids) >= self.response_length:
                    break

                # append the env observation as a user turn; its tokens are NOT generated
                messages.append({"role": "user", "content": obs["obs_str"]})
                new_ids = await self._tokenize_chat(messages)
                obs_tokens = new_ids[len(cur_ids):] if len(new_ids) > len(cur_ids) else []
                response_ids += obs_tokens
                response_mask += [0] * len(obs_tokens)  # 0 = observation -> masked out of loss
                cur_ids = new_ids
        finally:
            # always release the env (e.g. return a pooled remote WebShop session),
            # even if generate/step raises mid-episode.
            await env.close()

        # Per-episode tags the env optionally surfaces (string-valued -> kept in verl's
        # validation dump but SKIPPED by metric aggregation): goal_id (WebShop asin, for the
        # hardness-trajectories labelling pass) / task_type (ALFWorld, for the eval breakdown).
        # Present only when the env provides them (gated env-side), so normal runs are unchanged.
        reward_extra_info = {"traj_success": float(success)}
        for tag in ("goal_id", "task_type"):
            if info.get(tag) is not None:
                reward_extra_info[tag] = info[tag]

        return AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=turns,
            # episode reward minus the invalid-action penalty (coef * #invalid actions)
            reward_score=float(sum(env_rewards)) - self._invalid_penalty * n_invalid,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "reward_extra_info": reward_extra_info,
            },
        )
