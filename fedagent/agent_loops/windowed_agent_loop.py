"""WINDOWED (faithful) rollout — reproduces the verl-agent 0.3.1 paper rollout.

The paper ran ``env.history_length=2``: each turn the policy saw a FRESH prompt =
task + the most-recent 2 (obs, action) pairs + the current obs (NOT a growing full
concat), and EACH TURN was a separate training sample. This is the OPT-IN faithful mode;
the stock full-concat ``GymTextAgentLoop`` is unchanged and remains the other mode.

Division of labor (faithful to legacy, where history lived in env_manager.build_text_obs):
  * The ENV CLIENT (history_length>0) builds the full legacy windowed template and returns
    it as ``obs_str`` (see fedagent/envs/legacy_prompts.py). So this loop's per-turn prompt
    is exactly one user message = that template (legacy used a single user turn, no system
    message: rollout_loop.py:74 ``chat=[{content: obs, role: user}]``).
  * This loop generates per turn and emits ONE AgentLoopOutput PER TURN (response = the
    action only -> response_mask all 1s; no obs tokens in the response, unlike concat).
  * The episode return (sum of sparse step rewards) is broadcast to EVERY turn of the
    trajectory, and a per-episode ``traj_uid`` is tagged so the faithful GRPO advantage
    (grpo_traj) can group/normalize over N trajectories (not over per-turn samples).

The per-turn -> batch expansion + uid/traj_uid tagging is done by WindowedAgentLoopManager/
Worker (fedagent/agent_loops/windowed_manager.py), selected together via the rollout_mode switch.
"""
from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register

from fedagent.agent_loops.gym_text_agent_loop import GymTextAgentLoop
from fedagent.envs.registry import make_env


@register("gym_text_windowed")
class WindowedGymTextAgentLoop(GymTextAgentLoop):
    """Per-turn, windowed-history variant. ``run()`` (single-sample concat) is inherited so
    the class is harmless if invoked the stock way; the windowed worker calls
    ``run_episode_windowed`` to get one sample per turn."""

    async def run_episode_windowed(self, sampling_params: Dict[str, Any], validate: bool = False,
                                   **kwargs) -> List[AgentLoopOutput]:
        env_name = kwargs.get("env_name", "TinyGuess")
        env = make_env(env_name, kwargs.get("config", {}) or {})
        seed = int(kwargs.get("seed", 0))
        max_turns = int(kwargs.get("max_turns", 6))
        outputs: List[AgentLoopOutput] = []
        env_rewards: List[float] = []
        turn_invalid: List[float] = []   # per-turn 1.0 if THAT turn's action was invalid, else 0.0
        success = False
        traj_uid = uuid4().hex   # one id per trajectory (broadcast advantage groups by it)
        try:
            init_obs, _ = await env.reset(seed=seed)
            cur_obs = init_obs["obs_str"]    # env already built the full windowed template
            for _ in range(max_turns):
                # legacy chat is a SINGLE user message (no system turn): the instruction is
                # inside the windowed template the env returned.
                messages = [{"role": "user", "content": cur_obs}]
                prompt_ids = await self._tokenize_chat(messages)
                if len(prompt_ids) >= self._max_ctx - 1:        # never exceed the server ctx window
                    prompt_ids = prompt_ids[-(self._max_ctx - 1):]
                out = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
                action_ids = out.token_ids
                text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
                obs, reward, done, info = await env.step(text)
                env_rewards.append(float(reward))
                success = bool(info.get("success", success))
                turn_invalid.append(0.0 if info.get("is_action_valid", True) else 1.0)
                resp = action_ids[: self.response_length]
                # rollout log-probs (present iff rollout.calculate_log_probs=true), sliced with
                # resp; the response here is the action only, so no obs padding is needed.
                resp_lp = (out.log_probs[: len(resp)]
                           if out.log_probs is not None and len(out.log_probs) == len(action_ids)
                           else None)
                outputs.append(AgentLoopOutput(
                    prompt_ids=prompt_ids[-self.prompt_length:],
                    response_ids=resp,
                    response_mask=[1] * len(resp),     # response is the action only (no obs tokens)
                    response_logprobs=resp_lp,
                    num_turns=1,
                    reward_score=0.0,                  # set below: episode return broadcast
                    metrics={},
                    extra_fields={"turn_scores": [], "tool_rewards": [], "traj_uid": traj_uid,
                                  "reward_extra_info": {"traj_success": float(success)}},
                ))
                cur_obs = obs["obs_str"]
                if done:
                    break
        finally:
            await env.close()
        # Faithful legacy reward (per-turn samples): EpisodeRewardManager places the FULL episode
        # return at every turn's last token (normalize_by_length=False), and apply_invalid_action_
        # penalty then subtracts the penalty coef ONLY at turns whose OWN action was invalid -- a
        # per-turn deduction, NOT a uniform episode-level one. So broadcast the base episode return
        # and subtract the penalty per turn. (Concat mode = 1 sample/episode, so its episode-level
        # `- coef * n_invalid` is the correct collapse of the same thing.)
        # The invalid-action penalty is TRAIN-only (legacy applies apply_invalid_action_penalty in
        # the train reward path; val_reward_fn is the unpenalized episode return). So eval rows carry
        # the pure episode return -> the eval collapse (last turn) reports the faithful success/return.
        base_return = float(sum(env_rewards))
        for o, inv in zip(outputs, turn_invalid):
            penalty = 0.0 if validate else self._invalid_penalty * inv
            o.reward_score = base_return - penalty
            o.extra_fields["reward_extra_info"]["traj_success"] = float(success)
        return outputs
