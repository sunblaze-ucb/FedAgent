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

EVAL TRAJECTORY CAPTURE (``trajectory`` column in the val dump)
--------------------------------------------------------------
Eval COLLAPSES each episode to its last turn (windowed_manager.py: verl's ``_validate`` does
``test_batch.union(test_output_gen_batch)`` and requires 1:1, and ``summarize_val_dump`` means
over ROWS -- K rows per episode would weight long episodes more and silently corrupt
success_rate). So the val dump only ever showed the TERMINAL step. The other turns are not
recovered by expanding eval; they are carried as a payload on the row that survives.

``reward_extra_info`` is the existing channel for that (``goal_id``/``task_type`` already ride
it), and verl's ``_write_generations`` turns any length-matching key into a JSONL column
verbatim. Two constraints shape the implementation:

  * ``agent_loop.py`` takes the key set from ``reward_extra_infos[0]`` and then does
    ``info[key]`` for EVERY row -- a key present on only some rows is a KeyError. Attaching to
    ``outputs[-1]`` is exactly right: that is the row eval keeps, so every eval row has it, and
    TRAIN attaches nothing at all (key absent everywhere -> consistent).
  * The payload is a **dict, not a JSON string**. ``np.array([...])`` over strings yields a
    ``<U<maxlen>`` array that pads every row to the longest one at 4 bytes/char; over dicts it
    yields ``dtype=object`` and stores references. It also serializes as a nested object rather
    than an escaped blob. ``tests/test_windowed_eval_trajectory.py`` locks both.

**OPT-IN**: off unless ``FEDAGENT_EVAL_TRAJECTORY_DUMP=1``. The payload is the dominant term in
a dump's size (it scales with turns actually played, up to 50 for ALFWorld), so a run that only
needs the metric should not pay for it; turn it on for the runs you intend to read. TRAIN is
never affected. Concat mode needs nothing: its single sample's prompt IS the whole conversation.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register

from fedagent.agent_loops.gym_text_agent_loop import GymTextAgentLoop
from fedagent.envs.registry import make_env

_TRAJ_DUMP_ENV = "FEDAGENT_EVAL_TRAJECTORY_DUMP"


def eval_trajectory_dump_enabled() -> bool:
    """Whether eval rows carry the full-trajectory payload. **Default OFF** -- opt in with
    ``FEDAGENT_EVAL_TRAJECTORY_DUMP=1``. Read per episode (not cached at import) so a long-lived
    persistent worker picks up a flip without a restart."""
    return os.environ.get(_TRAJ_DUMP_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


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
        task_score_val = None    # WebShop partial-credit score [0,1] (info['task_score']); stays None if the env doesn't expose it (e.g. ALFWorld)
        # Per-episode string tags the env optionally surfaces (PARITY with the concat loop):
        # goal_id (WebShop; REQUIRED by the hardness-labelling aggregation) / task_type
        # (ALFWorld eval breakdown). String-valued -> kept in verl's validation dump, skipped
        # by metric aggregation. Without this, windowed labelling dumps carry no goal_id and
        # gen_hardness_trajectories aborts at aggregation.
        tag_vals: Dict[str, Any] = {}
        traj_uid = uuid4().hex   # one id per trajectory (broadcast advantage groups by it)
        # Eval-only full-trajectory record (see the module docstring). Built during the loop
        # because `cur_obs` and the generation prompt are gone by the time it ends.
        capture = bool(validate) and eval_trajectory_dump_enabled()
        turn_records: List[Dict[str, Any]] = []
        try:
            init_obs, _ = await env.reset(seed=seed)
            cur_obs = init_obs["obs_str"]    # env already built the full windowed template
            for _ in range(max_turns):
                # legacy chat is a SINGLE user message (no system turn): the instruction is
                # inside the windowed template the env returned.
                messages = [{"role": "user", "content": cur_obs}]
                prompt_ids = await self._tokenize_chat(messages)
                # Cap the GENERATION prompt at prompt_length -- the SAME budget the emitted
                # training sample is cut to below -- not at max_ctx-1. The legacy stack tokenized
                # ONCE at max_prompt_length and used that tensor for generation AND training
                # (rollout_loop.py:115-137); capping only at max_ctx-1 here let the policy act on
                # up to max_ctx-1 tokens while the stored sample was re-cut to prompt_length, so
                # prompts in (prompt_length, max_ctx-1] TRAINED on a head-truncated context (the
                # task line goes first under left-truncation) while carrying the full reward.
                # min() keeps the server-ctx guard for configs where prompt_length >= max_ctx.
                cap = min(self.prompt_length, self._max_ctx - 1)
                prompt_truncated = len(prompt_ids) > cap
                if prompt_truncated:
                    prompt_ids = prompt_ids[-cap:]
                out = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
                action_ids = out.token_ids
                text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
                obs, reward, done, info = await env.step(text)
                env_rewards.append(float(reward))
                success = bool(info.get("success", success))
                if info.get("task_score") is not None:
                    task_score_val = float(info["task_score"])   # last (terminal) turn carries the episode's partial-credit score
                for tag in ("goal_id", "task_type"):
                    if info.get(tag) is not None:
                        tag_vals[tag] = info[tag]
                turn_invalid.append(0.0 if info.get("is_action_valid", True) else 1.0)
                resp = action_ids[: self.response_length]
                # rollout log-probs (present iff rollout.calculate_log_probs=true), sliced with
                # resp; the response here is the action only, so no obs padding is needed.
                resp_lp = (out.log_probs[: len(resp)]
                           if out.log_probs is not None and len(out.log_probs) == len(action_ids)
                           else None)
                if capture:
                    # `obs` = the env's windowed template for THIS turn (pre chat-template);
                    # `prompt` = what the model actually saw (chat-templated, left-truncated at
                    # `cap`, special tokens kept). They differ by exactly that rendering, which is
                    # why both are recorded. `action` is the response: in windowed mode the
                    # response IS the action (no obs tokens), so one field covers both -- but the
                    # stored training sample is cut to response_length, so flag when that bites.
                    turn_records.append({
                        "turn": len(turn_records),
                        "obs": cur_obs,
                        "prompt": self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
                        "action": text,
                        "reward": float(reward),
                        "done": bool(done),
                        "valid": bool(info.get("is_action_valid", True)),
                        "prompt_truncated": prompt_truncated,
                        "response_truncated": len(action_ids) > len(resp),
                    })
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
            if task_score_val is not None:
                o.extra_fields["reward_extra_info"]["task_score"] = task_score_val
            for tag, v in tag_vals.items():
                o.extra_fields["reward_extra_info"][tag] = v
        if capture and outputs:
            # LAST turn only -- that is the one eval keeps (windowed_manager collapses to
            # turn_outputs[-1]), so every eval row ends up carrying the key exactly once. Putting
            # it on all turns would be equivalent but would hold K copies of the payload.
            # Self-contained on purpose: whoever extracts just this column gets the tags too.
            outputs[-1].extra_fields["reward_extra_info"]["trajectory"] = {
                "n_turns": len(turn_records),
                "success": float(success),
                "episode_return": base_return,
                "task_score": task_score_val,
                **tag_vals,
                "turns": turn_records,
            }
        return outputs
