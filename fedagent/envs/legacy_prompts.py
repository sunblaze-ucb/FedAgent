"""Legacy verl-agent windowed-history prompts — VERBATIM from the 0.3.1 baseline.

The FedAgent paper ran verl-agent-fedagent with ``env.history_length=2`` (the
``ppo_trainer.yaml`` default; no run overrides it). Each turn the policy saw a FRESH
prompt = task + the most-recent ``history_length`` (observation, action) pairs + the current
observation — NOT a growing full-trajectory concat. These templates + ``build_*_obs`` replicate
that exactly so the windowed (faithful) rollout mode reproduces the paper's observation.

Sources (verbatim):
  - templates: agent_system/environments/prompts/{alfworld,webshop}.py
  - build logic: agent_system/environments/env_manager.py build_text_obs (alfworld :85-110, webshop :379-410)

Equivalence-critical details preserved:
  * history is the last ``history_length`` (raw obs, action) pairs, formatted as
    ``[Observation N: '<obs>', Action N: '<action>']`` (N is the absolute step number),
    newline-joined then ``.strip()``-ed.
  * ALFWORLD_TEMPLATE_NO_HIS has NO task line; WEBSHOP_TEMPLATE_NO_HIS has one.
  * WebShop falls back to NO_HIS when the rendered obs exceeds 13000 chars.
  * admissible/available action formatting matches the env clients' ``_fmt_actions``
    (alfworld ``"\n ".join(f"'{s}'")``; webshop ``"\n".join(f"'{s}',")``).
"""
from __future__ import annotations

from typing import Dict, List

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

def _action_history(memory: List[Dict[str, str]], history_length: int) -> tuple[str, int]:
    """Format the last ``history_length`` (obs, action) pairs exactly like env_manager.

    Returns (action_history_text, valid_history_length). ``memory`` entries are
    ``{"text_obs": <obs BEFORE the action>, "action": <action text>}`` — the obs exactly as
    the legacy fed manager stored ``pre_text_obs``: for WebShop that is the ``format_obs``
    output (prefix-stripped, parts single-quoted — see webshop_env._format_obs), for
    ALFWorld the raw text (its fed manager has no format hook).
    """
    recent = memory[-history_length:]
    valid = len(recent)
    start_index = len(memory) - valid
    parts = ""
    for j, rec in enumerate(recent):
        step_number = start_index + j + 1
        parts += f"\n[Observation {step_number}: '{rec['text_obs']}', Action {step_number}: '{rec['action']}']"
    return parts.strip(), valid


def build_alfworld_obs(*, task: str, memory: List[Dict[str, str]], current_obs: str,
                       admissible_str: str, history_length: int, init: bool) -> str:
    """Replicate env_manager.build_text_obs (ALFWorld). ``admissible_str`` already formatted."""
    if init or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=current_obs, admissible_actions=admissible_str)
    action_history, valid = _action_history(memory, history_length)
    return ALFWORLD_TEMPLATE.format(
        task_description=task,
        step_count=len(memory),
        history_length=valid,
        action_history=action_history,
        current_step=len(memory) + 1,
        current_observation=current_obs,
        admissible_actions=admissible_str,
    )


def build_webshop_obs(*, task: str, memory: List[Dict[str, str]], current_obs: str,
                      available_str: str, history_length: int, init: bool) -> str:
    """Replicate env_manager.build_text_obs (WebShop), incl. the 13000-char NO_HIS fallback."""
    if init or history_length <= 0:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task, current_observation=current_obs, available_actions=available_str)
    action_history, valid = _action_history(memory, history_length)
    obs = WEBSHOP_TEMPLATE.format(
        task_description=task,
        step_count=len(memory),
        history_length=valid,
        action_history=action_history,
        current_step=len(memory) + 1,
        current_observation=current_obs,
        available_actions=available_str,
    )
    if len(obs) > 13000:   # legacy guard: overly long history -> drop to no-history template
        obs = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task, current_observation=current_obs, available_actions=available_str)
    return obs
