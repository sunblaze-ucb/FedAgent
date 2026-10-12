"""GymTextAgentLoop._tokenize_chat returns list[int] under transformers>=5, whose
apply_chat_template(tokenize=True) defaults to a BatchEncoding (pydantic then rejects
AgentLoopOutput.prompt_ids on the first rollout). The stub tokenizer mimics the 5.x default, so
the test also guards the contract on 4.x CI. No model download, no GPU."""
import asyncio
import types

import pytest

pytest.importorskip("verl")
from transformers import BatchEncoding  # noqa: E402

from fedagent.agent_loops.gym_text_agent_loop import GymTextAgentLoop  # noqa: E402

IDS = [151644, 872, 198, 13048]
MESSAGES = [{"role": "user", "content": "Make your first guess as <answer>N</answer>."}]


class Tok5:
    """transformers>=5 apply_chat_template: tokenize=True returns a BatchEncoding unless
    return_dict=False."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True,
                            return_dict=True, **kwargs):
        self.calls.append(dict(kwargs, return_dict=return_dict))
        if tokenize and return_dict:
            return BatchEncoding({"input_ids": list(IDS), "attention_mask": [1] * len(IDS)})
        return list(IDS)


def _tokenize(template_kwargs):
    tok = Tok5()

    async def go():
        stub = types.SimpleNamespace(loop=asyncio.get_running_loop(), tokenizer=tok,
                                     apply_chat_template_kwargs=template_kwargs)
        return await GymTextAgentLoop._tokenize_chat(stub, MESSAGES)

    return asyncio.run(go()), tok


def test_returns_token_list_not_batch_encoding():
    ids, _ = _tokenize({})
    assert ids == IDS and type(ids) is list


def test_user_template_kwargs_pass_through():
    _, tok = _tokenize({"enable_thinking": False})
    assert tok.calls == [{"enable_thinking": False, "return_dict": False}]


def test_user_return_dict_cannot_break_the_contract():
    ids, tok = _tokenize({"return_dict": True})       # no duplicate-keyword TypeError
    assert ids == IDS and tok.calls[0]["return_dict"] is False
