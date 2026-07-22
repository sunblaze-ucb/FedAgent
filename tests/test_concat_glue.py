"""Regression test for the concat agent-loop inter-turn glue (fedagent.agent_loops._concat_glue).

Locks the Qwen2.5-vs-Qwen3 behaviour:

- Qwen2.5-style template (no <think> scaffold): the anchored glue equals the legacy blind
  token slice exactly — 0 mis-slice, no regression.
- Qwen3-style template (<think> scaffold in the generation prompt, dropped in history): the
  legacy blind slice EATS the leading <|im_end|> turn boundary every turn; the anchored glue
  keeps the full, correct boundary.

Pure offline (no verl, no torch, no network): gym_text_agent_loop.py imports verl, so the
glue logic lives in the verl-free _concat_glue module and is tested here directly.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.agent_loops._concat_glue import inter_turn_glue_ids  # noqa: E402

IM_END, IM_START = 1000, 1001


class MockTok:
    """Char-level tokenizer, except <|im_end|>/<|im_start|> map to single ids — like a real
    tokenizer's special tokens (so terminator dedup and boundary checks are realistic)."""
    SPECIALS = {"<|im_end|>": IM_END, "<|im_start|>": IM_START}

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            for sp, sid in self.SPECIALS.items():
                if s.startswith(sp, i):
                    out.append(sid)
                    i += len(sp)
                    break
            else:
                out.append(ord(s[i]))
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        rev = {v: k for k, v in self.SPECIALS.items()}
        return "".join(rev.get(i, chr(i)) if not (skip_special_tokens and i in rev) else "" for i in ids)


TOK = MockTok()
PREFIX = "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nGuess 1-50<|im_end|>\n<|im_start|>assistant\n"
SCAFFOLD = "<think>\n\n</think>\n\n"
ACTION = "<answer>25</answer>"
OBS = "Too high."
# the glue that SHOULD follow the action: assistant-close + user obs + next generation prompt
GLUE_NOSCAF = f"<|im_end|>\n<|im_start|>user\n{OBS}<|im_end|>\n<|im_start|>assistant\n"
GLUE_SCAF = GLUE_NOSCAF + SCAFFOLD


def legacy_blind_slice(rt_prev, rt_next, action):
    """Reproduce gym_text_agent_loop.py's old obs_tokens = new_ids[len(cur_ids):]."""
    buf = TOK.encode(rt_prev) + TOK.encode(action)     # cur_ids = prompt + raw action
    new_ids = TOK.encode(rt_next)
    return new_ids[len(buf):] if len(new_ids) > len(buf) else []


def test_qwen25_no_scaffold_no_regression():
    rt_prev = PREFIX                                    # gen prompt ends at assistant\n (no scaffold)
    rt_next = PREFIX + ACTION + GLUE_NOSCAF
    correct = TOK.encode(GLUE_NOSCAF)
    new_glue = inter_turn_glue_ids(rt_prev, rt_next, ACTION, TOK, gen_last_id=None)
    old_obs = legacy_blind_slice(rt_prev, rt_next, ACTION)
    assert new_glue == correct, "anchored glue must be exact on Qwen2.5"
    assert old_obs == correct, "legacy slice is already correct on Qwen2.5 (0 mis-slice)"
    assert new_glue == old_obs, "no regression: identical to legacy on non-scaffold templates"
    assert new_glue[0] == IM_END, "glue starts with the assistant-closing <|im_end|>"
    print("qwen2.5 (no scaffold): anchored == legacy == correct, boundary intact  OK")


def test_qwen3_scaffold_fix():
    rt_prev = PREFIX + SCAFFOLD                         # gen prompt ends with the <think> scaffold
    rt_next = PREFIX + ACTION + GLUE_SCAF               # history re-renders the action WITHOUT scaffold
    correct = TOK.encode(GLUE_SCAF)
    new_glue = inter_turn_glue_ids(rt_prev, rt_next, ACTION, TOK, gen_last_id=None)
    old_obs = legacy_blind_slice(rt_prev, rt_next, ACTION)
    # the fix: anchored glue is exactly correct, boundary preserved
    assert new_glue == correct, "anchored glue must recover the full boundary on Qwen3"
    assert new_glue[0] == IM_END, "anchored glue keeps the assistant-closing <|im_end|>"
    # the bug: legacy slice eats the leading boundary (offset by the scaffold length)
    assert old_obs != correct, "legacy slice must differ (it mis-slices on Qwen3)"
    assert old_obs[0] != IM_END, "legacy slice ATE the <|im_end|> turn boundary"
    eaten = len(correct) - len(old_obs)
    assert eaten == len(TOK.encode(SCAFFOLD)), "mis-slice equals the scaffold token length"
    print(f"qwen3 (scaffold): anchored recovers full boundary; legacy ate {eaten} leading tokens  OK")


def test_terminator_dedup():
    rt_prev = PREFIX + SCAFFOLD
    rt_next = PREFIX + ACTION + GLUE_SCAF
    # sampler already emitted <|im_end|> as the action's last token -> must NOT be doubled
    glue = inter_turn_glue_ids(rt_prev, rt_next, ACTION, TOK, gen_last_id=IM_END)
    assert glue[0] != IM_END, "leading <|im_end|> dropped when the action already ended with it"
    assert glue == TOK.encode(GLUE_SCAF)[1:], "exactly one terminator removed"
    print("terminator dedup: no doubled <|im_end|> when gen ended with one  OK")


def test_action_not_found_falls_back():
    rt_prev = PREFIX + SCAFFOLD
    rt_next = PREFIX + ACTION + GLUE_SCAF
    assert inter_turn_glue_ids(rt_prev, rt_next, "NONEXISTENT-ACTION", TOK) is None
    assert inter_turn_glue_ids(rt_prev, rt_next, "", TOK) is None
    print("fallback: returns None when the action can't be located (caller keeps legacy slice)  OK")


if __name__ == "__main__":
    test_qwen25_no_scaffold_no_regression()
    test_qwen3_scaffold_fix()
    test_terminator_dedup()
    test_action_not_found_falls_back()
    print("\nALL CONCAT-GLUE REGRESSION TESTS PASS")
