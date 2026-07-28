"""Pure, dependency-free helpers for the concat agent-loop's inter-turn glue.

Kept import-light (no verl, no torch) so it can be unit-tested offline —
``gym_text_agent_loop.py`` itself imports verl and cannot be imported without the full stack.

Background — why this exists
----------------------------
The concat loop builds ONE token sequence per episode: ``prompt + Σ_t (action_t + glue_t)``,
where ``glue_t`` is the inter-turn text between one action and the next generation point
(assistant-close ``<|im_end|>`` + the observation as a user turn + the next generation prompt).
Actions are the RAW sampled token ids (so their rollout logprobs stay aligned); the glue is
masked out of the loss (``response_mask = 0``).

The historical way to get the glue was a blind TOKEN slice::

    obs_tokens = new_ids[len(cur_ids):]            # cur_ids == prompt + raw action ids

which assumes the fresh full re-render ``new_ids`` has ``cur_ids`` as a token prefix. That is
FALSE for reasoning-model templates: the turn-t generation prompt ends with a scaffold
(Qwen3 / Qwen3.5: ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``) that is DROPPED when the
same turn is re-rendered as completed history. So ``cur_ids`` (which ends in that scaffold +
the raw action) is not a prefix of ``new_ids``, and the slice is offset by the scaffold length
— on Qwen3.5 it silently eats the ``<|im_end|>\\n<|im_start|>user`` turn boundary every turn
(~4 tokens). On Qwen2.5 (no scaffold) the slice is exactly correct (0 offset), which is why the
loop was fine when it was written. See ``inter_turn_glue_ids`` for the fix.
"""
from __future__ import annotations

from typing import List, Optional


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def inter_turn_glue_ids(
    rt_prev: str,
    rt_next: str,
    action_text: str,
    tokenizer,
    gen_last_id: Optional[int] = None,
) -> Optional[List[int]]:
    """Token ids of the glue that FOLLOWS ``action_text`` in the next rendered prompt.

    Recovers the glue by a **string** diff of the two rendered prompts (never a token diff, so
    it is immune to BPE context-dependence):

    - ``rt_prev`` = this turn's rendered prompt (ends in any generation-prompt scaffold);
    - ``rt_next`` = next turn's rendered prompt (re-renders this turn's action as history,
      WITHOUT the scaffold, then appends the observation and the next generation prompt).

    ``rt_prev`` and ``rt_next`` diverge exactly where this turn's action content begins, so we
    anchor at ``common_prefix_len`` and take everything in ``rt_next`` after the action text.
    The ``rfind`` window ``[0, cp + len(action_text))`` keeps a later echo of the action inside
    the *next observation* from being mistaken for the action itself.

    ``gen_last_id``: last id of the RAW sampled action. If it equals the glue's leading
    turn-terminator id (i.e. the sampler already emitted ``<|im_end|>`` and it is trained as
    part of the action), that terminator is dropped from the glue so it is not doubled.

    REASONING-MODEL HISTORY REWRITE (2026-07-28 fix): Qwen3-family templates REWRITE a
    completed assistant turn when it is re-rendered as history — the Jinja does
    ``content.split('</think>')[-1].lstrip('\n')``, deleting the whole ``<think>…</think>``
    block — so the RAW sampled text never occurs in ``rt_next`` and the full-text anchor
    misses on every *valid* action (the system prompts mandate ``<think>``, and think-less
    actions are scored invalid). The kept form is a SUFFIX of the raw action, so when the
    full text misses we retry with (a) the known ``</think>``-strip candidate, then (b) each
    newline-delimited tail of the action, longest first (template rewrites drop a PREFIX at
    a block boundary, so the kept tail starts after a newline). Tails shorter than
    ``_MIN_SUFFIX_ANCHOR`` are junk-prone (a 1–2 char tail matches almost anywhere), so
    below the floor we return ``None`` and the caller's legacy fallback applies, exactly as
    before.

    Returns the glue ids, or ``None`` if the action text cannot be located (the caller should
    fall back to the legacy token slice — never worse than before).
    """
    if not action_text:
        return None
    cp = common_prefix_len(rt_prev, rt_next)
    matched, pos = action_text, rt_next.rfind(action_text, 0, cp + len(action_text))
    if pos < 0:
        matched, pos = _anchor_rewritten_action(rt_next, action_text, cp)
    if pos < 0:
        return None
    glue = rt_next[pos + len(matched):]
    ids = [int(x) for x in tokenizer.encode(glue, add_special_tokens=False)]
    if gen_last_id is not None and ids and ids[0] == int(gen_last_id):
        ids = ids[1:]
    return ids


_MIN_SUFFIX_ANCHOR = 8   # shortest action suffix we trust as an anchor (see docstring above)


def _anchor_rewritten_action(rt_next: str, action_text: str, cp: int):
    """Locate the template-REWRITTEN form of ``action_text`` in ``rt_next``: the known
    Qwen3-family ``</think>``-strip first, else each newline-delimited tail of the action
    (longest first — a template rewrite drops a PREFIX at a block boundary). Every candidate
    uses the same anchored window as the full-text probe (match must START at or before the
    divergence point ``cp``). Returns ``(matched_text, pos)`` or ``("", -1)``."""
    # (a) the known rewrite: history keeps only the post-</think> tail, leading newlines stripped
    kept = action_text.split("</think>")[-1].lstrip("\n")
    if len(kept) >= _MIN_SUFFIX_ANCHOR and kept != action_text:
        pos = rt_next.rfind(kept, 0, cp + len(kept))
        if pos >= 0:
            return kept, pos

    # (b) generic prefix-strip: try the tail after each newline (ascending newline index ==
    # descending tail length, so the FIRST anchoring tail is the longest — the true kept form,
    # not a junk sub-tail). Cost: one windowed rfind per action line, only on the miss path.
    start = 0
    while True:
        nl = action_text.find("\n", start)
        if nl < 0:
            return "", -1
        start = nl + 1
        kept = action_text[start:].lstrip("\n")
        if len(kept) < _MIN_SUFFIX_ANCHOR:
            return "", -1
        pos = rt_next.rfind(kept, 0, cp + len(kept))
        if pos >= 0:
            return kept, pos
