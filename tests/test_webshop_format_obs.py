"""Regression test for the WebShop fed-baseline observation rendering (audit webshop-env-1).

The paper's federated runs (main_ppo_fed.py -> fed_make_envs) post-process EVERY WebShop
observation with ``WebshopEnvironmentManager.format_obs`` (fed_env_manager.py:387-398): drop
the page prefix up to and including the instruction segment, single-quote each remaining
`` [SEP] `` part. The overlay client previously fed the engine-raw text into the prompt AND
the history memory -- a per-turn train+eval prompt divergence, WebShop-only (ALFWorld's fed
manager has no such hook). This test locks ``webshop_env._format_obs`` byte-for-byte against
a verbatim re-implementation of the fed original.

Pure offline: only the client module is imported (httpx, no verl/torch/network).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedagent.envs.webshop.webshop_env import _extract_task, _format_obs  # noqa: E402

TASK = "i need a gluten free, dairy free vanilla cake mix, and price lower than 40.00 dollars"

LANDING = f"WebShop [SEP] Instruction: [SEP] {TASK} [SEP] Search"
RESULTS = (
    f"Instruction: [SEP] {TASK} [SEP] Back to Search [SEP] Page 1 (Total results: 50) "
    f"[SEP] Next > [SEP] B078GWRC1J [SEP] Bright Citrus Deodorant by BeeFriendly [SEP] $10.99 "
    f"[SEP] B08KBVJ4XN [SEP] Vanilla Cake Mix, Gluten Free [SEP] $15.95"
)
ITEM = (
    f"Instruction: [SEP] {TASK} [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 16 ounce "
    f"[SEP] Price: $15.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews "
    f"[SEP] Buy Now"
)


# --- verbatim re-implementation of the fed originals (the oracle) --------------------------
# fed_env_manager.py:379-385 (extract_task) and :387-398 (format_obs), batch signature kept.

def fed_extract_task(text_obs):
    tasks = []
    for obs in text_obs:
        parts = obs.split(" [SEP] ")
        assert parts[1] == "Instruction:"
        tasks.append(parts[2])
    return tasks


def fed_format_obs(text_obs, tasks):
    postprocess_text_obs = []
    for i in range(len(text_obs)):
        parts = text_obs[i].split(" [SEP] ")
        try:
            index = parts.index(tasks[i])
            reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index + 1 :])
        except:  # noqa: E722 -- the original's bare except
            reformatted_obs = text_obs[i]
        postprocess_text_obs.append(reformatted_obs)
    return postprocess_text_obs


def test_extract_task_matches_fed_oracle():
    assert _extract_task(LANDING) == fed_extract_task([LANDING])[0] == TASK


def test_format_obs_matches_fed_oracle_bytewise():
    task = fed_extract_task([LANDING])[0]
    for page in (LANDING, RESULTS, ITEM):
        assert _format_obs(page, task) == fed_format_obs([page], [task])[0]


def test_landing_page_formats_to_quoted_search():
    assert _format_obs(LANDING, TASK) == "'Search'"


def test_instruction_prefix_dropped_and_parts_quoted():
    out = _format_obs(RESULTS, TASK)
    assert "Instruction:" not in out and TASK not in out
    assert out.startswith("'Back to Search'")
    parts = out.split(" [SEP] ")
    assert all(p.startswith("'") and p.endswith("'") for p in parts)
    assert "'$15.95'" in parts and "'B078GWRC1J'" in parts


def test_task_absent_falls_back_to_raw():
    page = "Thank you for shopping with us! [SEP] Your code: [SEP] None"
    assert _format_obs(page, TASK) == page   # the original's bare-except raw fallback


def test_stripped_match_guard():
    # cosmetic whitespace drift between the stored task and the page segment must not
    # silently disable the formatting (defensive extension over the fed original, which
    # would have fallen back to raw here). Two spaces before [SEP]: after the
    # " [SEP] " split the task segment carries a trailing space -> exact .index misses.
    page = f"Instruction: [SEP] {TASK}  [SEP] Back to Search [SEP] Buy Now"
    out = _format_obs(page, TASK)
    assert out == "'Back to Search' [SEP] 'Buy Now'"
