"""Regression test for the WebShop option-reward shape bug (docs/bugfixes.md 2026-07-25).

Stock ``get_reward`` compared two DIFFERENT shapes: ``list(options.values())`` (plain values,
what the agent clicked) against ``goal['goal_options'].items()`` ((key, value) tuples, what the
goal wants). ``normalize_color`` then rewrote only the clicked side -- ``norm_color in
color_string`` is a SUBSTRING test on ``str`` but a MEMBERSHIP test on ``tuple``, so tuples fell
through the loop and were returned unchanged, with no error. The two sides were then compared
with ``fuzz.token_set_ratio(...) > 85`` and could never agree once normalization actually
changed the clicked value (``'r.brown2060' -> 'brown'``, ``'fashion black girl01' -> 'ash'``
via "f-ash-ion", ``'rectangular' -> 'tan'`` via "rec-tan-gular").

458 of 6725 option-bearing goals (6.81%) could not reach 1.0 no matter what the agent did.

The decisive guard is ``test_oracle_best_purchase_scores_one``: buying the target product,
clicking every required option, at a qualifying price must score exactly 1.0 for EVERY goal in
the pool. No model, no rollout, no GPU. It failed on 458 goals before the fix.

Needs the vendored engine's deps (spacy + en_core_web_sm, thefuzz, rich) and its data files;
skips cleanly when they are absent. Set ``WEBSHOP_ORACLE_TEST_LIMIT=N`` to bound the sweep
(default: whole pool, ~35 s).
"""
import os
import random
import sys

import pytest

ENGINE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
    'fedagent/envs/webshop/engine/webshop',
))


@pytest.fixture(scope='module')
def engine():
    """Import the vendored engine, or skip. Chdir: its data paths are BASE_DIR-relative."""
    if ENGINE not in sys.path:
        sys.path.insert(0, ENGINE)
    try:
        from web_agent_site.engine import goal as goal_mod
        from web_agent_site.engine import normalize as norm_mod
    except Exception as exc:                                   # noqa: BLE001 - any dep gap skips
        pytest.skip(f'vendored WebShop engine not importable: {exc}')
    return goal_mod, norm_mod


@pytest.fixture(scope='module')
def pool(engine):
    """(product_item_dict, product_prices, goals) built exactly as SimServer builds them."""
    goal_mod, _ = engine
    from web_agent_site.engine.engine import load_products
    from web_agent_site.utils import DEFAULT_ATTR_PATH, DEFAULT_FILE_PATH

    if not os.path.exists(DEFAULT_FILE_PATH):
        pytest.skip(f'WebShop catalog missing: {DEFAULT_FILE_PATH}')

    cwd = os.getcwd()
    os.chdir(ENGINE)
    try:
        random.seed(42)                                        # SimServer's seeded window
        _, product_item_dict, product_prices, _ = load_products(
            filepath=DEFAULT_FILE_PATH, attrpath=DEFAULT_ATTR_PATH, human_goals=0,
        )
        goals = goal_mod.get_goals(_all(product_item_dict), product_prices, 0)
    finally:
        os.chdir(cwd)
    return product_item_dict, product_prices, goals


def _all(product_item_dict):
    return list(product_item_dict.values())


def _oracle_purchase(product, goal):
    """The clicked-options dict for a perfect play, keyed as the env keys it.

    web_agent_text_env.item_page stores ``session['options'][clickable['name'].lower()]``,
    i.e. the option CATEGORY lowercased -> the clicked button text. Returns None when the
    product page does not offer some required value (then the goal is not oracle-solvable
    for reasons unrelated to this bug -- currently never happens).
    """
    available = product.get('options', {}) or {}
    wanted = goal['goal_options']
    if not isinstance(wanted, dict):
        return {}
    clicked = {}
    for key, value in wanted.items():
        values = next((v for k, v in available.items() if k.lower() == key.lower()), None)
        if values is None or value not in values:
            return None
        clicked[key.lower()] = value
    return clicked


# --- the two defects, in isolation ---------------------------------------------------------

def test_normalize_color_rejects_non_str(engine):
    """The silence that hid the bug. A tuple must raise, not pass through unchanged."""
    _, norm_mod = engine
    with pytest.raises(TypeError):
        norm_mod.normalize_color(('color', 'r.brown2060'))
    # the str path is untouched: substring match, first COLOR_SET hit wins
    assert norm_mod.normalize_color('r.brown2060') == 'brown'
    assert norm_mod.normalize_color('16x24 inch') == '16x24 inch'


def test_option_reward_is_shape_symmetric(engine):
    """Both arguments are lists of plain values; clicking exactly the goal scores 1.0."""
    goal_mod, _ = engine
    wanted = {'color': 'r.brown2060', 'size': '6.5'}
    clicked = list(wanted.values())
    r_option, n_matches = goal_mod.get_option_reward(clicked, list(wanted.values()))
    assert (r_option, n_matches) == (1.0, 2)
    # and the old (key, value) shape is now loud rather than silently wrong
    with pytest.raises(TypeError):
        goal_mod.get_option_reward(clicked, list(wanted.items()))


# --- the guard that actually matters -------------------------------------------------------

def test_oracle_best_purchase_scores_one(engine, pool):
    """Perfect play must score exactly 1.0 for EVERY goal in the pool.

    Buy the target product, click every required option, at the target's own price (goals
    sample ``price_upper`` strictly above it, so the price term always passes). Anything
    below 1.0 means some component of the reward cannot be satisfied by any agent -- the
    goal is unwinnable by construction, and both the RL reward and every hardness label
    derived from it are wrong on that goal.
    """
    goal_mod, _ = engine
    product_item_dict, product_prices, goals = pool

    limit = int(os.environ.get('WEBSHOP_ORACLE_TEST_LIMIT', 0)) or len(goals)
    unreachable, unconstructible = [], []

    for goal in goals[:limit]:
        product = product_item_dict[goal['asin']]
        price = product_prices[goal['asin']]
        if goal['price_upper'] > 0 and price > goal['price_upper']:
            continue                                  # genuinely price-capped, not our concern
        clicked = _oracle_purchase(product, goal)
        if clicked is None:
            unconstructible.append(goal['asin'])
            continue
        reward = goal_mod.get_reward(product, goal, price=price, options=clicked)
        if abs(reward - 1.0) > 1e-9:
            unreachable.append((goal['asin'], goal['goal_options'], round(reward, 4)))

    assert not unreachable, (
        f'{len(unreachable)} of {min(limit, len(goals))} goals cannot reach 1.0 under perfect '
        f'play (option-matching path regressed?). First 5: {unreachable[:5]}'
    )
    assert not unconstructible, (
        f'{len(unconstructible)} goals require an option value the product page never offers: '
        f'{unconstructible[:5]}'
    )
