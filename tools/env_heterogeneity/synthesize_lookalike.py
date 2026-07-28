"""Generate adversarial lookalike products for paper Variant 4 (Lookalike Injection).

PROVENANCE: recovered 2026-07-28 from the original experiment repo
(project/federated_agent/tools/env_heterogeneity/synthesize_lookalike.py, where it
produced the four checked-in pools under config/2026_env_heterogeneity/variant_b/).
Only this docstring and the two path constants below were adapted to this repo's
layout; the generation body is byte-identical. Regenerating reproduces the
checked-in assets content-identically (price/size/price_color byte-identical;
the color file differs only in JSON serialization formatting).

For each of the 1000 catalog products, produce one clone-with-attack lookalike per
pool where the attacked option exists (EXACT text clone — document text untouched,
so the lookalike ties with the target under BM25):

  v_price       (1000): pricing pushed to $9999 (> any goal price_upper) → r_price = 0
  v_color        (286): customization_options['color'] swapped to non-matching colors
                        (fuzz.token_set_ratio < 85) → option-match miss
  v_size         (268): customization_options['size'] swapped likewise
  v_price_color  (286): price AND color attacked jointly

The agent is forced to read pricing/options before buying — different pools force
different attribute-checking policies → per-client pi* divergence.

Outputs JSON arrays of raw products (same schema as items_shuffle_1000.json),
ready to be appended to SimServer.full_products via env_kwargs['extra_products'].

Run:
    python tools/env_heterogeneity/synthesize_lookalike.py

Output:
    data/env_heterogeneity/lookalike_data/lookalike_v_{price,color,size,price_color}.json
"""
import json
import random
import re
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTS_FILE = REPO_ROOT / 'fedagent/envs/webshop/engine/webshop/data/items_shuffle_1000.json'
OUT_DIR = REPO_ROOT / 'data/env_heterogeneity/lookalike_data'
SEED = 99999

ATTACK_PRICE_TAG = '$9999.0'

# Non-matching color thesaurus: each common color maps to far-off neighbors so
# fuzz.token_set_ratio (the WebShop reward function) returns < 85 (no match).
COLOR_SWAPS = {
    'blue':   ['red', 'orange', 'yellow'],
    'red':    ['blue', 'green', 'purple'],
    'green':  ['red', 'pink', 'orange'],
    'black':  ['white', 'yellow', 'red'],
    'white':  ['black', 'red', 'green'],
    'pink':   ['black', 'green', 'blue'],
    'purple': ['orange', 'yellow', 'green'],
    'yellow': ['blue', 'purple', 'black'],
    'orange': ['blue', 'green', 'purple'],
    'gray':   ['red', 'green', 'yellow'],
    'grey':   ['red', 'green', 'yellow'],
    'brown':  ['blue', 'pink', 'green'],
    'beige':  ['black', 'red', 'blue'],
    'navy':   ['orange', 'yellow', 'pink'],
    'silver': ['red', 'green', 'orange'],
    'gold':   ['blue', 'green', 'purple'],
}
DEFAULT_SWAP = ['neon-fuchsia', 'electric-magenta', 'iridescent-teal']


def _detect_color_word(value):
    """Return first known-color substring in value (lowercased)."""
    s = (value or '').lower()
    for color in COLOR_SWAPS:
        if re.search(rf'\b{re.escape(color)}\b', s):
            return color
    return None


def perturb_color_options(options_dict):
    """Return a new options dict where any color option is swapped to non-matching values.

    Input shape (raw items_shuffle format):
      'customization_options': {
          'color': [{'value': 'navy blue', 'image': '...'}, {'value': 'red', ...}],
          'size':  [{'value': '12 inch', ...}],
      }

    We rewrite each color entry's 'value' to a swapped color while keeping all
    non-color options as-is.
    """
    if not options_dict or not isinstance(options_dict, dict):
        return options_dict
    new_dict = {}
    for option_name, contents in options_dict.items():
        if option_name.lower() != 'color' or not contents:
            new_dict[option_name] = contents
            continue
        new_contents = []
        for entry in contents:
            new_entry = dict(entry) if isinstance(entry, dict) else entry
            if isinstance(new_entry, dict) and 'value' in new_entry:
                old_color = _detect_color_word(new_entry['value'])
                swap_pool = COLOR_SWAPS.get(old_color, DEFAULT_SWAP) if old_color else DEFAULT_SWAP
                new_value = random.choice(swap_pool)
                new_entry['value'] = new_value
            new_contents.append(new_entry)
        new_dict[option_name] = new_contents
    return new_dict


def make_price_lookalike(target):
    lk = deepcopy(target)
    lk['asin'] = f'LK_PRICE_{target["asin"]}'
    lk['pricing'] = ATTACK_PRICE_TAG
    return lk


def make_color_lookalike(target, rng):
    lk = deepcopy(target)
    lk['asin'] = f'LK_COLOR_{target["asin"]}'
    co = lk.get('customization_options')
    lk['customization_options'] = perturb_color_options(co)
    return lk


# Adjacent-size swaps; goal-targeting size will be replaced with non-matching one.
SIZE_SWAPS = {
    'small': ['x-large', 'xx-large', 'large'],
    'medium': ['x-small', 'xx-small', 'xx-large'],
    'large': ['x-small', 'small', 'xx-small'],
    'x-small': ['x-large', 'large', 'xx-large'],
    'x-large': ['x-small', 'small', 'xx-small'],
    'xx-small': ['x-large', 'xx-large', 'large'],
    'xx-large': ['x-small', 'xx-small', 'small'],
    'xxl': ['xs', 's', 'xxs'],
    'xs': ['xxl', 'xl', 'l'],
    's': ['xxl', 'xl', 'l'],
    'm': ['xxs', 'xs', 'xxl'],
    'l': ['xxs', 'xs', 's'],
    'xl': ['xs', 'xxs', 's'],
}
DEFAULT_SIZE_SWAP = ['xxxxxxx-small', 'unknown-size', 'one-size-fits-no-one']


def _detect_size_word(value):
    s = (value or '').lower().strip()
    for size in SIZE_SWAPS:
        if re.search(rf'\b{re.escape(size)}\b', s):
            return size
    return None


def perturb_size_options(options_dict):
    if not options_dict or not isinstance(options_dict, dict):
        return options_dict
    new_dict = {}
    for option_name, contents in options_dict.items():
        if option_name.lower() != 'size' or not contents:
            new_dict[option_name] = contents
            continue
        new_contents = []
        for entry in contents:
            new_entry = dict(entry) if isinstance(entry, dict) else entry
            if isinstance(new_entry, dict) and 'value' in new_entry:
                old_size = _detect_size_word(new_entry['value'])
                swap_pool = SIZE_SWAPS.get(old_size, DEFAULT_SIZE_SWAP) if old_size else DEFAULT_SIZE_SWAP
                new_entry['value'] = random.choice(swap_pool)
            new_contents.append(new_entry)
        new_dict[option_name] = new_contents
    return new_dict


def make_size_lookalike(target, rng):
    lk = deepcopy(target)
    lk['asin'] = f'LK_SIZE_{target["asin"]}'
    co = lk.get('customization_options')
    lk['customization_options'] = perturb_size_options(co)
    return lk


def make_price_color_lookalike(target, rng):
    """Combined price+color attack on the same product (forces agent to check BOTH)."""
    lk = deepcopy(target)
    lk['asin'] = f'LK_PCOL_{target["asin"]}'
    lk['pricing'] = ATTACK_PRICE_TAG
    co = lk.get('customization_options')
    lk['customization_options'] = perturb_color_options(co)
    return lk


def main():
    random.seed(SEED)
    rng = random.Random(SEED)

    products = json.load(PRODUCTS_FILE.open())
    print(f'[lookalike] {len(products)} target products loaded from {PRODUCTS_FILE.name}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- v_price ----------
    price_lookalikes = [make_price_lookalike(p) for p in products]
    price_path = OUT_DIR / 'lookalike_v_price.json'
    json.dump(price_lookalikes, price_path.open('w'))
    print(f'  v_price: wrote {len(price_lookalikes)} lookalikes -> {price_path.name}')

    # ---------- v_color ----------
    def _has_nonempty_color(p):
        co = p.get('customization_options') or {}
        for k, v in co.items():
            if k.lower() == 'color' and v:  # contents must be non-empty list
                return True
        return False
    color_eligible = [p for p in products if _has_nonempty_color(p)]
    print(f'  v_color: {len(color_eligible)}/{len(products)} products have a color option')
    color_lookalikes = [make_color_lookalike(p, rng) for p in color_eligible]
    color_path = OUT_DIR / 'lookalike_v_color.json'
    json.dump(color_lookalikes, color_path.open('w'))
    print(f'  v_color: wrote {len(color_lookalikes)} lookalikes -> {color_path.name}')

    # ---------- v_size ----------
    def _has_nonempty_size(p):
        co = p.get('customization_options') or {}
        for k, v in co.items():
            if k.lower() == 'size' and v:
                return True
        return False
    size_eligible = [p for p in products if _has_nonempty_size(p)]
    print(f'  v_size: {len(size_eligible)}/{len(products)} products have a size option')
    size_lookalikes = [make_size_lookalike(p, rng) for p in size_eligible]
    size_path = OUT_DIR / 'lookalike_v_size.json'
    json.dump(size_lookalikes, size_path.open('w'))
    print(f'  v_size: wrote {len(size_lookalikes)} lookalikes -> {size_path.name}')

    # ---------- v_price_color (combined attack) ----------
    pcol_lookalikes = [make_price_color_lookalike(p, rng) for p in color_eligible]
    pcol_path = OUT_DIR / 'lookalike_v_price_color.json'
    json.dump(pcol_lookalikes, pcol_path.open('w'))
    print(f'  v_price_color: wrote {len(pcol_lookalikes)} lookalikes -> {pcol_path.name}')

    # ---------- sanity: verify fields look right on a sample ----------
    print('\n[lookalike] sample v_price[0]:')
    sp = price_lookalikes[0]
    print(f'  asin={sp["asin"]} pricing={sp["pricing"]!r} (orig={products[0]["pricing"]!r})')
    print(f'  name={sp["name"][:60]!r}')

    if color_lookalikes:
        sc = color_lookalikes[0]
        orig = next((p for p in products if p['asin'] == sc['asin'].replace('LK_COLOR_', '')), None)
        print('\n[lookalike] sample v_color[0]:')
        print(f'  asin={sc["asin"]}')
        co_old = orig['customization_options'] if orig else {}
        co_new = sc['customization_options']
        print(f'  customization_options(old) = {dict(list(co_old.items())[:1]) if co_old else co_old}')
        print(f'  customization_options(new) = {dict(list(co_new.items())[:1]) if co_new else co_new}')

    print('\n[lookalike] done.')


if __name__ == '__main__':
    main()
