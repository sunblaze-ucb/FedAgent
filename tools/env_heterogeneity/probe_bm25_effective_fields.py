"""Re-measure Variant 2/3 search divergence on the EFFECTIVE per-client indexes.

Why this exists: the paper's Variant-2/3 probe statistics were quoted from a Phase-0
probe whose field subsets name `description`/`features` — keys that do not exist on
the products SimServer actually indexes. In the shipped engine, `load_products`
synthesizes `Title` (:= name), `Description` (:= full_description) and `BulletPoints`
(:= small_description) but never lowercase `description`/`features`, and
`InMemoryBM25Searcher` silently skips missing fields with an all-empty fallback to
`name`. The EFFECTIVE doc text per pool entry is therefore:

    full    = name + Title + BulletPoints     (Title==name -> name tokens doubled)
    name    = name + Title                    (name tokens doubled)
    desc    = name                            (all-empty fallback)
    bullets = BulletPoints

This probe replays real agent queries (recovered from the original training sweep
logs with the same `grep search[..] | sort -u` extraction the Phase-0 probe used;
300 unique vs the 287 available mid-sweep) against searchers built EXACTLY like the
service builds them: raw catalog -> load_products-style normalization -> the
verbatim InMemoryBM25Searcher doc construction. Reported numbers are what the
training clients actually experienced; use them for the paper's Variant-2/3
divergence statistics.

Run (needs rank_bm25; the webshop service env has it):
    /home/xbb9020/.conda/envs/verl-agent-webshop/bin/python \
        tools/env_heterogeneity/probe_bm25_effective_fields.py
"""
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTS_FILE = REPO_ROOT / 'fedagent/envs/webshop/engine/webshop/data/items_shuffle_1000.json'
QUERIES_FILE = REPO_ROOT / 'data/env_heterogeneity/probe_queries_agent_300.txt'
K = 10

# Pools under measurement, in the SHIPPED assignment order (webshop_env_variants.py).
V2_POOL = [
    {'name': 'full',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
    {'name': 'name',    'fields': ['name', 'Title'],                                            'k1': 1.2, 'b': 0.75},
    {'name': 'desc',    'fields': ['description'],                                              'k1': 1.2, 'b': 0.75},
    {'name': 'bullets', 'fields': ['BulletPoints'],                                             'k1': 1.2, 'b': 0.75},
]
V2_POOL_EXT = V2_POOL + [
    {'name': 'features',      'fields': ['features'],                                'k1': 1.2, 'b': 0.75},
    {'name': 'name_bullets',  'fields': ['name', 'Title', 'BulletPoints'],           'k1': 1.2, 'b': 0.75},
    {'name': 'desc_features', 'fields': ['description', 'features'],                 'k1': 1.2, 'b': 0.75},
    {'name': 'no_name',       'fields': ['description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
]
V3_POOL = [
    {'name': 'full',        'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
    {'name': 'full_b=0.0',  'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.00},
    {'name': 'full_k1=0.3', 'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.3, 'b': 0.75},
    {'name': 'full_k1=5.0', 'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 5.0, 'b': 0.75},
]

_BM25_TOKEN_RE = re.compile(r'\w+')


def _bm25_tokenize(text):
    """Verbatim engine._bm25_tokenize."""
    return _BM25_TOKEN_RE.findall((text or '').lower())


def normalize_products(raw):
    """The load_products transforms that matter for doc text (engine.py 509-539)."""
    out, seen = [], set()
    for p in raw:
        asin = p['asin']
        if asin == 'nan' or (len(asin) > 10 and not asin.startswith('LK_')) or asin in seen:
            continue
        seen.add(asin)
        q = dict(p)
        q['Title'] = p['name']
        q['Description'] = p['full_description']
        sd = p['small_description']
        q['BulletPoints'] = sd if isinstance(sd, list) else [sd]
        out.append(q)
    return out


class Searcher:
    """Verbatim InMemoryBM25Searcher doc construction + search (engine.py 244-281)."""

    def __init__(self, products, fields, k1, b):
        self._doc_ids, docs = [], []
        self.nonempty_parts = 0
        for p in products:
            parts = []
            for f in fields:
                v = p.get(f, '')
                if isinstance(v, list):
                    parts.append(' '.join(str(x) for x in v))
                elif v:
                    parts.append(str(v))
            text = ' '.join(parts).strip() or str(p.get('name', ''))
            self.nonempty_parts += bool(parts)
            self._doc_ids.append(p['asin'])
            docs.append(text)
        self._bm25 = BM25Okapi([_bm25_tokenize(d) for d in docs], k1=float(k1), b=float(b))

    def top(self, query, k=K):
        toks = _bm25_tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        return [self._doc_ids[i] for i in np.argsort(-scores)[:k]]


def probe(pool, products, queries, label):
    searchers = {v['name']: Searcher(products, v['fields'], v['k1'], v['b']) for v in pool}
    print(f'\n== {label} ==')
    for v in pool:
        s = searchers[v['name']]
        print(f'   {v["name"]:14s} fields={v["fields"]} -> products with any requested '
              f'field non-empty: {s.nonempty_parts}/{len(products)}')
    rankings = {n: [s.top(q) for q in queries] for n, s in searchers.items()}
    pair_stats = []
    for a, b in combinations([v['name'] for v in pool], 2):
        js, t1 = [], 0
        for ra, rb in zip(rankings[a], rankings[b]):
            if not ra or not rb:
                continue
            sa, sb = set(ra), set(rb)
            js.append(len(sa & sb) / max(1, len(sa | sb)))
            t1 += ra[0] != rb[0]
        pair_stats.append((a, b, float(np.mean(js)), t1 / max(1, len(js))))
    print(f'   {"pair":34s} {"mean_J@10":>9s} {"top1!=%":>8s}')
    for a, b, mj, t1 in sorted(pair_stats, key=lambda r: r[2]):
        print(f'   {a:15s} vs {b:15s} {mj:9.3f} {100*t1:7.1f}%')
    pool_mean_j = float(np.mean([r[2] for r in pair_stats]))
    pool_mean_t1 = float(np.mean([r[3] for r in pair_stats]))
    print(f'   POOL AVG over {len(pair_stats)} pairs: mean_J@10={pool_mean_j:.3f} '
          f'top1!={100*pool_mean_t1:.1f}%')
    return pool_mean_j, pool_mean_t1


def main():
    raw = json.load(open(PRODUCTS_FILE))
    products = normalize_products(raw)
    queries = [q.strip() for q in QUERIES_FILE.read_text().splitlines() if q.strip()]
    print(f'[probe] {len(products)} normalized products, {len(queries)} replayed agent queries, K={K}')
    probe(V2_POOL, products, queries, 'Variant 2 pool |V|=4 (EFFECTIVE fields)')
    probe(V2_POOL_EXT, products, queries, 'Variant 2 pool |V|=8 (EFFECTIVE fields)')
    probe(V3_POOL, products, queries, 'Variant 3 pool |V|=4 (k1/b corners, shipped order)')


if __name__ == '__main__':
    main()
