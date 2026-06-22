"""
"""
import os
import re
import json
import random
from collections import defaultdict
from ast import literal_eval
from decimal import Decimal

import cleantext
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from flask import render_template_string
from rich import print
# pyserini's LuceneSearcher requires JVM (jnius); imported lazily in
# init_search_engine so InMemoryBM25Searcher can be used in environments
# without a JDK (e.g. login-node smoke tests).

from web_agent_site.utils import (
    BASE_DIR,
    DEFAULT_FILE_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Default 50 (legacy). For env-level heterogeneity experiments, set
# WEBSHOP_SEARCH_RETURN_N=200 so that BM25 top-K stays large after per-client
# catalog filtering (engine.py:172 filters out ASINs not in product_item_dict).
SEARCH_RETURN_N = int(os.environ.get('WEBSHOP_SEARCH_RETURN_N', 50))
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs']
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    if keywords[0] == '<r>':
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == '<a>':
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>':
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    elif keywords[0] == '<q>':
        query = ' '.join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p['query'] == query]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


# ============================================================
# Transition-level env heterogeneity: in-memory BM25 backend
# ============================================================
# See docs/heterogeneity.md (BM25 Reweighting).
# Mimics LuceneSearcher's two used methods (.search, .doc) so it can be
# swapped in via env_kwarg `bm25_in_memory_config` without touching
# get_top_n_product_from_keywords. Per-client variant = (fields, k1, b)
# triple → distinct BM25 ranking for the same query → transition T_k differs.

_BM25_TOKEN_RE = re.compile(r'\b\w+\b')

def _bm25_tokenize(text):
    """Lowercase + word-boundary tokenization. Lucene does stemming + stopwords on top."""
    return _BM25_TOKEN_RE.findall((text or '').lower())


class _InMemoryHit:
    __slots__ = ('docid', 'score')

    def __init__(self, docid, score):
        self.docid = docid
        self.score = score


class _InMemoryDoc:
    __slots__ = ('_raw',)

    def __init__(self, raw_json_str):
        self._raw = raw_json_str

    def raw(self):
        return self._raw


class InMemoryBM25Searcher:
    """Drop-in replacement for LuceneSearcher used by env-level T_k variants.

    Constructor:
        products: list of product dicts (must contain 'asin' + the requested fields)
        fields:   list of field names whose text gets concatenated into the doc
                  for BM25 indexing (e.g. ['name', 'Title', 'description'])
        k1, b:    BM25 hyperparameters (passed straight to BM25Okapi)
    """

    def __init__(self, products, fields, k1=1.2, b=0.75):
        self.fields = list(fields)
        self.k1 = float(k1)
        self.b = float(b)

        self._doc_ids = []
        self._doc_strings = []
        for p in products:
            parts = []
            for f in self.fields:
                v = p.get(f, '')
                if isinstance(v, list):
                    parts.append(' '.join(str(x) for x in v))
                elif v:
                    parts.append(str(v))
            text = ' '.join(parts).strip() or str(p.get('name', ''))
            self._doc_ids.append(p['asin'])
            self._doc_strings.append(text)

        tokenized = [_bm25_tokenize(d) for d in self._doc_strings]
        if not tokenized or all(len(t) == 0 for t in tokenized):
            raise ValueError(
                f'InMemoryBM25Searcher: empty corpus after tokenization '
                f'(fields={self.fields}, n_products={len(products)}). '
                f'Check that requested fields exist on the products.'
            )
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._asin_to_idx = {asin: i for i, asin in enumerate(self._doc_ids)}

    def search(self, query, k=50):
        q_tokens = _bm25_tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # top-k by descending score; cheap argsort is fine for n=1000
        import numpy as _np
        top_idx = _np.argsort(-scores)[:k]
        return [_InMemoryHit(self._doc_ids[i], float(scores[i])) for i in top_idx]

    def doc(self, docid):
        idx = self._asin_to_idx.get(docid)
        if idx is None:
            return _InMemoryDoc(json.dumps({'id': docid, 'contents': ''}))
        return _InMemoryDoc(json.dumps({
            'id': docid,
            'contents': self._doc_strings[idx],
        }))


# ============================================================
# Search engine TYPE wrappers — for "Search Variant" experiment design.
# See docs/heterogeneity.md (search backend axis).
#
# Each wrapper takes a base searcher (typically InMemoryBM25Searcher) and
# modifies the .search() behavior. Together they form 4 variants that
# break different baseline-policy assumptions while preserving reward gradient.
# ============================================================
class _SearchEngineWrapperBase:
    def __init__(self, base):
        self.base = base
    def search(self, query, k=50):
        return self.base.search(query, k=k)
    def doc(self, docid):
        return self.base.doc(docid)


class ShuffledTopKSearcher(_SearchEngineWrapperBase):
    """BM25 ranks top-shuffle_k as usual, then SHUFFLES before returning top k.
    Forces agent to scan all returned items (can't trust position 1)."""
    def __init__(self, base, shuffle_k=50, seed=42):
        super().__init__(base)
        self.shuffle_k = int(shuffle_k)
        self._rng = random.Random(seed)
    def search(self, query, k=50):
        hits = list(self.base.search(query, k=max(self.shuffle_k, k)))
        self._rng.shuffle(hits)
        return hits[:k]


class InvertedTopKSearcher(_SearchEngineWrapperBase):
    """BM25 returns top-K but in REVERSE order (rank-K first, rank-1 last)."""
    def search(self, query, k=50):
        return list(self.base.search(query, k=k))[::-1]


class PartialRandomSearcher(_SearchEngineWrapperBase):
    """With prob `random_prob` returns random k products; else BM25 top-K.
    Forces agent to verify each search result rather than trust the engine."""
    def __init__(self, base, asin_pool, random_prob=0.5, seed=42):
        super().__init__(base)
        self.asin_pool = list(asin_pool)
        self.p = float(random_prob)
        self._rng = random.Random(seed)
    def search(self, query, k=50):
        if self._rng.random() < self.p:
            n = min(k, len(self.asin_pool))
            sampled = self._rng.sample(self.asin_pool, n)
            return [_InMemoryHit(docid=a, score=0.0) for a in sampled]
        return self.base.search(query, k=k)


def _build_search_variant(base, variant_cfg):
    """Wrap a base searcher with the requested search-variant behavior."""
    vtype = variant_cfg.get('type', 'bm25_default')
    if vtype == 'bm25_default':
        return base
    if vtype == 'bm25_shuffle':
        return ShuffledTopKSearcher(
            base, shuffle_k=variant_cfg.get('shuffle_k', 50),
            seed=variant_cfg.get('seed', 42),
        )
    if vtype == 'bm25_invert':
        return InvertedTopKSearcher(base)
    if vtype == 'bm25_partial':
        return PartialRandomSearcher(
            base, asin_pool=base._doc_ids,
            random_prob=variant_cfg.get('random_prob', 0.5),
            seed=variant_cfg.get('seed', 42),
        )
    raise ValueError(f"unknown search_engine_variant type={vtype!r}")


def init_search_engine(num_products=None, in_memory_config=None, products=None,
                       search_engine_variant=None):
    """Initialize the search backend used by SimServer.

    Routing (priority order):
      0. search_engine_variant provided -> InMemoryBM25 base + behavior wrapper
         (search variant axis: bm25_shuffle / bm25_invert / bm25_partial / bm25_default).
         Requires `products`. May combine with in_memory_config (which seeds the base).
      1. in_memory_config provided -> InMemoryBM25Searcher (BM25 Reweighting variants).
         Requires `products` to be passed (the full product list).
      2. otherwise -> default LuceneSearcher with the legacy `indexes_*` dir.
    """
    if search_engine_variant is not None:
        if products is None:
            raise ValueError(
                'init_search_engine: search_engine_variant requires products list'
            )
        base_cfg = {
            'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'],
            'k1': 1.2, 'b': 0.75,
        }
        if in_memory_config is not None:
            base_cfg.update({k: v for k, v in in_memory_config.items() if k in ('fields', 'k1', 'b')})
        base = InMemoryBM25Searcher(
            products=products, fields=base_cfg['fields'],
            k1=base_cfg['k1'], b=base_cfg['b'],
        )
        return _build_search_variant(base, search_engine_variant)

    if in_memory_config is not None:
        if products is None:
            raise ValueError(
                'init_search_engine: in_memory_config requires products list'
            )
        return InMemoryBM25Searcher(
            products=products,
            fields=in_memory_config['fields'],
            k1=in_memory_config.get('k1', 1.2),
            b=in_memory_config.get('b', 0.75),
        )

    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        # for other numbers of products, use the default indexes
        # print(f"Warning: num_products={num_products} using default search engine indexes")
        # indexes = 'indexes'
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    from pyserini.search.lucene import LuceneSearcher  # lazy: needs JVM
    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'../search_engine/{indexes}'))
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def load_products(filepath, attrpath, num_products=None, human_goals=True,
                  catalog_filter_asins=None, extra_products=None):
    """Load products from disk.

    Args:
      catalog_filter_asins: if not None, restrict the loaded product list to
        ASINs in this set. Used by env-level heterogeneity experiments — each
        federated client gets a different subset of distractor ASINs while all
        share the target ASINs (those referenced by goals).
        See docs/heterogeneity.md
      extra_products: optional list of raw-format products to append after the
        on-disk catalog (and after catalog_filter_asins is applied). Used by
        lookalike injection (transition-level adversarial lookalike) to inject attack
        clones that share text fields with target products.
        See docs/heterogeneity.md
    """
    # TODO: move to preprocessing step -> enforce single source of truth
    with open(filepath) as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
    # Apply per-client catalog filter BEFORE num_products truncation so that
    # (catalog_filter_asins, num_products) are independently controllable.
    if catalog_filter_asins is not None:
        catalog_set = set(catalog_filter_asins)
        before = len(products)
        products = [p for p in products if p['asin'] in catalog_set]
        print(f'[load_products] catalog filter: {before} -> {len(products)} products')
    # Append adversarial lookalike products (lookalike injection) AFTER catalog filter and
    # num_products truncation logic, but BEFORE per-product processing — so
    # lookalikes go through identical Title/Description/options/pricing parsing.
    if extra_products:
        before = len(products)
        products = list(products) + clean_product_keys(list(extra_products))
        print(f'[load_products] extra_products injected: {before} -> {len(products)} '
              f'(+{len(extra_products)} lookalikes)')
    
    # with open(DEFAULT_REVIEW_PATH) as f:
    #     reviews = json.load(f)
    all_reviews = dict()
    all_ratings = dict()
    # for r in reviews:
    #     all_reviews[r['asin']] = r['reviews']
    #     all_ratings[r['asin']] = r['average_rating']

    if human_goals:
        with open(HUMAN_ATTR_PATH) as f:
            human_attributes = json.load(f)
    with open(attrpath) as f:
        attributes = json.load(f)
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes = json.load(f)
    print('Attributes loaded.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        products = products[:num_products]
    # add a debug flag, only print detailed info for the first 3 products
    debug_count = 0
    max_debug_products = 3
    
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan':
            continue
        # Original Amazon ASINs are 10 chars; the >10 filter sieves out junk.
        # Lookalike-injection products use the LK_ prefix and are intentionally longer.
        if len(asin) > 10 and not asin.startswith('LK_'):
            continue

        if asin in asins:
            continue
        else:
            asins.add(asin)

        products[i]['category'] = p['category']
        products[i]['query'] = p['query']
        products[i]['product_category'] = p['product_category']

        products[i]['Title'] = p['name']
        products[i]['Description'] = p['full_description']
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p['small_description'] \
            if isinstance(p['small_description'], list) else [p['small_description']]
            
        # print detailed info for the first few products as samples
        # if debug_count < max_debug_products:
        #     print(f"\n{'='*80}")
        #     print(f"Product sample #{debug_count + 1}")
        #     print(f"{'='*80}")
        #     print(f"ASIN: {asin}")
        #     print(f"Title: {p['name']}")
        #     print(f"Category: {p['category']}")
        #     print(f"Product category: {p['product_category']}")
        #     print(f"Query: {p['query']}")
        #     print(f"Price: {p.get('pricing', 'N/A')}")
        #     print(f"Description length: {len(p['full_description']) if p['full_description'] else 0} chars")
        #     print(f"Number of bullet points: {len(p['small_description']) if isinstance(p['small_description'], list) else 1}")
        #     if p['small_description']:
        #         print(f"Bullet point sample: {p['small_description'][:2] if isinstance(p['small_description'], list) else p['small_description']}")
        #     customization_options = p.get('customization_options', {})
        #     if isinstance(customization_options, dict):
        #         print(f"Customization options: {list(customization_options.keys())}")
        #     else:
        #         print(f"Customization options: {type(customization_options)} (non-dict type)")
        #     print(f"Number of images: {len(p.get('images', []))}")
        #     if p.get('images'):
        #         print(f"Main image URL: {p['images'][0]}")
        #     debug_count += 1

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            pricing = [
                float(Decimal(re.sub(r'[^\d.]', '', price)))
                for price in pricing.split('$')[1:]
            ]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p['customization_options']
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image

        # without color, size, price, availability
        # if asin in attributes and 'attributes' in attributes[asin]:
        #     products[i]['Attributes'] = attributes[asin]['attributes']
        # else:
        #     products[i]['Attributes'] = ['DUMMY_ATTR']
        # products[i]['instruction_text'] = \
        #     attributes[asin].get('instruction', None)
        # products[i]['instruction_attributes'] = \
        #     attributes[asin].get('instruction_attributes', None)

        # without color, size, price, availability
        if asin in attributes and 'attributes' in attributes[asin]:
            products[i]['Attributes'] = attributes[asin]['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        if human_goals:
            if asin in human_attributes:
                products[i]['instructions'] = human_attributes[asin]
        else:
            # Lookalike-injection products (LK_* asin) aren't in items_ins_v2_*.json.
            # Setting instruction_text=None makes get_synthetic_goals skip them
            # so we never generate goals targeting fake products.
            if asin in attributes:
                products[i]['instruction_text'] = \
                    attributes[asin].get('instruction', None)
                products[i]['instruction_attributes'] = \
                    attributes[asin].get('instruction_attributes', None)
            else:
                products[i]['instruction_text'] = None
                products[i]['instruction_attributes'] = None
                
        # print attribute info
        # if debug_count <= max_debug_products:
        #     print(f"Attributes: {products[i]['Attributes']}")
        #     if human_goals and asin in human_attributes:
        #         print(f"Human instruction: {human_attributes[asin]}")
        #     elif not human_goals and asin in attributes:
        #         print(f"Instruction text: {attributes[asin].get('instruction', 'N/A')}")
        #         print(f"Instruction attributes: {attributes[asin].get('instruction_attributes', 'N/A')}")
        #     print(f"{'='*80}")

        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()

        all_products.append(products[i])

    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    
    # # print loading summary
    # print(f"\n{'='*80}")
    # print("Data loading summary:")
    # print(f"{'='*80}")
    # print(f"Total number of products: {len(all_products)}")
    # print(f"Number of attribute types: {len(attribute_to_asins)}")
    # print(f"First 10 attribute examples: {list(attribute_to_asins.keys())[:10]}")
    # print(f"Price range: {min(product_prices.values()) if product_prices else 'N/A'} - {max(product_prices.values()) if product_prices else 'N/A'}")
    # print(f"{'='*80}\n")
    
    return all_products, product_item_dict, product_prices, attribute_to_asins
