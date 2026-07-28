import os
import sys
import json
from tqdm import tqdm
sys.path.insert(0, '../')

from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
from web_agent_site.engine.engine import load_products

all_products, *_ = load_products(filepath=DEFAULT_FILE_PATH, attrpath=DEFAULT_ATTR_PATH)


docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    options = p.get('options', {})
    for option_name, option_contents in options.items():
        option_contents_text = ', '.join(option_contents)
        option_texts.append(f'{option_name}: {option_contents_text}')
    option_text = ', and '.join(option_texts)

    doc = dict()
    doc['id'] = p['asin']
    doc['contents'] = ' '.join([
        p['Title'],
        p['Description'],
        p['BulletPoints'][0],
        option_text,
    ]).lower()
    doc['product'] = p
    docs.append(doc)


# Fresh clone: search_engine/resources* is untracked (built locally, not shipped), so create
# each output directory before opening the file -- otherwise open(..., 'w+') raises
# FileNotFoundError. (2026-07-28, back-ported from AccelAgent.)
outpath = './resources_100/documents.jsonl'
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, 'w+') as f:
    for doc in docs[:100]:
        f.write(json.dumps(doc) + '\n')

outpath = './resources/documents.jsonl'
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, 'w+') as f:
    for doc in docs:
        f.write(json.dumps(doc) + '\n')

outpath = './resources_1k/documents.jsonl'
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, 'w+') as f:
    for doc in docs[:1000]:
        f.write(json.dumps(doc) + '\n')

outpath = './resources_100k/documents.jsonl'
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, 'w+') as f:
    for doc in docs[:100000]:
        f.write(json.dumps(doc) + '\n')
