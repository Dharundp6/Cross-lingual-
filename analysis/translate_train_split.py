# -*- coding: utf-8 -*-
"""Translate the German training split into English so the system can be evaluated in the
language of the task.

Why this is needed. The deployed task supplies English questions; the only labelled cohort
large enough to support a paired comparison is the German training split. Two parts of the
system are therefore measured off-task in the archived experiments: the router, whose
prompt reads the question, and the rule bundles, whose triggers are English keywords and so
almost never fire on German text. Translating the split once lets both be measured in
English against the same gold citations.

The translation preserves every legal citation verbatim, so the identifier evidence the
task rewards is neither created nor destroyed by the translation step.

Model: deepseek-flash (DeepSeek-V4.1-Flash). Only the assistant `content` is kept; the
provider's reasoning trace is discarded, because an earlier cache accepted the trace as a
fallback and stored deliberation instead of translations for 45 of 150 queries.

Usage:  python analysis/translate_train_split.py [--limit N]
Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_router import load_key  # noqa: E402
from fix_transfer_translations import is_bad  # noqa: E402
from router_language_transfer import TRANSLATE_SYS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'train_queries_en.json'
SEED_CACHE = ROOT / 'analysis' / '_transfer_translations.json'
DS_URL = 'https://api.deepseek.com/v1/chat/completions'
MODEL = 'deepseek-flash'
CONCURRENCY = 8


def call_content(key, text, max_tokens):
    h = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    payload = {'model': MODEL, 'temperature': 0.0, 'max_tokens': max_tokens,
               'messages': [{'role': 'system', 'content': TRANSLATE_SYS},
                            {'role': 'user', 'content': str(text).strip()[:1800]}]}
    for a in range(6):
        try:
            r = requests.post(DS_URL, headers=h, json=payload, timeout=300)
            if r.status_code == 200:
                return (r.json()['choices'][0]['message'].get('content') or '').strip()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
                continue
            return ''
        except requests.RequestException:
            time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    key = load_key()
    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    qtext = {str(a): b for a, b in zip(train['query_id'], train['query'])}

    out = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    if not out and SEED_CACHE.exists():
        seed = json.loads(SEED_CACHE.read_text(encoding='utf-8'))
        out = {q: t for q, t in seed.items() if t and not is_bad(t)}
        print(f'seeded {len(out)} verified translations from the earlier transfer cache')

    todo = [q for q in qtext if not out.get(q) or is_bad(out.get(q))]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(qtext)} queries; {len(out)} already translated; {len(todo)} to fetch', flush=True)

    def fetch(q):
        for budget in (8000, 16000, 32000):
            t = call_content(key, qtext[q], budget)
            if t and not is_bad(t):
                return q, t
        return q, None

    t0 = time.time()
    done = fail = 0
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        for i, (q, t) in enumerate(ex.map(fetch, todo), 1):
            if t:
                out[q] = t
                done += 1
            else:
                fail += 1
            if i % 25 == 0:
                OUT.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
                print(f'  {i}/{len(todo)}  ok={done} failed={fail}  '
                      f'[{time.time() - t0:.0f}s]', flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(f'translated {len(out)}/{len(qtext)}; {fail} failed  [{time.time() - t0:.0f}s]')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
