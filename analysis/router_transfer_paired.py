# -*- coding: utf-8 -*-
"""Paired German/English test of the domain router: does the label survive the language boundary?

The archived router was fitted, prompted and measured on the German training split, while
the deployed task supplies English questions. The manuscript stated that transfer as
untested. This measures it.

DESIGN
    Same queries in both arms, same classifier, same prompt, same completion budget.
    DE arm  the original German question text from Data/train.csv
    EN arm  the same question translated DE->EN, citations preserved verbatim
    Both arms are classified NOW, by the same model, so a difference between them is a
    difference in language and not in model version. (The archived DE predictions were
    produced by an earlier snapshot behind the same provider alias, so comparing fresh
    English against archived German would confound language with model drift.)

    McNemar's exact test on the discordant pairs is the right test here: the arms are
    paired on the query, so only the queries where the two arms disagree carry
    information about a language effect.

MODEL
    deepseek-flash (DeepSeek-V4.1-Flash). The legacy alias deepseek-v4-flash routes to
    this model; deepseek-v4-pro is scheduled for retirement and is also routed to it.
    Naming it explicitly is what makes the run repeatable.

Usage:  python analysis/router_transfer_paired.py --n 150
Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_router import SYS, load_key, parse  # noqa: E402
from llm_router import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'router_transfer_paired.json'
CACHE = ROOT / 'analysis' / '_transfer_translations.json'
ACCURACY = ROOT / 'analysis' / 'router_accuracy_deepseek.csv'
DS_URL = 'https://api.deepseek.com/v1/chat/completions'
MODEL = 'deepseek-flash'
CONCURRENCY = 6
BUDGETS = (8000, 16000)


def call(key, messages, max_tokens, retries=6):
    h = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    payload = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0.0}
    for a in range(retries):
        try:
            r = requests.post(DS_URL, headers=h, json=payload, timeout=300)
            if r.status_code == 200:
                m = r.json()['choices'][0]['message']
                return m.get('content') or m.get('reasoning_content') or ''
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
                continue
            return ''
        except requests.RequestException:
            time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
    return ''


def classify(key, text):
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': f'LEGAL QUESTION:\n{str(text).strip()[:1800]}\n\nJSON:'}]
    for budget in BUDGETS:
        obj = parse(call(key, msgs, budget))
        dom = (obj or {}).get('domain')
        if dom in LABELS:
            return dom
    return None


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    key = load_key()
    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    qtext = {str(a): b for a, b in zip(train['query_id'], train['query'])}
    cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}

    acc = pd.read_csv(ACCURACY)
    rows = [r for _, r in acc.iterrows()
            if str(r['gold_domain']) in LABELS and str(r['query_id']) in qtext]
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    qids = [str(r['query_id']) for r in sample]
    gold = {str(r['query_id']): str(r['gold_domain']) for r in sample}
    from fix_transfer_translations import is_bad
    dropped = [q for q in qids if not cache.get(q) or is_bad(cache[q])]
    qids = [q for q in qids if q not in set(dropped)]
    if dropped:
        print(f'dropped {len(dropped)} queries whose cached translation is a reasoning trace rather than a translation: {dropped}')
    print(f'paired sample n={len(qids)} (seed {args.seed}); both arms classified with {MODEL}',
          flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        de = dict(zip(qids, ex.map(lambda q: classify(key, qtext[q]), qids)))
    print(f'  DE arm: {sum(1 for v in de.values() if v)}/{len(qids)} classified '
          f'[{time.time() - t0:.0f}s]', flush=True)
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        en = dict(zip(qids, ex.map(lambda q: classify(key, cache[q]), qids)))
    print(f'  EN arm: {sum(1 for v in en.values() if v)}/{len(qids)} classified '
          f'[{time.time() - t0:.0f}s]', flush=True)

    paired = [q for q in qids if de.get(q) and en.get(q)]
    dropout = 1 - len(paired) / len(qids)
    if dropout > 0.10:
        print(f'ABORT: {dropout:.1%} of queries produced no label in one arm; the survivors '
              'are a biased subset. No statistics written.')
        return

    de_ok = {q: de[q] == gold[q] for q in paired}
    en_ok = {q: en[q] == gold[q] for q in paired}
    b = sum(1 for q in paired if de_ok[q] and not en_ok[q])
    c = sum(1 for q in paired if en_ok[q] and not de_ok[q])
    de_acc = sum(de_ok.values()) / len(paired)
    en_acc = sum(en_ok.values()) / len(paired)
    agree = sum(1 for q in paired if de[q] == en[q]) / len(paired)

    res = {
        'design': 'paired DE/EN, same queries, same model, same prompt, same budget',
        'model': MODEL, 'n_sampled': len(qids), 'n_paired': len(paired), 'seed': args.seed,
        'n_dropped_bad_translation': len(dropped),
        'de_accuracy': round(de_acc, 4), 'en_accuracy': round(en_acc, 4),
        'delta_en_minus_de': round(en_acc - de_acc, 4),
        'label_agreement': round(agree, 4),
        'discordant_de_only_correct': b, 'discordant_en_only_correct': c,
        'mcnemar_exact_p': round(mcnemar(b, c), 4),
        'per_query': [{'query_id': q, 'gold': gold[q], 'de': de[q], 'en': en[q]} for q in paired],
    }
    OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print('=' * 60)
    print(f'paired n            {len(paired)}')
    print(f'DE accuracy         {de_acc:.4f}')
    print(f'EN accuracy         {en_acc:.4f}')
    print(f'delta (EN - DE)     {en_acc - de_acc:+.4f}')
    print(f'label agreement     {agree:.4f}')
    print(f'discordant  DE-only {b}  EN-only {c}   McNemar exact p = {mcnemar(b, c):.4f}')
    print('=' * 60)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main()
