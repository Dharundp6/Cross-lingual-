# -*- coding: utf-8 -*-
"""DeepSeek legal-domain router -- cross-model comparison against the Claude router.

Runs the SAME two-step task, the SAME label set, and the SAME blind inputs as the
Claude router in analysis/llm_router.py, so the two are directly comparable:

  step 1: predict the Swiss law codes the authoritative answer would cite
  step 2: map those codes through the 39-code DOMAIN table to one of six labels

Output is scored by the identical code path (llm_router.score_external), so any
difference in the numbers is a difference in the model, not the harness.

USAGE
  python analysis/deepseek_router.py --split train             # 993 queries
  python analysis/deepseek_router.py --split train --limit 50  # smoke run

Requires DEEP_SEEK in .env. Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_router import LABELS, cits, domain_of_gold, is_court, load_train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS_URL = 'https://api.deepseek.com/v1/chat/completions'
DS_MODEL = 'deepseek-v4-flash'
CONCURRENCY = 8

SYS = """You are a Swiss legal domain classifier. Classify the legal question into EXACTLY ONE domain.

STEP 1: predict which Swiss law codes the authoritative answer would cite (1-5 abbreviations).
STEP 2: map them through this table; the domain holding the most of your codes wins; break ties alphabetically.

  Criminal:                 StGB, StPO, StBOG, JStG, VStrR
  Civil & commercial:       ZGB, OR, ZPO, GBV, IPRG, HRegV, PartG
  Public & administrative:  BV, VwVG, RPG, USG, NHG, AIG, DSG, URG, BGO, RTVG, BGF, SVG
  Tax, social & financial:  DBG, MWSTG, StHG, ATSG, UVG, AHVG, IVG, BVG, FINMAG, FIDLEG, KAG, VAG
  Procedure & enforcement:  BGG, SchKG, BGFA

CRITICAL: any code NOT in the table counts as "Other" (e.g. MSchG, UWG, PatG, LugUe, AsylG, KVG, VStG,
FusG, BankG, JStPO, BEG, GSchG, ArG, KG, PrHG, IRSG, FinfraG, or any SR-number-only law). "Other" is NOT
an uncertainty fallback -- it is correct whenever the dominant law is outside the table. About a fifth of
real cases are "Other", so do not avoid it.

Permitted labels, verbatim:
"Civil & commercial", "Criminal", "Other", "Procedure & enforcement", "Public & administrative", "Tax, social & financial"

Reply with JSON ONLY:
{"codes": ["ZGB","ZPO"], "domain": "Civil & commercial", "confidence": 0.8}"""


def load_key():
    for line in (ROOT / '.env').read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if line.startswith('DEEP_SEEK') and '=' in line:
            return line.split('=', 1)[1].strip().strip('"\'')
    raise SystemExit('DEEP_SEEK not found in .env')


def call_ds(key, messages, max_tokens=2500, retries=6):
    """Exponential backoff with jitter -- bursty concurrency draws 429s.

    max_tokens must be generous: deepseek-v4-flash is a REASONING model that
    emits a separate `reasoning_content` field and consumes the token budget
    thinking before it writes `content`. At 300 tokens it returned
    finish_reason='length' with empty content on 54% of queries -- the long-
    reasoning ones -- which would have silently biased the comparison toward
    easy queries. If content is still empty we fall back to any JSON object
    found in the reasoning trace.
    """
    h = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    payload = {'model': DS_MODEL, 'messages': messages,
               'max_tokens': max_tokens, 'temperature': 0.0}
    for a in range(retries):
        try:
            r = requests.post(DS_URL, headers=h, json=payload, timeout=300)
            if r.status_code == 200:
                msg = r.json()['choices'][0]['message']
                return msg.get('content') or msg.get('reasoning_content') or ''
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
                continue
            return None
        except requests.RequestException:
            time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
    return None


def parse(raw):
    """Return the LAST JSON object carrying a 'domain' key.

    Last, not first: when we fall back to a reasoning trace the text may contain
    several candidate objects, and the model's conclusion is the final one.
    """
    if not raw:
        return None
    best = None
    for m in re.finditer(r'\{[^{}]*\}', raw, re.S):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and 'domain' in o:
            best = o
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='analysis/router_deepseek_predictions.csv')
    args = ap.parse_args()

    key = load_key()

    if args.split == 'train':
        qids, queries, labels = load_train(limit=args.limit)
    else:
        df = pd.read_csv(ROOT / 'Data/val.csv')
        if args.limit:
            df = df.head(args.limit)
        qids = df['query_id'].astype(str).tolist()
        queries = df['query'].astype(str).tolist()
        labels = [domain_of_gold(set(cits(g))) for g in df['gold_citations']]

    print(f'DeepSeek router: {DS_MODEL}, {len(qids)} {args.split} queries, '
          f'concurrency={CONCURRENCY}', flush=True)

    out = [None] * len(qids)
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def work(i):
        msgs = [{'role': 'system', 'content': SYS},
                {'role': 'user', 'content': f'LEGAL QUESTION:\n{queries[i][:2200]}\n\nJSON:'}]
        p = parse(call_ds(key, msgs)) or {}
        d = p.get('domain')
        out[i] = {'query_id': qids[i],
                  'domain': d if d in LABELS else '',
                  'codes': ';'.join(str(c).strip() for c in (p.get('codes') or []) if str(c).strip())}
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f'  {done[0]}/{len(qids)}  [{time.time() - t0:.0f}s]', flush=True)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, range(len(qids))))

    df_out = pd.DataFrame(out)
    n_bad = int((df_out['domain'] == '').sum())
    outp = ROOT / args.out
    df_out.to_csv(outp, index=False)
    print(f'\nwrote {outp}   unparseable/failed: {n_bad}/{len(qids)}  '
          f'[{time.time() - t0:.0f}s]')
    print('now score with:')
    print(f'  python analysis/llm_router.py --score-predictions {args.out} --label deepseek')
    return 0


if __name__ == '__main__':
    sys.exit(main())
