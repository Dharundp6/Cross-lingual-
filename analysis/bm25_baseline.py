# -*- coding: utf-8 -*-
"""End-to-end BM25 baseline over the statute corpus.

Chapter 2 argues that lexical evidence is strong in this task, and Chapter 7 concludes that
identifiers surviving translation are the most reliable signal. Neither claim had a lexical
comparator: BM25 appeared only on the court side, where it scores zero. This supplies the
missing baseline on the statute side, end to end, scored on the answer set.

Two conditions, because the argument has two halves:
  de  the German question against the German corpus, the in-language upper bound for a
      lexical retriever on this task
  en  the English question against the same German corpus, the condition the deployed
      system actually faces

The truncation depth K is swept on the 497-query selection half and applied once to the
disjoint 496, so the baseline is granted the same protocol as the systems it is compared
against and is not handicapped by an arbitrary depth.

Usage:  python analysis/bm25_baseline.py
Run from the repository root.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivered_system as ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'bm25_baseline.json'
KS = [1, 2, 3, 5, 8, 10, 15, 20, 30]
TOK = re.compile(r'[\w.]+', re.UNICODE)


def tokenize(s):
    """Keep legal notation: 'Art.', 'Abs.', code abbreviations and numbers stay intact."""
    return [t for t in TOK.findall(str(s).lower()) if t not in {'.', ''}]


def main():
    t0 = time.time()
    laws = pd.read_csv(ROOT / 'Data' / 'laws_de.csv').fillna('')
    texts = (laws['citation'].astype(str) + ' ' + laws.get('title', '').astype(str) + ' '
             + laws.get('text', '').astype(str)).tolist()
    cits = laws['citation'].astype(str).tolist()
    print(f'{len(cits):,} statute passages; tokenising...', flush=True)
    corpus = [tokenize(t) for t in texts]
    print(f'  tokenised [{time.time() - t0:.0f}s]; indexing...', flush=True)
    bm25 = BM25Okapi(corpus)
    print(f'  indexed [{time.time() - t0:.0f}s]', flush=True)

    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    en = json.loads((ROOT / 'analysis' / 'train_queries_en.json').read_text(encoding='utf-8'))

    qids, de_text, en_text, gold = [], [], [], []
    for _, r in train.iterrows():
        qid = str(r['query_id'])
        gs = {c for c in ds.parse(r['gold_citations']) if not ds.is_court(c) and c in ds.key_set}
        if not gs or not en.get(qid):
            continue
        qids.append(qid)
        de_text.append(str(r['query']))
        en_text.append(en[qid])
        gold.append(gs)
    print(f'{len(qids)} queries with statute gold and an English translation', flush=True)

    sel = list(range(0, len(qids), 2))
    ev = list(range(1, len(qids), 2))
    report = {'n_queries': len(qids), 'n_select': len(sel), 'n_eval': len(ev), 'k_grid': KS}

    for lang, qs in (('de', de_text), ('en', en_text)):
        ranked = []
        for i, q in enumerate(qs):
            s = bm25.get_scores(tokenize(q))
            top = np.argpartition(-s, max(KS))[:max(KS)]
            ranked.append([cits[j] for j in top[np.argsort(-s[top])]])
            if (i + 1) % 100 == 0:
                print(f'  [{lang}] scored {i + 1}/{len(qs)}  [{time.time() - t0:.0f}s]', flush=True)
        f1_at = {}
        for K in KS:
            f1_at[K] = [ds.f1(list(dict.fromkeys(ranked[i][:K] + [ds.BOILER])), gold[i])
                        for i in range(len(qids))]
        sel_scores = {K: float(np.mean([f1_at[K][i] for i in sel])) for K in KS}
        bestK = max(sel_scores, key=sel_scores.get)
        eval_score = float(np.mean([f1_at[bestK][i] for i in ev]))
        recall_any = float(np.mean([len(set(ranked[i][:max(KS)]) & gold[i]) > 0
                                    for i in range(len(qids))]))
        report[lang] = {
            'selection_half_scores': sel_scores, 'k_selected_on_selection_half': bestK,
            'eval_macro_f1': eval_score,
            'eval_scores_all_k': {K: float(np.mean([f1_at[K][i] for i in ev])) for K in KS},
            f'share_of_queries_with_any_gold_in_top_{max(KS)}': recall_any,
        }
        print(f'[{lang}] K={bestK} selected on the {len(sel)}-query half; '
              f'eval Macro-F1 {eval_score:.5f}; any gold in top {max(KS)}: {recall_any:.3f}',
              flush=True)

    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'wrote {OUT}  [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
