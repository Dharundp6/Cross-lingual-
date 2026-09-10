# -*- coding: utf-8 -*-
"""Apply the admission bar as a measurement rather than as a framing device.

Equation 5.4 says an addition raises expected Macro-F1 only when its probability of being
correct exceeds the per-query bar

    pi* = F1(before) / 2

Until now the dissertation derived that rule and then reasoned with it, because per-addition
correctness was never logged. This logs it. For each mechanism that adds citations to an
answer, and for each query, it records

    the answer before the mechanism runs, and its F1
    every citation the mechanism adds, and whether that citation is in the gold set
    the per-query bar pi* implied by the answer before the addition

and then reports the observed precision of the additions against the mean bar they had to
clear. A mechanism whose observed precision sits below the bar is predicted to lower the
score, and the measured change in Macro-F1 is reported beside the prediction so the two can
be compared.

This also explains why one mechanism can help on one evaluation set and hurt on another:
the bar rises with the quality of the answer it is added to and falls as the required set
grows, so the same precision clears the bar in one regime and fails it in the other.

Usage:  python analysis/admission_bar_measured.py [--lang de|en]
Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivered_system as ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='de', choices=['de', 'en'])
    args = ap.parse_args()

    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    en = {}
    if args.lang == 'en':
        en = json.loads((ROOT / 'analysis' / 'train_queries_en.json').read_text(encoding='utf-8'))

    qids, qtext, gold = [], [], []
    for _, r in train.iterrows():
        qid = str(r['query_id'])
        gs = {c for c in ds.parse(r['gold_citations']) if not ds.is_court(c) and c in ds.key_set}
        if not gs or (args.lang == 'en' and not en.get(qid)):
            continue
        qids.append(qid)
        qtext.append(en[qid] if args.lang == 'en' else str(r['query']))
        gold.append(gs)
    print(f'{len(qids)} queries ({args.lang})', flush=True)

    cocite = ds.build_graph()
    parafreq = ds.build_parafreq()

    STAGES = ['explicit1', 'family', 'stpo', 'domain2', 'pil', 'domain3', 'explicit2', 'lawname']

    records = []
    rows = []

    def measure(name, before_fn, after_fn):
        n_add = n_corr = 0
        bars, f1_b, f1_a, fired = [], [], [], 0
        for i in range(len(qids)):
            b = before_fn(i)
            a = after_fn(i)
            added = [c for c in a if c not in set(b)]
            fb = ds.f1(b, gold[i])
            fa = ds.f1(a, gold[i])
            f1_b.append(fb)
            f1_a.append(fa)
            if added:
                fired += 1
                n_add += len(added)
                n_corr += sum(1 for c in added if c in gold[i])
                bars.append(fb / 2)
                records.append({'query_id': qids[i], 'mechanism': name,
                                'n_added': len(added),
                                'n_added_correct': sum(1 for c in added if c in gold[i]),
                                'f1_before': fb, 'f1_after': fa, 'bar': fb / 2,
                                'added': added})
        pi = n_corr / n_add if n_add else float('nan')
        bar = float(np.mean(bars)) if bars else float('nan')
        rows.append({
            'mechanism': name,
            'queries_fired_on': fired,
            'additions': n_add,
            'additions_correct': n_corr,
            'observed_precision': pi,
            'mean_bar_pi_star': bar,
            'clears_bar': bool(pi > bar) if n_add else None,
            'macro_f1_before': float(np.mean(f1_b)),
            'macro_f1_after': float(np.mean(f1_a)),
            'macro_f1_change': float(np.mean(f1_a) - np.mean(f1_b)),
        })
        print(f'  {name:34s} fired={fired:4d} adds={n_add:5d} pi={pi:.4f} '
              f'bar={bar:.4f} {"CLEARS" if pi > bar else "below":6s} '
              f'dF1={rows[-1]["macro_f1_change"]:+.5f}', flush=True)

    lev_full = [ds.levers(q) for q in qtext]
    print('per-mechanism admission-bar measurement:', flush=True)

    # each lever stage, added on top of everything before it in the shipped order
    for k in range(len(STAGES)):
        before = [[ds.BOILER] + ds.levers(qtext[i], stages=tuple(STAGES[:k])) for i in range(len(qids))]
        after = [[ds.BOILER] + ds.levers(qtext[i], stages=tuple(STAGES[:k + 1])) for i in range(len(qids))]
        measure(STAGES[k], lambda i, b=before: b[i], lambda i, a=after: a[i])

    # the anchored expansion, added on top of the full lever chain
    base = [list(dict.fromkeys([ds.BOILER] + lev_full[i])) for i in range(len(qids))]
    withkg = [list(dict.fromkeys(base[i] + ds.kg_expand(qtext[i], cocite, parafreq)))
              for i in range(len(qids))]
    measure('anchored co-citation expansion', lambda i: base[i], lambda i: withkg[i])

    # the fixed procedural citation, measured against the answer without it
    nb = [[c for c in withkg[i] if c != ds.BOILER] for i in range(len(qids))]
    measure('fixed procedural citation', lambda i: nb[i], lambda i: withkg[i])

    # how the bar depends on the size of the required set
    sizes = np.array([len(g) for g in gold])
    f1s = np.array([ds.f1(withkg[i], gold[i]) for i in range(len(qids))])
    by_size = {}
    for lo, hi, label in ((1, 2, '1-2'), (3, 5, '3-5'), (6, 10, '6-10'), (11, 10 ** 6, '11+')):
        m = (sizes >= lo) & (sizes <= hi)
        if m.any():
            by_size[label] = {'n_queries': int(m.sum()),
                              'mean_f1': float(f1s[m].mean()),
                              'mean_bar_pi_star': float(f1s[m].mean() / 2)}

    out = ROOT / f'analysis/admission_bar_measured_{args.lang}.json'
    out.write_text(json.dumps({'lang': args.lang, 'n_queries': len(qids),
                               'mechanisms': rows, 'bar_by_required_set_size': by_size},
                              indent=2), encoding='utf-8')
    perq = ROOT / f'analysis/admission_bar_per_addition_{args.lang}.json'
    perq.write_text(json.dumps(records, indent=1), encoding='utf-8')
    print('\nbar by size of the required set:')
    for k, v in by_size.items():
        print(f'  gold size {k:5s} n={v["n_queries"]:4d}  mean F1 {v["mean_f1"]:.4f}  '
              f'bar {v["mean_bar_pi_star"]:.4f}')
    print(f'\nwrote {out}\nwrote {perq}')


if __name__ == '__main__':
    main()
