# -*- coding: utf-8 -*-
"""The Macro-F1 ceiling implied by emitting statutes only.

Design requirement 3 restricts the delivered system to statute citations, while court
authorities are a substantial share of the required set. A reader cannot interpret any
reported score without knowing what was attainable under that restriction. This computes
it exactly: for each query the best possible answer under the restriction is the statute
part of the gold set, so

    F1_max(q) = 2 |G_statute(q)| / (|G_statute(q)| + |G(q)|)

and the ceiling is the mean over queries. It is an upper bound no configuration can pass,
not an estimate of anything.

Run from the repository root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivered_system as ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'statute_ceiling.json'


def ceiling(df):
    rows = []
    for _, r in df.iterrows():
        g = [c for c in ds.parse(r['gold_citations'])]
        if not g:
            continue
        gs = [c for c in g if not ds.is_court(c)]
        gs_in = [c for c in gs if c in ds.key_set]
        rows.append({
            'n_gold': len(g), 'n_statute': len(gs), 'n_statute_in_corpus': len(gs_in),
            'f1_max': 2 * len(gs) / (len(gs) + len(g)),
            'f1_max_corpus': 2 * len(gs_in) / (len(gs_in) + len(g)) if (len(gs_in) + len(g)) else 0.0,
        })
    a = pd.DataFrame(rows)
    return {
        'n_queries': int(len(a)),
        'mean_gold_per_query': float(a.n_gold.mean()),
        'court_share_of_gold': float(1 - a.n_statute.sum() / a.n_gold.sum()),
        'macro_f1_ceiling_statute_only': float(a.f1_max.mean()),
        'macro_f1_ceiling_statute_in_corpus': float(a.f1_max_corpus.mean()),
        'queries_with_no_statute_gold': int((a.n_statute == 0).sum()),
    }


def main():
    out = {}
    for name in ('train', 'val', 'test'):
        p = ROOT / 'Data' / f'{name}.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if 'gold_citations' not in df.columns:
            out[name] = {'note': 'no gold column in the supplied file'}
            continue
        out[name] = ceiling(df)
        print(name, json.dumps(out[name], indent=2))
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
