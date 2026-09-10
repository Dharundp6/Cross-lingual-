# -*- coding: utf-8 -*-
"""Why can the learned label not act on the delivered system? Count the opportunities.

A ranking prior can only change an answer when the system makes an ordering decision that
the prior can reach. In the delivered rule-based system the only such decision is the
top-N cut over co-citation neighbours, and that cut binds only when a query names an
article AND that article has more than N neighbours above the gate. This counts both.

Run from the repository root.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivered_system as ds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'end_to_end_reach.json'


def main():
    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    cocite = ds.build_graph()

    n = 0
    with_anchor = 0
    cut_binds = 0
    n_lever_only = 0
    picks = []
    for _, r in train.iterrows():
        q = str(r['query'])
        gs = {c for c in ds.parse(r['gold_citations']) if not ds.is_court(c) and c in ds.key_set}
        if not gs:
            continue
        n += 1
        aks = ds.anchor_keys(q)
        if aks:
            with_anchor += 1
        binds = False
        for ak in aks:
            above = [w for w in cocite.get(ak, Counter()).values() if w >= ds.GATE]
            if len(above) > ds.N_NEIGHBOURS:
                binds = True
        if binds:
            cut_binds += 1
        lv = ds.levers(q)
        if not lv:
            n_lever_only += 1
        picks.append(len([ds.BOILER] + lv))

    res = {
        'n_queries': n,
        'queries_naming_an_article': with_anchor,
        'share_naming_an_article': round(with_anchor / n, 4),
        'queries_where_the_topN_cut_binds': cut_binds,
        'share_where_the_prior_could_change_the_answer': round(cut_binds / n, 4),
        'queries_emitting_only_the_fixed_procedural_citation': n_lever_only,
        'gate': ds.GATE, 'n_neighbours': ds.N_NEIGHBOURS,
    }
    OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
