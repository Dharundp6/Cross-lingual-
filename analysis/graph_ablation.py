# -*- coding: utf-8 -*-
"""Leave-one-out ablation of the anchored co-citation expansion.

The expansion is one of the two components claimed as the author's own, and it is built
from four decisions taken together:

  anchor      seed only from articles the question names verbatim
  cap         keep at most the top five neighbours of each seed
  gate        keep only neighbours co-cited at least three times
  hub filter  ignore decisions that cite more than 25 articles when building the graph

The recorded comparison put this configuration against five broader graph policies and
reported that it scored highest. That says the bundle is better than its alternatives; it
does not say which of the four decisions produces the effect. This switches each decision
off in turn, holding the other three and the rest of the system fixed, and scores the
system's emitted answer set.

Because the expansion only fires on queries that name an article, the table reports the
score on all queries and on the anchored subset, where the rule is actually active.

Usage:  python analysis/graph_ablation.py [--lang de|en]
Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

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

    print('building graphs...', flush=True)
    g_std = ds.build_graph(hub_limit=ds.HUB_LIMIT)
    g_nohub = ds.build_graph(hub_limit=None)
    parafreq = ds.build_parafreq()
    e_std = sum(len(v) for v in g_std.values()); e_nohub = sum(len(v) for v in g_nohub.values())
    print(f'  standard {len(g_std):,} nodes / {e_std:,} edges; without the hub filter '
          f'{len(g_nohub):,} nodes / {e_nohub:,} edges', flush=True)

    anchored_ix = [i for i in range(len(qids)) if ds.anchor_keys(qtext[i])]
    print(f'  {len(anchored_ix)} queries name at least one article', flush=True)

    CONFIGS = {
        'anchored expansion (as shipped)':
            dict(graph=g_std, anchored=True, n=ds.N_NEIGHBOURS, gate=ds.GATE),
        'without the explicit anchor':
            dict(graph=g_std, anchored=False, n=ds.N_NEIGHBOURS, gate=ds.GATE),
        'without the five-neighbour cap':
            dict(graph=g_std, anchored=True, n=10 ** 6, gate=ds.GATE),
        'without the weight threshold':
            dict(graph=g_std, anchored=True, n=ds.N_NEIGHBOURS, gate=1),
        'without the long-decision filter':
            dict(graph=g_nohub, anchored=True, n=ds.N_NEIGHBOURS, gate=ds.GATE),
        'no expansion at all':
            dict(graph=None, anchored=True, n=0, gate=ds.GATE),
    }

    rows = []
    for name, cfg in CONFIGS.items():
        sets = []
        for i in range(len(qids)):
            lv = ds.levers(qtext[i])
            if cfg['graph'] is None or cfg['n'] == 0:
                add = []
            elif cfg['anchored']:
                add = ds.kg_expand(qtext[i], cfg['graph'], parafreq, gate=cfg['gate'], n=cfg['n'])
            else:
                # seed from every article the levers already emitted, not only named ones
                seeds = set()
                for c in lv:
                    m = ds.re.match(r'Art\.\s*([0-9][0-9a-z]*(?:bis|ter|quater)?)\b.*?\s(\S+)$', c)
                    if m:
                        seeds.add((m.group(1), m.group(2)))
                seeds |= ds.anchor_keys(qtext[i])
                add = []
                for ak in seeds:
                    for nb, w in cfg['graph'].get(ak, Counter()).most_common(cfg['n']):
                        if w >= cfg['gate']:
                            forms = ds.PARA.get((nb[1], nb[0]), [])
                            if forms:
                                add.append(sorted(forms, key=lambda c: (-parafreq.get(c, 0), len(c)))[0])
            sets.append(list(dict.fromkeys([ds.BOILER] + lv + add)))
        f1s = np.array([ds.f1(sets[i], gold[i]) for i in range(len(qids))])
        rows.append({
            'configuration': name,
            'macro_f1_all': float(f1s.mean()),
            'macro_f1_anchored_subset': float(f1s[anchored_ix].mean()) if anchored_ix else None,
            'mean_picks_per_query': float(np.mean([len(s) for s in sets])),
            'mean_added_by_expansion': float(np.mean([len(s) for s in sets])
                                             - np.mean([len(ds.levers(qtext[i])) + 1
                                                        for i in range(len(qids))])),
        })
        print(f'  {name:36s} all={rows[-1]["macro_f1_all"]:.5f}  '
              f'anchored={rows[-1]["macro_f1_anchored_subset"]:.5f}  '
              f'picks={rows[-1]["mean_picks_per_query"]:.2f}', flush=True)

    out = ROOT / f'analysis/graph_ablation_{args.lang}.json'
    out.write_text(json.dumps({'lang': args.lang, 'n_queries': len(qids),
                               'n_anchored': len(anchored_ix), 'rows': rows}, indent=2),
                   encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
