# -*- coding: utf-8 -*-
"""Does the learned legal-area label improve the DELIVERED system's citation set?

The recorded router experiment reordered a 150-candidate dense statute pool. The delivered
rule-based system does not contain that pool, so the recorded contrast measured a component
in isolation. This measures the label end to end, on the answer set the system actually
emits, at the two points inside the delivered pipeline where an ordering decision is made:

  ARM A  the delivered system exactly as shipped
  ARM B  the same system with the learned label added to the co-citation neighbour weight,
         so it changes WHICH neighbours survive the top-N cut

alpha is selected on the 497-query selection half and applied once to the disjoint 496,
so unlike the recorded component experiment nothing is tuned on the scored half.

Everything else is held fixed: same lever stages, same gate, same N, same boilerplate,
same gold. With alpha = 0 the two arms are identical by construction, which is asserted.

Per-query answer sets and F1 for both arms are written out, so the admission bar can be
applied to the additions rather than argued about.

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
from llm_router import DOMAIN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'analysis' / 'end_to_end_router.json'
PERQ = ROOT / 'analysis' / 'end_to_end_router_per_query.json'
ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def main():
    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    router = pd.read_csv(ROOT / 'analysis' / 'router_claude_predictions.csv')
    rdom = {str(a): b for a, b in zip(router['query_id'], router['domain'])}

    print('building co-citation graph...', flush=True)
    cocite = ds.build_graph()
    parafreq = ds.build_parafreq()
    print(f'  {len(cocite):,} statute nodes', flush=True)

    qids, qtext, gold_all, gold_stat = [], [], [], []
    for _, r in train.iterrows():
        g = [c for c in ds.parse(r['gold_citations'])]
        gs = {c for c in g if not ds.is_court(c) and c in ds.key_set}
        if not gs:
            continue
        qids.append(str(r['query_id']))
        qtext.append(str(r['query']))
        gold_all.append(set(g))
        gold_stat.append(gs)
    print(f'  {len(qids)} training queries with at least one statute gold citation', flush=True)

    # identity check: alpha = 0 must reproduce the shipped answer set exactly
    for i in range(len(qids)):
        a = ds.emit(qtext[i], cocite, parafreq)
        b = ds.emit(qtext[i], cocite, parafreq, domain=rdom.get(qids[i]), alpha=0.0,
                    domain_of_code=DOMAIN)
        assert a == b, f'IDENTITY FAILED on {qids[i]}'
    print('  PASS identity: alpha=0 reproduces the shipped answer set on every query', flush=True)

    # answer sets per alpha
    sets = {}
    for alpha in ALPHAS:
        sets[alpha] = [ds.emit(qtext[i], cocite, parafreq, domain=rdom.get(qids[i]),
                               alpha=alpha, domain_of_code=DOMAIN) for i in range(len(qids))]
        print(f'  alpha={alpha}: mean picks/query '
              f'{np.mean([len(s) for s in sets[alpha]]):.2f}', flush=True)

    sel = list(range(0, len(qids), 2))
    ev = list(range(1, len(qids), 2))

    def mean_f1(idx, alpha, gold):
        return float(np.mean([ds.f1(sets[alpha][i], gold[i]) for i in idx]))

    report = {'n_queries': len(qids), 'n_select': len(sel), 'n_eval': len(ev),
              'alphas': ALPHAS, 'graph': {'gate': ds.GATE, 'n_neighbours': ds.N_NEIGHBOURS,
                                          'hub_limit': ds.HUB_LIMIT}}

    for gname, gold in (('statute_only', gold_stat), ('all_gold', gold_all)):
        sel_scores = {a: mean_f1(sel, a, gold) for a in ALPHAS}
        best_alpha = max(sel_scores, key=sel_scores.get)
        base_eval = mean_f1(ev, 0.0, gold)
        tuned_eval = mean_f1(ev, best_alpha, gold)
        per_q_base = [ds.f1(sets[0.0][i], gold[i]) for i in ev]
        per_q_tuned = [ds.f1(sets[best_alpha][i], gold[i]) for i in ev]
        d = np.array(per_q_tuned) - np.array(per_q_base)
        rng = np.random.default_rng(20260910)
        draws = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        changed = int(sum(1 for i in ev if sets[best_alpha][i] != sets[0.0][i]))
        report[gname] = {
            'selection_half_scores': sel_scores, 'alpha_selected_on_selection_half': best_alpha,
            'eval_half_scores_all_alphas': {a: mean_f1(ev, a, gold) for a in ALPHAS},
            'eval_half_sets_changed_all_alphas': {
                a: int(sum(1 for i in ev if sets[a][i] != sets[0.0][i])) for a in ALPHAS},
            'eval_base': base_eval, 'eval_tuned': tuned_eval,
            'delta': tuned_eval - base_eval,
            'delta_ci95': [float(lo), float(hi)],
            'n_eval_queries_whose_answer_set_changed': changed,
        }
        print(f'\n[{gname}] alpha selected on the 497 half = {best_alpha}')
        print(f'  eval base  {base_eval:.5f}')
        print(f'  eval tuned {tuned_eval:.5f}   delta {tuned_eval - base_eval:+.5f} '
              f'[{lo:+.5f}, {hi:+.5f}]')
        print(f'  answer set changed on {changed}/{len(ev)} scored queries')

    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    best = report['statute_only']['alpha_selected_on_selection_half']
    PERQ.write_text(json.dumps([
        {'query_id': qids[i], 'gold_statute': sorted(gold_stat[i]),
         'arm_a': sets[0.0][i], 'arm_b': sets[best][i],
         'f1_a': ds.f1(sets[0.0][i], gold_stat[i]), 'f1_b': ds.f1(sets[best][i], gold_stat[i]),
         'in_eval_half': i in set(ev)}
        for i in range(len(qids))], indent=1), encoding='utf-8')
    print(f'\nwrote {OUT}\nwrote {PERQ}')


if __name__ == '__main__':
    main()
