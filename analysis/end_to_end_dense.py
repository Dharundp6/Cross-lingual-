# -*- coding: utf-8 -*-
"""End-to-end effect of the learned legal-area label on the retrieval pipeline's answer set.

This is the end-to-end counterpart to the component experiment. The component experiment
reordered a candidate pool and scored the pool. Here the prior is placed inside the
pipeline that assembles the submitted answer, and what is scored is the answer itself:
the emitted citation set, after the base ordering, the truncation, the deterministic lever
additions and the fixed procedural citation.

  ARM A  the pipeline as shipped: the base ordering carries a keyword code bonus
         (alpha_code = 0.05 over the codes the keyword rules extract from the question)
  ARM B  the same pipeline with the learned legal-area label added to the same ordering
         step, as a second soft bonus

Nothing else differs: same encoder, same corpus, same truncation depth, same lever stages,
same boilerplate, same gold. alpha_domain = 0 makes the two arms identical, which is
asserted before anything is scored.

PROTOCOL. alpha_domain is selected on the 497-query selection half and applied once to the
disjoint 496-query half. The truncation depth is held at the shipped constant K = 18 for
both arms and both halves, so no output-size parameter is fitted on the scored half. That
is the one respect in which this supersedes the recorded component experiment.

Per-query answer sets, gold sets and F1 are written for both arms, so every addition can be
scored against the admission bar rather than reasoned about.

Usage:  python analysis/end_to_end_dense.py [--lang de|en]
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
from llm_router import DOMAIN, expected_codes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAW_EMB = ROOT / 'assets' / 'hf' / 'law_embs_legal_v3.npy'
QUERIES = ROOT / 'current' / '_hf_assets' / 'queries_v3.npz'
ALPHA_CODE = 0.05        # shipped keyword code bonus
K_BASE = 18              # shipped truncation depth
POOL = 200               # shipped reordering window
ALPHA_DOMAIN = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]


def norm(a):
    a = np.asarray(a).astype(np.float32)
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='de', choices=['de', 'en'])
    args = ap.parse_args()

    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    router = pd.read_csv(ROOT / 'analysis' / 'router_claude_predictions.csv')
    rdom = {str(a): b for a, b in zip(router['query_id'], router['domain'])}

    en = {}
    if args.lang == 'en':
        p = ROOT / 'analysis' / 'train_queries_en.json'
        en = json.loads(p.read_text(encoding='utf-8'))
        print(f'English text available for {len(en)} queries')

    keep, qids, qtext, gold_all, gold_stat = [], [], [], [], []
    for _, r in train.iterrows():
        qid = str(r['query_id'])
        g = ds.parse(r['gold_citations'])
        gs = {c for c in g if not ds.is_court(c) and c in ds.key_set}
        ok = bool(gs) and (args.lang == 'de' or bool(en.get(qid)))
        keep.append(ok)
        if ok:
            qids.append(qid)
            qtext.append(en[qid] if args.lang == 'en' else str(r['query']))
            gold_all.append(set(g))
            gold_stat.append(gs)
    mask = np.array(keep)
    print(f'{len(qids)} queries scored ({args.lang})', flush=True)

    print('loading corpus embeddings...', flush=True)
    L = norm(np.load(LAW_EMB))
    Q = norm(np.load(QUERIES)['train'])[mask]
    code_of = [ds.PARA and c.split()[-1] for c in ds.cits]
    CODE = np.array([c.split()[-1] if c.split() else '' for c in ds.cits], dtype=object)
    dom_of_cit = np.array([DOMAIN.get(c, None) for c in CODE], dtype=object)

    print('building co-citation graph...', flush=True)
    cocite = ds.build_graph()
    parafreq = ds.build_parafreq()

    # per-query pool of POOL candidates, their scores and their lever additions
    pools, sims, lever_adds, dense18 = [], [], [], []
    for i in range(len(qids)):
        s = Q[i] @ L.T
        top = np.argpartition(-s, POOL)[:POOL]
        top = top[np.argsort(-s[top])]
        pools.append(top)
        sims.append(s[top])
        lv = ds.levers(qtext[i]) + ds.kg_expand(qtext[i], cocite, parafreq)
        lever_adds.append(list(dict.fromkeys(lv)))
        dense18.append({ds.cits[j] for j in top[:K_BASE]} | {ds.BOILER})
        if (i + 1) % 200 == 0:
            print(f'  pooled {i + 1}/{len(qids)}', flush=True)

    def answer(i, alpha_domain):
        top, s = pools[i], sims[i]
        bonus = ALPHA_CODE * np.array([CODE[j] in expected_codes(qtext[i]) for j in top], dtype=np.float32)
        if alpha_domain:
            d = rdom.get(qids[i])
            if d:
                bonus = bonus + alpha_domain * np.array([dom_of_cit[j] == d for j in top],
                                                        dtype=np.float32)
        order = top[np.argsort(-(s + bonus), kind='stable')]
        base = [ds.cits[j] for j in order[:K_BASE]] + [ds.BOILER]
        adds = [c for c in lever_adds[i] if c not in dense18[i]]
        return list(dict.fromkeys(base + adds))

    sets = {}
    for a in ALPHA_DOMAIN:
        sets[a] = [answer(i, a) for i in range(len(qids))]
    assert sets[0.0] == [answer(i, 0.0) for i in range(len(qids))]
    print('PASS identity: alpha_domain=0 is the shipped ordering', flush=True)

    sel = list(range(0, len(qids), 2))
    ev = list(range(1, len(qids), 2))
    report = {'lang': args.lang, 'n_queries': len(qids), 'n_select': len(sel), 'n_eval': len(ev),
              'alpha_code': ALPHA_CODE, 'k_base': K_BASE, 'pool': POOL,
              'alpha_domain_grid': ALPHA_DOMAIN,
              'k_selected_on': 'shipped constant, not fitted on either half'}

    for gname, gold in (('statute_only', gold_stat), ('all_gold', gold_all)):
        f1s = {a: [ds.f1(sets[a][i], gold[i]) for i in range(len(qids))] for a in ALPHA_DOMAIN}
        sel_scores = {a: float(np.mean([f1s[a][i] for i in sel])) for a in ALPHA_DOMAIN}
        best = max(sel_scores, key=sel_scores.get)
        base_eval = float(np.mean([f1s[0.0][i] for i in ev]))
        tuned_eval = float(np.mean([f1s[best][i] for i in ev]))
        d = np.array([f1s[best][i] - f1s[0.0][i] for i in ev])
        rng = np.random.default_rng(20260910)
        draws = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        report[gname] = {
            'selection_half_scores': sel_scores,
            'alpha_domain_selected_on_selection_half': best,
            'eval_half_scores_all_alphas': {a: float(np.mean([f1s[a][i] for i in ev]))
                                            for a in ALPHA_DOMAIN},
            'eval_base': base_eval, 'eval_tuned': tuned_eval, 'delta': tuned_eval - base_eval,
            'delta_ci95': [float(lo), float(hi)],
            'n_eval_answer_sets_changed': int(sum(1 for i in ev if sets[best][i] != sets[0.0][i])),
            'mean_picks_per_query': float(np.mean([len(s) for s in sets[best]])),
        }
        print(f'\n[{gname}] alpha_domain selected on the {len(sel)}-query half = {best}')
        print(f'  eval base  {base_eval:.5f}')
        print(f'  eval tuned {tuned_eval:.5f}   delta {tuned_eval - base_eval:+.5f} '
              f'[{lo:+.5f}, {hi:+.5f}]')
        print(f'  answer sets changed: {report[gname]["n_eval_answer_sets_changed"]}/{len(ev)}')

    out = ROOT / f'analysis/end_to_end_dense_{args.lang}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    best = report['statute_only']['alpha_domain_selected_on_selection_half']
    perq = ROOT / f'analysis/end_to_end_dense_{args.lang}_per_query.json'
    perq.write_text(json.dumps([
        {'query_id': qids[i], 'gold_statute': sorted(gold_stat[i]),
         'arm_a': sets[0.0][i], 'arm_b': sets[best][i],
         'f1_a': ds.f1(sets[0.0][i], gold_stat[i]), 'f1_b': ds.f1(sets[best][i], gold_stat[i]),
         'in_eval_half': i in set(ev)} for i in range(len(qids))], indent=1), encoding='utf-8')
    print(f'\nwrote {out}\nwrote {perq}')


if __name__ == '__main__':
    main()
