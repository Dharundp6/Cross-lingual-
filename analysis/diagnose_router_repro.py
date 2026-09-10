# -*- coding: utf-8 -*-
"""Which archived artefact reproduces each recorded router run?

The manuscript records that re-deriving the archived per-query table did not
reproduce the recorded routing score exactly. This locates the cause: the
`router_accuracy_*.csv` tables carry a gold column and a prediction column but
no model-returned `codes` column, so a rebuild from them silently substitutes
keyword-derived codes for the beta term. The `router_*_predictions.csv` files do
carry `codes`. Run from the repository root.
"""
import json
import pandas as pd

REC = {
    'claude': 'analysis/router_rerank_results.json',
    'keyword': 'analysis/router_rerank_keyword.json',
    'deepseek': 'analysis/router_rerank_deepseek.json',
}
REPRO = {
    'claude': 'analysis/router_rerank_repro_predictionscsv.json',
    'keyword': 'analysis/router_rerank_repro_keyword.json',
    'deepseek': 'analysis/router_rerank_repro_deepseek.json',
}

print('=== per-run reproduction from the *_predictions.csv artefacts ===')
for name in REC:
    a = json.load(open(REC[name]))['split_half']
    b = json.load(open(REPRO[name]))['split_half']
    ok = (a['tuned_per_query_f1'] == b['tuned_per_query_f1']
          and a['base_per_query_f1'] == b['base_per_query_f1']
          and a['eval_query_ids'] == b['eval_query_ids'])
    print(f'{name:9s} recorded tuned={a["tuned"]:.9f} delta={a["delta"]:+.9f} '
          f'(a={a["alpha"]}, b={a["beta"]}, K={a["k"]})')
    print(f'{"":9s} repro    tuned={b["tuned"]:.9f} delta={b["delta"]:+.9f} '
          f'(a={b["alpha"]}, b={b["beta"]}, K={b["k"]})  per-query identical={ok}')

print()
print('=== where the beta term got its codes ===')
grids = {n: json.load(open(p))['splits']['train']['grid'] for n, p in REC.items()}
zero = {n: {k: v for k, v in g.items() if k.startswith('0.0|')} for n, g in grids.items()}
print('alpha=0 rows depend only on the code set:')
print('  deepseek == keyword :', zero['deepseek'] == zero['keyword'])
print('  deepseek == claude  :', zero['deepseek'] == zero['claude'])
print('  keyword  == claude  :', zero['keyword'] == zero['claude'])

print()
print('=== archived artefact columns ===')
for f in ['router_claude_predictions', 'router_deepseek_predictions', 'router_deepseek_all',
          'router_keyword_predictions', 'router_accuracy_claude', 'router_accuracy_deepseek']:
    d = pd.read_csv(f'analysis/{f}.csv')
    print(f'{f:32s} rows={len(d):5d} cols={list(d.columns)}')
