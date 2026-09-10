# -*- coding: utf-8 -*-
"""Reproduce the keyword and DeepSeek router control runs from the archived prediction files
and compare per-query F1 vectors with the recorded JSON. Run from the repository root."""
import json, subprocess, sys

RUNS = {
    'keyword': 'analysis/router_keyword_predictions.csv',
    'deepseek': 'analysis/router_deepseek_predictions.csv',
}
for name, router in RUNS.items():
    out = f'analysis/router_rerank_repro_{name}.json'
    subprocess.run([sys.executable, 'analysis/router_rerank_experiment.py', '--router', router, '--out', out],
                   check=True, capture_output=True)
    a = json.load(open(f'analysis/router_rerank_{name}.json'))['split_half']
    b = json.load(open(out))['split_half']
    keys = ['alpha', 'beta', 'k', 'base', 'tuned', 'delta']
    print(name, 'recorded', {k: a[k] for k in keys})
    print(name, 'repro   ', {k: b[k] for k in keys})
    print(name, 'per-query identical:', a['tuned_per_query_f1'] == b['tuned_per_query_f1'],
          'base identical:', a['base_per_query_f1'] == b['base_per_query_f1'])
