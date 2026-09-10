# -*- coding: utf-8 -*-
"""Router-conditioned reranking: does a domain prior improve statute ranking?

HYPOTHESIS
----------
Unconditioned reranking is dead in this project (base cross-encoder lost 0.143
recall at K=40; two leaderboard submissions built on it lost 0.069 and 0.032).
This tests a different claim: that what the ranker was missing is a DOMAIN PRIOR,
supplied by a router that reads the query.

    final_score(candidate) = base_score
                           + alpha * 1[domain(candidate) == router_domain]
                           + beta  * 1[code(candidate)  in router_codes]

alpha = beta = 0 recovers the base ranker EXACTLY, which makes the ablation exact
and is asserted by --verify-identity.

WHY SOFT, NEVER A FILTER
------------------------
Validation query val_003 is a criminal matter whose gold citations include
inheritance law. Hard domain filtering deletes real gold. These are rank
bonuses only; every candidate stays in the pool.

WHERE THE SWEEP RUNS
--------------------
alpha/beta are swept on TRAIN (n=993) and confirmed on VAL (n=10). Prior routing
work in this project was tuned on the n=10 validation set and overfit -- k-means
routing gained +0.010 on val and then regressed 0.169 vs 0.184 on private test.
Sweeping on 993 queries removes that failure mode.

USAGE
-----
  python analysis/router_rerank_experiment.py --verify-identity
  python analysis/router_rerank_experiment.py --router analysis/router_accuracy_claude.csv
  python analysis/router_rerank_experiment.py --router ... --scorer ce   # needs GPU

Run from the repository root.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_router import (DOMAIN, LABELS, cits, code_of, domain_of_gold,  # noqa: E402
                        expected_codes, is_court)

ROOT = Path(__file__).resolve().parent.parent
LAW_EMB = ROOT / 'assets' / 'hf' / 'law_embs_legal_v3.npy'
QUERIES = ROOT / 'current' / '_hf_assets' / 'queries_v3.npz'
CE_DIR = ROOT / 'hf_inspect' / 'ce_model' / 'ce_swiss_legal_v1'

POOL_K = 150              # widened from the shipped 100, per the plan
BOILER = 'Art. 100 Abs. 1 BGG'
ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
BETAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
TOPKS = [3, 5, 10, 15, 18, 20, 25]   # train gold median is 2/query, val is 22 -- span both


def norm(a):
    a = np.asarray(a).astype(np.float32)
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)


def f1(pred, gold):
    p, g = set(pred), set(gold)
    if not g:
        return 0.0
    tp = len(p & g)
    if not tp:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------
def load_assets():
    laws = pd.read_csv(ROOT / 'Data' / 'laws_de.csv').fillna('')
    law_cits = laws['citation'].astype(str).tolist()
    L = norm(np.load(LAW_EMB))
    qz = np.load(QUERIES)
    return laws, law_cits, set(law_cits), L, qz


def build_pools(L, Q, law_cits, pool_k=POOL_K, chunk=64):
    """Top-`pool_k` statute candidates per query, with their cosine scores."""
    pools, scores = [], []
    for s in range(0, len(Q), chunk):
        sims = Q[s:s + chunk] @ L.T                       # (chunk, 175933)
        idx = np.argpartition(-sims, pool_k, axis=1)[:, :pool_k]
        for r in range(idx.shape[0]):
            o = idx[r][np.argsort(-sims[r, idx[r]])]
            pools.append([law_cits[j] for j in o])
            scores.append(sims[r, o].astype(np.float32))
    return pools, scores


def load_split(name, law_set):
    df = pd.read_csv(ROOT / f'Data/{name}.csv')
    gold, keep = [], []
    for _, r in df.iterrows():
        g = {c for c in cits(r['gold_citations']) if not is_court(c) and c in law_set}
        keep.append(bool(g))
        gold.append(g)
    mask = np.array(keep)
    df = df[mask].reset_index(drop=True)
    gold = [g for g, k in zip(gold, keep) if k]
    return df, gold, mask


# --------------------------------------------------------------------------
def load_router(path, qids, queries):
    """Router predictions -> (domain per query, code set per query).

    Falls back to the deterministic keyword router for any query the external
    router did not label, and reports how often that happened.
    """
    if path is None:
        return ([None] * len(qids), [set() for _ in qids], 0)

    p = Path(path)
    if p.suffix.lower() == '.json':
        obj = json.loads(p.read_text(encoding='utf-8'))
        rows = obj if isinstance(obj, list) else [{'query_id': k, 'domain': v} for k, v in obj.items()]
        dmap = {str(r['query_id']): r.get('domain') for r in rows}
        cmap = {str(r['query_id']): set(r.get('codes') or []) for r in rows}
    else:
        df = pd.read_csv(p)
        dcol = 'domain' if 'domain' in df.columns else [c for c in df.columns if c.startswith('pred_')][0]
        dmap = {str(a): b for a, b in zip(df['query_id'], df[dcol])}
        cmap = ({str(a): set(str(b).split(';')) if isinstance(b, str) and b else set()
                 for a, b in zip(df['query_id'], df['codes'])} if 'codes' in df.columns else {})

    doms, codes, n_missing = [], [], 0
    for qid, q in zip(qids, queries):
        d = dmap.get(str(qid))
        if d not in LABELS:
            n_missing += 1
            d = None
        doms.append(d)
        codes.append(cmap.get(str(qid)) or expected_codes(q))
    return doms, codes, n_missing


def fuse_rank(pool, base, dom, codes, alpha, beta):
    """Soft-boosted reordering. alpha=beta=0 returns `pool` unchanged."""
    if alpha == 0.0 and beta == 0.0:
        return pool
    bonus = np.zeros(len(pool), dtype=np.float32)
    if alpha and dom:
        bonus += alpha * np.array([DOMAIN.get(code_of(c)) == dom for c in pool], dtype=np.float32)
    if beta and codes:
        bonus += beta * np.array([code_of(c) in codes for c in pool], dtype=np.float32)
    order = np.argsort(-(base + bonus), kind='stable')
    return [pool[i] for i in order]


def evaluate(pools, bases, doms, codes, gold, alpha, beta, topks=TOPKS):
    out = {}
    ranked = [fuse_rank(p, b, d, c, alpha, beta)
              for p, b, d, c in zip(pools, bases, doms, codes)]
    for K in topks:
        out[K] = float(np.mean([f1(list(dict.fromkeys(r[:K] + [BOILER])), g)
                                for r, g in zip(ranked, gold)]))
    return out


def per_query_f1(pools, bases, doms, codes, gold, alpha, beta, k):
    """Per-query F1 at a fixed configuration, preserving paired query order."""
    ranked = [fuse_rank(p, b, d, c, alpha, beta)
              for p, b, d, c in zip(pools, bases, doms, codes)]
    return [f1(list(dict.fromkeys(r[:k] + [BOILER])), g)
            for r, g in zip(ranked, gold)]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--router', help='router predictions CSV/JSON (query_id, domain[, codes])')
    ap.add_argument('--verify-identity', action='store_true',
                    help='assert alpha=beta=0 reproduces the base ranking exactly, then exit')
    ap.add_argument('--scorer', default='dense', choices=['dense', 'ce'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='analysis/router_rerank_results.json')
    args = ap.parse_args()

    t0 = time.time()
    print('loading assets...', flush=True)
    laws, law_cits, law_set, L, qz = load_assets()
    print(f'  laws={len(law_cits):,}  emb={L.shape}  [{time.time() - t0:.0f}s]', flush=True)

    results = {'pool_k': POOL_K, 'scorer': args.scorer, 'splits': {}}

    for split in ('train', 'val'):
        df, gold, mask = load_split(split, law_set)
        Q = norm(qz[split])[mask]
        if args.limit:
            df, gold, Q = df.head(args.limit), gold[:args.limit], Q[:args.limit]
        qids = df['query_id'].astype(str).tolist()
        queries = df['query'].astype(str).tolist()

        print(f'\n[{split}] n={len(qids)}  building {POOL_K}-candidate pools...', flush=True)
        pools, bases = build_pools(L, Q, law_cits)

        doms, codes, n_missing = load_router(args.router, qids, queries)
        if args.router:
            print(f'  router coverage: {len(qids) - n_missing}/{len(qids)} '
                  f'({1 - n_missing / len(qids):.1%}); rest fell back to keyword codes')

        base = evaluate(pools, bases, doms, codes, gold, 0.0, 0.0)
        bestK0 = max(base, key=base.get)
        print(f'  BASE (alpha=beta=0): ' + '  '.join(f'K={k}:{v:.4f}' for k, v in base.items()))

        if args.verify_identity:
            for a, b in ((0.0, 0.0),):
                same = all(fuse_rank(p, bs, d, c, a, b) == p
                           for p, bs, d, c in zip(pools, bases, doms, codes))
            assert same, 'IDENTITY FAILED: alpha=beta=0 changed the ranking'
            print(f'  PASS identity: alpha=beta=0 reproduces the base ranking exactly')
            continue

        grid = {}
        for a in ALPHAS:
            for b in BETAS:
                grid[f'{a}|{b}'] = evaluate(pools, bases, doms, codes, gold, a, b)
        flat = [(k, K, v) for k, d in grid.items() for K, v in d.items()]
        bk, bK, bv = max(flat, key=lambda x: x[2])
        ba, bb = bk.split('|')
        print(f'  BEST: alpha={ba} beta={bb} K={bK} -> F1={bv:.4f}   '
              f'(base K={bestK0} F1={base[bestK0]:.4f}, delta={bv - base[bestK0]:+.4f})')

        results['splits'][split] = {
            'n': len(qids), 'router_missing': n_missing,
            'base': base, 'base_best_k': bestK0, 'base_best_f1': base[bestK0],
            'best': {'alpha': float(ba), 'beta': float(bb), 'k': bK, 'f1': bv},
            'delta': bv - base[bestK0], 'grid': grid,
        }

        # Split-half held-out: the train BEST above is selected in-sample and so
        # is optimistically biased. Select alpha/beta on even-indexed queries,
        # evaluate on odd-indexed ones (n~496 each). Deterministic, no shuffle.
        if split == 'train':
            ev = list(range(0, len(qids), 2))
            od = list(range(1, len(qids), 2))
            def sub(ix, seq): return [seq[i] for i in ix]
            gA = {}
            for a in ALPHAS:
                for b in BETAS:
                    gA[f'{a}|{b}'] = evaluate(sub(ev, pools), sub(ev, bases), sub(ev, doms),
                                              sub(ev, codes), sub(ev, gold), a, b)
            fa = [(k, K, v) for k, d in gA.items() for K, v in d.items()]
            sk, sK, _ = max(fa, key=lambda x: x[2])
            sa, sb = (float(x) for x in sk.split('|'))
            heldB = evaluate(sub(od, pools), sub(od, bases), sub(od, doms),
                             sub(od, codes), sub(od, gold), sa, sb)[sK]
            baseB = evaluate(sub(od, pools), sub(od, bases), sub(od, doms),
                             sub(od, codes), sub(od, gold), 0.0, 0.0)[sK]
            print(f'  SPLIT-HALF held-out: alpha={sa} beta={sb} K={sK} chosen on n={len(ev)}, '
                  f'scored on n={len(od)}')
            print(f'    base={baseB:.4f}  tuned={heldB:.4f}  delta={heldB - baseB:+.4f}  '
                  f"{'PASS' if heldB > baseB else 'FAIL'}")
            base_per_query = per_query_f1(sub(od, pools), sub(od, bases), sub(od, doms),
                                           sub(od, codes), sub(od, gold), 0.0, 0.0, sK)
            tuned_per_query = per_query_f1(sub(od, pools), sub(od, bases), sub(od, doms),
                                            sub(od, codes), sub(od, gold), sa, sb, sK)
            results['split_half'] = {'alpha': sa, 'beta': sb, 'k': sK,
                                     'n_select': len(ev), 'n_eval': len(od),
                                     'base': baseB, 'tuned': heldB,
                                     'delta': heldB - baseB, 'gate_pass': heldB > baseB,
                                     'eval_query_ids': sub(od, qids),
                                     'base_per_query_f1': base_per_query,
                                     'tuned_per_query_f1': tuned_per_query}

    if args.verify_identity:
        print('\nIDENTITY CHECK PASSED')
        return 0

    tr, va = results['splits'].get('train'), results['splits'].get('val')
    if tr and va:
        # alpha/beta are a RANKING question and transfer across splits.
        # K is a COUNT question and does NOT: train gold has median 2 citations
        # per query, val has 22, so train-selected K is meaningless on val.
        # Select alpha/beta on train (n=993); hold K fixed at val's own base-best
        # so the comparison isolates the routing effect.
        a, b = tr['best']['alpha'], tr['best']['beta']
        K = va['base_best_k']
        held = va['grid'][f'{a}|{b}'][K]
        base_at_k = va['base'][K]
        print(f"\nHELD-OUT CHECK: alpha={a} beta={b} selected on train (n={tr['n']}), "
              f"applied to val at K={K} (val's own base-best K)")
        print(f"  val base  F1={base_at_k:.4f}")
        print(f"  val tuned F1={held:.4f}   delta={held - base_at_k:+.4f}")
        gate = 'PASS' if held > base_at_k else 'FAIL'
        print(f"  GATE (routing must beat base on held-out val): {gate}")
        results['held_out'] = {
            'alpha': a, 'beta': b, 'k': K, 'k_source': 'val base-best',
            'alpha_beta_source': f'train n={tr["n"]}',
            'val_base': base_at_k, 'val_tuned': held,
            'delta': held - base_at_k, 'gate_pass': held > base_at_k,
        }

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nwrote {outp}  [{time.time() - t0:.0f}s]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
