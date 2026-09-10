# -*- coding: utf-8 -*-
"""Full IR + classification metric suite for the dissertation evaluation chapter.

Assets (memory-verified, from Dharun72/llm-agentic-precomputed-v3 ONLY):
  - assets/hf/law_embs_legal_v3.npy        FT e5-legal statute embeddings, row-aligned to laws_de.csv
  - current/_hf_assets/queries_v3.npz       canonical query embeddings (val/test/train)
  - current/_hf_assets/court_embs_e5legal_full.npy + court_ids_full.pkl   FT court embeddings
AVOID: Dharun72/KaggleComp (reranker/embeddings) per user instruction.

Computes, on the labelled splits (train n=1139, val n=10):
  Recall@K, MRR, mAP@K, and Macro-F1 / Micro-F1 for the dense STATUTE retriever,
  plus the dense COURT retriever on val (the known cross-lingual court wall).
  Stratified on train by legal domain and by query complexity (word count).
Outputs analysis/metrics_suite.json and prints a summary.
"""
import re, json, pickle, sys, io
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def cits(s): return [x.strip() for x in str(s).split(';') if x.strip()]
DOCKET = re.compile(r'^\d+[A-Z]+_\d+/\d+')
def is_court(c): return c.startswith('BGE') or bool(DOCKET.match(c))
def norm(a):
    a = np.asarray(a).astype(np.float32); n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.clip(n, 1e-9, None)

# code (last token) -> legal domain
DOMAIN = {
 'StGB':'Criminal','StPO':'Criminal','StBOG':'Criminal','JStG':'Criminal','VStrR':'Criminal',
 'ZGB':'Civil & commercial','OR':'Civil & commercial','ZPO':'Civil & commercial','GBV':'Civil & commercial',
 'IPRG':'Civil & commercial','HRegV':'Civil & commercial','PartG':'Civil & commercial',
 'BV':'Public & administrative','VwVG':'Public & administrative','RPG':'Public & administrative',
 'USG':'Public & administrative','NHG':'Public & administrative','AIG':'Public & administrative',
 'DSG':'Public & administrative','URG':'Public & administrative','BGO':'Public & administrative',
 'RTVG':'Public & administrative','BGF':'Public & administrative',
 'DBG':'Tax, social & financial','MWSTG':'Tax, social & financial','StHG':'Tax, social & financial',
 'ATSG':'Tax, social & financial','UVG':'Tax, social & financial','AHVG':'Tax, social & financial',
 'IVG':'Tax, social & financial','BVG':'Tax, social & financial','FINMAG':'Tax, social & financial',
 'FIDLEG':'Tax, social & financial','KAG':'Tax, social & financial','VAG':'Tax, social & financial',
 'BGG':'Procedure & enforcement','SchKG':'Procedure & enforcement','BGFA':'Procedure & enforcement',
 'SVG':'Public & administrative',
}
def code_of(c):
    toks = c.split(); return toks[-1] if toks else ''
def domain_of_gold(goldset):
    from collections import Counter
    doms = Counter(DOMAIN.get(code_of(c), 'Other') for c in goldset if not is_court(c))
    return doms.most_common(1)[0][0] if doms else 'Other'

# ---------- load statute assets ----------
laws = pd.read_csv('Data/laws_de.csv').fillna('')
law_cits = laws['citation'].astype(str).tolist()
law_set = set(law_cits)
L = norm(np.load('assets/hf/law_embs_legal_v3.npy'))          # (175933,1024)
qz = np.load('current/_hf_assets/queries_v3.npz')

def load_split(name):
    df = pd.read_csv(f'Data/{name}.csv')
    gold = [set(cits(r['gold_citations'])) for _, r in df.iterrows()]
    ql = [len(str(r['query']).split()) for _, r in df.iterrows()]
    return df, gold, ql

RECALL_KS = [5,10,20,50,100,200,500,1000,5000]
MAP_KS = [3,5,10]
F1_KS = [5,10,18]

def statute_metrics(qv, gold_sets, qlens):
    """Return per-query records with gold statute ranks + aggregate metrics."""
    Q = norm(qv)
    recs = []
    for i in range(len(gold_sets)):
        g = {c for c in gold_sets[i] if not is_court(c) and c in law_set}
        if not g:
            continue
        sims = L @ Q[i]                       # (175933,)
        order = np.argsort(-sims)             # full ranking
        pos = {law_cits[j]: r for r, j in enumerate(order) if law_cits[j] in g}
        ranks = sorted(pos[c] for c in g)     # 0-based ranks of gold
        topcits = [law_cits[j] for j in order[:max(F1_KS + [max(RECALL_KS)])]]
        recs.append({'gold': g, 'ranks': ranks, 'top': topcits,
                     'qlen': qlens[i], 'domain': domain_of_gold(gold_sets[i])})
    return recs

def aggregate(recs):
    out = {}
    # Recall@K (macro over queries)
    out['recall_at_k'] = {}
    for K in RECALL_KS:
        rs = [sum(r < K for r in rec['ranks']) / len(rec['gold']) for rec in recs]
        out['recall_at_k'][K] = float(np.mean(rs))
    # MRR (first relevant)
    out['mrr'] = float(np.mean([1.0 / (rec['ranks'][0] + 1) for rec in recs]))
    # mAP@K
    out['map_at_k'] = {}
    for K in MAP_KS:
        aps = []
        for rec in recs:
            hit = 0; s = 0.0
            for k in range(K):
                if k in [r for r in rec['ranks']]:
                    hit += 1; s += hit / (k + 1)
            aps.append(s / min(len(rec['gold']), K))
        out['map_at_k'][K] = float(np.mean(aps))
    # Macro & Micro F1 at cutoff K
    out['f1'] = {}
    for K in F1_KS:
        f1s = []; TP = FP = FN = 0
        for rec in recs:
            pred = set(rec['top'][:K]); g = rec['gold']
            tp = len(pred & g); fp = len(pred - g); fn = len(g - pred)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
            TP += tp; FP += fp; FN += fn
        micro_p = TP / (TP + FP) if TP + FP else 0.0
        micro_r = TP / (TP + FN) if TP + FN else 0.0
        out['f1'][K] = {'macro': float(np.mean(f1s)),
                        'micro': float(2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0),
                        'micro_p': float(micro_p), 'micro_r': float(micro_r)}
    return out

results = {'assets': 'llm-agentic-precomputed-v3 (law_embs_legal_v3 + queries_v3 + court_embs_e5legal_full)',
           'note': 'dense retriever, statute leg; court leg on val only. KaggleComp assets avoided.'}

for split in ['val', 'train']:
    df, gold, ql = load_split(split)
    recs = statute_metrics(qz[split], gold, ql)
    results[f'statute_{split}'] = aggregate(recs)
    results[f'statute_{split}']['n_queries'] = len(recs)
    print(f'\n=== STATUTE dense retriever — {split} (n={len(recs)}) ===')
    print('  Recall@K :', {k: round(v, 3) for k, v in results[f'statute_{split}']['recall_at_k'].items()})
    print('  MRR      :', round(results[f'statute_{split}']['mrr'], 3))
    print('  mAP@K    :', {k: round(v, 3) for k, v in results[f'statute_{split}']['map_at_k'].items()})
    for K in F1_KS:
        f = results[f'statute_{split}']['f1'][K]
        print(f'  @K={K:>2}: Macro-F1={f["macro"]:.3f}  Micro-F1={f["micro"]:.3f}  (micro P={f["micro_p"]:.3f} R={f["micro_r"]:.3f})')
    # ---- stratification (train only; val too small) ----
    if split == 'train':
        # by domain
        strat = {}
        for key, kind in [('domain', lambda r: r['domain'])]:
            pass
        bydom = {}
        for rec in recs:
            bydom.setdefault(rec['domain'], []).append(rec)
        results['strat_domain'] = {}
        print('\n  --- by legal domain ---')
        for dom, rs in sorted(bydom.items(), key=lambda kv: -len(kv[1])):
            agg = aggregate(rs)
            results['strat_domain'][dom] = {'n': len(rs), 'recall_100': agg['recall_at_k'][100],
                                            'mrr': agg['mrr'], 'macro_f1_10': agg['f1'][10]['macro'],
                                            'micro_f1_10': agg['f1'][10]['micro']}
            print(f'    {dom:<26} n={len(rs):<4} R@100={agg["recall_at_k"][100]:.3f} MRR={agg["mrr"]:.3f} '
                  f'MacroF1@10={agg["f1"][10]["macro"]:.3f} MicroF1@10={agg["f1"][10]["micro"]:.3f}')
        # by complexity (word-count quartiles)
        lens = sorted(rec['qlen'] for rec in recs)
        qs = [lens[int(len(lens) * p)] for p in (0.25, 0.5, 0.75)]
        def bucket(n):
            if n <= qs[0]: return f'Q1 short (<={qs[0]}w)'
            if n <= qs[1]: return f'Q2 ({qs[0]+1}-{qs[1]}w)'
            if n <= qs[2]: return f'Q3 ({qs[1]+1}-{qs[2]}w)'
            return f'Q4 long (>{qs[2]}w)'
        bycx = {}
        for rec in recs:
            bycx.setdefault(bucket(rec['qlen']), []).append(rec)
        results['strat_complexity'] = {}
        print('\n  --- by query complexity (word count quartiles) ---')
        for b in sorted(bycx):
            rs = bycx[b]; agg = aggregate(rs)
            results['strat_complexity'][b] = {'n': len(rs), 'recall_100': agg['recall_at_k'][100],
                                              'mrr': agg['mrr'], 'macro_f1_10': agg['f1'][10]['macro'],
                                              'micro_f1_10': agg['f1'][10]['micro']}
            print(f'    {b:<20} n={len(rs):<4} R@100={agg["recall_at_k"][100]:.3f} MRR={agg["mrr"]:.3f} '
                  f'MacroF1@10={agg["f1"][10]["macro"]:.3f} MicroF1@10={agg["f1"][10]["micro"]:.3f}')

# ---------- court leg (val only) ----------
print('\n=== COURT dense retriever — val (the cross-lingual court wall) ===')
court_ids = pickle.load(open('current/_hf_assets/court_ids_full.pkl', 'rb'))
# map citation -> list of row indices (a ruling appears as multiple text chunks)
from collections import defaultdict
cid2rows = defaultdict(list)
for i, c in enumerate(court_ids):
    cid2rows[c].append(i)
dfv, goldv, qlv = load_split('val')
CE = np.load('current/_hf_assets/court_embs_e5legal_full.npy', mmap_mode='r')  # (2476315,1024) f16
Qv = norm(qz['val'])
court_ks = [100, 500, 1000, 2000, 5000]
recalls = {K: [] for K in court_ks}; mrrs = []
for i in range(len(goldv)):
    g = {c for c in goldv[i] if is_court(c) and c in cid2rows}
    if not g:
        continue
    sims = (np.asarray(CE, dtype=np.float32) @ Qv[i])          # (2.47M,)  ~5GB read
    order = np.argsort(-sims)
    rank_of_row = np.empty(len(order), dtype=np.int64); rank_of_row[order] = np.arange(len(order))
    # best (min) rank across a citation's chunks
    grank = {c: min(rank_of_row[r] for r in cid2rows[c]) for c in g}
    ranks = sorted(grank.values())
    for K in court_ks:
        recalls[K].append(sum(r < K for r in ranks) / len(g))
    mrrs.append(1.0 / (ranks[0] + 1))
results['court_val'] = {'recall_at_k': {K: float(np.mean(v)) for K, v in recalls.items()},
                        'mrr': float(np.mean(mrrs)), 'n_queries': len(mrrs)}
print('  Recall@K :', {k: round(v, 3) for k, v in results['court_val']['recall_at_k'].items()})
print('  MRR      :', round(results['court_val']['mrr'], 4))

json.dump(results, open('analysis/metrics_suite.json', 'w'), indent=2)
print('\nwrote analysis/metrics_suite.json')
