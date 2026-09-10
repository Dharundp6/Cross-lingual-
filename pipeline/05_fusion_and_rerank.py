# -*- coding: utf-8 -*-
"""
CELL 5 — Fusion + cross-encoder reranking.

For each test query:
  1. Reciprocal Rank Fusion (RRF) of BM25 laws + BM25 court + Dense laws + mentions
  2. Cross-encoder rerank of top-150 candidates
  3. Pick top-K (calibrated on val)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RRF_K_CONST = 60
RERANK_K = 150

def rrf_fuse(*rankings, k=RRF_K_CONST):
    """RRF fusion of multiple (cit, score) ranked lists.
    Returns dict {cit: rrf_score}.
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (cit, _) in enumerate(ranking):
            scores[cit] += 1.0 / (k + rank + 1)
    return scores

# --- Build laws_text lookup for reranker ---
laws_text_map = {laws['citation'].iloc[i]: laws_text[i] for i in range(len(laws))}
# Note: court_by_cit already maps citation → list of paragraph texts
def court_text_concat(cit):
    rows = court_by_cit.get(cit, [])
    return ' '.join(rows)[:1500]

def fuse_per_q(qid):
    rankings = [
        bm25_laws_per_q.get(qid, []),
        bm25_court_per_q.get(qid, []),
        dense_laws_per_q.get(qid, []),
        # Mentions get top-rank boost
        [(c, 1.0) for c in mention_per_q.get(qid, [])],
    ]
    return rrf_fuse(*rankings)

# --- Load cross-encoder ---
print(f'Loading cross-encoder from {RERANK_DIR}...')
t0 = time.time()
tok_r = AutoTokenizer.from_pretrained(str(RERANK_DIR))
mdl_r = AutoModelForSequenceClassification.from_pretrained(str(RERANK_DIR)).to(DEVICE).eval()
print(f'  loaded in {time.time()-t0:.0f}s')

def rerank(query, candidates, top_n=RERANK_K, batch_size=32, max_length=384):
    """Rerank (cit, text) pairs using cross-encoder."""
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        enc = tok_r([q for q, _ in batch], [t for _, t in batch],
                    return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(DEVICE)
        with torch.no_grad():
            out = mdl_r(**enc)
        s = out.logits.squeeze(-1).cpu().numpy()
        scores.extend(s.tolist())
    scored = [(candidates[j][0], scores[j]) for j in range(len(candidates))]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]

# --- Per-query: fuse, get text for top-RERANK_K, rerank ---
court_pat = re.compile(r'^(BGE|\d+[A-Z]_)')

def get_cit_text(cit):
    if court_pat.match(cit):
        return court_text_concat(cit)
    return laws_text_map.get(cit, '')

ranked_per_q = {}
print(f'\nFusing + reranking {len(test)} test queries...')
t0 = time.time()
for _, row in test.iterrows():
    qid = row['query_id']
    qde = str(row['query_de'])
    fused = fuse_per_q(qid)
    # Top RERANK_K candidates by fusion score
    top_cands = sorted(fused.items(), key=lambda x: -x[1])[:RERANK_K]
    cand_with_text = [(c, get_cit_text(c)) for c, _ in top_cands]
    reranked = rerank(qde, cand_with_text, top_n=RERANK_K)
    ranked_per_q[qid] = reranked
print(f'  reranked in {time.time()-t0:.0f}s')

# Free reranker
del mdl_r, tok_r
gc.collect()
torch.cuda.empty_cache()
