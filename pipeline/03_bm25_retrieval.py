# -*- coding: utf-8 -*-
"""
CELL 3 — BM25 retrieval over laws + court_considerations.

For each test query (German):
  - BM25 top-200 over laws_de
  - BM25 top-300 over court_considerations
  - Plus explicit-mention extraction (queries that mention "Art. N CODE" boost that cit)

Output:
  bm25_laws_per_q: {qid: [(cit, score), ...]}
  bm25_court_per_q: {qid: [(cit, score), ...]}
"""
import time
from rank_bm25 import BM25Okapi

TOK_PAT = re.compile(r'[\wäöüÄÖÜß.]+', re.UNICODE)
def tok_bm25(s):
    """Tokenizer preserving legal notation (Art., Abs., numbers)."""
    return [t.lower() for t in TOK_PAT.findall(str(s))]

# --- Build BM25 index over laws_de.csv ---
t0 = time.time()
laws_text = (laws['citation'].astype(str) + ' ' +
             laws['text'].astype(str) + ' ' +
             laws['title'].astype(str)).tolist()
laws_tok = [tok_bm25(t) for t in laws_text]
laws_cits = laws['citation'].astype(str).tolist()
print(f'laws tokenized in {time.time()-t0:.0f}s')

t0 = time.time()
bm25_laws = BM25Okapi(laws_tok)
print(f'BM25 laws built in {time.time()-t0:.0f}s')

BM25_LAWS_K = 200

# --- Run BM25 for each test query ---
bm25_laws_per_q = {}
for _, row in test.iterrows():
    qid = row['query_id']
    qde = str(row['query_de'])
    scores = bm25_laws.get_scores(tok_bm25(qde))
    top_idx = np.argsort(-scores)[:BM25_LAWS_K]
    bm25_laws_per_q[qid] = [(laws_cits[i], float(scores[i])) for i in top_idx]
print(f'BM25 laws scored {len(test)} test queries in {time.time()-t0:.0f}s')

# --- Build BM25 over court_considerations (streamed in chunks due to size) ---
# court_considerations has 2.47M rows. Loading all into BM25 requires ~16GB RAM.
# Compromise: aggregate by citation (one BM25 doc per court ref, concatenating all rows).
print('\nBuilding court BM25 index (aggregated by citation)...')
t0 = time.time()
court_by_cit = defaultdict(list)
for chunk in pd.read_csv(DATA / 'court_considerations.csv', chunksize=200_000):
    for _, r in chunk.iterrows():
        court_by_cit[str(r['citation'])].append(str(r['text']))
court_cits = list(court_by_cit.keys())
court_text = [' '.join(court_by_cit[c]) for c in court_cits]
print(f'  aggregated {len(court_cits):,} court refs in {time.time()-t0:.0f}s')

# Tokenize + BM25
t0 = time.time()
court_tok = [tok_bm25(t) for t in court_text]
print(f'  court tokenized in {time.time()-t0:.0f}s')
t0 = time.time()
bm25_court = BM25Okapi(court_tok)
print(f'  court BM25 built in {time.time()-t0:.0f}s')

# Free intermediate
del court_text, court_tok
gc.collect()

BM25_COURT_K = 300

bm25_court_per_q = {}
t0 = time.time()
for _, row in test.iterrows():
    qid = row['query_id']
    qde = str(row['query_de'])
    scores = bm25_court.get_scores(tok_bm25(qde))
    top_idx = np.argsort(-scores)[:BM25_COURT_K]
    bm25_court_per_q[qid] = [(court_cits[i], float(scores[i])) for i in top_idx]
print(f'BM25 court scored {len(test)} test queries in {time.time()-t0:.0f}s')

# --- Explicit-mention extraction ---
# For each query, find "Art. N CODE" patterns and ensure those candidates are in pool
MENTION_PAT = re.compile(r'Art\.\s*(\d+[a-z]?)(?:\s+Abs\.\s*(\d+[a-z]*))?(?:\s+lit\.\s*[a-z]+)?\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü]+\.?)', re.UNICODE)

def extract_mentions(qtext):
    """Extract (art, abs, code) cit forms from query text."""
    out = []
    for m in MENTION_PAT.finditer(qtext):
        art = m.group(1)
        abs_ = m.group(2)
        code = m.group(3)
        if abs_:
            out.append(f'Art. {art} Abs. {abs_} {code}')
        out.append(f'Art. {art} {code}')
    return out

laws_set = set(laws_cits)
mention_per_q = {}
for _, row in test.iterrows():
    qid = row['query_id']
    qen = str(row['query'])
    qde = str(row['query_de'])
    mentions = set(extract_mentions(qen) + extract_mentions(qde))
    # Filter to corpus-existing
    valid = [c for c in mentions if c in laws_set]
    mention_per_q[qid] = valid
n_mentions = sum(len(v) for v in mention_per_q.values())
print(f'Mention extraction: {n_mentions} corpus-valid mentions across {len(test)} queries')
