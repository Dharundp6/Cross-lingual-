# -*- coding: utf-8 -*-
"""
CELL 4 — Dense retrieval over laws_de using multilingual-e5-large.

For each test query: encode query → top-200 laws by cosine similarity.
Court refs are too many (2.47M) for dense — stick with BM25 for those.
"""
import torch
from transformers import AutoTokenizer, AutoModel

print(f'Loading e5 model from {E5_DIR}...')
t0 = time.time()
tok_e = AutoTokenizer.from_pretrained(str(E5_DIR))
mdl_e = AutoModel.from_pretrained(str(E5_DIR)).to(DEVICE).eval()
print(f'  loaded in {time.time()-t0:.0f}s, params {sum(p.numel() for p in mdl_e.parameters())/1e6:.0f}M')

def mean_pool(h, mask):
    mask = mask.unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def encode(texts, prefix='passage: ', batch_size=8, max_length=512):
    """Encode texts to L2-normalized embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + str(t) for t in texts[i:i+batch_size]]
        enc = tok_e(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(DEVICE)
        with torch.no_grad():
            out = mdl_e(**enc)
        embs = mean_pool(out.last_hidden_state, enc['attention_mask'])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs).astype('float32')

# Encode laws_de
print('\nEncoding laws_de...')
t0 = time.time()
laws_embs = encode(laws_text, prefix='passage: ', batch_size=16)
print(f'  laws encoded: {laws_embs.shape} in {time.time()-t0:.0f}s')

# Encode test queries (German)
print('\nEncoding test queries...')
test_q_embs = encode(test['query_de'].tolist(), prefix='query: ', batch_size=4)

# Encode val queries (for later use)
val_q_embs = encode(val['query_de'].tolist(), prefix='query: ', batch_size=4)

# Free e5 model
del mdl_e, tok_e
gc.collect()
torch.cuda.empty_cache()

# Cosine sim search (dot product since both are L2-normalized)
DENSE_LAWS_K = 200

dense_laws_per_q = {}
for i, (_, row) in enumerate(test.iterrows()):
    qid = row['query_id']
    sims = laws_embs @ test_q_embs[i]
    top_idx = np.argsort(-sims)[:DENSE_LAWS_K]
    dense_laws_per_q[qid] = [(laws_cits[j], float(sims[j])) for j in top_idx]
print(f'\nDense retrieval done for {len(test)} test queries')

# Free laws_embs (large)
del laws_embs
gc.collect()
