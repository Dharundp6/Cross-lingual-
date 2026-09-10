# -*- coding: utf-8 -*-
"""Encode the English training questions with the shipped fine-tuned encoder.

The archived query matrix `queries_v3.npz` holds embeddings of the German training text.
Measuring the delivered system on English input needs the English text encoded the same
way, so this reproduces the shipped recipe and then applies it to the translations.

The recipe is verified before it is used: the German text is re-encoded first and its
cosine similarity against the archived matrix is reported. A mean cosine at 1.0 means the
recipe matches the one that produced the archive, and only then are the English embeddings
written. If it does not match, the run stops rather than silently producing embeddings from
a different pipeline than the one the corpus side was built with.

Usage:  python analysis/encode_train_en.py
Run from the repository root.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / '_models' / 'kagglecomp' / 'e5-legal-finetuned'
ARCHIVE = ROOT / 'current' / '_hf_assets' / 'queries_v3.npz'
OUT = ROOT / 'analysis' / 'train_queries_en_emb.npy'

PREFIX = 'query: '
MAX_LEN = 256
BATCH = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode(tok, mdl, texts):
    """CLS pooling, L2 normalised - the recipe the archived matrix was built with."""
    out = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        b = [PREFIX + str(t) for t in texts[i:i + BATCH]]
        enc = tok(b, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        # via tolist(): torch and numpy are ABI-mismatched in this environment, so the
        # tensor->array bridge raises. The values are identical either way.
        out.append(np.asarray(h.cpu().tolist(), dtype='float32'))
        if (i // BATCH) % 10 == 0:
            print(f'  encoded {min(i + BATCH, len(texts))}/{len(texts)} '
                  f'[{time.time() - t0:.0f}s]', flush=True)
    return np.vstack(out)


def main():
    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    en = json.loads((ROOT / 'analysis' / 'train_queries_en.json').read_text(encoding='utf-8'))
    missing = [str(q) for q in train['query_id'] if not en.get(str(q))]
    print(f'{len(train)} training queries; {len(missing)} without an English translation')

    print(f'loading encoder on {DEVICE}...', flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    mdl = AutoModel.from_pretrained(str(MODEL)).to(DEVICE).eval()

    print('verifying the recipe against the archived German embeddings...', flush=True)
    probe = list(train['query'].astype(str))[:64]
    got = encode(tok, mdl, probe)
    want = np.load(ARCHIVE)['train'][:64].astype('float32')
    want = want / np.clip(np.linalg.norm(want, axis=1, keepdims=True), 1e-9, None)
    cos = float(np.mean(np.sum(got * want, axis=1)))
    print(f'  mean cosine against the archive: {cos:.6f}')
    if cos < 0.999:
        print('RECIPE MISMATCH - refusing to write English embeddings from a different '
              'pipeline than the corpus side was built with.')
        sys.exit(1)

    print('encoding the English text...', flush=True)
    texts = [en.get(str(q), str(t)) for q, t in zip(train['query_id'], train['query'])]
    emb = encode(tok, mdl, texts)
    np.save(OUT, emb)
    print(f'wrote {OUT}  shape {emb.shape}')


if __name__ == '__main__':
    main()
