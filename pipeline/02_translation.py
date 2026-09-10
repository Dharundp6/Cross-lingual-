# -*- coding: utf-8 -*-
"""
CELL 2 — Translation: English query → German.

Uses Helsinki-NLP/opus-mt-en-de loaded from Kaggle dataset.
Queries are translated once and cached.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {DEVICE}')

if IS_KAGGLE:
    TRANS_DIR = Path('/kaggle/input/opus-mt-en-de')
else:
    TRANS_DIR = Path('models/opus-mt-en-de')

print(f'Loading translation model from {TRANS_DIR}...')
t0 = time.time()
tok_t = AutoTokenizer.from_pretrained(str(TRANS_DIR))
mdl_t = AutoModelForSeq2SeqLM.from_pretrained(str(TRANS_DIR)).to(DEVICE).eval()
print(f'  loaded in {time.time()-t0:.0f}s')

def translate_batch(texts, batch_size=4, max_length=512):
    """Translate English texts to German in batches."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Truncate long inputs (queries can be 1500+ chars)
        enc = tok_t(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(DEVICE)
        with torch.no_grad():
            gen = mdl_t.generate(**enc, max_length=max_length, num_beams=2)
        out.extend(tok_t.batch_decode(gen, skip_special_tokens=True))
    return out

# Translate test queries
t0 = time.time()
test_de = translate_batch(test['query'].tolist(), batch_size=2)
test['query_de'] = test_de
print(f'Translated {len(test_de)} test queries in {time.time()-t0:.0f}s')

# Translate val queries (for FT and for boilerplate-ADD calibration)
val_de = translate_batch(val['query'].tolist(), batch_size=2)
val['query_de'] = val_de
print(f'Translated {len(val_de)} val queries in {time.time()-t0:.0f}s')

# Free translation model VRAM
del mdl_t, tok_t
gc.collect()
torch.cuda.empty_cache()
print('Translation model unloaded.')
