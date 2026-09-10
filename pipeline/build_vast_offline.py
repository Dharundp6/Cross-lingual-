# -*- coding: utf-8 -*-
"""Wrap the PURE offline_reproducible.ipynb with a Vast setup preamble so we can CONFIRM
its true from-scratch score on a Vast GPU (val F1, no submission needed). The pure pipeline
cells are copied VERBATIM — only a deps/creds/download preamble is prepended. On Vast
(no /kaggle dir) the notebook's own 'local' branch uses models/ + Data/, which the preamble
populates. NOTHING from proven CSVs / curation JSON / external APIs is added — stays pure.

Run: python build_vast_offline.py  ->  vast_offline_reproducible.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC  = HERE / 'offline_reproducible.ipynb'
OUT  = HERE / 'vast_offline_reproducible.ipynb'

pre = []
def md(s):  pre.append(("markdown", s))
def code(s): pre.append(("code", s))

md(r"""# Vast confirmation run — PURE offline pipeline (no API, no CSV, no precomputed embeddings)

Wraps `offline_reproducible.ipynb` (verified pure: BM25 + dense e5 + CE rerank + content DROP +
val-grounded boilerplate ADD) with a Vast setup preamble. Encodes everything from scratch with
stock models — this is the genuinely-reproducible floor. Reports **val macro-F1** so we confirm
the score WITHOUT a Kaggle submission.

Honest expectation: this is the from-scratch pipeline (without the service-curated configuration). Whatever it scores here IS your pure-offline ceiling.""")

code(r"""# Preamble 1 — deps + LIVE (unbuffered) output
import sys, subprocess, functools
try: sys.stdout.reconfigure(line_buffering=True)   # flush on every newline (live progress)
except Exception: pass
print = functools.partial(print, flush=True)        # persists across cells (shared notebook namespace)
subprocess.check_call([sys.executable,'-m','pip','install','-q','torch','transformers',
    'sentencepiece','sacremoses','rank_bm25','pandas','numpy','scikit-learn','huggingface_hub','kaggle'])
print('deps ok — output is now live (line-buffered)')""")

code(r'''# Preamble 2 — Kaggle creds (paste key; DO NOT commit notebook with the key filled in)
import os, json
from pathlib import Path
KAGGLE_USERNAME = "dharunps"
KAGGLE_KEY      = "PASTE_YOUR_KAGGLE_KEY_HERE"   # <- paste once on the Vast box
assert KAGGLE_KEY and KAGGLE_KEY != "PASTE_YOUR_KAGGLE_KEY_HERE", "paste your Kaggle key"
os.environ["KAGGLE_USERNAME"]=KAGGLE_USERNAME; os.environ["KAGGLE_KEY"]=KAGGLE_KEY
kd=Path.home()/".kaggle"; kd.mkdir(exist_ok=True)
(kd/"kaggle.json").write_text(json.dumps({"username":KAGGLE_USERNAME,"key":KAGGLE_KEY}))
try:(kd/"kaggle.json").chmod(0o600)
except Exception:pass
print("kaggle creds set:",KAGGLE_USERNAME)''')

code(r"""# Preamble 3 — download stock models + competition data to the paths the pure notebook expects
from huggingface_hub import snapshot_download
from pathlib import Path
import glob, zipfile
MODELS={'models/multilingual-e5-large':'intfloat/multilingual-e5-large',
        'models/mmarco-mMiniLMv2-L12-H384-v1':'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        'models/opus-mt-en-de':'Helsinki-NLP/opus-mt-en-de'}
for d,repo in MODELS.items():
    if not (Path(d)/'config.json').exists():
        print('downloading',repo,'->',d); snapshot_download(repo_id=repo, local_dir=d)
    else: print('present:',d)
Path('Data').mkdir(exist_ok=True)
if not (Path('Data')/'laws_de.csv').exists():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi(); api.authenticate()
    print('downloading competition data...')
    api.competition_download_files('llm-agentic-legal-information-retrieval', path='Data', quiet=False)
    for z in glob.glob('Data/*.zip'): zipfile.ZipFile(z).extractall('Data')
print('models + data ready; the pure pipeline cells below now run on the local branch (IS_KAGGLE=False)')""")

# ---- assemble: preamble + verbatim pure cells ----
src_nb = json.load(open(SRC, encoding='utf-8'))
cells=[]
for ctype, s in pre:
    c={"cell_type":ctype,"metadata":{},"source":s.splitlines(keepends=True)}
    if ctype=="code": c["outputs"]=[]; c["execution_count"]=None
    cells.append(c)
cells += src_nb['cells']   # verbatim pure cells

nb={"cells":cells,"metadata":src_nb.get('metadata',{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}}),
    "nbformat":4,"nbformat_minor":5}
json.dump(nb, open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'wrote {OUT}  ({len(cells)} cells = {len(pre)} preamble + {len(src_nb["cells"])} pure)')
