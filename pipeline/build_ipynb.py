# -*- coding: utf-8 -*-
"""
Build a single Jupyter notebook from cells 01-07.
Output: offline_reproducible.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

CELLS = [
    ('md', '# LLM Agentic Legal Information Retrieval — Offline Reproducible Notebook\n\n'
           '**Pipeline:** BM25 + Dense (multilingual-e5-large) + Cross-Encoder rerank '
           '(mmarco-mMiniLMv2) + drop layer (operative-vs-dispositif regex) + '
           'boilerplate-ADD (val-grounded).\n\n'
           '**No external API calls.** All models loaded from Kaggle datasets. '
           'Reproducible offline within 12-hour Kaggle notebook limit.\n\n'
           '**Expected output:** `/kaggle/working/submission.csv`'),
    ('py', '01_setup_and_data.py'),
    ('md', '## Translation: English → German\n\nQueries arrive in English. The corpus '
           '(laws_de.csv and court_considerations.csv) is German+French. We translate '
           'queries to German for better lexical match with the corpus.'),
    ('py', '02_translation.py'),
    ('md', '## Retrieval Stage 1: BM25 over laws + court\n\nBM25 captures explicit-vocabulary '
           'matches. Also runs explicit-mention extraction: queries that mention '
           '"Art. N CODE" boost that cit into the candidate pool.'),
    ('py', '03_bm25_retrieval.py'),
    ('md', '## Retrieval Stage 2: Dense (multilingual-e5-large)\n\nDense semantic retrieval '
           'over laws_de. Captures cross-lingual semantic matches that BM25 misses. Court '
           'corpus too large for dense (2.47M docs) — sticks with BM25 there.'),
    ('py', '04_dense_retrieval.py'),
    ('md', '## Fusion + Cross-Encoder Reranking\n\nReciprocal Rank Fusion (RRF) merges '
           'BM25 laws + BM25 court + dense laws + explicit mentions. Top-150 candidates '
           'are reranked by mmarco-mMiniLMv2 cross-encoder.'),
    ('py', '05_fusion_and_rerank.py'),
    ('md', '## Drop Layer (operative-vs-dispositif content classification)\n\nDrops court refs '
           'whose paragraph text is non-doctrinal: issue-framing transitions, party-argument '
           'summaries, case-specific procedural admissibility, pure fact narratives. \n\n'
           'All rules grounded in val analysis. Pure regex over court_considerations.csv text.'),
    ('py', '06_drop_layer.py'),
    ('md', '## Boilerplate-ADD Layer (val-grounded breakthrough)\n\n**The mechanism:** val gold '
           'shows LEXam annotators systematically add procedural-boilerplate articles based on '
           'query case type. Train has 4.1 gold cits/q; val has 25.1 cits/q. 9/10 val queries '
           'cite Art. 100 Abs. 1 BGG (Supreme Court 30-day appeal deadline).\n\n'
           '**Confirmed on public LB:** +0.024 lift from this layer alone (v313/v314).'),
    ('py', '07_boilerplate_add.py'),
    ('md', '## Done\n\nSubmission written to `/kaggle/working/submission.csv`. Reproducibility:\n\n'
           '- No external API calls ✓\n'
           '- All models from Kaggle datasets ✓\n'
           '- 12hr runtime budget: ~15-20 min actual ✓\n'
           '- Deterministic given fixed weights ✓'),
]

cells = []
for ct, content in CELLS:
    if ct == 'md':
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': content.split('\n'),
        })
    else:
        path = ROOT / content
        src = path.read_text(encoding='utf-8')
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': src.split('\n'),
        })
        # Add trailing newline-as-string per Jupyter format
        cells[-1]['source'] = [(l + '\n' if i < len(cells[-1]['source']) - 1 else l)
                                for i, l in enumerate(cells[-1]['source'])]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.10',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

out = ROOT / 'offline_reproducible.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Wrote {out}')
print(f'  cells: {len(cells)} ({sum(1 for c in cells if c["cell_type"]=="code")} code, '
      f'{sum(1 for c in cells if c["cell_type"]=="markdown")} markdown)')
