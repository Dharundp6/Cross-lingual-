# Cross-Lingual Legal Citation Retrieval for Swiss Law

Code and recorded results for the MSc dissertation *Cross-Lingual Legal Citation Retrieval for Swiss Law* (University of Glasgow, 2026). Given an English legal question, the system returns the Swiss statute paragraphs it should cite, from a German-language corpus.

## What is here

| Folder | Contents |
|---|---|
| `pipeline/` | The reproducible offline system: explicit-citation extraction, rule bundles, anchored co-citation expansion, and the dense-retrieval stages. `offline_reproducible.ipynb` runs end to end on Kaggle without internet. |
| `analysis/` | The experiments reported in the dissertation: the legal-area router and its ranking prior (`llm_router.py`, `router_rerank_experiment.py`), the paired 497/496 comparison and its re-split sensitivity, the end-to-end comparison (`end_to_end_dense.py`), the admission-bar audit, the graph ablation and the statute-only ceiling. |
| `analysis/results/` | Recorded outputs behind every number in the results chapter, including per-query answer sets, so each comparison can be regenerated rather than re-run. |
| `notebooks/` | The Vast.ai / Colab notebooks used for fine-tuning and the full experiment sweep. Outputs are cleared; paths and keys are placeholders. |
| `figures/` | Scripts that draw the results-chapter figures from the recorded outputs. |

## Data and models

The competition data (`laws_de.csv`, court decisions, `train.csv`) is not redistributed here. Model checkpoints, corpus embeddings and precomputed indices are on Hugging Face:

- `Dharun72/llm-agentic-precomputed-v3` — statute embeddings, BM25 indices, query embeddings
- `Dharun72/swiss-legal-checkpoints` — fine-tuned encoder and cross-encoder
- `Dharun72/llm-agentic-swiss-legal-checkpoints` — adapters, co-citation index, translated training split

Scripts expect the data under `Data/` and the checkpoints under `_models/`; each script's docstring says what it reads and writes.

## Reproducing the headline comparisons

```bash
pip install -r requirements.txt
python analysis/repro_router_controls.py       # regenerates the router comparison from archived predictions
python analysis/end_to_end_dense.py --lang de   # end-to-end effect of the domain prior
python analysis/admission_bar_measured.py       # admission-bar audit
python analysis/graph_ablation.py               # leave-one-out graph ablation
```

Language-model calls (`llm_router.py`, `deepseek_router.py`, `translate_train_split.py`) need an API key in the environment and are not prospectively reproducible, since a provider may update the model behind a stable identifier; the archived predictions in `analysis/results/` are the authoritative record.
