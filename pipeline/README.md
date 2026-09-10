# Offline pipeline

The deterministic system the dissertation reports, as one Kaggle notebook and the stage scripts it is built from.

| File | Role |
|---|---|
| `offline_reproducible.ipynb` | Runs end to end with no internet: encodes each query at runtime, retrieves from the statute corpus, applies the rule bundles and the anchored co-citation expansion, writes `submission.csv`. |
| `submission_offline.ipynb` | Rule bundles only, no graph expansion. |
| `submission_offline_kg.ipynb` | Rule bundles plus the anchored expansion. |
| `submission_offline_llm.ipynb` | Adds an optional local instruct model for doctrinal naming; falls back to the rules when absent. |
| `01_…07_*.py` | The stages in order: data, translation, BM25, dense retrieval, fusion and reranking, drop rules, fixed additions. `build_ipynb.py` assembles them into the notebook. |
| `vast_offline_reproducible.ipynb`, `build_vast_offline.py` | The same pipeline packaged for a Vast.ai GPU instance. |

Query encoding uses CLS pooling with a `query: ` prefix at `max_length=256`, the recipe the corpus embeddings were built with; `analysis/encode_train_en.py` checks that recipe against the archived query matrix before writing anything.
