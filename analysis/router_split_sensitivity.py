"""Post-hoc split sensitivity and alpha/beta ablation for the stored Claude router.

No model calls are made. The script reconstructs candidate pools from the pinned
embedding and evaluates only archived router predictions. Because the 100 random
splits were not pre-registered, they are reported as a robustness sensitivity check,
not an additional independent test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from router_rerank_experiment import (
    ALPHAS, BETAS, ROOT, build_pools, f1, fuse_rank, load_assets, load_router,
    load_split, norm,
)

OUT = ROOT / "analysis" / "router_split_sensitivity.json"
ROUTER = ROOT / "analysis" / "router_accuracy_claude.csv"
SOURCE = ROOT / "analysis" / "router_rerank_results.json"
K = 5
SEED = 20260831
N_SPLITS = 100
N_BOOTSTRAP = 10_000
BOILER = "Art. 100 Abs. 1 BGG"


def per_query_scores(pools, bases, domains, codes, gold, alpha, beta):
    return np.asarray([
        f1(list(dict.fromkeys(fuse_rank(p, b, d, c, alpha, beta)[:K] + [BOILER])), g)
        for p, b, d, c, g in zip(pools, bases, domains, codes, gold)
    ], dtype=float)


def ci(delta, rng):
    index = rng.integers(0, len(delta), size=(N_BOOTSTRAP, len(delta)))
    means = delta[index].mean(axis=1)
    return [float(np.quantile(means, q)) for q in (0.025, 0.975)]


def main():
    laws, citations, law_set, embedding, queries = load_assets()
    df, gold, mask = load_split("train", law_set)
    query_vectors = norm(queries["train"])[mask]
    qids = df["query_id"].astype(str).tolist()
    questions = df["query"].astype(str).tolist()
    pools, bases = build_pools(embedding, query_vectors, citations)
    domains, codes, missing = load_router(ROUTER, qids, questions)
    if missing:
        raise RuntimeError(f"Expected complete Claude coverage, found {missing} missing rows")

    scores = {(0.0, 0.0): per_query_scores(pools, bases, domains, codes, gold, 0.0, 0.0)}
    for alpha in ALPHAS:
        for beta in BETAS:
            if (alpha, beta) != (0.0, 0.0):
                scores[(alpha, beta)] = per_query_scores(
                    pools, bases, domains, codes, gold, alpha, beta)

    base = scores[(0.0, 0.0)]
    configs = [(a, b) for a in ALPHAS for b in BETAS if (a, b) != (0.0, 0.0)]
    rng = np.random.default_rng(SEED)
    deltas, selected = [], []
    n = len(qids)
    select_n = n // 2 + 1
    for _ in range(N_SPLITS):
        select_idx = rng.choice(n, size=select_n, replace=False)
        eval_mask = np.ones(n, dtype=bool)
        eval_mask[select_idx] = False
        eval_idx = np.flatnonzero(eval_mask)
        best = max(configs, key=lambda pair: scores[pair][select_idx].mean())
        deltas.append(float((scores[best][eval_idx] - base[eval_idx]).mean()))
        selected.append({"alpha": best[0], "beta": best[1]})

    original = json.loads(SOURCE.read_text(encoding="utf-8"))["split_half"]
    original_idx = np.asarray([qids.index(qid) for qid in original["eval_query_ids"]])
    ablations = {}
    for name, pair in {
        "base": (0.0, 0.0),
        "alpha_only": (0.05, 0.0),
        "beta_only": (0.0, 0.10),
        "alpha_and_beta": (0.05, 0.10),
    }.items():
        delta = scores[pair][original_idx] - base[original_idx]
        ablations[name] = {
            "alpha": pair[0], "beta": pair[1],
            "mean_f1": float(scores[pair][original_idx].mean()),
            "delta_vs_base": float(delta.mean()),
            "ci95": [0.0, 0.0] if name == "base" else ci(delta, np.random.default_rng(SEED)),
        }

    result = {
        "scope": "post-hoc random-split sensitivity; no additional model calls",
        "n_queries": n, "k": K, "n_splits": N_SPLITS, "seed": SEED,
        "selection_size": select_n, "evaluation_size": n - select_n,
        "split_deltas": deltas,
        "positive_split_fraction": float(np.mean(np.asarray(deltas) > 0)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "interval_95": [float(np.quantile(deltas, q)) for q in (0.025, 0.975)],
        "selected_configurations": selected,
        "original_split_ablation": ablations,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()