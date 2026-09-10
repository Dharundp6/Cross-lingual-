#!/usr/bin/env python3
"""Paired bootstrap intervals for the fixed split-half router comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEED = 20260824
N_BOOT = 10_000
FILES = {
    "claude": ROOT / "router_rerank_results.json",
    "keyword": ROOT / "router_rerank_keyword.json",
    "deepseek": ROOT / "router_rerank_deepseek.json",
}


def load(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    split = json.loads(FILES[name].read_text(encoding="utf-8"))["split_half"]
    return (np.asarray(split["base_per_query_f1"], dtype=float),
            np.asarray(split["tuned_per_query_f1"], dtype=float),
            split["eval_query_ids"])


def interval(delta: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    draws = delta[rng.integers(0, len(delta), size=(N_BOOT, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(delta.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def main() -> None:
    base_c, claude, ids_c = load("claude")
    base_k, keyword, ids_k = load("keyword")
    base_d, deepseek, ids_d = load("deepseek")
    if (ids_c != ids_k or ids_c != ids_d
            or not np.array_equal(base_c, base_k)
            or not np.array_equal(base_c, base_d)):
        raise ValueError("Conditions are not paired on the same held-out queries and baseline.")
    rng = np.random.default_rng(SEED)
    report = {
        "n_eval": int(len(base_c)), "seed": SEED, "bootstrap_replicates": N_BOOT,
        "base_mean_f1": float(base_c.mean()), "keyword_mean_f1": float(keyword.mean()),
        "deepseek_mean_f1": float(deepseek.mean()), "claude_mean_f1": float(claude.mean()),
        "claude_minus_base": interval(claude - base_c, rng),
        "keyword_minus_base": interval(keyword - base_c, rng),
        "claude_minus_keyword": interval(claude - keyword, rng),
        "deepseek_minus_base": interval(deepseek - base_c, rng),
    }
    output = ROOT / "router_rerank_paired_bootstrap.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()