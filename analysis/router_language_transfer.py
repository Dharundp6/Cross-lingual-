"""Does the domain router survive the language boundary?

The router's recorded accuracy was measured on `Data/train.csv`, whose queries are
German. The deployed task supplies English fact patterns. That is a language
transfer the manuscript never tested, so the router's contribution to the English
task rested on an assumption.

Paired design, same queries in both arms:
  DE arm  archived predictions from `router_deepseek_predictions.csv` (unchanged)
  EN arm  each sampled query translated DE->EN, then classified with the SAME
          system prompt and the same client as the archived run

Because both arms score the same queries against the same gold domains, the
comparison isolates language: any gap is the cost of crossing it, not a
difference in question difficulty.

Reuses `deepseek_router.SYS`, `call_ds` and `parse` verbatim so the EN arm is
method-identical to the archived DE arm.

Usage:
    python analysis/router_language_transfer.py --n 150
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_router import SYS, call_ds, load_key, parse  # noqa: E402
from llm_router import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "router_language_transfer.json"
ARCHIVE = ROOT / "analysis" / "router_deepseek_predictions.csv"
ACCURACY = ROOT / "analysis" / "router_accuracy_deepseek.csv"
CONCURRENCY = 8

TRANSLATE_SYS = (
    "You translate Swiss legal questions from German to English. Preserve every "
    "legal citation exactly as written (for example 'Art. 221 Abs. 1 StPO' stays "
    "'Art. 221 Abs. 1 StPO'); preserve names, dates and amounts. Translate the "
    "surrounding prose into fluent legal English. Reply with the translation only, "
    "no preamble."
)


def translate(key, text):
    msgs = [{"role": "system", "content": TRANSLATE_SYS},
            {"role": "user", "content": str(text).strip()[:1800]}]
    return call_ds(key, msgs, max_tokens=2500)


def classify(key, text):
    """Same prompt as the archived run, with a far larger completion budget.

    deepseek-v4-flash reasons before it answers, and at 2,500 tokens it returned
    finish_reason='length' with empty content on 64% of queries: precisely the
    long-reasoning ones. Those are not missing at random, so the survivors are
    the easy queries and any accuracy computed over them is inflated. The budget
    is therefore set well above the observed reasoning length, and escalated once
    if the model still runs out.
    """
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"LEGAL QUESTION:\n{str(text).strip()[:1800]}\n\nJSON:"}]
    for budget in (8000, 16000):
        raw = call_ds(key, msgs, max_tokens=budget)
        obj = parse(raw)
        dom = (obj or {}).get("domain")
        if dom in LABELS:
            return dom
    return None


def mcnemar(b, c):
    """Exact two-sided binomial McNemar on the discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = load_key()
    train = pd.read_csv(ROOT / "Data" / "train.csv")
    qtext = {str(a): b for a, b in zip(train["query_id"], train["query"])}

    acc = pd.read_csv(ACCURACY)
    gold_col = "gold_domain"
    pred_col = [c for c in acc.columns if c.startswith("pred_")][0]
    rows = [r for _, r in acc.iterrows()
            if str(r[gold_col]) in LABELS and str(r["query_id"]) in qtext]
    print(f"archived DE arm: {len(rows)} queries with gold domain and text")

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    qids = [str(r["query_id"]) for r in sample]
    gold = {str(r["query_id"]): str(r[gold_col]) for r in sample}
    de_pred = {str(r["query_id"]): str(r[pred_col]) for r in sample}
    print(f"sampled n={len(qids)} (seed {args.seed})\n")

    t0 = time.time()
    cache_path = ROOT / "analysis" / "_transfer_translations.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo_tr = [q for q in qids if not cache.get(q)]
    print(f"translating DE->EN ... ({len(cache)} cached, {len(todo_tr)} to fetch)", flush=True)
    if todo_tr:
        with ThreadPoolExecutor(CONCURRENCY) as ex:
            for q, v in zip(todo_tr, ex.map(lambda q: translate(key, qtext[q]), todo_tr)):
                if v:
                    cache[q] = v
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    en_text = {q: cache.get(q) for q in qids}
    n_tr = sum(1 for v in en_text.values() if v)
    print(f"  translated {n_tr}/{len(qids)}  [{time.time()-t0:.0f}s]\n", flush=True)

    print("classifying EN ...", flush=True)
    todo = [q for q in qids if en_text.get(q)]
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        en_pred = dict(zip(todo, ex.map(lambda q: classify(key, en_text[q]), todo)))
    print(f"  classified {sum(1 for v in en_pred.values() if v)}/{len(todo)}"
          f"  [{time.time()-t0:.0f}s]\n", flush=True)

    paired = [q for q in qids if en_pred.get(q)]
    dropout = 1 - len(paired) / len(qids)
    if dropout > 0.10:
        print(f"\nABORT: {dropout:.1%} of queries produced no English label.\n"
              "Classification failures concentrate on long-reasoning queries, so the\n"
              "remainder is a biased easy subset and any accuracy computed over it\n"
              "would overstate both arms. Raise the completion budget and re-run.\n"
              "No statistics written.")
        OUT.with_suffix(".aborted.json").write_text(
            json.dumps({"n_sampled": len(qids), "n_paired": len(paired),
                        "dropout": round(dropout, 4)}, indent=2), encoding="utf-8")
        return

    de_ok = {q: de_pred[q] == gold[q] for q in paired}
    en_ok = {q: en_pred[q] == gold[q] for q in paired}
    b = sum(1 for q in paired if de_ok[q] and not en_ok[q])
    c = sum(1 for q in paired if en_ok[q] and not de_ok[q])
    de_acc = sum(de_ok.values()) / len(paired)
    en_acc = sum(en_ok.values()) / len(paired)
    agree = sum(1 for q in paired if de_pred[q] == en_pred[q]) / len(paired)
    p = mcnemar(b, c)

    res = {
        "n_sampled": len(qids), "n_paired": len(paired), "seed": args.seed,
        "model": "deepseek-v4-flash",
        "de_accuracy": round(de_acc, 4), "en_accuracy": round(en_acc, 4),
        "delta_en_minus_de": round(en_acc - de_acc, 4),
        "label_agreement": round(agree, 4),
        "discordant_de_only_correct": b, "discordant_en_only_correct": c,
        "mcnemar_exact_p": round(p, 4),
        "n_translation_failed": len(qids) - n_tr,
        "n_classification_failed": len(todo) - sum(1 for v in en_pred.values() if v),
        "per_query": [{"query_id": q, "gold": gold[q], "de": de_pred[q],
                       "en": en_pred[q]} for q in paired],
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"paired n              {len(paired)}")
    print(f"DE accuracy           {de_acc:.4f}   (archived predictions)")
    print(f"EN accuracy           {en_acc:.4f}   (translated, same queries)")
    print(f"delta (EN - DE)       {en_acc-de_acc:+.4f}")
    print(f"label agreement       {agree:.4f}")
    print(f"discordant  DE-only {b}   EN-only {c}   McNemar exact p = {p:.4f}")
    print("=" * 60)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
