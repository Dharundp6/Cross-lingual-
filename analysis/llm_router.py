# -*- coding: utf-8 -*-
"""Inference-time legal-domain router for the Swiss legal citation pipeline.

WHY THIS EXISTS
---------------
The six domain labels reported in the stratified evaluation
(`analysis/eval_metrics_suite.py:28-48`) are derived by majority vote over each
query's GOLD citations. That is an evaluation-time ORACLE: it cannot run at
inference. This module supplies the missing inference-time classifier.

Three routers, scored against the same gold-derived labels:
  1. `MajorityRouter`     - always predicts the most frequent train domain (floor)
  2. `KeywordRouter`      - KW2CODES rules lifted from `full_runtime_pipeline.py:145-160`
  3. `LLMRouter`          - Qwen2.5-7B-Instruct 4-bit, constrained JSON, temperature 0

The LLM router must beat the keyword router to justify its cost. That comparison
runs over ~993 train queries, NOT the n=10 validation set -- prior routing work
in this project was judged on n=10 and overfit (k-means routing gained +0.010 on
val then regressed 0.169 vs 0.184 private).

USAGE
-----
  python analysis/llm_router.py --self-test          # no GPU, no model, parity checks
  python analysis/llm_router.py --baseline           # majority + keyword on train
  python analysis/llm_router.py --llm                # adds the Qwen router (needs GPU)
  python analysis/llm_router.py --llm --limit 40     # quick smoke run

Run from the repository root.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:  # idempotent -- safe when this module is imported after stdout is already set up
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
EVAL_SUITE = ROOT / "analysis" / "eval_metrics_suite.py"
RUNTIME_PIPELINE = ROOT / "analysis" / "full_runtime_pipeline.py"

# --------------------------------------------------------------------------
# Label set -- MUST stay identical to eval_metrics_suite.py:28-48.
# Replicated rather than imported because that module loads a 720 MB embedding
# matrix at import time. `--self-test` enforces parity against the source.
# --------------------------------------------------------------------------
DOMAIN = {
    'StGB': 'Criminal', 'StPO': 'Criminal', 'StBOG': 'Criminal', 'JStG': 'Criminal', 'VStrR': 'Criminal',
    'ZGB': 'Civil & commercial', 'OR': 'Civil & commercial', 'ZPO': 'Civil & commercial', 'GBV': 'Civil & commercial',
    'IPRG': 'Civil & commercial', 'HRegV': 'Civil & commercial', 'PartG': 'Civil & commercial',
    'BV': 'Public & administrative', 'VwVG': 'Public & administrative', 'RPG': 'Public & administrative',
    'USG': 'Public & administrative', 'NHG': 'Public & administrative', 'AIG': 'Public & administrative',
    'DSG': 'Public & administrative', 'URG': 'Public & administrative', 'BGO': 'Public & administrative',
    'RTVG': 'Public & administrative', 'BGF': 'Public & administrative',
    'DBG': 'Tax, social & financial', 'MWSTG': 'Tax, social & financial', 'StHG': 'Tax, social & financial',
    'ATSG': 'Tax, social & financial', 'UVG': 'Tax, social & financial', 'AHVG': 'Tax, social & financial',
    'IVG': 'Tax, social & financial', 'BVG': 'Tax, social & financial', 'FINMAG': 'Tax, social & financial',
    'FIDLEG': 'Tax, social & financial', 'KAG': 'Tax, social & financial', 'VAG': 'Tax, social & financial',
    'BGG': 'Procedure & enforcement', 'SchKG': 'Procedure & enforcement', 'BGFA': 'Procedure & enforcement',
    'SVG': 'Public & administrative',
}
LABELS = ['Civil & commercial', 'Criminal', 'Other', 'Procedure & enforcement',
          'Public & administrative', 'Tax, social & financial']

DOCKET = re.compile(r'^\d+[A-Z]+_\d+/\d+')


def cits(s):
    return [x.strip() for x in str(s).split(';') if x.strip()]


def is_court(c):
    return c.startswith('BGE') or bool(DOCKET.match(c))


def code_of(c):
    toks = c.split()
    return toks[-1] if toks else ''


def domain_of_gold_legacy(goldset):
    """Verbatim replica of eval_metrics_suite.py:45-48.

    NOT DETERMINISTIC. `goldset` is a Python set, so on a tied majority the
    winner depends on set iteration order, which depends on string hash
    randomisation (PYTHONHASHSEED). 63 of the 993 scored train queries (6.3%)
    have a tied majority domain, so the published per-domain stratified counts
    are not reproducible run-to-run. Kept only to measure that drift.
    """
    doms = Counter(DOMAIN.get(code_of(c), 'Other') for c in goldset if not is_court(c))
    return doms.most_common(1)[0][0] if doms else 'Other'


def domain_of_gold(goldset):
    """Deterministic gold-derived domain label.

    Same majority vote as the legacy function, but ties are broken
    alphabetically instead of by set iteration order. Identical output for the
    930 untied queries; reproducible for the 63 tied ones.
    """
    doms = Counter(DOMAIN.get(code_of(c), 'Other') for c in goldset if not is_court(c))
    if not doms:
        return 'Other'
    top = max(doms.values())
    return sorted(d for d, n in doms.items() if n == top)[0]


# --------------------------------------------------------------------------
# Keyword rules -- MUST stay identical to full_runtime_pipeline.py:145-160.
# --------------------------------------------------------------------------
KW2CODES = [
    (['divorce', 'spouse', 'marriage', 'marri', 'maintenance', 'custody', 'child', 'matrimon', 'separat',
      'husband', 'wife', 'guardian', 'lien', 'mortgage', 'inherit', 'will', 'estate', 'succession'], {'ZGB'}),
    (['accused', 'prosecut', 'detention', 'robbery', 'criminal', 'offence', 'offense', 'penal', 'theft',
      'juvenile', 'dna'], {'StPO', 'StGB', 'JStPO'}),
    (['contract', 'mandate', 'lease', 'tenancy', 'landlord', 'freight', 'forwarder', 'rent', 'partnership',
      'agent', 'liabilit', 'tort', 'damages', 'obligation'], {'OR'}),
    (['foreign', 'recogni', 'jurisdiction', 'international', 'apostille', 'probate', 'cross-border'], {'IPRG'}),
    (['accident', 'occupational', 'disability', 'insurance', 'insured', 'uvg', 'pension', 'invalid'],
     {'UVG', 'ATSG', 'IVG', 'UVV', 'IVV'}),
    (['bankrupt', 'debt enforcement', 'insolven', 'seizure', 'attachment'], {'SchKG'}),
    (['trademark', 'domain', 'unfair competition', 'copyright'], {'MSchG', 'UWG', 'URG'}),
    (['tax', 'withholding'], {'DBG', 'VStG'}),
    (['product liab', 'defect'], {'PrHG'}),
]
_EXPLICIT_CODE_RX = re.compile(r'\bArt(?:icle|\.)?\s*[0-9][0-9a-z]*[^.]{0,12}?\b([A-Z][A-Za-z]{1,7})\b')


def expected_codes(q):
    """Verbatim replica of full_runtime_pipeline.py:155-160."""
    ql = q.lower()
    codes = set()
    for kws, cs in KW2CODES:
        if any(k in ql for k in kws):
            codes |= cs
    for m in _EXPLICIT_CODE_RX.finditer(q):
        codes.add(m.group(1))
    return codes


def codes_to_domain(codes):
    """Map a predicted code set to a single domain.

    Deterministic tie-break (alphabetical) so results are reproducible -- this
    differs from domain_of_gold, which inherits Counter's insertion order. The
    gold path is left untouched so labels stay bit-identical to the metrics suite.
    """
    doms = Counter(DOMAIN[c] for c in codes if c in DOMAIN)
    if not doms:
        return 'Other'
    top = max(doms.values())
    return sorted(d for d, n in doms.items() if n == top)[0]


# --------------------------------------------------------------------------
# Routers
# --------------------------------------------------------------------------
class MajorityRouter:
    """Predicts the most frequent training domain. The floor any router must clear."""
    name = 'majority'

    def __init__(self, fallback='Civil & commercial'):
        self.fallback = fallback

    def fit(self, labels):
        self.fallback = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, query):
        return self.fallback, set(), 1.0


class KeywordRouter:
    """Deterministic KW2CODES router -- the baseline the LLM must beat."""
    name = 'keyword'

    def predict(self, query):
        codes = expected_codes(query)
        return codes_to_domain(codes), codes, 1.0 if codes else 0.0


ROUTER_SYS = """You are a Swiss legal domain classifier. Classify the legal question into EXACTLY ONE domain.

DOMAINS:
- "Criminal": criminal offences and criminal procedure (StGB, StPO, JStG, VStrR)
- "Civil & commercial": private law, family, succession, property, contract, company, private international law (ZGB, OR, ZPO, IPRG, GBV, HRegV)
- "Public & administrative": constitutional rights, administrative procedure, planning, environment, migration, data protection, IP, road traffic (BV, VwVG, RPG, USG, AIG, DSG, URG, SVG)
- "Tax, social & financial": taxation, social insurance, pensions, financial market regulation (DBG, MWSTG, ATSG, UVG, AHVG, IVG, BVG, FINMAG)
- "Procedure & enforcement": appeals to the Federal Supreme Court, debt enforcement and bankruptcy, legal profession (BGG, SchKG, BGFA)
- "Other": none of the above fits

Reply with JSON ONLY, no prose:
{"domain": "<one domain string exactly as written above>", "dominant_codes": ["<up to 3 Swiss law abbreviations>"], "confidence": <0.0-1.0>}"""


class LLMRouter:
    """Qwen2.5-7B-Instruct 4-bit, constrained JSON, temperature 0.

    Loading pattern mirrors current/_local_llm_picker_v10.py:54-73.
    Falls back to the keyword router whenever the model output cannot be parsed
    into a valid label, so the router never emits an out-of-vocabulary domain.
    """
    name = 'llm'

    def __init__(self, model_id='Qwen/Qwen2.5-7B-Instruct', quant='4bit', device_map='auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f'Loading router model: {model_id} quant={quant}', flush=True)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs = {'device_map': device_map, 'trust_remote_code': True}
        if quant == '4bit':
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        elif quant == '8bit':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs['torch_dtype'] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self._torch = torch
        self.fallback = KeywordRouter()
        self.n_fallback = 0
        print(f'  loaded in {time.time() - t0:.1f}s', flush=True)

    def _chat(self, user, max_new_tokens=96):
        msgs = [{'role': 'system', 'content': ROUTER_SYS}, {'role': 'user', 'content': user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors='pt').to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def predict(self, query):
        raw = self._chat(f'LEGAL QUESTION:\n{str(query).strip()[:1800]}\n\nJSON:')
        parsed = _parse_json(raw)
        dom = (parsed or {}).get('domain')
        if dom not in LABELS:
            self.n_fallback += 1
            return self.fallback.predict(query)
        codes = {str(c).strip() for c in (parsed.get('dominant_codes') or []) if str(c).strip()}
        try:
            conf = float(parsed.get('confidence', 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return dom, codes, conf


def _parse_json(raw):
    """Extract the first JSON object from a model response."""
    if not raw:
        return None
    m = re.search(r'\{.*?\}', raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(y_true, y_pred, labels=LABELS):
    """Accuracy, macro-F1, per-class precision/recall/F1, and a confusion matrix."""
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    per_class = {}
    f1s = []
    for l in labels:
        i = idx[l]
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[l] = {'support': int(cm[i, :].sum()), 'precision': prec, 'recall': rec, 'f1': f1}
        f1s.append(f1)

    return {
        'n': len(y_true),
        'accuracy': float(np.trace(cm) / max(cm.sum(), 1)),
        'macro_f1': float(np.mean(f1s)),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
    }


def print_report(name, res):
    print(f'\n=== router: {name} ===')
    print(f"n={res['n']}  accuracy={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")
    print(f"{'domain':<26} {'supp':>5} {'prec':>7} {'rec':>7} {'F1':>7}")
    for l in res['labels']:
        c = res['per_class'][l]
        print(f"{l:<26} {c['support']:>5} {c['precision']:>7.3f} {c['recall']:>7.3f} {c['f1']:>7.3f}")
    print('\nconfusion matrix (rows = gold, cols = predicted)')
    cm = np.array(res['confusion_matrix'])
    short = [l[:10] for l in res['labels']]
    print(' ' * 26 + ''.join(f'{s:>11}' for s in short))
    for i, l in enumerate(res['labels']):
        print(f'{l:<26}' + ''.join(f'{v:>11}' for v in cm[i]))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_train(match_metrics_suite=True, limit=None):
    """Load train queries with their gold-derived domain labels.

    match_metrics_suite=True reproduces the exact filter used by
    eval_metrics_suite.statute_metrics: keep only queries with at least one
    non-court gold citation present verbatim in laws_de.csv. That yields the
    n=993 population the stratified metrics were computed over.
    """
    df = pd.read_csv(ROOT / 'Data' / 'train.csv')
    keep_mask = None
    if match_metrics_suite:
        laws = pd.read_csv(ROOT / 'Data' / 'laws_de.csv', usecols=['citation'])
        law_set = set(laws['citation'].astype(str))
        keep_mask = [
            any((not is_court(c)) and c in law_set for c in cits(r['gold_citations']))
            for _, r in df.iterrows()
        ]
        df = df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    queries = df['query'].astype(str).tolist()
    labels = [domain_of_gold(set(cits(g))) for g in df['gold_citations']]
    return df['query_id'].tolist(), queries, labels


# --------------------------------------------------------------------------
# External router I/O (external classification)
# --------------------------------------------------------------------------
def dump_batches(outdir, batch_size=100, limit=None, max_chars=2200):
    """Write query batches for an external classifier.

    Gold citations are deliberately EXCLUDED so the external router classifies
    blind. Query text is truncated to `max_chars`; domain routing does not need
    the full fact pattern and the cap keeps batches within a sane prompt size.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qids, queries, _ = load_train(limit=limit)
    n = 0
    for b, start in enumerate(range(0, len(qids), batch_size)):
        chunk = [{'query_id': qids[i], 'query': queries[i][:max_chars]}
                 for i in range(start, min(start + batch_size, len(qids)))]
        (outdir / f'batch_{b:02d}.json').write_text(
            json.dumps(chunk, indent=1, ensure_ascii=False), encoding='utf-8')
        n += len(chunk)
    print(f'wrote {b + 1} batches ({n} queries, no gold labels) to {outdir}')
    print('LABELS:', LABELS)
    return 0


def _load_external(path):
    p = Path(path)
    if p.suffix.lower() == '.json':
        obj = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        return {str(r['query_id']): str(r['domain']) for r in obj}
    df = pd.read_csv(p)
    col = 'domain' if 'domain' in df.columns else df.columns[1]
    return {str(a): str(b) for a, b in zip(df['query_id'], df[col])}


def score_external(path, label, out, limit=None, exclude_missing=False):
    """Score an externally produced set of domain labels against gold."""
    preds_map = _load_external(path)
    qids, queries, labels = load_train(limit=limit)

    missing = [q for q in qids if preds_map.get(q) not in LABELS]
    invalid = sorted({v for v in preds_map.values() if v not in LABELS})
    if invalid:
        print(f'WARNING  {len(invalid)} out-of-vocabulary labels present: {invalid[:6]}')
    if missing:
        if exclude_missing:
            print(f'NOTE  excluding {len(missing)} queries without a genuine external prediction')
        else:
            print(f'WARNING  {len(missing)} queries have no prediction; '
                  f'falling back to the keyword router for those')
            print(f'         first few: {missing[:8]}')

    kr = KeywordRouter()
    eval_qids, eval_labels, preds = [], [], []
    for qid, q, y_true in zip(qids, queries, labels):
        v = preds_map.get(qid)
        if exclude_missing and v not in LABELS:
            continue
        eval_qids.append(qid)
        eval_labels.append(y_true)
        preds.append(v if v in LABELS else kr.predict(q)[0])

    res = score(eval_labels, preds)
    res['n_missing'] = len(missing)
    res['n_invalid_labels'] = len(invalid)
    res['coverage'] = 1.0 - len(missing) / max(len(qids), 1)
    print_report(label, res)

    kw = score(labels, [kr.predict(q)[0] for q in queries])
    gate = 'PASS' if res['macro_f1'] > kw['macro_f1'] else 'FAIL'
    print(f"\nKILL GATE (router must beat keyword baseline):")
    print(f"  {label:<10} accuracy={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")
    print(f"  {'keyword':<10} accuracy={kw['accuracy']:.4f}  macro-F1={kw['macro_f1']:.4f}")
    print(f"  -> {gate}  (coverage {res['coverage']:.1%})")
    res['gate_beats_keyword'] = res['macro_f1'] > kw['macro_f1']

    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(outp.read_text(encoding='utf-8')) if outp.exists() else {}
    existing[label] = res
    existing['keyword'] = kw
    outp.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    pd.DataFrame({'query_id': eval_qids, 'gold_domain': eval_labels,
                  f'pred_{label}': preds}).to_csv(
        outp.with_name(outp.stem + f'_{label}.csv'), index=False)
    print(f'\nwrote {outp}')
    return 0


# --------------------------------------------------------------------------
# Parity self-test
# --------------------------------------------------------------------------
def _literal_from_source(path, varname):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == varname:
                    return ast.literal_eval(node.value)
    raise AssertionError(f'{varname} not found in {path}')


def self_test():
    ok = True

    src_domain = _literal_from_source(EVAL_SUITE, 'DOMAIN')
    if src_domain == DOMAIN:
        print(f'PASS  DOMAIN parity with eval_metrics_suite.py ({len(DOMAIN)} codes)')
    else:
        ok = False
        only_src = set(src_domain) - set(DOMAIN)
        only_here = set(DOMAIN) - set(src_domain)
        changed = {k for k in set(src_domain) & set(DOMAIN) if src_domain[k] != DOMAIN[k]}
        print(f'FAIL  DOMAIN drift -- missing here: {sorted(only_src)}; '
              f'extra here: {sorted(only_here)}; relabelled: {sorted(changed)}')

    src_kw = _literal_from_source(RUNTIME_PIPELINE, 'KW2CODES')
    if [(list(k), set(v)) for k, v in src_kw] == [(list(k), set(v)) for k, v in KW2CODES]:
        print(f'PASS  KW2CODES parity with full_runtime_pipeline.py ({len(KW2CODES)} rules)')
    else:
        ok = False
        print('FAIL  KW2CODES drift vs full_runtime_pipeline.py')

    derived = sorted(set(DOMAIN.values()) | {'Other'})
    if derived == sorted(LABELS):
        print(f'PASS  LABELS cover all domains ({len(LABELS)})')
    else:
        ok = False
        print(f'FAIL  LABELS mismatch: derived={derived} declared={sorted(LABELS)}')

    # Counts published in metrics_suite.json / the dissertation artifact.
    published = {'Civil & commercial': 323, 'Other': 211, 'Public & administrative': 190,
                 'Tax, social & financial': 125, 'Criminal': 106, 'Procedure & enforcement': 38}
    try:
        qids, queries, labels = load_train()

        if len(labels) == sum(published.values()):
            print(f'PASS  scored population n={len(labels)} matches published total')
        else:
            ok = False
            print(f'FAIL  scored population n={len(labels)} != published {sum(published.values())}')

        # Quantify the non-determinism in the legacy labeller.
        df = pd.read_csv(ROOT / 'Data' / 'train.csv').set_index('query_id')
        tied = 0
        for qid in qids:
            doms = Counter(DOMAIN.get(code_of(c), 'Other')
                           for c in cits(df.loc[qid, 'gold_citations']) if not is_court(c))
            if doms and sum(1 for v in doms.values() if v == max(doms.values())) > 1:
                tied += 1

        got = Counter(labels)
        print(f'\ntrain population: n={len(labels)}   tied-majority queries: {tied} '
              f'({tied / len(labels):.1%}, label was hash-order dependent before the fix)')
        print(f"{'domain':<26} {'deterministic':>14} {'published':>10} {'delta':>7}")
        for l in LABELS:
            d = got[l] - published.get(l, 0)
            print(f'{l:<26} {got[l]:>14} {published.get(l, 0):>10} {d:>+7}')
        if dict(got) != published:
            print('NOTE  per-domain counts differ from the published stratified metrics.\n'
                  '      Cause: eval_metrics_suite.domain_of_gold breaks tied majorities by\n'
                  '      set iteration order. The deterministic labeller above supersedes it.\n'
                  '      The published stratified metrics should be regenerated.')

        kr = KeywordRouter()
        res = score(labels, [kr.predict(q)[0] for q in queries])
        print_report('keyword (self-test)', res)
    except FileNotFoundError as e:
        print(f'SKIP  data-dependent checks -- {e}')

    print('\nSELF-TEST', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true', help='parity checks only, no GPU')
    ap.add_argument('--baseline', action='store_true', help='score majority + keyword routers')
    ap.add_argument('--llm', action='store_true', help='additionally score the Qwen router (needs GPU)')
    ap.add_argument('--model-id', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--quant', default='4bit', choices=['4bit', '8bit', 'fp16'])
    ap.add_argument('--limit', type=int, default=None, help='cap query count (smoke runs)')
    ap.add_argument('--out', default='analysis/router_accuracy.json')
    ap.add_argument('--dump-batches', metavar='DIR',
                    help='write query batches (query_id + query text ONLY, no gold) for an '
                         'external router such as external classification')
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--score-predictions', metavar='FILE',
                    help='score an external router: CSV/JSON mapping query_id -> domain')
    ap.add_argument('--label', default='external', help='name for the scored external router')
    ap.add_argument('--exclude-missing', action='store_true',
                    help='score only genuine external predictions; never backfill missing rows')
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.dump_batches:
        return dump_batches(args.dump_batches, args.batch_size, args.limit)
    if args.score_predictions:
        return score_external(args.score_predictions, args.label, args.out, args.limit,
                              args.exclude_missing)
    if not (args.baseline or args.llm):
        ap.error('choose --self-test, --baseline, --llm, --dump-batches, or --score-predictions')

    qids, queries, labels = load_train(limit=args.limit)
    print(f'loaded {len(queries)} train queries with gold-derived domain labels', flush=True)

    results, predictions = {}, {'query_id': qids, 'gold_domain': labels}

    for router in (MajorityRouter().fit(labels), KeywordRouter()):
        preds = [router.predict(q)[0] for q in queries]
        results[router.name] = score(labels, preds)
        predictions[f'pred_{router.name}'] = preds
        print_report(router.name, results[router.name])

    if args.llm:
        router = LLMRouter(model_id=args.model_id, quant=args.quant)
        preds, codes_out, confs = [], [], []
        t0 = time.time()
        for i, q in enumerate(queries):
            d, c, cf = router.predict(q)
            preds.append(d)
            codes_out.append(';'.join(sorted(c)))
            confs.append(cf)
            if (i + 1) % 50 == 0:
                print(f'  routed {i + 1}/{len(queries)}  [{time.time() - t0:.0f}s]', flush=True)
        results['llm'] = score(labels, preds)
        results['llm']['n_unparseable_fallbacks'] = router.n_fallback
        predictions['pred_llm'] = preds
        predictions['llm_codes'] = codes_out
        predictions['llm_confidence'] = confs
        print_report('llm', results['llm'])
        print(f'  unparseable model outputs fell back to keyword: {router.n_fallback}')

        kw_f1 = results['keyword']['macro_f1']
        gate = 'PASS' if results['llm']['macro_f1'] > kw_f1 else 'FAIL'
        print(f"\nKILL GATE (router must beat keyword baseline): "
              f"llm macro-F1 {results['llm']['macro_f1']:.4f} vs keyword {kw_f1:.4f} -> {gate}")
        results['gate_beats_keyword'] = results['llm']['macro_f1'] > kw_f1

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    pred_path = out.with_name(out.stem + '_predictions.csv')
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    print(f'\nwrote {out}\nwrote {pred_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
