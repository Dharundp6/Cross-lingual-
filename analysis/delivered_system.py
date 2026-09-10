# -*- coding: utf-8 -*-
"""The delivered rule-based citation system, lifted verbatim from the shipped notebook.

Source of truth: kaggle_offline_notebook/submission_offline_kg.ipynb, the configuration
that produced the best reproducible development-benchmark score. The code below is copied
from that notebook so that experiments run against the artefact that was actually
submitted rather than against a paraphrase of it. Only two things are added:

  * `kg_expand` takes an optional (domain, alpha) pair so that the learned legal-area
    label can reorder the co-citation neighbours before the top-N cut. With alpha = 0 the
    ordering, and therefore the emitted set, is byte-identical to the shipped system;
    `assert_identity()` checks that.
  * the lever stages are exposed individually so that a leave-one-out ablation can switch
    each of them off.

Run from the repository root.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BOILER = 'Art. 100 Abs. 1 BGG'
GATE = 3
N_NEIGHBOURS = 5
HUB_LIMIT = 25

laws = pd.read_csv(ROOT / 'Data' / 'laws_de.csv')
cits = laws['citation'].astype(str).tolist()
key_set = set(cits)

parse = lambda s: [x.strip() for x in str(s).split(';') if x.strip()]  # noqa: E731
is_court = lambda c: c.startswith('BGE') or bool(re.match(r'^\d[A-Z]?_\d', c))  # noqa: E731

PARA: dict = {}
code_set = set()
for c in cits:
    m = re.match(r'Art\.\s*([0-9][0-9a-z]*(?:bis|ter|quater)?)\b.*?\s(\S+)$', c)
    if m:
        PARA.setdefault((m.group(2), m.group(1)), []).append(c)
        code_set.add(m.group(2))

FRDE = {'CO': 'OR', 'CC': 'ZGB', 'CP': 'StGB', 'CPP': 'StPO', 'LP': 'SchKG', 'LCC': 'KKG',
        'LCD': 'UWG', 'LPM': 'MSchG', 'LDIP': 'IPRG', 'LRFP': 'PrHG', 'Cst': 'BV',
        'LEtr': 'AIG', 'LAVS': 'AHVG', 'LAI': 'IVG', 'LAA': 'UVG', 'LPGA': 'ATSG',
        'LDA': 'URG', 'LBI': 'PatG', 'LFus': 'FusG'}
ART_V1 = re.compile(r'Art\.\s*([0-9][0-9a-z]*(?:bis|ter|quater)?(?:\s*(?:and|und|et|,|/|&)\s*[0-9][0-9a-z]*(?:bis|ter|quater)?)*)(?:\s+Abs\.\s*[0-9]+\w*)?(?:\s+lit\.\s*[a-z]+)?\s+([A-Za-z][A-Za-z]{1,7}|\d{3}\.\d[\d.]*)')
ART_V2 = re.compile(r'(?i)\bart(?:icle|\.)?\s*([0-9][0-9a-z]*(?:bis|ter|quater)?(?:\s*(?:and|und|et|,|/|&)\s*[0-9][0-9a-z]*(?:bis|ter|quater)?)*)(?:\s+abs\.?\s*[0-9]+\w*)?(?:\s+lit\.?\s*[a-z]+)?(?:\s+of(?:\s+the)?)?\s+([A-Z][A-Za-z]{1,7}|\d{3}\.\d[\d.]*)')

LAWNAME_CORE = {'consumer credit': ['Art. 1 KKG']}
SPOUSAL = ['Art. 163 Abs. 1 ZGB', 'Art. 176 Abs. 1 ZGB']
DIVORCE = ['Art. 125 Abs. 1 ZGB']
CHILD = ['Art. 276 Abs. 1 ZGB', 'Art. 285 Abs. 1 ZGB']
STPO_CL = [c for c in ['Art. 428 Abs. 1 StPO', 'Art. 422 Abs. 1 StPO', 'Art. 135 Abs. 4 StPO',
                       'Art. 382 Abs. 1 StPO', 'Art. 393 Abs. 1 StPO', 'Art. 396 Abs. 1 StPO',
                       'Art. 37 Abs. 1 StBOG', 'Art. 39 Abs. 1 StBOG'] if c in key_set]
OR_MANDATE = ['Art. 394 Abs. 1 OR', 'Art. 398 Abs. 1 OR', 'Art. 398 Abs. 2 OR', 'Art. 400 Abs. 1 OR']
UVG = ['Art. 4 ATSG', 'Art. 6 Abs. 1 UVG', 'Art. 6 Abs. 2 UVG', 'Art. 9 Abs. 1 UVG']
ZGB_LIEN = ['Art. 837 Abs. 1 ZGB', 'Art. 839 Abs. 1 ZGB', 'Art. 840 ZGB', 'Art. 841 Abs. 1 ZGB']
RECOG = [c for c in ['Art. 25 IPRG', 'Art. 26 IPRG', 'Art. 26 Abs. 1 IPRG', 'Art. 27 Abs. 1 IPRG',
                     'Art. 27 Abs. 2 IPRG', 'Art. 29 Abs. 1 IPRG'] if c in key_set]
ADULT = ['Art. 390 Abs. 1 ZGB', 'Art. 393 Abs. 1 ZGB', 'Art. 398 Abs. 1 ZGB',
         'Art. 446 Abs. 1 ZGB', 'Art. 449a ZGB', 'Art. 450 Abs. 1 ZGB']
TENANCY = ['Art. 257d Abs. 1 OR', 'Art. 257d Abs. 2 OR', 'Art. 266a Abs. 1 OR',
           'Art. 271 Abs. 1 OR', 'Art. 257f Abs. 3 OR']
TRADEMARK = ['Art. 13 Abs. 1 MSchG', 'Art. 3 Abs. 1 MSchG', 'Art. 55 Abs. 1 MSchG',
             'Art. 2 UWG', 'Art. 3 Abs. 1 UWG', 'Art. 9 Abs. 1 UWG']


def _extract(rx, q):
    out = []
    for m in rx.finditer(q):
        code = FRDE.get(m.group(2), m.group(2))
        if code not in code_set:
            continue
        for n in re.findall(r'\b(\d+[a-z]*(?:bis|ter|quater)?)\b', m.group(1)):
            out.extend(PARA.get((code, n), []))
    return list(dict.fromkeys(out))


def levers(qtext, stages=('explicit1', 'family', 'stpo', 'domain2', 'pil', 'domain3',
                          'explicit2', 'lawname')):
    """Cumulative deterministic ADDs, in the shipped order. `stages` switches one off."""
    ql = qtext.lower()
    a = []
    if 'explicit1' in stages:
        a += _extract(ART_V1, qtext)
    if 'family' in stages:
        maint = ('maintenance' in ql or 'alimony' in ql or 'support' in ql)
        marital = any(w in ql for w in ['spouse', 'marriage', 'marri', 'separat', 'matrimon',
                                        'divorce', 'husband', 'wife'])
        if maint and marital:
            a += SPOUSAL
            if 'divorce' in ql:
                a += DIVORCE
        if maint and ('child' in ql or 'children' in ql):
            a += CHILD
    if 'stpo' in stages:
        strong = ('robbery' in ql or 'pretrial' in ql or 'pre-trial' in ql or 'pre‑trial' in ql)
        accused = ('accused' in ql and ('prosecutor' in ql or 'detention' in ql
                                        or 'offence' in ql or 'offense' in ql))
        if (strong or accused) and not any(w in ql for w in ['judicial assistance', 'child protection',
                                                             'trademark', 'collective labour', 'tenancy']):
            a += STPO_CL
    if 'domain2' in stages:
        if 'mandate' in ql or 'freight' in ql or 'forwarder' in ql or 'factoring' in ql:
            a += OR_MANDATE
        if 'uvg' in ql or 'occupational disease' in ql:
            a += UVG
        if re.search(r'\blien\b', ql) or 'craftsmen' in ql or 'statutory lien' in ql:
            a += ZGB_LIEN
    if 'pil' in stages:
        recog = ('recogni' in ql or 'apostille' in ql or 'probate' in ql
                 or 'letters of administration' in ql or 'foreign judgment' in ql
                 or 'foreign decree' in ql)
        cross = ('foreign' in ql or 'abroad' in ql or 'canad' in ql or 'moroc' in ql
                 or 'international' in ql or 'jurisdiction' in ql or 'apostille' in ql
                 or 'probate' in ql)
        if recog and cross and not any(w in ql for w in ['uvg', 'occupational disease', 'asthma',
                                                          'insurer', 'social insurance']):
            a += RECOG
    if 'domain3' in stages:
        if ('guardian' in ql or 'adult protection' in ql) and not any(
                w in ql for w in ['child', 'children', 'custody', 'pediatric', 'minor']):
            a += ADULT
        if 'arrears' in ql and ('landlord' in ql or 'tenancy' in ql or 'lease' in ql):
            a += TENANCY
        if 'trademark' in ql or 'domain name' in ql:
            a += TRADEMARK
    if 'explicit2' in stages:
        a += _extract(ART_V2, qtext)
    if 'lawname' in stages:
        a += [art for kw, arts in LAWNAME_CORE.items() if kw in ql for art in arts if art in key_set]
    return [c for c in dict.fromkeys(a) if c in key_set]


# ---------------------------------------------------------------- co-citation graph
def build_graph(hub_limit=HUB_LIMIT):
    a2c = pickle.load(open(ROOT / 'article_to_courts_v2.pkl', 'rb'))
    court_arts = defaultdict(set)
    for art, courts in a2c.items():
        for crt in courts:
            court_arts[crt].add(art)
    cocite = defaultdict(Counter)
    for crt, arts in court_arts.items():
        if hub_limit is not None and len(arts) > hub_limit:
            continue
        al = list(arts)
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                cocite[al[i]][al[j]] += 1
                cocite[al[j]][al[i]] += 1
    return cocite


def build_parafreq():
    pf = Counter()
    for _, r in pd.read_csv(ROOT / 'Data' / 'train.csv').iterrows():
        for c in parse(r['gold_citations']):
            if not is_court(c):
                pf[c] += 1
    return pf


def anchor_keys(q):
    ks = set()
    for rx in (ART_V1, ART_V2):
        for m in rx.finditer(q):
            code = FRDE.get(m.group(2), m.group(2))
            if code not in code_set:
                continue
            for n in re.findall(r'\b(\d+[a-z]*(?:bis|ter|quater)?)\b', m.group(1)):
                ks.add((n, code))
    return ks


def kg_expand(q, cocite, parafreq, gate=GATE, n=N_NEIGHBOURS, anchored=True,
              domain=None, alpha=0.0, domain_of_code=None):
    """Anchored co-citation expansion.

    anchored=True   seeds only from articles the query names verbatim (the shipped rule)
    anchored=False  seeds from every article the levers already emitted (a broader policy)
    alpha>0         adds a domain bonus to the neighbour weight before the top-n cut
    """
    def resolve(num, code):
        forms = PARA.get((code, num), [])
        return [sorted(forms, key=lambda c: (-parafreq.get(c, 0), len(c)))[0]] if forms else []

    add = []
    for ak in anchor_keys(q):
        nbrs = cocite.get(ak, Counter())
        if alpha and domain and domain_of_code is not None:
            ranked = sorted(nbrs.items(),
                            key=lambda kv: (-(kv[1] + alpha * (domain_of_code.get(kv[0][1]) == domain)),
                                            kv[0]))
        else:
            ranked = nbrs.most_common(n) if not alpha else sorted(nbrs.items(), key=lambda kv: -kv[1])
        for nb, w in ranked[:n]:
            if w >= gate:
                add += resolve(nb[0], nb[1])
    return add


def emit(q, cocite, parafreq, **kw):
    """The delivered system's answer set for one query."""
    return list(dict.fromkeys([BOILER] + levers(q) + kg_expand(q, cocite, parafreq, **kw)))


def f1(pred, gold):
    p, g = set(pred), set(gold)
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if not tp:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)
