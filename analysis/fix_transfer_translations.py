# -*- coding: utf-8 -*-
"""Re-translate cached DE->EN transfer translations that are reasoning traces, not translations.

The archived cache was filled by call_ds, which falls back to reasoning_content when the
completion budget is exhausted. For 45 of 150 queries that fallback stored the model's
deliberation (often containing the German source) instead of an English translation.
This script re-requests those entries, keeps `content` only, and verifies the result.
Run from the repository root.
"""
import json, re, sys, time, random, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'analysis')
from deepseek_router import load_key, DS_URL, DS_MODEL
from router_language_transfer import TRANSLATE_SYS
import pandas as pd

ROOT = Path('.')
CACHE = ROOT / 'analysis' / '_transfer_translations.json'
train = pd.read_csv(ROOT / 'Data' / 'train.csv')
qtext = {str(a): b for a, b in zip(train['query_id'], train['query'])}
cache = json.loads(CACHE.read_text(encoding='utf-8'))

META = re.compile(r"^(We need|The user|I need|Let me|Let us|We must|Need |Okay|Alright|First,)")
GERMAN = re.compile(r"\b(und|der|die|das|nicht|Sachverhalt|Frage|ist|wird)\b")


def is_bad(s):
    s = (s or '').strip()
    if len(s) < 40:
        return True
    if META.match(s):
        return True
    if re.search(r'\btranslat', s[:300], re.I):
        return True
    if GERMAN.search(s):
        return True
    return False


def call_content(key, text, max_tokens):
    h = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    payload = {'model': DS_MODEL, 'temperature': 0.0, 'max_tokens': max_tokens,
               'messages': [{'role': 'system', 'content': TRANSLATE_SYS},
                            {'role': 'user', 'content': str(text).strip()[:1800]}]}
    for a in range(6):
        try:
            r = requests.post(DS_URL, headers=h, json=payload, timeout=300)
            if r.status_code == 200:
                m = r.json()['choices'][0]['message']
                return (m.get('content') or '').strip()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
                continue
            return ''
        except requests.RequestException:
            time.sleep(min(2 ** a, 30) + random.uniform(0, 1.5))
    return ''


def fix(q):
    key = load_key()
    for budget in (8000, 16000, 32000):
        out = call_content(key, qtext[q], budget)
        if out and not is_bad(out):
            return q, out, budget
    return q, None, None


if __name__ == '__main__':
    bad = [q for q, t in cache.items() if is_bad(t)]
    print('bad entries:', len(bad), flush=True)
    with ThreadPoolExecutor(6) as ex:
        for q, out, budget in ex.map(fix, bad):
            if out:
                cache[q] = out
                print('fixed', q, 'budget', budget, 'len', len(out), flush=True)
            else:
                print('FAILED', q, flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    still = [q for q, t in cache.items() if is_bad(t)]
    print('remaining bad:', len(still), still)
