# -*- coding: utf-8 -*-
"""Fill the gaps the DeepSeek translation run left, from the archived translation.

The run left 28 of 1,139 training questions untranslated. Dropping them would make the
English cohort a different set of queries from the German one, which is exactly the
confound the paired design exists to avoid. The archived file
`training/lexam_train_translated.csv` in the project's Hugging Face repository covers all
1,139, so it supplies the gaps.

The two sources are not equivalent, so which query came from which is recorded rather than
merged silently: the archived translation is visibly lossier and a few of its rows were
never translated at all. Only rows that pass the same English-text check as the primary run
are accepted, and the sidecar records the provenance of every query so the downstream
experiments can be re-run on the primary subset alone.

Usage:  python analysis/fill_translation_gaps.py
Run from the repository root.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / 'analysis' / 'train_queries_en.json'
SIDECAR = ROOT / 'analysis' / 'train_queries_en_source.json'

META = re.compile(r"^(We need|The user|I need|Let me|Let us|We must|Need |Okay|Alright|First,)")
# whole-sentence German, not the loan words and statute names English legal prose keeps
GERMAN_RUN = re.compile(
    r'\b(ist|sind|wird|werden|nicht|und|der|die|das|dem|den|eine|einen|einer)\b'
    r'(?:\W+\w+){0,4}\W+'
    r'\b(ist|sind|wird|werden|nicht|und|der|die|das|dem|den|eine|einen|einer)\b')


def is_bad(s):
    s = (s or '').strip()
    if len(s) < 40:
        return 'short'
    if META.match(s):
        return 'reasoning-trace'
    if re.search(r'\btranslat', s[:300], re.I):
        return 'meta'
    if len(GERMAN_RUN.findall(s)) >= 3:
        return 'german'
    return None


def main():
    cur = json.loads(TARGET.read_text(encoding='utf-8'))
    train = pd.read_csv(ROOT / 'Data' / 'train.csv')
    all_ids = [str(q) for q in train['query_id']]
    have = {k for k, v in cur.items() if v and not is_bad(v)}
    missing = [q for q in all_ids if q not in have]
    print(f'{len(have)} usable from the primary run; {len(missing)} gaps')

    p = hf_hub_download('Dharun72/llm-agentic-swiss-legal-checkpoints',
                        'training/lexam_train_translated.csv', repo_type='dataset')
    arch = {str(r['query_id']): str(r['query_en'])
            for _, r in pd.read_csv(p).fillna('').iterrows()}

    filled, refused = 0, {}
    for q in missing:
        why = is_bad(arch.get(q))
        if why:
            refused[why] = refused.get(why, 0) + 1
            continue
        cur[q] = arch[q]
        filled += 1

    source = {q: ('primary' if q in have else ('archive' if cur.get(q) else 'none'))
              for q in all_ids}
    counts = {}
    for v in source.values():
        counts[v] = counts.get(v, 0) + 1

    io.open(TARGET, 'w', encoding='utf-8').write(json.dumps(cur, ensure_ascii=False, indent=1))
    SIDECAR.write_text(json.dumps(source, indent=1), encoding='utf-8')
    print(f'filled {filled} from the archive; refused {refused or "none"}')
    print('provenance:', counts)
    print(f'wrote {TARGET}\nwrote {SIDECAR}')


if __name__ == '__main__':
    main()
