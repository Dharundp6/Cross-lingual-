# -*- coding: utf-8 -*-
"""
CELL 1 — Setup, imports, and data loading.

Offline Kaggle notebook for LLM Agentic Legal Information Retrieval competition.
All assets loaded from Kaggle datasets — no external API calls.

Required Kaggle inputs (mount as datasets):
  /kaggle/input/llm-agentic-legal-information-retrieval/
    ├── train.csv, val.csv, test.csv
    ├── laws_de.csv
    └── court_considerations.csv
  /kaggle/input/multilingual-e5-large/   (intfloat/multilingual-e5-large, 560MB)
  /kaggle/input/mmarco-mminilmv2/        (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, 120MB)

Output:
  /kaggle/working/submission.csv
"""
import os, sys, re, gc, time, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Runtime detection
IS_KAGGLE = os.path.exists('/kaggle')
if IS_KAGGLE:
    DATA = Path('/kaggle/input/llm-agentic-legal-information-retrieval')
    E5_DIR = Path('/kaggle/input/multilingual-e5-large')
    RERANK_DIR = Path('/kaggle/input/mmarco-mminilmv2')
    OUT_PATH = Path('/kaggle/working/submission.csv')
else:
    DATA = Path('Data')
    E5_DIR = Path('models/multilingual-e5-large')
    RERANK_DIR = Path('models/mmarco-mMiniLMv2-L12-H384-v1')
    OUT_PATH = Path('submission.csv')

print(f'IS_KAGGLE: {IS_KAGGLE}')
print(f'DATA: {DATA}')
print(f'OUT: {OUT_PATH}')

# Load competition data
train = pd.read_csv(DATA / 'train.csv')
val = pd.read_csv(DATA / 'val.csv')
test = pd.read_csv(DATA / 'test.csv')
laws = pd.read_csv(DATA / 'laws_de.csv').fillna('')

print(f'\nLoaded:')
print(f'  train: {len(train)} queries (mean {train["gold_citations"].astype(str).str.split(";").str.len().mean():.1f} cits/q)')
print(f'  val:   {len(val)} queries (mean {val["gold_citations"].astype(str).str.split(";").str.len().mean():.1f} cits/q)')
print(f'  test:  {len(test)} queries')
print(f'  laws:  {len(laws):,} entries')

# Notebook self-translates English queries to German via Helsinki-NLP/opus-mt-en-de
# OR uses pre-translated val_queries_de.csv / test_queries_de.csv if available
# For reproducibility we include the translation step in the pipeline
