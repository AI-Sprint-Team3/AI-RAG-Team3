# config/path.py
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 폴더
DATA_DIR = os.path.join(PROJECT_ROOT, "data/release/20250929/data")

# 문서/임베딩 저장 폴더
DOCS_PATH = os.path.join(DATA_DIR, "docs_merged.jsonl")