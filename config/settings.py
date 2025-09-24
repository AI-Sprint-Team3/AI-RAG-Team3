# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# === OpenAI API ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Chroma 설정 ===
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")  # 기본값 localhost
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))   # 기본값 8002
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")

# === 파일 경로 관련 ===
CSV_PATH = os.getenv("CSV_PATH", "data/raw/data_list.csv")
FILE_PATH = os.getenv("FILE_PATH", "data/raw/files")

# === LLM 관련 ===
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")

