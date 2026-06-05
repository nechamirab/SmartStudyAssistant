from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# PDF RAG defaults. Keep these in one place so the demo is easy to tune.
CHUNK_SIZE = _int_env("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 160)
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")
RETRIEVAL_TOP_K = _int_env("RETRIEVAL_TOP_K", 6)
MIN_RETRIEVAL_SCORE = _float_env("MIN_RETRIEVAL_SCORE", 0.08)

# Local RAG defaults. MiniLM + FAISS keep retrieval local; no hosted model is
# used for PDF indexing or question answering.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "minilm")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MOCK_EMBEDDING_DIM = _int_env("MOCK_EMBEDDING_DIM", 128)
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss")

# General AI tutor and Groq exam generation settings. Secrets are read from
# environment variables or local ignored config files, never hardcoded.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY_PATH = ROOT_DIR / "config" / "groq_api_key.txt"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = _float_env("LLM_TEMPERATURE", 0.2)
GENERAL_AI_TEMPERATURE = _float_env("GENERAL_AI_TEMPERATURE", 0.4)
LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 2200)

NOT_FOUND_ANSWER = "I could not find this clearly in the uploaded material."


def read_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def read_groq_api_key(path: str | Path = GROQ_API_KEY_PATH) -> str:
    env_value = os.getenv("GROQ_API_KEY", "").strip()
    if env_value:
        return env_value

    key_path = Path(path)
    if not key_path.exists():
        return ""
    value = key_path.read_text(encoding="utf-8").strip()
    for line in value.splitlines():
        clean = line.strip()
        if clean.startswith("GROQ_API_KEY="):
            return clean.split("=", 1)[1].strip()
    if value.startswith("GROQ_API_KEY="):
        value = value.split("=", 1)[1].strip()
    if "=" in value:
        return ""
    return value
