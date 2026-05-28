from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from core.config import MOCK_EMBEDDING_DIM


class EmbeddingProviderError(Exception):
    """Raised when an embedding provider cannot create vectors."""


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy of a vector."""
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


class BaseEmbeddingProvider(Protocol):
    """Common interface for all embedding backends."""

    provider_name: str
    model_name: str

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed documents in batches."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a retrieval query."""

    async def aembed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Async document embedding helper."""
        return await asyncio.to_thread(self.embed_texts, texts, batch_size)

    async def aembed_query(self, query: str) -> list[float]:
        """Async query embedding helper."""
        return await asyncio.to_thread(self.embed_query, query)


class EmbeddingCache:
    """Small persistent SQLite cache keyed by provider, model, and text hash."""

    def __init__(self, path: str | Path = ".cache/embeddings.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get(self, provider: str, model: str, text: str) -> list[float] | None:
        key = self._key(provider, model, text)
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return list(json.loads(row[0]))

    def set(self, provider: str, model: str, text: str, vector: Sequence[float]) -> None:
        key = self._key(provider, model, text)
        payload = json.dumps(list(vector))
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, provider, model, text_hash, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    provider,
                    model,
                    self._text_hash(text),
                    payload,
                    time.time(),
                ),
            )

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @classmethod
    def _key(cls, provider: str, model: str, text: str) -> str:
        return f"{provider}:{model}:{cls._text_hash(text)}"


@dataclass
class MockEmbeddingProvider:
    """Deterministic offline provider for tests and reproducible smoke runs."""

    model_name: str = "mock-hash-v1"
    dimension: int = MOCK_EMBEDDING_DIM
    provider_name: str = "mock"
    normalize_embeddings: bool = True

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        query = (query or "").strip()
        if not query:
            raise EmbeddingProviderError("Query cannot be empty.")
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for i, ch in enumerate((text or "")[:5000]):
            index = (ord(ch) + i) % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if self.normalize_embeddings:
            return [v / norm for v in vector] if norm else vector
        return vector


class SentenceTransformersEmbeddingProvider:
    """
    Local HuggingFace/SentenceTransformers provider.

    Provider aliases:
    - sentence-transformers: default all-MiniLM-L6-v2
    - bge: BAAI/bge-small-en-v1.5
    - e5: intfloat/e5-small-v2 with E5 query/passsage prefixes
    - huggingface: explicit or default SentenceTransformers model
    """

    DEFAULT_MODELS = {
        "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "bge": "BAAI/bge-small-en-v1.5",
        "e5": "intfloat/e5-small-v2",
    }

    def __init__(
        self,
        provider_name: str,
        model_name: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name or self.DEFAULT_MODELS.get(
            provider_name,
            self.DEFAULT_MODELS["sentence-transformers"],
        )
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        model = self._load_model()
        prepared = [self._prepare_document(text) for text in texts]
        vectors = model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [self._to_list(vector) for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        query = (query or "").strip()
        if not query:
            raise EmbeddingProviderError("Query cannot be empty.")
        model = self._load_model()
        vectors = model.encode(
            [self._prepare_query(query)],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return self._to_list(vectors[0])

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingProviderError(
                "sentence-transformers is not installed. "
                "Install optional dependencies with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _prepare_document(self, text: str) -> str:
        if self.provider_name == "e5":
            return f"passage: {text or ''}"
        return text or ""

    def _prepare_query(self, query: str) -> str:
        if self.provider_name == "e5":
            return f"query: {query}"
        return query

    @staticmethod
    def _to_list(vector) -> list[float]:
        if hasattr(vector, "tolist"):
            return [float(value) for value in vector.tolist()]
        return [float(value) for value in vector]


class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider with batch support."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        self.provider_name = "openai"
        self.model_name = model_name

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbeddingProviderError(
                "OpenAI package is not installed. Run: pip install openai"
            ) from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingProviderError("OPENAI_API_KEY is missing.")

        client = OpenAI(api_key=api_key)
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                response = client.embeddings.create(model=self.model_name, input=batch)
                vectors.extend([item.embedding for item in response.data])
            return vectors
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI embedding failed: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        query = (query or "").strip()
        if not query:
            raise EmbeddingProviderError("Query cannot be empty.")
        return self.embed_texts([query], batch_size=1)[0]


class CachedEmbeddingProvider:
    """Decorator that adds persistent caching to any embedding provider."""

    def __init__(self, provider: BaseEmbeddingProvider, cache: EmbeddingCache) -> None:
        self.provider = provider
        self.cache = cache
        self.provider_name = provider.provider_name
        self.model_name = provider.model_name

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        results: list[list[float] | None] = []
        misses: list[str] = []
        miss_positions: list[int] = []

        for text in texts:
            cached = self.cache.get(self.provider_name, self.model_name, text)
            results.append(cached)
            if cached is None:
                miss_positions.append(len(results) - 1)
                misses.append(text)

        if misses:
            vectors = self.provider.embed_texts(misses, batch_size=batch_size)
            for position, text, vector in zip(miss_positions, misses, vectors):
                self.cache.set(self.provider_name, self.model_name, text, vector)
                results[position] = vector

        return [vector or [] for vector in results]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], batch_size=1)[0]

    async def aembed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts, batch_size)

    async def aembed_query(self, query: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, query)


class EmbeddingProviderRegistry:
    """Factory for supported embedding providers."""

    SENTENCE_TRANSFORMER_ALIASES = {
        "sentence-transformers",
        "sentence_transformers",
        "sentence-transformer",
        "minilm",
        "all-minilm-l6-v2",
        "huggingface",
        "hf",
        "bge",
        "e5",
    }

    @classmethod
    def create(
        cls,
        provider: str,
        model: str | None = None,
        normalize_embeddings: bool = True,
        cache_enabled: bool = True,
        cache_path: str | Path = ".cache/embeddings.sqlite3",
    ) -> BaseEmbeddingProvider:
        normalized = (provider or "mock").strip().lower()
        if normalized == "mock":
            base: BaseEmbeddingProvider = MockEmbeddingProvider(
                model_name=model or "mock-hash-v1",
                normalize_embeddings=normalize_embeddings,
            )
        elif normalized == "openai":
            base = OpenAIEmbeddingProvider(model_name=model or "text-embedding-3-small")
        elif normalized in cls.SENTENCE_TRANSFORMER_ALIASES:
            provider_name = cls._canonical_sentence_provider(normalized)
            base = SentenceTransformersEmbeddingProvider(
                provider_name,
                model,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            raise EmbeddingProviderError(f"Unsupported embedding provider: {provider}")

        if cache_enabled:
            return CachedEmbeddingProvider(base, EmbeddingCache(cache_path))
        return base

    @staticmethod
    def _canonical_sentence_provider(provider: str) -> str:
        if provider in {
            "sentence_transformers",
            "sentence-transformer",
            "minilm",
            "all-minilm-l6-v2",
        }:
            return "sentence-transformers"
        if provider == "hf":
            return "huggingface"
        return provider
