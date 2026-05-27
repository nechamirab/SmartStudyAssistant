from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, MOCK_EMBEDDING_DIM
from embeddings.providers import EmbeddingProviderError, EmbeddingProviderRegistry


class EmbeddingError(Exception):
    """Raised when embedding creation fails."""
    pass


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Represents an embedding vector generated for a chunk.
    Attributes:
        chunk_id: The ID of the chunk.
        vector: The embedding vector.
    """
    chunk_id: str
    vector: list[float]


class EmbeddingService:
    """
    Service responsible for generating embeddings for document chunks and queries.
    Supports:
    - mock embeddings for offline/testing mode
    - OpenAI embeddings for real semantic search
    """

    def __init__(
        self,
        provider: str = EMBEDDING_PROVIDER,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        cache_enabled: bool = True,
        cache_path: str = ".cache/embeddings.sqlite3",
        fallback_to_mock: bool = True,
    ):
        self.requested_provider = provider
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.cache_enabled = cache_enabled
        self.cache_path = cache_path
        self.fallback_to_mock = fallback_to_mock
        self.fallback_reason: str | None = None
        self.embedding_dimension: int | None = None
        try:
            self._provider = EmbeddingProviderRegistry.create(
                provider=provider,
                model=model,
                normalize_embeddings=normalize_embeddings,
                cache_enabled=cache_enabled,
                cache_path=cache_path,
            )
            self.provider = self._provider.provider_name
            self.model = self._provider.model_name
        except EmbeddingProviderError as e:
            raise EmbeddingError(str(e)) from e

    def embed_texts(self, chunks: list) -> list[EmbeddingResult]:
        """
        Generate embeddings for a list of document chunks.
        Args:
            chunks: List of DocumentChunk objects.
        Returns:
            List of EmbeddingResult objects.
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            vectors = self._provider.embed_texts(
                [chunk.text for chunk in chunks],
                batch_size=self.batch_size,
            )
            self._record_dimension(vectors)
            return [
                EmbeddingResult(chunk_id=chunk.chunk_id, vector=vector)
                for chunk, vector in zip(chunks, vectors)
            ]
        except EmbeddingProviderError as e:
            if self._can_fallback():
                self._fallback_to_mock(str(e))
                return self.embed_texts(chunks)
            raise EmbeddingError(str(e)) from e

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding vector for a user query.
        Args:
            query: The user's question.
        Returns:
            Embedding vector as a list of floats.
        Raises:
            EmbeddingError: If query is empty or embedding generation fails.
        """
        query = (query or "").strip()
        if not query:
            raise EmbeddingError("Query cannot be empty.")

        try:
            vector = self._provider.embed_query(query)
            self._record_dimension([vector])
            return vector
        except EmbeddingProviderError as e:
            if self._can_fallback():
                self._fallback_to_mock(str(e))
                return self.embed_query(query)
            raise EmbeddingError(str(e)) from e

    async def aembed_texts(self, chunks: list) -> list[EmbeddingResult]:
        """Async version of embed_texts for ingestion pipelines and dashboards."""
        try:
            vectors = await self._provider.aembed_texts(
                [chunk.text for chunk in chunks],
                batch_size=self.batch_size,
            )
            self._record_dimension(vectors)
            return [
                EmbeddingResult(chunk_id=chunk.chunk_id, vector=vector)
                for chunk, vector in zip(chunks, vectors)
            ]
        except AttributeError:
            return await asyncio.to_thread(self.embed_texts, chunks)
        except EmbeddingProviderError as e:
            if self._can_fallback():
                self._fallback_to_mock(str(e))
                return await asyncio.to_thread(self.embed_texts, chunks)
            raise EmbeddingError(str(e)) from e

    async def aembed_query(self, query: str) -> list[float]:
        """Async version of embed_query."""
        query = (query or "").strip()
        if not query:
            raise EmbeddingError("Query cannot be empty.")
        try:
            vector = await self._provider.aembed_query(query)
            self._record_dimension([vector])
            return vector
        except AttributeError:
            return await asyncio.to_thread(self.embed_query, query)
        except EmbeddingProviderError as e:
            if self._can_fallback():
                self._fallback_to_mock(str(e))
                return await asyncio.to_thread(self.embed_query, query)
            raise EmbeddingError(str(e)) from e

    def _embed_text(self, text: str) -> list[float]:
        """
        Route embedding generation according to the configured provider.
        """
        if self.provider == "mock":
            return self._mock_embed(text)

        if self.provider == "openai":
            return self._openai_embed(text)

        raise EmbeddingError(f"Unsupported embedding provider: {self.provider}")

    @staticmethod
    def _mock_embed(text: str) -> list[float]:
        """
        Generate a deterministic mock embedding for offline development.

        This is not semantically meaningful like real embeddings,
        but it is useful for testing the pipeline end-to-end.
        """
        dim = MOCK_EMBEDDING_DIM
        vector = [0.0] * dim

        for i, ch in enumerate((text or "")[:5000]):
            index = (ord(ch) + i) % dim
            vector[index] += 1.0

        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector

        return [v / norm for v in vector]

    def _openai_embed(self, text: str) -> list[float]:
        """
        Generate a real embedding using OpenAI API.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbeddingError(
                "OpenAI package is not installed. Run: python -m pip install openai"
            ) from e

        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is missing.")

        try:
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            message = str(e)

            if "insufficient_quota" in message or "429" in message:
                raise EmbeddingError(
                    "OpenAI quota exceeded. Check billing or usage limits."
                ) from e

            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

    def _record_dimension(self, vectors: list[list[float]]) -> None:
        """Store the last observed embedding dimension for reports."""
        if vectors and vectors[0]:
            self.embedding_dimension = len(vectors[0])

    def _can_fallback(self) -> bool:
        """Only fallback once, and never fallback from an already-mock provider."""
        return self.fallback_to_mock and self.provider != "mock"

    def _fallback_to_mock(self, reason: str) -> None:
        """Switch to deterministic mock embeddings when optional providers fail."""
        self.fallback_reason = reason
        print(
            "  ⚠️  Embedding provider unavailable; falling back to mock embeddings. "
            f"Reason: {reason}"
        )
        self._provider = EmbeddingProviderRegistry.create(
            provider="mock",
            model="mock-hash-v1",
            normalize_embeddings=self.normalize_embeddings,
            cache_enabled=self.cache_enabled,
            cache_path=self.cache_path,
        )
        self.provider = self._provider.provider_name
        self.model = self._provider.model_name
