from __future__ import annotations

from dataclasses import dataclass

from core.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, MOCK_EMBEDDING_DIM


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

    def __init__(self, provider: str = EMBEDDING_PROVIDER, model: str = EMBEDDING_MODEL):
        self.provider = provider
        self.model = model

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
        results: list[EmbeddingResult] = []

        for chunk in chunks:
            vector = self._embed_text(chunk.text)
            results.append(
                EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    vector=vector,
                )
            )

        return results

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

        return self._embed_text(query)

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