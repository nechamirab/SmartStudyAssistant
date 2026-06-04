from __future__ import annotations

from dataclasses import dataclass

from services.embedding_service import EmbeddingError, EmbeddingService
from services.vector_store_service import SearchResult
from vectorstores.base import BaseVectorStore


class RetrievalError(Exception):
    """Raised when retrieval fails."""


@dataclass(frozen=True)
class RetrievalResponse:
    query: str
    results: list[SearchResult]


class RetrievalService:
    """Embed a question and retrieve the most similar PDF chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        chunks: list | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunks = chunks or getattr(vector_store, "chunks", [])

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> RetrievalResponse:
        query = (query or "").strip()
        if not query:
            raise RetrievalError("Query cannot be empty.")
        if top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")

        try:
            query_vector = self.embedding_service.embed_query(query)
            results = self.vector_store.search(
                query_vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            return RetrievalResponse(query=query, results=results)
        except EmbeddingError as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc
