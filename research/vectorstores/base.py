from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult


class VectorStoreError(Exception):
    """Raised when a vector store backend cannot complete an operation."""


@dataclass(frozen=True)
class SearchResult:
    """A scored retrieval hit from any vector store backend."""

    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class VectorStoreStats:
    """Operational statistics for dashboards and benchmark reports."""

    backend: str
    collection: str
    num_vectors: int
    persisted: bool = False
    path: str | None = None


class BaseVectorStore(Protocol):
    """Common interface implemented by all vector database backends."""

    backend_name: str
    collection_name: str

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return indexed chunks for diagnostics and lexical retrieval."""

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> None:
        """Add or update chunks and vectors."""

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """Return top-k nearest chunks, optionally filtered by metadata."""

    def delete(self, chunk_ids: Sequence[str] | None = None, source_id: str | None = None) -> int:
        """Delete indexed chunks by chunk IDs or source ID. Returns deletion count."""

    def save(self, path: str | Path | None = None) -> None:
        """Persist the index if the backend supports it."""

    def load(self, path: str | Path | None = None) -> None:
        """Load a persisted index if the backend supports it."""

    def stats(self) -> VectorStoreStats:
        """Return backend statistics."""
