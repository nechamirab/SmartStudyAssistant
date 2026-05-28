from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult
from vectorstores.base import SearchResult, VectorStoreError, VectorStoreStats


class InMemoryVectorStore:
    """
    Dependency-free cosine-similarity store.

    It is useful for deterministic tests and small experiments, and it can be
    persisted to JSON so benchmark runs are reproducible without external
    services.
    """

    backend_name = "memory"

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_path = Path(persist_path) if persist_path else None
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Expose indexed chunks for hybrid retrieval diagnostics."""
        return self._chunks.copy()

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> None:
        """Add new vectors or replace existing chunk IDs."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings length mismatch.")

        by_id = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_id in by_id:
                index = by_id[chunk.chunk_id]
                self._chunks[index] = chunk
                self._vectors[index] = list(embedding.vector)
            else:
                self._chunks.append(chunk)
                self._vectors.append(list(embedding.vector))

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """Find the most similar chunks to the query vector."""
        if top_k <= 0:
            return []

        scores: list[SearchResult] = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if not self._matches_filter(chunk, metadata_filter):
                continue
            score = self._cosine_similarity(query_vector, vector)
            scores.append(SearchResult(chunk=chunk, score=score))

        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]

    def delete(
        self,
        chunk_ids: Sequence[str] | None = None,
        source_id: str | None = None,
    ) -> int:
        """Delete chunks by ID or source document."""
        chunk_id_set = set(chunk_ids or [])
        keep_chunks: list[DocumentChunk] = []
        keep_vectors: list[list[float]] = []
        deleted = 0

        for chunk, vector in zip(self._chunks, self._vectors):
            should_delete = bool(chunk_id_set and chunk.chunk_id in chunk_id_set)
            should_delete = should_delete or bool(source_id and chunk.source_id == source_id)
            if should_delete:
                deleted += 1
                continue
            keep_chunks.append(chunk)
            keep_vectors.append(vector)

        self._chunks = keep_chunks
        self._vectors = keep_vectors
        return deleted

    def save(self, path: str | Path | None = None) -> None:
        """Persist the collection to JSON."""
        target = Path(path) if path else self.persist_path
        if target is None:
            raise VectorStoreError("No persist path configured for memory vector store.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend_name,
            "collection": self.collection_name,
            "records": [
                {
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "source_id": chunk.source_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "parent_id": chunk.parent_id,
                        "metadata": chunk.metadata,
                    },
                    "vector": vector,
                }
                for chunk, vector in zip(self._chunks, self._vectors)
            ],
        }
        target.write_text(json.dumps(payload, indent=2))

    def load(self, path: str | Path | None = None) -> None:
        """Load a JSON-persisted collection."""
        target = Path(path) if path else self.persist_path
        if target is None or not target.exists():
            raise VectorStoreError(f"Persisted vector store not found: {target}")

        payload = json.loads(target.read_text())
        records = payload.get("records", [])
        self.collection_name = str(payload.get("collection", self.collection_name))
        self._chunks = []
        self._vectors = []
        for record in records:
            chunk_data = record.get("chunk", {})
            self._chunks.append(DocumentChunk(**chunk_data))
            self._vectors.append([float(value) for value in record.get("vector", [])])

    def stats(self) -> VectorStoreStats:
        """Return store statistics for observability and dashboards."""
        return VectorStoreStats(
            backend=self.backend_name,
            collection=self.collection_name,
            num_vectors=len(self._chunks),
            persisted=self.persist_path.exists() if self.persist_path else False,
            path=str(self.persist_path) if self.persist_path else None,
        )

    @staticmethod
    def _matches_filter(chunk: DocumentChunk, metadata_filter: dict | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if key == "source_id":
                actual = chunk.source_id
            elif key == "page_number":
                actual = chunk.page_number
            else:
                actual = chunk.metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
