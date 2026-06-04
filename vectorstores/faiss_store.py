from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult
from vectorstores.base import SearchResult, VectorStoreError, VectorStoreStats


class FaissVectorStore:
    """Local FAISS cosine-similarity vector store."""

    backend_name = "faiss"

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_path = Path(persist_path) if persist_path else None
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []
        self._index = None

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self._chunks.copy()

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings length mismatch.")

        by_id = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        for chunk, embedding in zip(chunks, embeddings):
            vector = list(embedding.vector)
            if chunk.chunk_id in by_id:
                index = by_id[chunk.chunk_id]
                self._chunks[index] = chunk
                self._vectors[index] = vector
            else:
                self._chunks.append(chunk)
                self._vectors.append(vector)
        self._rebuild_index()

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or not self._chunks:
            return []
        if self._index is None:
            self._rebuild_index()

        query = self._normalize(np.array([list(query_vector)], dtype="float32"))
        limit = min(len(self._chunks), max(top_k * 4, top_k))
        scores, indices = self._index.search(query, limit)

        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            if not self._matches_filter(chunk, metadata_filter):
                continue
            results.append(SearchResult(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def delete(self, chunk_ids: Sequence[str] | None = None, source_id: str | None = None) -> int:
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
        self._rebuild_index()
        return deleted

    def save(self, path: str | Path | None = None) -> None:
        raise VectorStoreError("FAISS persistence is not enabled in this simplified demo.")

    def load(self, path: str | Path | None = None) -> None:
        raise VectorStoreError("FAISS persistence is not enabled in this simplified demo.")

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            backend=self.backend_name,
            collection=self.collection_name,
            num_vectors=len(self._chunks),
            persisted=False,
            path=str(self.persist_path) if self.persist_path else None,
        )

    def _rebuild_index(self) -> None:
        if not self._vectors:
            self._index = None
            return
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("faiss-cpu is not installed. Install it with: pip install faiss-cpu") from exc

        matrix = self._normalize(np.array(self._vectors, dtype="float32"))
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index

    @staticmethod
    def _normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

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
