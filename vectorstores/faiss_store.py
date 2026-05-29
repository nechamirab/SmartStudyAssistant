from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult
from vectorstores.base import SearchResult, VectorStoreError, VectorStoreStats
from vectorstores.memory import InMemoryVectorStore


class FaissVectorStore:
    """FAISS-backed cosine search with a memory fallback for filtered queries."""

    backend_name = "faiss"

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_path = Path(persist_path) if persist_path else None
        self._memory = InMemoryVectorStore(collection_name=collection_name)
        self._index = None
        self._dimension = 0

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self._memory.chunks

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> None:
        self._memory.add(chunks, embeddings)
        self._rebuild_index()

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        if metadata_filter:
            return self._memory.search(query_vector, top_k=top_k, metadata_filter=metadata_filter)
        if self._index is None:
            return []

        try:
            import numpy as np
        except ImportError as e:
            raise VectorStoreError("numpy is required for FAISS search.") from e

        query = np.array([list(query_vector)], dtype="float32")
        self._normalize(query)
        scores, positions = self._index.search(query, top_k)
        chunks = self._memory.chunks
        results: list[SearchResult] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(chunks):
                continue
            results.append(SearchResult(chunk=chunks[position], score=float(score)))
        return results

    def delete(
        self,
        chunk_ids: Sequence[str] | None = None,
        source_id: str | None = None,
    ) -> int:
        deleted = self._memory.delete(chunk_ids=chunk_ids, source_id=source_id)
        self._rebuild_index()
        return deleted

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.persist_path
        if target is None:
            raise VectorStoreError("No persist path configured for FAISS vector store.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(
                {
                    "collection": self.collection_name,
                    "chunks": self._memory.chunks,
                    "vectors": self._memory._vectors,
                },
                f,
            )

    def load(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.persist_path
        if target is None or not target.exists():
            raise VectorStoreError(f"Persisted FAISS vector store not found: {target}")
        with target.open("rb") as f:
            payload = pickle.load(f)
        self.collection_name = str(payload.get("collection", self.collection_name))
        self._memory._chunks = list(payload.get("chunks", []))
        self._memory._vectors = [list(vector) for vector in payload.get("vectors", [])]
        self._rebuild_index()

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            backend=self.backend_name,
            collection=self.collection_name,
            num_vectors=len(self._memory.chunks),
            persisted=self.persist_path.exists() if self.persist_path else False,
            path=str(self.persist_path) if self.persist_path else None,
        )

    def _rebuild_index(self) -> None:
        vectors = self._memory._vectors
        if not vectors:
            self._index = None
            self._dimension = 0
            return
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise VectorStoreError(
                "faiss-cpu and numpy are required for --vector-store faiss. "
                "Install optional dependencies with: pip install faiss-cpu numpy"
            ) from e

        matrix = np.array(vectors, dtype="float32")
        self._normalize(matrix)
        self._dimension = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(matrix)

    @staticmethod
    def _normalize(matrix) -> None:
        try:
            import numpy as np
        except ImportError as e:
            raise VectorStoreError("numpy is required for FAISS normalization.") from e
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
