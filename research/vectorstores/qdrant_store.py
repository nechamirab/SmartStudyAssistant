from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult
from vectorstores.base import SearchResult, VectorStoreError, VectorStoreStats


class QdrantVectorStore:
    """Local persistent Qdrant collection wrapper."""

    backend_name = "qdrant"

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: str | Path | None = ".vectorstores/qdrant",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise VectorStoreError(
                "qdrant-client is not installed. Install optional dependencies with: pip install qdrant-client"
            ) from e

        self.collection_name = collection_name
        self.persist_path = Path(persist_path or ".vectorstores/qdrant")
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.persist_path))
        self._chunks_by_id: dict[str, DocumentChunk] = {}
        self._point_ids: dict[str, int] = {}
        self._next_point_id = 1

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks_by_id.values())

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings length mismatch.")
        if not chunks:
            return

        self._ensure_collection(len(embeddings[0].vector))
        try:
            from qdrant_client.models import PointStruct
        except ImportError as e:
            raise VectorStoreError("qdrant-client models could not be imported.") from e

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = self._point_ids.get(chunk.chunk_id, self._next_point_id)
            if chunk.chunk_id not in self._point_ids:
                self._next_point_id += 1
            self._point_ids[chunk.chunk_id] = point_id
            self._chunks_by_id[chunk.chunk_id] = chunk
            points.append(
                PointStruct(
                    id=point_id,
                    vector=list(embedding.vector),
                    payload=self._payload(chunk),
                )
            )
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as e:
            raise VectorStoreError("qdrant-client models could not be imported.") from e

        query_filter = None
        if metadata_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )

        raw = self._client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
        )
        results: list[SearchResult] = []
        for point in raw:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", point.id))
            chunk = self._chunks_by_id.get(chunk_id) or self._chunk_from_payload(payload)
            results.append(SearchResult(chunk=chunk, score=float(point.score)))
        return results

    def delete(
        self,
        chunk_ids: Sequence[str] | None = None,
        source_id: str | None = None,
    ) -> int:
        if chunk_ids:
            ids = [self._point_ids[chunk_id] for chunk_id in chunk_ids if chunk_id in self._point_ids]
            for chunk_id in chunk_ids:
                self._chunks_by_id.pop(chunk_id, None)
                self._point_ids.pop(chunk_id, None)
        elif source_id:
            chunk_ids = [
                chunk.chunk_id
                for chunk in self._chunks_by_id.values()
                if chunk.source_id == source_id
            ]
            ids = [self._point_ids[chunk_id] for chunk_id in chunk_ids if chunk_id in self._point_ids]
            for chunk_id in chunk_ids:
                self._chunks_by_id.pop(chunk_id, None)
                self._point_ids.pop(chunk_id, None)
        else:
            ids = []
        if ids:
            self._client.delete(collection_name=self.collection_name, points_selector=ids)
        return len(ids)

    def save(self, path: str | Path | None = None) -> None:
        return None

    def load(self, path: str | Path | None = None) -> None:
        return None

    def stats(self) -> VectorStoreStats:
        try:
            count = self._client.count(collection_name=self.collection_name).count
        except Exception:
            count = len(self._chunks_by_id)
        return VectorStoreStats(
            backend=self.backend_name,
            collection=self.collection_name,
            num_vectors=count,
            persisted=True,
            path=str(self.persist_path),
        )

    def _ensure_collection(self, dimension: int) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise VectorStoreError("qdrant-client models could not be imported.") from e
        collections = self._client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    @staticmethod
    def _payload(chunk: DocumentChunk) -> dict:
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_id": chunk.source_id,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char or 0,
                "end_char": chunk.end_char or 0,
                "parent_id": chunk.parent_id or "",
            }
        )
        return metadata

    @staticmethod
    def _chunk_from_payload(payload: dict) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=str(payload.get("chunk_id", "")),
            page_number=int(payload.get("page_number", 0) or 0),
            text=str(payload.get("text", "") or ""),
            source_id=str(payload.get("source_id", "") or ""),
            start_char=int(payload.get("start_char", 0) or 0),
            end_char=int(payload.get("end_char", 0) or 0),
            parent_id=str(payload.get("parent_id", "") or "") or None,
            metadata=payload,
        )
