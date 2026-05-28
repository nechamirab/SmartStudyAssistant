from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult
from vectorstores.base import SearchResult, VectorStoreError, VectorStoreStats


class ChromaVectorStore:
    """ChromaDB persistent collection wrapper."""

    backend_name = "chroma"

    def __init__(
        self,
        collection_name: str = "default",
        persist_path: str | Path | None = ".vectorstores/chroma",
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise VectorStoreError(
                "chromadb is not installed. Install optional dependencies with: pip install chromadb"
            ) from e

        self.collection_name = collection_name
        self.persist_path = Path(persist_path or ".vectorstores/chroma")
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        self._collection = self._client.get_or_create_collection(collection_name)
        self._chunks_by_id: dict[str, DocumentChunk] = {}

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

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._metadata(chunk) for chunk in chunks]
        self._collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(embedding.vector) for embedding in embeddings],
            metadatas=metadatas,
        )
        for chunk in chunks:
            self._chunks_by_id[chunk.chunk_id] = chunk

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        where = self._where(metadata_filter)
        raw = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, distance, document, metadata in zip(ids, distances, documents, metadatas):
            chunk = self._chunks_by_id.get(chunk_id) or self._chunk_from_metadata(
                chunk_id,
                document,
                metadata or {},
            )
            results.append(SearchResult(chunk=chunk, score=1.0 / (1.0 + float(distance))))
        return results

    def delete(
        self,
        chunk_ids: Sequence[str] | None = None,
        source_id: str | None = None,
    ) -> int:
        if chunk_ids:
            ids = list(chunk_ids)
        elif source_id:
            ids = [
                chunk.chunk_id
                for chunk in self._chunks_by_id.values()
                if chunk.source_id == source_id
            ]
        else:
            ids = []
        if ids:
            self._collection.delete(ids=ids)
            for chunk_id in ids:
                self._chunks_by_id.pop(chunk_id, None)
        return len(ids)

    def save(self, path: str | Path | None = None) -> None:
        return None

    def load(self, path: str | Path | None = None) -> None:
        return None

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            backend=self.backend_name,
            collection=self.collection_name,
            num_vectors=self._collection.count(),
            persisted=True,
            path=str(self.persist_path),
        )

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict:
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "source_id": chunk.source_id,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char or 0,
                "end_char": chunk.end_char or 0,
                "parent_id": chunk.parent_id or "",
            }
        )
        return metadata

    @staticmethod
    def _where(metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        return {key: {"$eq": value} for key, value in metadata_filter.items()}

    @staticmethod
    def _chunk_from_metadata(chunk_id: str, text: str, metadata: dict) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=chunk_id,
            page_number=int(metadata.get("page_number", 0) or 0),
            text=text,
            source_id=str(metadata.get("source_id", "") or ""),
            start_char=int(metadata.get("start_char", 0) or 0),
            end_char=int(metadata.get("end_char", 0) or 0),
            parent_id=str(metadata.get("parent_id", "") or "") or None,
            metadata=metadata,
        )
