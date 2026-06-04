from __future__ import annotations

from pathlib import Path

from vectorstores.base import BaseVectorStore, VectorStoreError
from vectorstores.faiss_store import FaissVectorStore
from vectorstores.memory import InMemoryVectorStore


class VectorStoreFactory:
    """Create the single vector store used by the demo."""

    @staticmethod
    def create(
        backend: str = "memory",
        collection_name: str = "default",
        persist_path: str | Path | None = None,
    ) -> BaseVectorStore:
        normalized = (backend or "memory").strip().lower()
        if normalized in {"memory", "in-memory", "in_memory"}:
            return InMemoryVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
            )
        if normalized in {"faiss", "faiss-cpu"}:
            return FaissVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
            )
        raise VectorStoreError(
            f"Unsupported vector store backend: {backend}. "
            "Supported backends: memory, faiss."
        )
