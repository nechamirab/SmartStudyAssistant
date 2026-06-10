from dataclasses import dataclass
from typing import List

from services.embedding_service import EmbeddingService, EmbeddingError
from services.vector_store_service import SearchResult
from retrieval.hybrid import BM25Retriever, HybridRetriever
from reranking.rerankers import BaseReranker
from vectorstores.base import BaseVectorStore


class RetrievalError(Exception):
    """Raised when retrieval fails."""
    pass


@dataclass(frozen=True)
class RetrievalResponse:
    """
    Represents the retrieval output for a user query.
    Attributes:
        query: The original user query.
        results: The most relevant search results.
    """
    query: str
    results: List[SearchResult]


class RetrievalService:
    """
    Service responsible for semantic retrieval.
    It converts the user query into an embedding vector and
    retrieves the most similar chunks from the vector store.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        chunks: list | None = None,
        retrieval_mode: str = "semantic",
        reranker: BaseReranker | None = None,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retrieval_mode = retrieval_mode
        self.reranker = reranker
        self.bm25_retriever = BM25Retriever(chunks or getattr(vector_store, "chunks", []))
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=self.bm25_retriever,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> RetrievalResponse:
        """
        Retrieve the most relevant chunks for a user query.
        Args:
            query: The user's question.
            top_k: Number of top results to return.
        Returns:
            RetrievalResponse containing the original query and matching chunks.
        Raises:
            RetrievalError: If embedding generation or search fails.
        """
        query = (query or "").strip()
        if not query:
            raise RetrievalError("Query cannot be empty.")

        if top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")

        try:
            query_vector = self.embedding_service.embed_query(query)
            mode = (self.retrieval_mode or "semantic").lower()
            if mode == "semantic":
                results = self.vector_store.search(
                    query_vector,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                )
            elif mode == "bm25":
                results = [
                    SearchResult(chunk=item.chunk, score=item.score)
                    for item in self.bm25_retriever.search(query, top_k=top_k)
                    if self._matches_filter(item.chunk, metadata_filter)
                ]
            elif mode == "hybrid":
                candidate_k = max(top_k * 4, top_k)
                hybrid_results = self.hybrid_retriever.search(
                    query_vector=query_vector,
                    query=query,
                    top_k=candidate_k if self.reranker else top_k,
                )
                if self.reranker:
                    hybrid_results = self.reranker.rerank(query, hybrid_results, top_k=top_k)
                results = [
                    SearchResult(chunk=item.chunk, score=item.score)
                    for item in hybrid_results[:top_k]
                    if self._matches_filter(item.chunk, metadata_filter)
                ]
            else:
                raise RetrievalError(f"Unsupported retrieval mode: {self.retrieval_mode}")
            return RetrievalResponse(query=query, results=results)
        except EmbeddingError as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}") from e

    @staticmethod
    def _matches_filter(chunk, metadata_filter: dict | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            if key == "source_id":
                actual = chunk.source_id
            elif key == "page_number":
                actual = chunk.page_number
            else:
                actual = chunk.metadata.get(key)
            if actual != expected:
                return False
        return True
