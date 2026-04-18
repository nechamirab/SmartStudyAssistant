from dataclasses import dataclass
from typing import List

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult


@dataclass(frozen=True)
class SearchResult:
    """
    Represents a search result with similarity score.
    """
    chunk: DocumentChunk
    score: float


class VectorStoreService:
    """
    Simple in-memory vector store.
    Stores embeddings and allows similarity search using cosine similarity.
    """

    def __init__(self):
        self._chunks: List[DocumentChunk] = []
        self._vectors: List[List[float]] = []

    def add(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> None:
        """
        Store chunks and their corresponding embeddings.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")

        for chunk, embedding in zip(chunks, embeddings):
            self._chunks.append(chunk)
            self._vectors.append(embedding.vector)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[SearchResult]:
        """
        Find the most similar chunks to the query vector.
        """
        scores = []

        for chunk, vector in zip(self._chunks, self._vectors):
            score = self._cosine_similarity(query_vector, vector)
            scores.append(SearchResult(chunk=chunk, score=score))

        # sort descending
        scores.sort(key=lambda x: x.score, reverse=True)

        return scores[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)