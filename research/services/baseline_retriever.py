"""
Baseline Retrieval Methods for comparison with FAISS.

Why baselines matter:
- A good system should outperform naive methods
- Helps understand which retrieval approach is best
- Provides context for interpreting results

Baselines:
1. Keyword Search: TF-IDF-like word overlap
2. Random: Baseline for luck
3. FAISS: Current semantic approach (included for completeness)
"""

from __future__ import annotations

from typing import List
import random
from dataclasses import dataclass
from core.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    """Result from retrieval baseline."""
    chunk: DocumentChunk
    score: float


class BaselineRetriever:
    """Simple retrieval baselines for comparison."""

    def __init__(self, chunks: List[DocumentChunk]):
        """
        Initialize with document chunks.

        Args:
            chunks: List of DocumentChunk objects
        """
        self.chunks = chunks

    def retrieve_by_keyword_overlap(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[DocumentChunk]:
        """
        Keyword Search Baseline: Return chunks with highest word overlap with query.

        Why keyword search?
        - Simple, interpretable baseline
        - Shows if semantic embeddings are necessary
        - Works for technical documents with specific terminology

        Algorithm:
        1. Split query into words
        2. For each chunk, count matching words
        3. Return top-K by match count

        Args:
            query: User question
            top_k: Number of chunks to return

        Returns:
            List of DocumentChunk objects ranked by keyword overlap
        """
        query_words = set(query.lower().split())

        # Score each chunk by word overlap
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words & chunk_words)
            scored.append((chunk, overlap))

        # Sort by overlap (descending) and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_k]]

    def retrieve_by_question_word_presence(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[DocumentChunk]:
        """
        Enhanced Keyword Search: Prioritize chunks containing main query keywords.

        Why this variant?
        - Better than simple overlap for question-answering
        - Identifies chunks most likely to contain answer
        - Shows importance of semantic understanding

        Algorithm:
        1. Extract "important" words (longer words, likely concepts)
        2. Find chunks containing these words
        3. Rank by number of important words

        Args:
            query: User question
            top_k: Number of chunks to return

        Returns:
            List of DocumentChunk objects
        """
        # Extract words > 4 chars (likely meaningful content words)
        query_words = [
            w.lower() for w in query.split()
            if len(w) > 4
        ]

        scored = []
        for chunk in self.chunks:
            chunk_text_lower = chunk.text.lower()
            # Count how many important words appear
            matches = sum(
                1 for word in query_words
                if word in chunk_text_lower
            )
            scored.append((chunk, matches))

        # Sort by matches (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_k]]

    def retrieve_by_random(
        self,
        query: str,  # Unused, just for interface compatibility
        top_k: int = 3,
    ) -> List[DocumentChunk]:
        """
        Random Baseline: Return random chunks.

        Why random?
        - Worst-case baseline
        - Helps detect if system is better than guessing
        - Sanity check for evaluation

        Args:
            query: User question (ignored)
            top_k: Number of chunks to return

        Returns:
            List of random DocumentChunk objects
        """
        if len(self.chunks) <= top_k:
            return self.chunks.copy()
        return random.sample(self.chunks, k=top_k)

    def retrieve_by_chunk_order(
        self,
        query: str,  # Unused, just for interface compatibility
        top_k: int = 3,
    ) -> List[DocumentChunk]:
        """
        Positional Baseline: Return first K chunks (document order).

        Why document order?
        - Represents sequential reading
        - Shows if intelligent retrieval helps
        - Baseline for "just read from start"

        Args:
            query: User question (ignored)
            top_k: Number of chunks to return

        Returns:
            First top_k chunks by page order
        """
        return self.chunks[:top_k]


class RetrievalBaselines:
    """
    Wrapper to run multiple baselines for comparison.
    """

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.retriever = BaselineRetriever(chunks)

    def retrieve_all_baselines(
        self,
        query: str,
        top_k: int = 3,
    ) -> dict[str, List[DocumentChunk]]:
        """
        Run all baseline retrieval methods.

        Args:
            query: User question
            top_k: Number of chunks for each method

        Returns:
            Dict mapping baseline name to retrieved chunks
        """
        return {
            "keyword_overlap": self.retriever.retrieve_by_keyword_overlap(query, top_k),
            "important_words": self.retriever.retrieve_by_question_word_presence(query, top_k),
            "random": self.retriever.retrieve_by_random(query, top_k),
            "document_order": self.retriever.retrieve_by_chunk_order(query, top_k),
        }

    @staticmethod
    def chunks_to_text(chunks: List[DocumentChunk]) -> List[str]:
        """Convert chunks to text strings for evaluation."""
        return [chunk.text for chunk in chunks]
