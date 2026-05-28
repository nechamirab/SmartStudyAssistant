from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from core.models import DocumentChunk
from vectorstores.base import BaseVectorStore, SearchResult


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    method: str
    diagnostics: dict[str, float]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


class BM25Retriever:
    """BM25 keyword retriever for lexical baseline and hybrid fusion."""

    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0.0
        )
        self.doc_freq = self._document_frequencies(self.doc_tokens)

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored: list[ScoredChunk] = []
        query_counts = Counter(query_terms)
        for chunk, tokens, doc_len in zip(self.chunks, self.doc_tokens, self.doc_lengths):
            tf = Counter(tokens)
            score = 0.0
            for term, qf in query_counts.items():
                if term not in tf:
                    continue
                idf = self._idf(term)
                numerator = tf[term] * (self.k1 + 1)
                denominator = tf[term] + self.k1 * (
                    1 - self.b + self.b * doc_len / (self.avg_doc_len or 1)
                )
                score += idf * numerator / denominator * qf
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    method="bm25",
                    diagnostics={"bm25": score},
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    @staticmethod
    def _document_frequencies(docs: Iterable[list[str]]) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in docs:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies


class HybridRetriever:
    """Weighted retrieval fusion over semantic vector search and BM25."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def search(
        self,
        query_vector: list[float],
        query: str,
        top_k: int = 3,
        candidate_k: int | None = None,
    ) -> list[ScoredChunk]:
        candidate_k = candidate_k or max(top_k * 4, top_k)
        semantic = self.vector_store.search(query_vector, top_k=candidate_k)
        keyword = self.bm25_retriever.search(query, top_k=candidate_k)

        semantic_norm = self._normalize_semantic(semantic)
        keyword_norm = self._normalize_scored(keyword)
        by_id: dict[str, ScoredChunk] = {}

        for result in semantic:
            chunk_id = result.chunk.chunk_id
            by_id[chunk_id] = ScoredChunk(
                chunk=result.chunk,
                score=self.semantic_weight * semantic_norm.get(chunk_id, 0.0),
                method="hybrid",
                diagnostics={
                    "semantic": result.score,
                    "semantic_norm": semantic_norm.get(chunk_id, 0.0),
                    "bm25": 0.0,
                    "bm25_norm": 0.0,
                },
            )

        for result in keyword:
            chunk_id = result.chunk.chunk_id
            existing = by_id.get(chunk_id)
            keyword_score = self.keyword_weight * keyword_norm.get(chunk_id, 0.0)
            if existing:
                by_id[chunk_id] = ScoredChunk(
                    chunk=existing.chunk,
                    score=existing.score + keyword_score,
                    method="hybrid",
                    diagnostics={
                        **existing.diagnostics,
                        "bm25": result.score,
                        "bm25_norm": keyword_norm.get(chunk_id, 0.0),
                    },
                )
            else:
                by_id[chunk_id] = ScoredChunk(
                    chunk=result.chunk,
                    score=keyword_score,
                    method="hybrid",
                    diagnostics={
                        "semantic": 0.0,
                        "semantic_norm": 0.0,
                        "bm25": result.score,
                        "bm25_norm": keyword_norm.get(chunk_id, 0.0),
                    },
                )

        fused = list(by_id.values())
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    @staticmethod
    def _normalize_semantic(results: list[SearchResult]) -> dict[str, float]:
        return HybridRetriever._minmax(
            {result.chunk.chunk_id: result.score for result in results}
        )

    @staticmethod
    def _normalize_scored(results: list[ScoredChunk]) -> dict[str, float]:
        return HybridRetriever._minmax(
            {result.chunk.chunk_id: result.score for result in results}
        )

    @staticmethod
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        min_score = min(scores.values())
        max_score = max(scores.values())
        if math.isclose(max_score, min_score):
            return {key: 1.0 for key in scores}
        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }
