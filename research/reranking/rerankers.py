from __future__ import annotations

from typing import Protocol, Sequence

from retrieval.hybrid import ScoredChunk, tokenize


class BaseReranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return candidates ordered by answer usefulness."""


class HeuristicReranker:
    """Dependency-free reranker using lexical coverage as a transparent fallback."""

    name = "heuristic"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        query_terms = set(tokenize(query))
        rescored: list[ScoredChunk] = []
        for candidate in candidates:
            chunk_terms = set(tokenize(candidate.chunk.text))
            coverage = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0
            score = candidate.score + coverage
            rescored.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=score,
                    method=f"{candidate.method}+heuristic_rerank",
                    diagnostics={**candidate.diagnostics, "query_term_coverage": coverage},
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]


class CrossEncoderReranker:
    """SentenceTransformers CrossEncoder reranker with optional BGE models."""

    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        model = self._load_model()
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = model.predict(pairs)
        rescored = []
        for candidate, score in zip(candidates, scores):
            cross_score = float(score)
            rescored.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=cross_score,
                    method=f"{candidate.method}+cross_encoder",
                    diagnostics={**candidate.diagnostics, "cross_encoder": cross_score},
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoder reranking."
            ) from e
        self._model = CrossEncoder(self.model_name)
        return self._model
