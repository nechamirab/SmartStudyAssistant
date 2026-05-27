from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RetrievedContext:
    """Retrieved chunk plus retrieval metadata used for generation."""

    chunk_id: str
    text: str
    score: float
    source: str = ""
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """Citation rendered in answers and benchmark artifacts."""

    chunk_id: str
    label: str
    source: str
    page_number: int | None
    score: float


@dataclass(frozen=True)
class GenerationContext:
    """Inputs passed from retrieval into the generation layer."""

    question: str
    contexts: list[RetrievedContext]
    show_citations: bool = True


@dataclass(frozen=True)
class GenerationResult:
    """Structured grounded generation output."""

    answer: str
    citations: list[Citation]
    used_chunk_ids: list[str]
    confidence: float
    weak_context_warning: str | None = None
    provider: str = "mock"
    prompt: str = ""

    @property
    def confidence_level(self) -> str:
        """Human-readable confidence bucket for UI and JSON output."""
        if self.confidence >= 0.7:
            return "high"
        if self.confidence >= 0.35:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        """Structured answer payload used by the UI and demos."""
        return {
            "answer": self.answer,
            "citations": [citation.__dict__ for citation in self.citations],
            "used_chunks": self.used_chunk_ids,
            "confidence": self.confidence_level,
            "grounded": bool(self.citations or self.used_chunk_ids),
            "warning": self.weak_context_warning,
            "provider": self.provider,
        }


class LLMClient(Protocol):
    """Minimal interface for optional LLM-backed answer generation."""

    provider_name: str

    def generate(self, prompt: str) -> str:
        """Generate answer text from an already-grounded prompt."""
