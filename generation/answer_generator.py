from __future__ import annotations

import re

from generation.base import (
    GenerationContext,
    GenerationResult,
    LLMClient,
    RetrievedContext,
)
from generation.citation_formatter import CitationFormatter
from generation.mock_llm import MockLLM
from generation.openai_llm import OpenAILLM
from generation.prompt_builder import PromptBuilder


class AnswerGenerator:
    """Generate grounded answers from retrieved contexts."""

    def __init__(
        self,
        llm_provider: str = "mock",
        show_citations: bool = True,
        weak_score_threshold: float = 0.05,
    ) -> None:
        self.llm_provider = llm_provider
        self.show_citations = show_citations
        self.weak_score_threshold = weak_score_threshold
        self.prompt_builder = PromptBuilder()
        self.llm = self._create_llm(llm_provider)

    def generate(
        self,
        question: str,
        contexts: list[RetrievedContext],
    ) -> GenerationResult:
        generation_context = GenerationContext(
            question=question,
            contexts=contexts,
            show_citations=self.show_citations,
        )
        prompt = self.prompt_builder.build(generation_context)
        weak_warning = self._weak_context_warning(contexts)
        used_contexts = self._select_used_contexts(contexts)
        citations = [
            CitationFormatter.citation_for(context, index)
            for index, context in enumerate(used_contexts, 1)
        ]

        llm_answer = ""
        if self.llm_provider == "openai":
            llm_answer = self.llm.generate(prompt).strip()

        answer = llm_answer or self._deterministic_answer(
            question=question,
            contexts=used_contexts,
        )

        if self.show_citations and citations:
            citation_block = CitationFormatter.render_citation_block(citations)
            if citation_block:
                answer = f"{answer}\n\n{citation_block}"

        return GenerationResult(
            answer=answer,
            citations=citations,
            used_chunk_ids=[context.chunk_id for context in used_contexts],
            confidence=self._confidence(contexts, weak_warning),
            weak_context_warning=weak_warning,
            provider=self.llm.provider_name,
            prompt=prompt,
        )

    @staticmethod
    def from_search_results(
        results,
        llm_provider: str = "mock",
        show_citations: bool = True,
    ) -> "AnswerGenerator":
        return AnswerGenerator(llm_provider=llm_provider, show_citations=show_citations)

    @staticmethod
    def contexts_from_search_results(results) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        for result in results:
            chunk = result.chunk
            contexts.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(result.score),
                    source=chunk.source_id or str(chunk.metadata.get("source", "") or ""),
                    page_number=chunk.page_number,
                    metadata=chunk.metadata,
                )
            )
        return contexts

    def _deterministic_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
    ) -> str:
        if not contexts:
            return (
                "I do not have enough retrieved context to answer this question "
                "without guessing."
            )

        selected_sentences: list[str] = []
        for index, context in enumerate(contexts, 1):
            sentence = self._best_sentence(question, context.text)
            if not sentence:
                continue
            marker = CitationFormatter.inline_marker(index) if self.show_citations else ""
            selected_sentences.append(f"{sentence} {marker}".strip())

        if not selected_sentences:
            selected_sentences = [
                (
                    f"The retrieved context discusses: {self._trim(context.text, 220)} "
                    f"{CitationFormatter.inline_marker(index) if self.show_citations else ''}"
                ).strip()
                for index, context in enumerate(contexts[:2], 1)
            ]

        answer = " ".join(selected_sentences)
        return answer

    def _select_used_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        if not contexts:
            return []
        positive = [context for context in contexts if context.score > 0]
        return positive[: min(3, len(positive))] or contexts[:1]

    def _weak_context_warning(self, contexts: list[RetrievedContext]) -> str | None:
        if not contexts:
            return "Warning: no context was retrieved, so the answer may be incomplete."
        best_score = max(context.score for context in contexts)
        if best_score < self.weak_score_threshold:
            return "Warning: retrieved context appears weak, so confidence is low."
        return None

    def _confidence(
        self,
        contexts: list[RetrievedContext],
        weak_warning: str | None,
    ) -> float:
        if not contexts:
            return 0.0
        best = max(context.score for context in contexts)
        normalized = max(0.0, min(1.0, best))
        if weak_warning:
            normalized *= 0.45
        return round(normalized, 3)

    @staticmethod
    def _best_sentence(question: str, text: str) -> str:
        question_terms = set(re.findall(r"[A-Za-z0-9_]+", question.lower()))
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        best_sentence = ""
        best_overlap = -1
        for sentence in sentences:
            clean = sentence.strip()
            if not clean:
                continue
            terms = set(re.findall(r"[A-Za-z0-9_]+", clean.lower()))
            overlap = len(question_terms & terms)
            if overlap > best_overlap:
                best_sentence = clean
                best_overlap = overlap
        return AnswerGenerator._trim(best_sentence, 260)

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        text = " ".join((text or "").split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _create_llm(provider: str) -> LLMClient:
        if (provider or "mock").lower() == "openai":
            openai_llm = OpenAILLM()
            if openai_llm.available():
                return openai_llm
        return MockLLM()
