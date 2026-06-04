from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ChunkSpan:
    text: str
    start_char: int
    end_char: int
    parent_id: str | None = None
    metadata: dict | None = None


class ChunkingStrategy:
    """Base class for chunking strategies."""

    name = "base"

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        raise NotImplementedError


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()


class RecursiveCharacterTextSplitter(ChunkingStrategy):
    """Recursive-ish character splitter that prefers whitespace boundaries."""

    name = "recursive"

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []

        spans: list[ChunkSpan] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                cut = self._best_cut(text, start, end, chunk_size)
                if cut:
                    end = cut
            chunk_text = text[start:end].strip()
            if chunk_text:
                actual_start = start + len(text[start:end]) - len(text[start:end].lstrip())
                actual_end = actual_start + len(chunk_text)
                spans.append(ChunkSpan(chunk_text, actual_start, actual_end))
            if end >= text_length:
                break
            start = max(0, end - chunk_overlap)
        return spans

    @staticmethod
    def _best_cut(text: str, start: int, end: int, chunk_size: int) -> int | None:
        min_cut = start + int(chunk_size * 0.6)
        for separator in ("\n\n", "\n", ". ", " "):
            cut = text.rfind(separator, start, end)
            if cut > min_cut:
                return cut + len(separator.rstrip())
        return None


class SentenceChunker(ChunkingStrategy):
    """Group sentences into chunks that fit the target character budget."""

    name = "sentence"
    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []
        sentences = list(self._sentence_spans(text))
        spans: list[ChunkSpan] = []
        current: list[tuple[str, int, int]] = []
        current_len = 0

        for sentence, start, end in sentences:
            if current and current_len + len(sentence) + 1 > chunk_size:
                spans.append(self._join(current))
                current = self._overlap_tail(current, chunk_overlap)
                current_len = sum(len(item[0]) for item in current)
            current.append((sentence, start, end))
            current_len += len(sentence) + 1

        if current:
            spans.append(self._join(current))
        return spans

    def _sentence_spans(self, text: str) -> Iterable[tuple[str, int, int]]:
        cursor = 0
        for sentence in self.SENTENCE_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            start = text.find(sentence, cursor)
            end = start + len(sentence)
            cursor = end
            yield sentence, start, end

    @staticmethod
    def _join(parts: list[tuple[str, int, int]]) -> ChunkSpan:
        text = " ".join(part[0] for part in parts)
        return ChunkSpan(text=text, start_char=parts[0][1], end_char=parts[-1][2])

    @staticmethod
    def _overlap_tail(
        parts: list[tuple[str, int, int]],
        overlap_chars: int,
    ) -> list[tuple[str, int, int]]:
        if overlap_chars <= 0:
            return []
        tail: list[tuple[str, int, int]] = []
        total = 0
        for part in reversed(parts):
            tail.insert(0, part)
            total += len(part[0])
            if total >= overlap_chars:
                break
        return tail


class SlidingWindowChunker(ChunkingStrategy):
    """Fixed-width character windows for controlled ablation studies."""

    name = "sliding_window"

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []
        step = max(1, chunk_size - chunk_overlap)
        spans: list[ChunkSpan] = []
        for start in range(0, len(text), step):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                spans.append(ChunkSpan(chunk_text, start, end))
            if end >= len(text):
                break
        return spans


class TokenAwareChunker(ChunkingStrategy):
    """Whitespace-token chunker that approximates token budgets offline."""

    name = "token"

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []
        tokens = list(re.finditer(r"\S+", text))
        # Roughly four characters per token when no tokenizer is installed.
        max_tokens = max(1, chunk_size // 4)
        overlap_tokens = min(max_tokens - 1, max(0, chunk_overlap // 4))
        step = max(1, max_tokens - overlap_tokens)
        spans: list[ChunkSpan] = []
        for start_idx in range(0, len(tokens), step):
            window = tokens[start_idx:start_idx + max_tokens]
            if not window:
                break
            start = window[0].start()
            end = window[-1].end()
            spans.append(ChunkSpan(text[start:end], start, end))
            if start_idx + max_tokens >= len(tokens):
                break
        return spans


class ParentChildChunker(ChunkingStrategy):
    """Create child chunks and annotate them with a larger parent window."""

    name = "parent_child"

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        child_splitter = RecursiveCharacterTextSplitter()
        parent_size = max(chunk_size * 2, chunk_size + chunk_overlap)
        parents = child_splitter.split(text, parent_size, chunk_overlap)
        children: list[ChunkSpan] = []
        for parent_index, parent in enumerate(parents, 1):
            parent_id = f"parent_{parent_index}"
            for child in child_splitter.split(parent.text, chunk_size, chunk_overlap):
                children.append(
                    ChunkSpan(
                        text=child.text,
                        start_char=parent.start_char + child.start_char,
                        end_char=parent.start_char + child.end_char,
                        parent_id=parent_id,
                        metadata={"parent_text": parent.text},
                    )
                )
        return children


class SemanticChunker(ChunkingStrategy):
    """
    Lightweight semantic chunker.

    Full embedding-distance semantic splitting belongs in the research pipeline,
    but this offline implementation is a useful default: paragraphs remain intact
    and only overflow paragraphs are recursively split.
    """

    name = "semantic"

    def split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []
        recursive = RecursiveCharacterTextSplitter()
        spans: list[ChunkSpan] = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            start = text.find(paragraph)
            if len(paragraph) <= chunk_size:
                spans.append(ChunkSpan(paragraph, start, start + len(paragraph)))
            else:
                for span in recursive.split(paragraph, chunk_size, chunk_overlap):
                    spans.append(
                        ChunkSpan(
                            span.text,
                            start + span.start_char,
                            start + span.end_char,
                        )
                    )
        return spans


class ChunkingStrategyFactory:
    """Factory for named chunking strategies used by the PDF RAG pipeline."""

    STRATEGIES = {
        "recursive": RecursiveCharacterTextSplitter,
        "recursive_character": RecursiveCharacterTextSplitter,
        "sentence": SentenceChunker,
        "sentence_based": SentenceChunker,
        "sliding": SlidingWindowChunker,
        "sliding_window": SlidingWindowChunker,
        "token": TokenAwareChunker,
        "token_aware": TokenAwareChunker,
        "parent_child": ParentChildChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def create(cls, name: str | None) -> ChunkingStrategy:
        key = (name or "recursive").strip().lower()
        strategy_class = cls.STRATEGIES.get(key)
        if not strategy_class:
            supported = ", ".join(sorted(cls.STRATEGIES))
            raise ValueError(f"Unsupported chunking strategy '{name}'. Supported: {supported}")
        return strategy_class()
