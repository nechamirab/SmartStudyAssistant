from __future__ import annotations

import re
from typing import Any


class ContextRetrievalService:
    """Local lexical retrieval for selecting PDF context before AI calls."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
        "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "in", "into", "is", "it", "its", "of", "on", "or", "should",
        "that", "the", "their", "there", "these", "this", "those", "to",
        "was", "were", "what", "when", "where", "which", "why", "with",
        "would", "explain", "describe", "define", "tell", "about",
    }

    @classmethod
    def build_chunks_from_pages(cls, pages: list[Any], sections: list[Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        sorted_sections = sorted(
            sections or [],
            key=lambda section: int(getattr(section, "section_number", 0) or 0),
        )

        for page in pages or []:
            page_number = int(getattr(page, "page_number", 0) or 0)
            page_text = cls._clean_text(getattr(page, "text", "") or "")
            if not page_text:
                continue

            section = cls._section_for_page(page_number, sorted_sections)
            if section is None:
                section_number = 0
                section_title = "Unassigned section"
                start_page = page_number
                end_page = page_number
                key_concepts: list[str] = []
            else:
                section_number = int(getattr(section, "section_number", 0) or 0)
                section_title = str(getattr(section, "title", "") or f"Section {section_number}")
                start_page = int(getattr(section, "start_page", page_number) or page_number)
                end_page = int(getattr(section, "end_page", page_number) or page_number)
                key_concepts = list(getattr(section, "key_concepts", []) or [])

            for text_part in cls._split_text(page_text):
                chunks.append(
                    {
                        "section_number": section_number,
                        "section_title": section_title,
                        "start_page": start_page,
                        "end_page": end_page,
                        "page": page_number,
                        "text": text_part,
                        "key_concepts": key_concepts,
                    }
                )

        return chunks

    @classmethod
    def retrieve_relevant_chunks(
        cls,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        min_score: int = 1,
    ) -> list[dict[str, Any]]:
        query_tokens = cls._tokens(question)
        if not query_tokens:
            return []

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, chunk in enumerate(chunks or []):
            text_tokens = cls._tokens(str(chunk.get("text", "")))
            title_tokens = cls._tokens(str(chunk.get("section_title", "")))
            concept_tokens = cls._tokens(" ".join(str(item) for item in chunk.get("key_concepts", []) or []))

            overlap = len(query_tokens & text_tokens)
            title_overlap = len(query_tokens & title_tokens)
            concept_overlap = len(query_tokens & concept_tokens)
            score = overlap + (title_overlap * 3) + (concept_overlap * 2)

            if score >= min_score:
                ranked_chunk = dict(chunk)
                ranked_chunk["score"] = score
                scored.append((score, index, ranked_chunk))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[: max(1, int(top_k or 1))]]

    @staticmethod
    def format_chunks_for_prompt(chunks: list[dict[str, Any]], max_chars: int = 7000) -> str:
        parts: list[str] = []
        used = 0

        for chunk in chunks or []:
            header = (
                f"[Section {int(chunk.get('section_number', 0) or 0)} | "
                f"{chunk.get('section_title', 'Untitled section')} | "
                f"Page {int(chunk.get('page', 0) or 0)}]"
            )
            text = re.sub(r"\s+", " ", str(chunk.get("text", "") or "")).strip()
            if not text:
                continue

            block = f"{header}\n{text}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rsplit(" ", 1)[0].rstrip()
            if block:
                parts.append(block)
                used += len(block) + 2

        return "\n\n".join(parts).strip()

    @classmethod
    def retrieve_exam_context(cls, sections: list[Any], pages: list[Any], max_chars: int = 12000) -> str:
        if not sections or not pages or max_chars <= 0:
            return ""

        parts: list[str] = []
        used = 0
        ordered_sections = sorted(
            sections,
            key=lambda section: int(getattr(section, "section_number", 0) or 0),
        )
        section_count = max(1, len(ordered_sections))
        per_section_excerpt = max(300, min(1200, max_chars // section_count - 250))

        for section in ordered_sections:
            section_number = int(getattr(section, "section_number", 0) or 0)
            title = str(getattr(section, "title", "") or f"Section {section_number}")
            start_page = int(getattr(section, "start_page", 0) or 0)
            end_page = int(getattr(section, "end_page", start_page) or start_page)
            summary = str(getattr(section, "summary", "") or "").strip()
            concepts = ", ".join(str(item) for item in (getattr(section, "key_concepts", []) or [])[:8])
            section_text = cls._section_excerpt(pages, start_page, end_page, per_section_excerpt)

            block = "\n".join(
                item
                for item in [
                    f"Section {section_number}: {title}",
                    f"Pages: {start_page}-{end_page}" if start_page != end_page else f"Page: {start_page}",
                    f"Summary: {summary}" if summary else "",
                    f"Key concepts: {concepts}" if concepts else "",
                    f"Representative text: {section_text}" if section_text else "",
                ]
                if item
            )

            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rsplit(" ", 1)[0].rstrip()
            if block:
                parts.append(block)
                used += len(block) + 2

        return "\n\n".join(parts).strip()

    @classmethod
    def source_labels(cls, chunks: list[dict[str, Any]]) -> list[str]:
        labels: list[str] = []
        seen: set[tuple[int, int]] = set()
        for chunk in chunks or []:
            section_number = int(chunk.get("section_number", 0) or 0)
            page = int(chunk.get("page", 0) or 0)
            key = (section_number, page)
            if key in seen:
                continue
            seen.add(key)
            title = str(chunk.get("section_title", "") or "Untitled section")
            labels.append(f"Section {section_number}, Page {page}: {title}")
        return labels

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9\u0590-\u05ff]+", (value or "").lower())
            if len(token) > 1 and token not in cls.STOP_WORDS
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def _split_text(cls, text: str, max_chars: int = 1400, overlap: int = 160) -> list[str]:
        text = cls._clean_text(text)
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                split_at = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if split_at > start + max_chars // 2:
                    end = split_at + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _section_for_page(page_number: int, sections: list[Any]) -> Any | None:
        for section in sections:
            start_page = int(getattr(section, "start_page", 0) or 0)
            end_page = int(getattr(section, "end_page", start_page) or start_page)
            if start_page <= page_number <= end_page:
                return section
        return None

    @classmethod
    def _section_excerpt(cls, pages: list[Any], start_page: int, end_page: int, max_chars: int) -> str:
        text = " ".join(
            getattr(page, "text", "") or ""
            for page in pages
            if start_page <= int(getattr(page, "page_number", 0) or 0) <= end_page
        )
        return cls._clean_text(text)[:max_chars].rstrip()
