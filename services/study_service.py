from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StudySection:
    section_number: int
    title: str
    start_page: int
    end_page: int
    estimated_minutes: int
    difficulty: str
    summary: str
    learning_objectives: list[str] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)

    @property
    def page_label(self) -> str:
        if self.start_page == self.end_page:
            return f"Page {self.start_page}"
        return f"Pages {self.start_page}-{self.end_page}"


class StudyService:
    """Create a stable, human-readable study roadmap from extracted PDF pages."""

    STOP_WORDS = {
        "about", "after", "also", "because", "between", "chapter", "could", "every",
        "content", "from", "have", "into", "more", "page", "section", "should", "that",
        "their", "there", "these", "this", "through", "using", "were", "which",
        "with", "within", "would", "above", "below", "outlined", "source", "document",
        "material", "information", "example", "examples",
    }
    GENERIC_TITLE_WORDS = {"source page", "content", "section", "introduction", "document", "metadata"}
    GENERIC_SUMMARY_PATTERNS = {
        "what i have outlined",
        "the content above",
        "this content",
        "source page",
        "uploaded pdf",
        "chunk",
        "metadata",
    }

    def generate_study_plan(self, pages: list[Any], pages_per_section: int = 2) -> list[StudySection]:
        usable_pages = [page for page in pages if (getattr(page, "text", "") or "").strip()]
        if not usable_pages:
            return []

        sections: list[StudySection] = []
        for start in range(0, len(usable_pages), max(1, pages_per_section)):
            group = usable_pages[start : start + max(1, pages_per_section)]
            start_page = int(getattr(group[0], "page_number", start + 1))
            end_page = int(getattr(group[-1], "page_number", start_page))
            text = self._clean(" ".join(getattr(page, "text", "") or "" for page in group))
            concepts = self._key_concepts(text)
            sections.append(
                StudySection(
                    section_number=len(sections) + 1,
                    title=self._title(group, len(sections) + 1, start_page, end_page),
                    start_page=start_page,
                    end_page=end_page,
                    estimated_minutes=self._estimated_minutes(text),
                    difficulty=self._difficulty(text),
                    summary=self._summary(text, start_page, end_page),
                    learning_objectives=self._learning_objectives(text, concepts, start_page, end_page),
                    key_concepts=concepts,
                )
            )
        return sections

    def generate_study_plan_for_sessions(self, pages: list[Any], session_count: int) -> list[StudySection]:
        usable_pages = [page for page in pages if (getattr(page, "text", "") or "").strip()]
        if not usable_pages:
            return []

        target_count = max(1, min(int(session_count or 1), len(usable_pages)))
        base_size, extra = divmod(len(usable_pages), target_count)
        sections: list[StudySection] = []
        cursor = 0

        for index in range(target_count):
            group_size = base_size + (1 if index < extra else 0)
            group = usable_pages[cursor : cursor + group_size]
            cursor += group_size
            start_page = int(getattr(group[0], "page_number", cursor + 1))
            end_page = int(getattr(group[-1], "page_number", start_page))
            text = self._clean(" ".join(getattr(page, "text", "") or "" for page in group))
            concepts = self._key_concepts(text)
            sections.append(
                StudySection(
                    section_number=len(sections) + 1,
                    title=self._title(group, len(sections) + 1, start_page, end_page),
                    start_page=start_page,
                    end_page=end_page,
                    estimated_minutes=self._estimated_minutes(text),
                    difficulty=self._difficulty(text),
                    summary=self._summary(text, start_page, end_page),
                    learning_objectives=self._learning_objectives(text, concepts, start_page, end_page),
                    key_concepts=concepts,
                )
            )
        return sections

    @staticmethod
    def readable_page_count(pages: list[Any]) -> int:
        return sum(1 for page in pages if (getattr(page, "text", "") or "").strip())

    @classmethod
    def suggest_session_count(cls, text_or_pages: str | list[Any], page_count: int | None = None) -> int:
        text = cls._suggestion_text(text_or_pages)
        words = re.findall(r"\b\w+\b", text)
        effective_page_count = page_count
        if effective_page_count is None and isinstance(text_or_pages, list):
            effective_page_count = cls.readable_page_count(text_or_pages)
        return cls.suggest_session_count_from_size(len(words), effective_page_count)

    @staticmethod
    def suggest_session_count_from_size(word_count: int, page_count: int | None = None) -> int:
        if word_count <= 0:
            suggested = 5
        elif word_count < 1500:
            suggested = 3
        elif word_count < 3000:
            suggested = 5
        elif word_count < 6000:
            suggested = 7
        elif word_count < 10000:
            suggested = 10
        elif word_count < 15000:
            suggested = 12
        else:
            suggested = 15

        if page_count is not None and page_count > 24:
            suggested = max(suggested, min(15, round(page_count / 3)))
        return max(3, min(15, suggested))

    @staticmethod
    def _suggestion_text(text_or_pages: str | list[Any]) -> str:
        if isinstance(text_or_pages, str):
            return text_or_pages
        return " ".join(getattr(page, "text", "") or "" for page in text_or_pages)

    @staticmethod
    def section_text(pages: list[Any], section: StudySection) -> str:
        selected = [
            getattr(page, "text", "") or ""
            for page in pages
            if section.start_page <= int(getattr(page, "page_number", 0) or 0) <= section.end_page
        ]
        return "\n\n".join(text.strip() for text in selected if text.strip())

    @staticmethod
    def next_section_index(current_index: int, total_sections: int) -> int:
        if total_sections <= 0:
            return 0
        return min(max(0, current_index) + 1, total_sections - 1)

    def _title(self, pages: list[Any], section_number: int, start_page: int, end_page: int) -> str:
        for page in pages:
            for raw_line in (getattr(page, "text", "") or "").splitlines()[:12]:
                line = self._clean(raw_line)
                if self._looks_like_heading(line):
                    title = self._title_case(line)
                    if self._is_meaningful_title(title):
                        return f"Section {section_number}: {title}"

        text = self._clean(" ".join(getattr(page, "text", "") or "" for page in pages))
        concepts = self._key_concepts(text)
        if concepts:
            return f"Section {section_number}: {', '.join(concepts[:2])}"
        return f"Section {section_number}: Pages {start_page}-{end_page}"

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        if not line or len(line) > 90:
            return False
        words = line.split()
        if len(words) < 2 or len(words) > 10:
            return False
        if re.match(r"^\d+(\.\d+)*\s+\w+", line):
            return True
        letters = re.sub(r"[^A-Za-z]", "", line)
        return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) > 0.45

    @classmethod
    def _is_meaningful_title(cls, title: str) -> bool:
        normalized = re.sub(r"[^a-z0-9 ]+", "", (title or "").lower()).strip()
        if not normalized:
            return False
        if normalized in cls.GENERIC_TITLE_WORDS:
            return False
        if normalized.startswith("source page"):
            return False
        if len(normalized.split()) == 1 and normalized in {"section", "content", "introduction"}:
            return False
        return True

    def _key_concepts(self, text: str, limit: int = 5) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]{4,}", text)
        counts: dict[str, int] = {}
        display: dict[str, str] = {}
        positions: dict[str, int] = {}
        for position, word in enumerate(words):
            key = word.lower().strip("-")
            if key in self.STOP_WORDS or len(key) < 5:
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, self._title_case(word))
            positions.setdefault(key, position)
        ranked = sorted(counts, key=lambda item: (-counts[item], positions[item]))
        return [display[key] for key in ranked[:limit]]

    @staticmethod
    def _summary(text: str, start_page: int, end_page: int) -> str:
        sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cleaned = re.sub(r"\s+", " ", sentence or "").strip()
            lowered = cleaned.lower()
            if len(cleaned.split()) < 6:
                continue
            if any(pattern in lowered for pattern in StudyService.GENERIC_SUMMARY_PATTERNS):
                continue
            if re.search(r"\b(chunk|metadata|source|file name)\b", lowered):
                continue
            sentences.append(cleaned)
        if sentences:
            summary = " ".join(sentences[:3])
        else:
            summary = f"Study the main ideas and examples from pages {start_page}-{end_page}."
        return summary[:420].rstrip() + ("..." if len(summary) > 420 else "")

    @staticmethod
    def _learning_objectives(text: str, concepts: list[str], start_page: int, end_page: int) -> list[str]:
        selected = concepts[:4]
        if selected:
            objectives = [f"Explain {concept} using the section examples." for concept in selected[:2]]
            if len(selected) >= 3:
                objectives.append(f"Compare how {selected[1]} and {selected[2]} appear in the material.")
            objectives.append(f"Summarize the main argument from pages {start_page}-{end_page}.")
        else:
            objectives = [
                f"Summarize the main ideas from pages {start_page}-{end_page}.",
                "Identify the terms or examples that are most likely to appear on a quiz.",
                "Explain the section in your own words without looking at the PDF.",
            ]
        return objectives[:5]

    @staticmethod
    def _estimated_minutes(text: str) -> int:
        words = len(text.split())
        return max(8, min(45, round(words / 140) * 5 or 10))

    @staticmethod
    def _difficulty(text: str) -> str:
        words = text.split()
        if not words:
            return "Easy"
        avg_word_length = sum(len(word) for word in words) / len(words)
        long_words = sum(1 for word in words if len(word) >= 10)
        density = long_words / len(words)
        if avg_word_length > 6.3 or density > 0.22:
            return "Hard"
        if avg_word_length > 5.2 or density > 0.12:
            return "Medium"
        return "Easy"

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _title_case(text: str) -> str:
        text = re.sub(r"^\d+(\.\d+)*\s*", "", text).strip(" -:")
        return " ".join(word.capitalize() if not word.isupper() else word for word in text.split())
