from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from services.general_ai_service import GeneralAIService
from translations import normalize_language, study_plan_language_instruction

logger = logging.getLogger(__name__)


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
    time_explanation: str = ""
    time_breakdown: dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    workload_score: float = 0.0

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

    @staticmethod
    def build_study_plan_prompt(language: str = "en") -> str:
        return study_plan_language_instruction(language)

    def generate_study_plan(
        self,
        pages: list[Any],
        pages_per_section: int = 2,
        language: str = "en",
    ) -> list[StudySection]:
        language = normalize_language(language)
        usable_pages = [page for page in pages if (getattr(page, "text", "") or "").strip()]
        if not usable_pages:
            return []

        group_size = max(1, pages_per_section)
        target_count = max(1, (len(usable_pages) + group_size - 1) // group_size)
        ai_sections = self._generate_ai_study_plan_for_sessions(usable_pages, target_count, language)
        if ai_sections:
            return ai_sections

        sections: list[StudySection] = []
        for group in self._group_pages_by_workload(usable_pages, target_count):
            sections.append(self._build_section_from_pages(group, len(sections) + 1, language))
        return sections

    def generate_study_plan_for_sessions(
        self,
        pages: list[Any],
        session_count: int,
        language: str = "en",
    ) -> list[StudySection]:
        language = normalize_language(language)
        usable_pages = [page for page in pages if (getattr(page, "text", "") or "").strip()]
        if not usable_pages:
            return []

        target_count = max(1, min(int(session_count or 1), len(usable_pages)))
        ai_sections = self._generate_ai_study_plan_for_sessions(usable_pages, target_count, language)
        if ai_sections:
            return ai_sections

        sections: list[StudySection] = []
        for group in self._group_pages_by_workload(usable_pages, target_count):
            sections.append(self._build_section_from_pages(group, len(sections) + 1, language))
        return sections

    def _build_section_from_pages(
        self,
        pages: list[Any],
        section_number: int,
        language: str = "en",
    ) -> StudySection:
        start_page = int(getattr(pages[0], "page_number", section_number))
        end_page = int(getattr(pages[-1], "page_number", start_page))
        text = self._clean(" ".join(getattr(page, "text", "") or "" for page in pages))
        concepts = self._key_concepts(text)
        objectives = self._learning_objectives(text, concepts, start_page, end_page, language)
        difficulty = self._difficulty(text)
        estimate = self.estimate_section_minutes(text, concepts, objectives, difficulty)
        return StudySection(
            section_number=section_number,
            title=self._title(pages, section_number, start_page, end_page, language),
            start_page=start_page,
            end_page=end_page,
            estimated_minutes=int(estimate["estimated_minutes"]),
            difficulty=difficulty,
            summary=self._summary(text, start_page, end_page, language),
            learning_objectives=objectives,
            key_concepts=concepts,
            time_explanation=str(estimate["explanation"]),
            time_breakdown=dict(estimate),
            word_count=int(estimate["word_count"]),
            workload_score=float(estimate["workload_score"]),
        )

    @classmethod
    def estimate_section_minutes(
        cls,
        text: str,
        key_concepts: list[str] | None = None,
        learning_objectives: list[str] | None = None,
        difficulty: str | None = None,
    ) -> dict[str, Any]:
        word_count = len(re.findall(r"\b[\w\u0590-\u05ff]+\b", text or ""))
        concept_count = len(key_concepts or [])
        objective_count = len(learning_objectives or [])
        normalized_difficulty = cls._normalize_difficulty(difficulty or "", text or "")
        difficulty_multiplier = {
            "Easy": 1.0,
            "Medium": 1.2,
            "Hard": 1.45,
        }.get(normalized_difficulty, 1.2)

        reading_minutes = word_count / 160
        concept_minutes = concept_count * 2
        objective_minutes = objective_count * 1.5
        practice_minutes = 5
        workload_score = (
            reading_minutes + concept_minutes + objective_minutes + practice_minutes
        ) * difficulty_multiplier
        max_minutes = 90 if word_count >= 5000 else 60
        estimated_minutes = cls._clamp(
            cls._round_to_nearest_five(workload_score),
            minimum=8,
            maximum=max_minutes,
        )

        explanation = (
            f"Estimated {estimated_minutes} min: "
            f"{round(reading_minutes)} min reading + "
            f"{round(concept_minutes)} min concepts + "
            f"{round(objective_minutes)} min objectives + "
            f"{practice_minutes} min practice, {normalized_difficulty} difficulty."
        )
        return {
            "word_count": word_count,
            "reading_minutes": round(reading_minutes, 2),
            "concept_minutes": round(concept_minutes, 2),
            "objective_minutes": round(objective_minutes, 2),
            "practice_minutes": practice_minutes,
            "difficulty": normalized_difficulty,
            "difficulty_multiplier": difficulty_multiplier,
            "estimated_minutes": int(estimated_minutes),
            "workload_score": round(workload_score, 2),
            "explanation": explanation,
        }

    @classmethod
    def _group_pages_by_workload(cls, pages: list[Any], target_count: int) -> list[list[Any]]:
        if not pages:
            return []
        target_count = max(1, min(int(target_count or 1), len(pages)))
        heading_groups = cls._heading_page_groups(pages, target_count)
        if heading_groups:
            return heading_groups
        return cls._workload_page_groups(pages, target_count)

    @classmethod
    def _workload_page_groups(cls, pages: list[Any], target_count: int) -> list[list[Any]]:
        if target_count <= 1 or len(pages) <= 1:
            return [pages]

        page_workloads = [cls._page_workload(page) for page in pages]
        total_workload = sum(page_workloads) or len(pages)
        target_workload = total_workload / target_count
        groups: list[list[Any]] = []
        current: list[Any] = []
        current_workload = 0.0

        for index, page in enumerate(pages):
            current.append(page)
            current_workload += page_workloads[index]
            remaining_pages = len(pages) - index - 1
            remaining_groups = target_count - len(groups) - 1
            should_close = (
                remaining_groups > 0
                and current_workload >= target_workload
                and remaining_pages >= remaining_groups
            )
            if should_close:
                groups.append(current)
                current = []
                current_workload = 0.0

        if current:
            groups.append(current)

        groups = cls._split_until_target_count(groups, page_workloads, pages, target_count)
        return cls._merge_tiny_last_group(groups, target_workload)

    @classmethod
    def _heading_page_groups(cls, pages: list[Any], target_count: int) -> list[list[Any]]:
        heading_indexes = [
            index
            for index, page in enumerate(pages)
            if cls._page_starts_with_heading(getattr(page, "text", "") or "")
        ]
        if 0 not in heading_indexes:
            heading_indexes.insert(0, 0)
        heading_indexes = sorted(dict.fromkeys(heading_indexes))
        if len(heading_indexes) <= 1:
            return []

        groups = [
            pages[start_index : (heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(pages))]
            for position, start_index in enumerate(heading_indexes)
        ]
        groups = [group for group in groups if group]
        if len(groups) > target_count:
            groups = cls._merge_groups_to_count(groups, target_count)

        workloads = [sum(cls._page_workload(page) for page in group) for group in groups]
        total_workload = sum(workloads)
        target_workload = total_workload / max(1, len(groups))
        if len(groups) >= 2 and all(workload >= target_workload * 0.35 for workload in workloads):
            return groups
        return []

    @classmethod
    def _merge_groups_to_count(cls, groups: list[list[Any]], target_count: int) -> list[list[Any]]:
        groups = [list(group) for group in groups if group]
        while len(groups) > target_count and len(groups) > 1:
            workloads = [sum(cls._page_workload(page) for page in group) for group in groups]
            merge_index = min(range(len(groups) - 1), key=lambda index: workloads[index] + workloads[index + 1])
            groups[merge_index].extend(groups.pop(merge_index + 1))
        return groups

    @classmethod
    def _split_until_target_count(
        cls,
        groups: list[list[Any]],
        page_workloads: list[float],
        pages: list[Any],
        target_count: int,
    ) -> list[list[Any]]:
        page_index_by_id = {id(page): index for index, page in enumerate(pages)}
        while len(groups) < target_count:
            split_index = -1
            split_workload = -1.0
            for index, group in enumerate(groups):
                if len(group) < 2:
                    continue
                workload = sum(page_workloads[page_index_by_id[id(page)]] for page in group)
                if workload > split_workload:
                    split_index = index
                    split_workload = workload
            if split_index < 0:
                break
            group = groups.pop(split_index)
            midpoint = max(1, len(group) // 2)
            groups.insert(split_index, group[:midpoint])
            groups.insert(split_index + 1, group[midpoint:])
        return groups

    @classmethod
    def _merge_tiny_last_group(cls, groups: list[list[Any]], target_workload: float) -> list[list[Any]]:
        if len(groups) < 3:
            return groups
        last_workload = sum(cls._page_workload(page) for page in groups[-1])
        if last_workload < max(2.0, target_workload * 0.35):
            groups[-2].extend(groups.pop())
        return groups

    @classmethod
    def _page_workload(cls, page: Any) -> float:
        text = getattr(page, "text", "") or ""
        word_count = len(re.findall(r"\b[\w\u0590-\u05ff]+\b", text))
        if word_count <= 0:
            return 0.0
        difficulty = cls._difficulty(text)
        multiplier = {"Easy": 1.0, "Medium": 1.2, "Hard": 1.45}.get(difficulty, 1.2)
        return max(1.0, (word_count / 160) * multiplier)

    @classmethod
    def _page_starts_with_heading(cls, text: str) -> bool:
        for raw_line in (text or "").splitlines()[:8]:
            line = cls._clean(raw_line)
            if cls._looks_like_natural_boundary_heading(line):
                return True
        return False

    @staticmethod
    def _looks_like_natural_boundary_heading(line: str) -> bool:
        if not line or len(line) > 100:
            return False
        if re.match(r"^(chapter|lecture|section|unit|module)\s+\d+(\.\d+)?\b", line, flags=re.IGNORECASE):
            return True
        if re.match(r"^\d+(\.\d+)+\s+[A-Za-z\u0590-\u05ff]", line):
            return True
        if re.match(r"^\d+\s+[A-Z][A-Za-z][^\n]{2,80}$", line):
            return True
        words = line.split()
        if 2 <= len(words) <= 9:
            letters = re.sub(r"[^A-Za-z]", "", line)
            if letters and sum(1 for char in letters if char.isupper()) / len(letters) > 0.65:
                return True
        return False

    @staticmethod
    def _round_to_nearest_five(value: float) -> int:
        return max(5, int(round(value / 5) * 5))

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, value))

    def _generate_ai_study_plan_for_sessions(
        self,
        usable_pages: list[Any],
        target_count: int,
        language: str = "en",
    ) -> list[StudySection]:
        context = self._ai_page_context(usable_pages)
        if not context:
            return []

        system_prompt = (
            "You create exam-focused study plans from extracted PDF text. "
            "Return only valid JSON that follows the requested schema. "
            "Do not include markdown, commentary, citations, or extra keys."
        )
        language_name = "Hebrew" if normalize_language(language) == "he" else "English"
        prompt = (
            "Create a study plan by grouping the PDF pages into coherent study sessions.\n"
            f"{study_plan_language_instruction(language)}\n"
            f"Return exactly {target_count} sections.\n"
            f"Write titles, summaries, key concepts, and learning objectives in {language_name}.\n"
            "Use only the provided page text.\n"
            "Prefer topic boundaries over equal page counts, but keep page ranges ordered and non-overlapping.\n"
            "Each section must have this shape:\n"
            "{\n"
            '  "section_number": 1,\n'
            '  "title": "short topic title",\n'
            '  "start_page": 1,\n'
            '  "end_page": 2,\n'
            '  "estimated_minutes": 25,\n'
            '  "difficulty": "Easy|Medium|Hard",\n'
            '  "summary": "2-3 sentence student-facing summary",\n'
            '  "key_concepts": ["concept 1", "concept 2"],\n'
            '  "learning_objectives": ["objective 1", "objective 2"]\n'
            "}\n"
            "Return a JSON object with one top-level key named sections.\n\n"
            f"Readable page numbers: {', '.join(str(int(getattr(page, 'page_number', 0) or 0)) for page in usable_pages)}\n\n"
            f"PDF page text:\n{context}"
        )

        try:
            response = GeneralAIService().complete(
                system_prompt,
                prompt,
                language=language,
                response_format={"type": "json_object"},
            )
            if not response["ok"]:
                return []
            payload = self._parse_ai_json(response.get("answer", ""))
            return self._sections_from_ai_payload(payload, usable_pages, target_count, language)
        except Exception as exc:
            logger.debug("AI study-plan generation failed; using heuristic fallback: %s", exc)
            return []

    def _sections_from_ai_payload(
        self,
        payload: Any,
        usable_pages: list[Any],
        target_count: int,
        language: str = "en",
    ) -> list[StudySection]:
        if isinstance(payload, dict):
            raw_sections = payload.get("sections")
        else:
            raw_sections = payload
        if not isinstance(raw_sections, list) or len(raw_sections) != target_count:
            return []

        readable_page_numbers = sorted(
            int(getattr(page, "page_number", 0) or 0)
            for page in usable_pages
            if int(getattr(page, "page_number", 0) or 0) > 0
        )
        if not readable_page_numbers:
            return []

        first_page = readable_page_numbers[0]
        last_page = readable_page_numbers[-1]
        sections: list[StudySection] = []
        previous_end = first_page - 1
        covered_readable_pages: set[int] = set()

        for index, item in enumerate(raw_sections, start=1):
            if not isinstance(item, dict):
                return []

            start_page = self._coerce_int(item.get("start_page"), default=0)
            end_page = self._coerce_int(item.get("end_page"), default=0)
            if start_page < first_page or end_page > last_page or start_page > end_page:
                return []
            if start_page <= previous_end:
                return []
            pages_in_range = [
                page_number for page_number in readable_page_numbers if start_page <= page_number <= end_page
            ]
            if not pages_in_range:
                return []
            covered_readable_pages.update(pages_in_range)

            section_text = self._clean(
                " ".join(
                    getattr(page, "text", "") or ""
                    for page in usable_pages
                    if start_page <= int(getattr(page, "page_number", 0) or 0) <= end_page
                )
            )
            concepts = self._clean_string_list(item.get("key_concepts"), limit=6) or self._key_concepts(section_text)
            title = self._clean(str(item.get("title", "") or ""))
            if not self._is_meaningful_title(title):
                title = self._title(
                    [
                        page
                        for page in usable_pages
                        if start_page <= int(getattr(page, "page_number", 0) or 0) <= end_page
                    ],
                    index,
                    start_page,
                    end_page,
                    language,
                )
            else:
                title = self._with_section_prefix(title, index, language)

            summary = self._clean(str(item.get("summary", "") or ""))
            if not summary:
                summary = self._summary(section_text, start_page, end_page, language)
            difficulty = self._normalize_difficulty(str(item.get("difficulty", "") or ""), section_text)
            objectives = (
                self._clean_string_list(item.get("learning_objectives"), limit=5)
                or self._learning_objectives(section_text, concepts, start_page, end_page, language)
            )
            estimate = self.estimate_section_minutes(section_text, concepts, objectives, difficulty)

            sections.append(
                StudySection(
                    section_number=index,
                    title=title,
                    start_page=start_page,
                    end_page=end_page,
                    estimated_minutes=int(estimate["estimated_minutes"]),
                    difficulty=difficulty,
                    summary=summary[:420].rstrip() + ("..." if len(summary) > 420 else ""),
                    learning_objectives=objectives,
                    key_concepts=concepts,
                    time_explanation=str(estimate["explanation"]),
                    time_breakdown=dict(estimate),
                    word_count=int(estimate["word_count"]),
                    workload_score=float(estimate["workload_score"]),
                )
            )
            previous_end = end_page

        if covered_readable_pages != set(readable_page_numbers):
            return []
        return sections

    @classmethod
    def _parse_ai_json(cls, raw: str) -> Any:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start_candidates = [position for position in (text.find("{"), text.find("[")) if position >= 0]
            if not start_candidates:
                raise
            start = min(start_candidates)
            end = max(text.rfind("}"), text.rfind("]"))
            if end <= start:
                raise
            return json.loads(text[start : end + 1])

    @classmethod
    def _ai_page_context(cls, pages: list[Any], max_chars: int = 14000) -> str:
        if not pages:
            return ""
        per_page = max(120, min(1200, max_chars // max(1, len(pages))))
        parts = []
        for page in pages:
            page_number = int(getattr(page, "page_number", 0) or 0)
            text = cls._clean(getattr(page, "text", "") or "")
            if not text:
                continue
            parts.append(f"Page {page_number}\n{text[:per_page]}")
        return "\n\n".join(parts)[:max_chars]

    @staticmethod
    def _clean_string_list(raw: Any, limit: int = 5) -> list[str]:
        if not isinstance(raw, list):
            return []
        cleaned: list[str] = []
        for item in raw:
            value = re.sub(r"\s+", " ", str(item or "")).strip()
            if value and value not in cleaned:
                cleaned.append(value[:140])
            if len(cleaned) >= limit:
                break
        return cleaned

    @staticmethod
    def _coerce_int(raw: Any, default: int = 0) -> int:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_difficulty(raw: str, text: str) -> str:
        value = (raw or "").strip().lower()
        if value in {"easy", "medium", "hard"}:
            return value.capitalize()
        return StudyService._difficulty(text)

    @staticmethod
    def _with_section_prefix(title: str, section_number: int, language: str = "en") -> str:
        title = re.sub(r"^\d+(\.\d+)*\s*", "", title).strip(" -:")
        if normalize_language(language) == "he":
            if title.startswith("חלק"):
                return title
            return f"חלק {section_number}: {title}"
        if re.match(r"^section\s+\d+\s*:", title, flags=re.IGNORECASE):
            return title
        return f"Section {section_number}: {title}"

    @staticmethod
    def readable_page_count(pages: list[Any]) -> int:
        return sum(1 for page in pages if (getattr(page, "text", "") or "").strip())

    @classmethod
    def suggest_session_count(cls, text_or_pages: str | list[Any], page_count: int | None = None) -> int:
        text = cls._suggestion_text(text_or_pages)
        effective_page_count = page_count
        if effective_page_count is None and isinstance(text_or_pages, list):
            effective_page_count = cls.readable_page_count(text_or_pages)
        return cls.suggest_session_count_from_size(
            cls._total_workload_minutes(text),
            effective_page_count,
        )

    @staticmethod
    def suggest_session_count_from_size(total_workload_minutes: int | float, page_count: int | None = None) -> int:
        if total_workload_minutes <= 0:
            suggested = 1
        else:
            suggested = ceil(total_workload_minutes / 30)

        maximum = 15
        if page_count is not None and page_count > 0:
            maximum = min(maximum, max(1, int(page_count)))
        return max(1, min(maximum, suggested))

    @classmethod
    def _total_workload_minutes(cls, text: str) -> float:
        word_count = len(re.findall(r"\b[\w\u0590-\u05ff]+\b", text or ""))
        if word_count <= 0:
            return 0.0
        concept_count = min(10, len(cls()._key_concepts(text, limit=10)))
        practice_blocks = max(1, ceil(word_count / 1500))
        difficulty = cls._difficulty(text)
        multiplier = {"Easy": 1.0, "Medium": 1.2, "Hard": 1.45}.get(difficulty, 1.2)
        return ((word_count / 160) + (concept_count * 2) + (practice_blocks * 5)) * multiplier

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

    def _title(
        self,
        pages: list[Any],
        section_number: int,
        start_page: int,
        end_page: int,
        language: str = "en",
    ) -> str:
        section_prefix = "חלק" if normalize_language(language) == "he" else "Section"
        for page in pages:
            for raw_line in (getattr(page, "text", "") or "").splitlines()[:12]:
                line = self._clean(raw_line)
                if self._looks_like_heading(line):
                    title = self._title_case(line)
                    if self._is_meaningful_title(title):
                        return f"{section_prefix} {section_number}: {title}"

        text = self._clean(" ".join(getattr(page, "text", "") or "" for page in pages))
        concepts = self._key_concepts(text)
        if concepts:
            return f"{section_prefix} {section_number}: {', '.join(concepts[:2])}"
        if normalize_language(language) == "he":
            return f"חלק {section_number}: עמודים {start_page}-{end_page}"
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
        words = re.findall(r"[A-Za-z\u0590-\u05FF][A-Za-z0-9\u0590-\u05FF-]{4,}", text)
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
    def _summary(text: str, start_page: int, end_page: int, language: str = "en") -> str:
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
        elif normalize_language(language) == "he":
            summary = f"למדו את הרעיונות המרכזיים והדוגמאות מעמודים {start_page}-{end_page}."
        else:
            summary = f"Study the main ideas and examples from pages {start_page}-{end_page}."
        return summary[:420].rstrip() + ("..." if len(summary) > 420 else "")

    @staticmethod
    def _learning_objectives(
        text: str,
        concepts: list[str],
        start_page: int,
        end_page: int,
        language: str = "en",
    ) -> list[str]:
        if normalize_language(language) == "he":
            selected = concepts[:4]
            if selected:
                objectives = [f"להסביר את {concept} בעזרת הדוגמאות בחלק." for concept in selected[:2]]
                if len(selected) >= 3:
                    objectives.append(f"להשוות כיצד {selected[1]} ו-{selected[2]} מופיעים בחומר.")
                objectives.append(f"לסכם את הרעיון המרכזי מעמודים {start_page}-{end_page}.")
            else:
                objectives = [
                    f"לסכם את הרעיונות המרכזיים מעמודים {start_page}-{end_page}.",
                    "לזהות מונחים או דוגמאות שסביר שיופיעו בשאלון.",
                    "להסביר את החלק במילים שלכם בלי להסתכל ב-PDF.",
                ]
            return objectives[:5]

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
        return int(StudyService.estimate_section_minutes(text)["estimated_minutes"])

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
