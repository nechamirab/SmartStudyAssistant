from __future__ import annotations

import re
from typing import Any

from services.study_service import StudySection


class ExamGradingService:
    """Grade generated final exams and map misses back to study sections."""

    @classmethod
    def grade_exam(
        cls,
        exam: dict[str, Any],
        answers: dict[str, Any],
        sections: list[StudySection],
    ) -> dict[str, Any]:
        questions = [item for item in exam.get("questions", []) if isinstance(item, dict)]
        results: list[dict[str, Any]] = []
        weak_topics: list[str] = []
        weak_sections: list[str] = []
        correct_count = 0

        for question in questions:
            question_id = str(question.get("id", len(results) + 1))
            user_answer = str(answers.get(question_id, "") or "").strip()
            expected = str(question.get("answer", "") or "").strip()
            question_type = str(question.get("type", "short_answer"))
            is_correct = cls._is_correct(question_type, user_answer, expected)
            related_section = cls.related_section(question, sections)

            if is_correct:
                correct_count += 1
            else:
                topic = str(question.get("topic", "") or "Review").strip()
                if topic and topic not in weak_topics:
                    weak_topics.append(topic)
                if related_section and related_section.title not in weak_sections:
                    weak_sections.append(related_section.title)

            results.append(
                {
                    "id": question.get("id"),
                    "question": question.get("question", ""),
                    "type": question_type,
                    "user_answer": user_answer,
                    "expected_answer": expected,
                    "is_correct": is_correct,
                    "topic": question.get("topic", "Review"),
                    "related_section": related_section.title if related_section else "",
                }
            )

        total = max(1, len(questions))
        score = round(correct_count / total * 100)
        return {
            "score": score,
            "correct_count": correct_count,
            "wrong_count": len(questions) - correct_count,
            "total": len(questions),
            "weak_topics": weak_topics,
            "weak_sections": weak_sections,
            "results": results,
            "recommendation": cls.recommendation(weak_sections, weak_topics),
        }

    @staticmethod
    def recommendation(weak_sections: list[str], weak_topics: list[str]) -> str:
        if weak_sections:
            return f"Review {weak_sections[0]} first, then retry the missed questions."
        if weak_topics:
            return f"Review {weak_topics[0]} and write a one-sentence summary before retaking the exam."
        return "Good work. Revisit your notes briefly before moving on."

    @classmethod
    def related_section(cls, question: dict[str, Any], sections: list[StudySection]) -> StudySection | None:
        haystack = " ".join(
            str(question.get(key, "") or "")
            for key in ("question", "answer", "topic")
        )
        question_tokens = cls._tokens(haystack)
        if not question_tokens:
            return None

        best_section: StudySection | None = None
        best_score = 0
        for section in sections:
            section_text = " ".join(
                [
                    section.title,
                    section.summary,
                    " ".join(section.key_concepts),
                    " ".join(section.learning_objectives),
                ]
            )
            score = len(question_tokens & cls._tokens(section_text))
            if score > best_score:
                best_score = score
                best_section = section
        return best_section if best_score > 0 else None

    @classmethod
    def _is_correct(cls, question_type: str, user_answer: str, expected: str) -> bool:
        if not user_answer:
            return False
        if question_type in {"multiple_choice", "true_false"}:
            return cls._normalize(user_answer) == cls._normalize(expected)

        user_tokens = cls._tokens(user_answer)
        expected_tokens = cls._tokens(expected)
        if not expected_tokens:
            return bool(user_tokens)
        overlap = len(user_tokens & expected_tokens)
        return overlap >= max(1, min(3, round(len(expected_tokens) * 0.4)))

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip().lower()

    @staticmethod
    def _tokens(value: str) -> set[str]:
        stop_words = {
            "about", "answer", "based", "explain", "from", "idea", "important",
            "material", "question", "review", "section", "study", "this", "with",
        }
        return {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", (value or "").lower())
            if token not in stop_words
        }
