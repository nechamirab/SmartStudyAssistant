from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass
from typing import Any, List, Optional

from core.models import DocumentChunk
from services.general_ai_service import GeneralAIService
from translations import normalize_language, quiz_language_instruction


@dataclass(frozen=True)
class QuizQuestion:
    prompt: str
    options: List[str]
    answer: str
    explanation: Optional[str] = None
    citation: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None


class QuizService:
    """Simple quiz generator for document-based multiple-choice questions."""

    STOP_WORDS = {
        "the", "and", "with", "that", "this", "those", "their", "about",
        "which", "there", "where", "while", "because", "through", "between",
    }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        return [sentence.strip() for sentence in re.split(r'(?<=[.?!])\s+', text) if sentence.strip()]

    @staticmethod
    def _select_keyword(sentence: str) -> Optional[str]:
        words = [w.strip(".,;:()[]\"'`)" ) for w in sentence.split()]
        candidates = [w for w in words if len(w) > 5 and w.lower() not in QuizService.STOP_WORDS]
        return max(candidates, key=len) if candidates else None

    @staticmethod
    def _build_options(correct: str, pool: List[str], num_options: int = 4, language: str = "en") -> List[str]:
        options = [correct]
        seen = {correct.lower()}
        for candidate in pool:
            normalized = candidate.lower()
            if normalized in seen or len(candidate) <= 3:
                continue
            seen.add(normalized)
            options.append(candidate)
            if len(options) >= num_options:
                break
        fallback_terms = (
            ["ניתוח", "מסמך", "מושג", "תהליך", "שיטה", "רשומה"]
            if normalize_language(language) == "he"
            else ["analysis", "document", "concept", "process", "method", "record"]
        )
        for fallback in fallback_terms:
            if len(options) >= num_options:
                break
            if fallback.lower() in seen:
                continue
            seen.add(fallback.lower())
            options.append(fallback)
        return options

    @staticmethod
    def _extract_doc_data(item: Any) -> dict[str, Any]:
        if isinstance(item, DocumentChunk):
            return {
                "text": item.text,
                "source": "Uploaded PDF",
                "page": item.page_number,
            }

        if isinstance(item, dict):
            return {
                "text": item.get("text", ""),
                "source": item.get("source", "Uploaded PDF"),
                "page": item.get("page"),
            }

        metadata = dict(getattr(item, "metadata", {}) or {})
        return {
            "text": getattr(item, "page_content", ""),
            "source": metadata.get("source", "Uploaded PDF"),
            "page": metadata.get("page"),
        }

    @classmethod
    def _documents_to_context(cls, documents: List[Any], max_chars: int = 6000) -> str:
        parts: List[str] = []

        for item in documents:
            doc_data = cls._extract_doc_data(item)
            text = re.sub(r"\s+", " ", doc_data["text"] or "").strip()
            if not text:
                continue

            source = doc_data["source"] or "Uploaded PDF"
            page = doc_data["page"]
            label = f"{source} p.{page}" if page else source
            parts.append(f"[{label}]\n{text}")

            if sum(len(part) for part in parts) >= max_chars:
                break

        return "\n\n".join(parts)[:max_chars]

    @staticmethod
    def build_quiz_prompt(language: str = "en") -> str:
        return quiz_language_instruction(language)

    @classmethod
    def _generate_ai_questions(cls, documents: List[Any], num_questions: int, language: str = "en") -> List[QuizQuestion]:
        language = normalize_language(language)
        context = cls._documents_to_context(documents)

        if not context:
            return []

        seed = random.randint(1000, 999999)

        prompt = (
            "Create multiple-choice quiz questions from the study material below.\n"
            f"{cls.build_quiz_prompt(language)}\n"
            "Use only the provided material.\n"
            "Create different questions each time, using the variation seed.\n"
            "Return only valid JSON, without markdown.\n"
            "The JSON must be a list of objects.\n"
            "Each object must contain: prompt, options, answer, explanation, citation.\n"
            "Each question must have exactly 4 options.\n"
            "The answer must exactly match one of the options.\n"
            "Avoid simple fill-in-the-blank questions.\n"
            "Prefer conceptual understanding questions.\n\n"
            f"Number of questions: {num_questions}\n"
            f"Variation seed: {seed}\n\n"
            f"Study material:\n{context}"
        )

        response = GeneralAIService().ask([], prompt, language=language)

        if not response["ok"]:
            return []

        raw_answer = response.get("answer", "").strip()

        try:
            data = json.loads(raw_answer)
        except json.JSONDecodeError:
            return []

        questions: List[QuizQuestion] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            prompt_text = str(item.get("prompt", "")).strip()
            options = item.get("options", [])
            answer = str(item.get("answer", "")).strip()

            if not prompt_text or not isinstance(options, list) or len(options) != 4:
                continue

            clean_options = [str(option).strip() for option in options if str(option).strip()]

            if len(clean_options) != 4 or answer not in clean_options:
                continue

            questions.append(
                QuizQuestion(
                    prompt=prompt_text,
                    options=clean_options,
                    answer=answer,
                    explanation=str(item.get("explanation", "")).strip() or None,
                    citation=str(item.get("citation", "Uploaded PDF")).strip(),
                    source="AI Generated",
                    page=None,
                )
            )

            if len(questions) >= num_questions:
                break

        return questions

    @classmethod
    def generate_mcq(cls, chunks: List[DocumentChunk], num_questions: int = 3, language: str = "en") -> List[QuizQuestion]:
        """Generate simple multiple-choice questions from legacy chunk objects."""
        return cls.generate_from_documents(chunks, num_questions=num_questions, language=language)

    @classmethod
    def generate_from_documents(
        cls,
        documents: List[Any],
        num_questions: int = 3,
        language: str = "en",
    ) -> List[QuizQuestion]:
        """Generate AI-based multiple-choice questions with deterministic fallback."""
        language = normalize_language(language)
        ai_questions = cls._generate_ai_questions(documents, num_questions, language=language)

        if ai_questions:
            return ai_questions

        sentence_rows: List[dict[str, Any]] = []
        for item in documents:
            doc_data = cls._extract_doc_data(item)
            for sentence in cls._split_sentences(doc_data["text"]):
                if len(sentence.split()) < 8:
                    continue
                sentence_rows.append(
                    {
                        "sentence": sentence,
                        "source": doc_data["source"],
                        "page": doc_data["page"],
                    }
                )

        if not sentence_rows:
            return []

        keyword_pool: List[str] = []
        for row in sentence_rows:
            keyword = cls._select_keyword(row["sentence"])
            if keyword:
                keyword_pool.append(keyword)

        questions: List[QuizQuestion] = []
        seen_prompts = set()
        for row in sentence_rows:
            sentence = row["sentence"]
            keyword = cls._select_keyword(sentence)
            if not keyword:
                continue

            prompt = (
                f"השלימו את החסר: {sentence.replace(keyword, '_____', 1)}"
                if language == "he"
                else f"Fill in the blank: {sentence.replace(keyword, '_____', 1)}"
            )
            if prompt in seen_prompts:
                continue

            options = cls._build_options(keyword, keyword_pool, num_options=4, language=language)
            if len(options) < 2:
                continue

            page = row["page"]
            source = row["source"] or "Uploaded PDF"
            citation = f"{source} p.{page}" if page else source

            questions.append(
                QuizQuestion(
                    prompt=prompt,
                    options=options,
                    answer=keyword,
                    explanation=(
                        f"התשובה הנכונה היא '{keyword}'."
                        if language == "he"
                        else f"The correct answer is '{keyword}'."
                    ),
                    citation=citation,
                    source=source,
                    page=page,
                )
            )
            seen_prompts.add(prompt)

            if len(questions) >= num_questions:
                break

        return questions
