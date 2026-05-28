from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional

from core.models import DocumentChunk


@dataclass(frozen=True)
class QuizQuestion:
    prompt: str
    options: List[str]
    answer: str
    explanation: Optional[str] = None


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
    def _build_options(correct: str, pool: List[str], num_options: int = 4) -> List[str]:
        options = {correct}
        pool = [w for w in pool if w.lower() != correct.lower() and len(w) > 3]
        random.shuffle(pool)
        for candidate in pool:
            if len(options) >= num_options:
                break
            options.add(candidate)
        options_list = list(options)
        random.shuffle(options_list)
        return options_list

    @classmethod
    def generate_mcq(cls, chunks: List[DocumentChunk], num_questions: int = 3) -> List[QuizQuestion]:
        """Generate simple multiple-choice questions from document text."""
        sentences = []
        for chunk in chunks:
            sentences.extend(cls._split_sentences(chunk.text))

        candidates = [s for s in sentences if len(s.split()) >= 8]
        if not candidates:
            return []

        all_keywords = []
        for sentence in candidates:
            keyword = cls._select_keyword(sentence)
            if keyword:
                all_keywords.append(keyword)

        questions: List[QuizQuestion] = []
        for sentence in random.sample(candidates, min(num_questions, len(candidates))):
            keyword = cls._select_keyword(sentence)
            if not keyword:
                continue

            question_text = sentence.replace(keyword, "_____", 1)
            options = cls._build_options(keyword, all_keywords, num_options=4)
            if len(options) < 2:
                continue

            questions.append(
                QuizQuestion(
                    prompt=f"Fill in the blank: {question_text}",
                    options=options,
                    answer=keyword,
                    explanation=f"The correct answer is '{keyword}'.",
                )
            )

            if len(questions) >= num_questions:
                break

        return questions
