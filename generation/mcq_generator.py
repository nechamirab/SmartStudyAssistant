from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from generation.base import RetrievedContext
from generation.citation_formatter import CitationFormatter


@dataclass(frozen=True)
class MCQQuestion:
    """A multiple-choice question grounded in a retrieved/document chunk."""

    question: str
    options: list[str]
    correct_answer: str
    difficulty: str
    citation: dict

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "difficulty": self.difficulty,
            "citation": self.citation,
        }


class MCQGenerator:
    """Deterministic MCQ generator for offline demos and tests."""

    DIFFICULTY_PREFIX = {
        "easy": "What concept is described by this statement",
        "medium": "Which option best completes the study note",
        "hard": "Which statement is most strongly supported by the source",
    }

    def generate(
        self,
        contexts: list[RetrievedContext],
        count: int = 5,
        difficulty: str = "medium",
    ) -> list[MCQQuestion]:
        difficulty = difficulty if difficulty in self.DIFFICULTY_PREFIX else "medium"
        sentences = self._candidate_sentences(contexts)
        questions: list[MCQQuestion] = []

        for index, (sentence, context) in enumerate(sentences[:count], 1):
            answer = self._answer_phrase(sentence)
            distractors = self._distractors(answer, sentences, index)
            options = [answer, *distractors][:4]
            while len(options) < 4:
                options.append(f"Related concept {len(options)}")

            citation = CitationFormatter.citation_for(context, index)
            questions.append(
                MCQQuestion(
                    question=f"{self.DIFFICULTY_PREFIX[difficulty]}? ({index})",
                    options=options,
                    correct_answer=answer,
                    difficulty=difficulty,
                    citation=citation.__dict__,
                )
            )

        return questions

    @staticmethod
    def save_json(questions: list[MCQQuestion], path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([question.to_dict() for question in questions], indent=2))

    @staticmethod
    def save_markdown(questions: list[MCQQuestion], path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Generated Study Quiz\n\n"]
        for index, question in enumerate(questions, 1):
            lines.append(f"## Question {index}\n")
            lines.append(f"{question.question}\n\n")
            for option_index, option in enumerate(question.options, 1):
                lines.append(f"{option_index}. {option}\n")
            lines.append(f"\n**Correct answer:** {question.correct_answer}\n")
            lines.append(f"**Source:** {question.citation.get('label', 'source unavailable')}\n\n")
        target.write_text("".join(lines))

    def _candidate_sentences(
        self,
        contexts: list[RetrievedContext],
    ) -> list[tuple[str, RetrievedContext]]:
        candidates: list[tuple[str, RetrievedContext]] = []
        for context in contexts:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", context.text):
                clean = " ".join(sentence.split())
                if len(clean.split()) >= 7:
                    candidates.append((clean[:220], context))
        return candidates

    @staticmethod
    def _answer_phrase(sentence: str) -> str:
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", sentence)
        if len(words) <= 8:
            return " ".join(words)
        return " ".join(words[:8])

    @staticmethod
    def _distractors(
        answer: str,
        sentences: list[tuple[str, RetrievedContext]],
        offset: int,
    ) -> list[str]:
        phrases: list[str] = []
        for sentence, _context in sentences[offset:]:
            phrase = MCQGenerator._answer_phrase(sentence)
            if phrase and phrase.lower() != answer.lower() and phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) == 3:
                break
        fallback = ["The opposite conclusion", "An unrelated definition", "A random example"]
        return (phrases + fallback)[:3]
