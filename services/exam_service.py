from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from core.config import (
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    NOT_FOUND_ANSWER,
    read_groq_api_key,
)
from services.rag_service import PDFIndex


GROQ_LIMIT_MESSAGE = "Groq free API limit reached. Please try again later or reduce the number of questions."


class ExamGenerationError(Exception):
    """Raised when an AI quiz or exam cannot be generated from the uploaded PDF."""


@dataclass(frozen=True)
class ExamRequest:
    number_of_questions: int = 10
    question_types: list[str] = field(
        default_factory=lambda: ["multiple_choice", "true_false", "short_answer", "open_question"]
    )
    difficulty: str = "mixed"
    include_answer_key: bool = True

    # Backward-compatible inputs used by older API/UI tests. New code should use
    # number_of_questions and question_types.
    multiple_choice: int | None = None
    open_questions: int | None = None
    true_false: int | None = None
    short_answer: int | None = None

    def normalized(self) -> "ExamRequest":
        counts = {
            "multiple_choice": self.multiple_choice,
            "open_question": self.open_questions,
            "true_false": self.true_false,
            "short_answer": self.short_answer,
        }
        explicit_counts = {key: value for key, value in counts.items() if value is not None and value > 0}
        if explicit_counts:
            return ExamRequest(
                number_of_questions=sum(explicit_counts.values()),
                question_types=list(explicit_counts),
                difficulty=self.difficulty,
                include_answer_key=self.include_answer_key,
            )

        types = [self._normalize_type(value) for value in self.question_types if value]
        types = [value for value in types if value in FullExamService.QUESTION_TYPES]
        return ExamRequest(
            number_of_questions=max(1, min(int(self.number_of_questions), 40)),
            question_types=types or ["multiple_choice"],
            difficulty=self.difficulty,
            include_answer_key=bool(self.include_answer_key),
        )

    @staticmethod
    def _normalize_type(value: str) -> str:
        return value.strip().lower().replace("/", "_").replace(" ", "_")


class FullExamService:
    """Generate an AI Quiz / Exam with Groq using only retrieved PDF chunks."""

    DIFFICULTIES = {"easy", "medium", "hard", "mixed"}
    QUESTION_TYPES = {"multiple_choice", "true_false", "short_answer", "open_question"}
    GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, allow_test_fallback: bool = False) -> None:
        self.allow_test_fallback = allow_test_fallback

    def generate_exam(self, index: PDFIndex, request: ExamRequest) -> dict[str, Any]:
        if not index or not index.chunks:
            raise ExamGenerationError("Upload and process a PDF before generating an AI Quiz / Exam.")

        normalized = request.normalized()
        difficulty = normalized.difficulty if normalized.difficulty in self.DIFFICULTIES else "mixed"
        context = self._exam_context(index)
        if not context.strip():
            raise ExamGenerationError("The processed PDF does not contain enough text for an AI Quiz / Exam.")

        api_key = read_groq_api_key()
        if not api_key:
            if self.allow_test_fallback:
                return self._generate_test_fallback(index, normalized, difficulty)
            raise ExamGenerationError(
                "Groq API key is missing. Add it to config/groq_api_key.txt before generating an AI Quiz / Exam."
            )

        prompt = self.build_prompt(index.pdf_name, context, normalized, difficulty)
        return self._generate_with_groq(prompt, api_key)

    @staticmethod
    def build_prompt(
        pdf_name: str,
        context: str,
        request: ExamRequest,
        difficulty: str,
    ) -> str:
        schema = {
            "title": "AI Quiz / Exam title",
            "pdf_name": pdf_name,
            "grounding": "Use only uploaded PDF context",
            "fallback_message": NOT_FOUND_ANSWER,
            "questions": [
                {
                    "id": 1,
                    "type": "multiple_choice | true_false | short_answer | open_question",
                    "difficulty": "easy | medium | hard",
                    "question": "Question text",
                    "options": ["Only for multiple_choice"],
                    "answer": "Answer text or null when answer key is excluded",
                    "source_references": [{"page_number": 1, "chunk_id": "chunk-id"}],
                }
            ],
            "answer_key": [
                {
                    "id": 1,
                    "answer": "Answer text",
                    "source_references": [{"page_number": 1, "chunk_id": "chunk-id"}],
                }
            ],
        }
        return (
            "Create an AI Quiz / Exam using ONLY the uploaded PDF context from the retrieved PDF chunks below.\n"
            "Do not use outside knowledge or assumptions.\n"
            "If the context is not enough to create a requested question, include a question object whose "
            f"question says: {NOT_FOUND_ANSWER}\n"
            "Every real question must be based on specific information from the retrieved PDF chunks.\n"
            "Include source references where possible using page_number and chunk_id from the context labels.\n"
            "Return valid JSON only. Do not wrap it in Markdown.\n\n"
            f"PDF set: {pdf_name}\n"
            f"Number of questions: {request.number_of_questions}\n"
            f"Question types: {', '.join(request.question_types)}\n"
            f"Difficulty: {difficulty}\n"
            f"Include answer key: {request.include_answer_key}\n"
            f"Required JSON shape:\n{json.dumps(schema, indent=2)}\n\n"
            f"Retrieved PDF chunks:\n{context}"
        )

    def _generate_with_groq(self, prompt: str, api_key: str) -> dict[str, Any]:
        payload = {
            "model": GROQ_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate AI Quiz / Exam JSON only from retrieved PDF chunks. "
                        "Never add facts from outside the provided chunks."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            self.GROQ_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if self._is_groq_limit_error(exc.code, body):
                raise ExamGenerationError(GROQ_LIMIT_MESSAGE) from exc
            raise ExamGenerationError(f"Groq AI Quiz / Exam generation failed: {body or exc}") from exc
        except urllib.error.URLError as exc:
            raise ExamGenerationError(f"Groq AI Quiz / Exam generation failed: {exc}") from exc

        raw = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not raw:
            raise ExamGenerationError("Groq returned an empty AI Quiz / Exam.")

        payload = self._parse_json(raw)
        return self._normalize_payload(payload)

    def _exam_context(self, index: PDFIndex, max_chars: int = 12000) -> str:
        parts: list[str] = []
        total = 0
        for chunk in index.chunks:
            block = (
                f"[PDF: {chunk.source_id or index.pdf_name}; page_number: {chunk.page_number}; "
                f"chunk_id: {chunk.chunk_id}]\n"
                f"{chunk.text.strip()}"
            )
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def _generate_test_fallback(self, index: PDFIndex, request: ExamRequest, difficulty: str) -> dict[str, Any]:
        questions: list[dict[str, Any]] = []
        chunks = [chunk for chunk in index.chunks if chunk.text.strip()]
        for question_id in range(1, request.number_of_questions + 1):
            chunk = chunks[(question_id - 1) % len(chunks)]
            question_type = request.question_types[(question_id - 1) % len(request.question_types)]
            text = " ".join(chunk.text.split())
            stem = text[:180].rstrip()
            answer = text[:220].rstrip()
            question: dict[str, Any] = {
                "id": question_id,
                "type": question_type,
                "difficulty": difficulty,
                "question": f"Based on the retrieved PDF chunk, what is stated in this excerpt: {stem}?",
                "options": [],
                "answer": answer if request.include_answer_key else None,
                "source_references": [{"page_number": chunk.page_number, "chunk_id": chunk.chunk_id}],
            }
            if question_type == "multiple_choice":
                question["question"] = f"Which option is supported by the retrieved PDF chunk from page {chunk.page_number}?"
                question["options"] = [answer, "Not stated in the uploaded PDF", "Outside knowledge", "Unsupported claim"]
            elif question_type == "true_false":
                question["question"] = f"True or False: {stem}"
                question["answer"] = "True" if request.include_answer_key else None
            questions.append(question)

        return self._normalize_payload(
            {
                "title": f"AI Quiz / Exam: {index.pdf_name}",
                "pdf_name": index.pdf_name,
                "fallback_used": True,
                "fallback_note": "Test fallback used; add a Groq key for Groq-generated questions.",
                "questions": questions,
                "answer_key": [
                    {
                        "id": question["id"],
                        "answer": question["answer"],
                        "source_references": question["source_references"],
                    }
                    for question in questions
                    if request.include_answer_key
                ],
            }
        )

    @staticmethod
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("title", "AI Quiz / Exam")
        payload.setdefault("questions", [])
        payload.setdefault("answer_key", [])
        payload.setdefault("fallback_used", False)
        return payload

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ExamGenerationError("Groq returned invalid JSON for the AI Quiz / Exam.") from exc

    @staticmethod
    def _is_groq_limit_error(status_code: int, body: str) -> bool:
        lowered = (body or "").lower()
        return status_code == 429 or "rate_limit" in lowered or "quota" in lowered or "limit" in lowered
