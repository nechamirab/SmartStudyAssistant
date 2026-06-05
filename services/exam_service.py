from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from services.general_ai_service import GeneralAIService


@dataclass(frozen=True)
class ExamOptions:
    question_count: int = 10
    difficulty: str = "mixed"


class ExamService:
    """Generate a final exam from study context with defensive AI parsing."""

    def generate_final_exam(self, study_context: str, options: ExamOptions | None = None) -> dict[str, Any]:
        options = options or ExamOptions()
        context = (study_context or "").strip()
        if not context:
            return self._fallback_exam("No study context was available.", options)

        try:
            raw = self._call_ai(context, options)
            return self.normalize_payload(self.parse_json(raw), options)
        except Exception as exc:
            return self._fallback_exam(f"AI exam generation fell back safely: {exc}", options)

    def _call_ai(self, context: str, options: ExamOptions) -> str:
        provider = GeneralAIService().select_provider()
        if provider is None:
            raise RuntimeError("Set OPENAI_API_KEY or GROQ_API_KEY for AI-generated final exams.")

        prompt = (
            "Create a final exam from this study material. Return JSON only with keys "
            "title, questions, answer_key. Each question needs id, type, question, options, answer, topic.\n\n"
            "Use only these question types: multiple_choice, true_false, short_answer.\n"
            f"Question count: {max(1, min(options.question_count, 25))}\n"
            f"Difficulty: {options.difficulty}\n"
            f"Study material:\n{context[:12000]}"
        )
        messages = [
            {"role": "system", "content": "Return valid JSON only. Do not wrap output in Markdown."},
            {"role": "user", "content": prompt},
        ]
        if provider.name == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=provider.api_key)
            response = client.chat.completions.create(
                model=provider.model,
                messages=messages,
                temperature=0.25,
                response_format={"type": "json_object"},
            )
            return (response.choices[0].message.content or "").strip()

        payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": 0.25,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            GeneralAIService.GROQ_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    @staticmethod
    def parse_json(raw: str) -> dict[str, Any]:
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("Exam response must be a JSON object.")
        return payload

    @staticmethod
    def normalize_payload(payload: dict[str, Any], options: ExamOptions | None = None) -> dict[str, Any]:
        options = options or ExamOptions()
        questions = payload.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        normalized_questions: list[dict[str, Any]] = []
        allowed_types = {"multiple_choice", "true_false", "short_answer"}
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            question_type = str(item.get("type", "short_answer")).strip().lower().replace(" ", "_")
            if question_type not in allowed_types:
                question_type = "short_answer"
            normalized_questions.append(
                {
                    "id": int(item.get("id", index) or index),
                    "type": question_type,
                    "question": question,
                    "options": item.get("options") if isinstance(item.get("options"), list) else [],
                    "answer": str(item.get("answer", "") or "Review the related study material."),
                    "topic": str(item.get("topic", "General")),
                }
            )

        if not normalized_questions:
            raise ValueError("No valid questions were returned.")

        answer_key = payload.get("answer_key")
        if not isinstance(answer_key, list):
            answer_key = [{"id": item["id"], "answer": item["answer"]} for item in normalized_questions]

        return {
            "title": str(payload.get("title", "AI Final Exam")),
            "questions": normalized_questions[: max(1, min(options.question_count, 25))],
            "answer_key": answer_key,
            "fallback_used": bool(payload.get("fallback_used", False)),
        }

    def _fallback_exam(self, note: str, options: ExamOptions) -> dict[str, Any]:
        count = max(1, min(options.question_count, 10))
        types = ["multiple_choice", "true_false", "short_answer"]
        questions = [
            {
                "id": index,
                "type": types[(index - 1) % len(types)],
                "question": f"Review question {index}: explain or identify one important idea from the study material.",
                "options": ["A key idea from the PDF", "An unrelated topic", "A random definition", "None of the above"]
                if types[(index - 1) % len(types)] == "multiple_choice"
                else ["True", "False"] if types[(index - 1) % len(types)] == "true_false" else [],
                "answer": "A key idea from the PDF" if types[(index - 1) % len(types)] == "multiple_choice"
                else "True" if types[(index - 1) % len(types)] == "true_false"
                else "Use your section notes and explanations to support the answer.",
                "topic": "Review",
            }
            for index in range(1, count + 1)
        ]
        return {
            "title": "AI Final Exam",
            "questions": questions,
            "answer_key": [{"id": item["id"], "answer": item["answer"]} for item in questions],
            "fallback_used": True,
            "fallback_note": note,
        }
