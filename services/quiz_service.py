from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from core.models import DocumentChunk



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
    def _build_options(correct: str, pool: List[str], num_options: int = 4) -> List[str]:
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
        fallback_terms = ["analysis", "document", "concept", "process", "method", "record"]
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
    def generate_mcq(cls, chunks: List[DocumentChunk], num_questions: int = 3) -> List[QuizQuestion]:
        """Generate simple multiple-choice questions from legacy chunk objects."""
        return cls.generate_from_documents(chunks, num_questions=num_questions)

    @classmethod
    def generate_with_openai(
        cls,
        documents: List[Any],
        num_questions: int = 5,
        difficulty: str = "medium",
    ) -> List[QuizQuestion]:
        context_parts = []

        for item in documents[:10]:
            doc_data = cls._extract_doc_data(item)
            text = (doc_data["text"] or "").strip()

            if text:
                context_parts.append(
                    f"Source: {doc_data['source']} page {doc_data['page']}\n{text[:1200]}"
                )

        context = "\n\n".join(context_parts)

        if not context:
            return []

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return cls.generate_from_documents(documents, num_questions)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            prompt = f"""
            Create {num_questions} multiple-choice quiz questions based only on the context below.
            Difficulty: {difficulty}

            Difficulty rules:
            - easy: ask direct factual questions from the text.
            - medium: ask questions that require understanding relationships between concepts.
            - hard: ask scenario-based or comparison questions that require applying the document concepts.
            Return ONLY valid JSON in this exact format:
            [
              {{
                "question": "question text",
                "options": ["option A", "option B", "option C", "option D"],
                "answer": "the exact correct option",
                "explanation": "short explanation",
                "citation": "source and page"
              }}
            ]
            Rules:
            - The answer must be exactly one of the options.
            - Do not invent information.
            - Use only the provided context.
            - Make the questions useful for exam preparation.
            - Avoid repeating the same question style across difficulty levels.
            
            Context:
            {context}
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate grounded study quizzes from provided document context only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            print("OpenAI quiz generation succeeded")

            raw = response.choices[0].message.content or "[]"
            raw = raw.strip()

            if raw.startswith("```"):
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            data = json.loads(raw)

            questions: List[QuizQuestion] = []
            for item in data[:num_questions]:
                options = item.get("options", [])
                answer = item.get("answer", "")

                if not item.get("question") or not options or answer not in options:
                    continue

                questions.append(
                    QuizQuestion(
                        prompt=item.get("question", ""),
                        options=options,
                        answer=answer,
                        explanation=item.get("explanation"),
                        citation=item.get("citation"),
                    )
                )

            if questions:
                return questions

            return cls.generate_from_documents(documents, num_questions)


        except Exception as exc:

            print(f"OpenAI quiz generation failed: {exc}")

            return cls.generate_from_documents(documents, num_questions)


    @classmethod
    def generate_from_documents(cls, documents: List[Any], num_questions: int = 3) -> List[QuizQuestion]:
        """Generate deterministic fill-in-the-blank questions with citations."""
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

            prompt = f"Fill in the blank: {sentence.replace(keyword, '_____', 1)}"
            if prompt in seen_prompts:
                continue

            options = cls._build_options(keyword, keyword_pool, num_options=4)
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
                    explanation=f"The correct answer is '{keyword}'.",
                    citation=citation,
                    source=source,
                    page=page,
                )
            )
            seen_prompts.add(prompt)

            if len(questions) >= num_questions:
                break

        return questions
