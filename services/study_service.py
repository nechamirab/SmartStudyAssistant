from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from core.models import DocumentChunk, StudySection
from services.rag_service import PDFIndex, PDFRAGService
from services.source_utils import clean_section_title, format_source_label, normalize_bullets, sanitize_visible_text


class StudyServiceError(Exception):
    """Raised when a study artifact cannot be created from PDF content."""


class StudyService:
    """Create grounded study plans, explanations, quizzes, and exams from PDF chunks."""

    DIFFICULTIES = ("Easy", "Medium", "Hard")
    DEFAULT_EXPLANATION_LEVEL = "Default"
    QUESTION_TYPES = ("multiple_choice", "true_false", "short_answer")

    STOPWORDS = PDFRAGService.STOPWORDS | {
        "also",
        "can",
        "may",
        "one",
        "using",
        "used",
        "use",
        "will",
        "would",
    }

    def create_study_plan(self, index: PDFIndex, target_section_count: int | None = None) -> list[StudySection]:
        if not index or not index.chunks:
            raise StudyServiceError("Upload and process a PDF before creating a study plan.")

        groups = self._section_groups(index, target_section_count)
        sections: list[StudySection] = []
        for number, chunks in enumerate(groups, 1):
            section = self._build_section(index, chunks, number)
            sections.append(section)
            for chunk in chunks:
                chunk.metadata["section_id"] = section.section_id
                chunk.metadata["section_title"] = section.title

        return sections

    def explain_section(self, section: StudySection, index: PDFIndex, level: str | None = None) -> dict[str, Any]:
        level = level or self.DEFAULT_EXPLANATION_LEVEL
        chunks = self.section_chunks(index, section)
        content = self._join_chunks(chunks, max_chars=3500)
        if not content:
            raise StudyServiceError("This section does not contain enough text to explain.")

        sentences = self._important_sentences(content, limit=5)
        concepts = section.key_concepts or self._key_concepts(content, limit=6)
        explanation = self._exam_explanation(section, sentences, concepts)

        return {
            "section_id": section.section_id,
            "level": level,
            "explanation": explanation,
            "definitions": self._definitions(content, concepts),
            "important_points": sentences,
            "example_questions": self._example_questions(section, concepts),
            "sources": self._source_refs(chunks),
        }

    def generate_section_quiz(self, section: StudySection, index: PDFIndex, count: int = 6) -> dict[str, Any]:
        chunks = self.section_chunks(index, section)
        content = self._join_chunks(chunks, max_chars=4500)
        if not content:
            raise StudyServiceError("This section does not contain enough text for a quiz.")

        concepts = section.key_concepts or self._key_concepts(content, limit=8)
        sentences = self._important_sentences(content, limit=max(4, count))
        questions: list[dict[str, Any]] = []
        for number in range(1, count + 1):
            question_type = self.QUESTION_TYPES[(number - 1) % len(self.QUESTION_TYPES)]
            sentence = sentences[(number - 1) % len(sentences)]
            concept = concepts[(number - 1) % len(concepts)] if concepts else "this section"
            question = self._quiz_question(number, question_type, concept, sentence, concepts, chunks)
            questions.append(question)

        return {
            "title": f"Mini Quiz: {section.title}",
            "section_id": section.section_id,
            "questions": questions,
        }

    def grade_quiz(self, quiz: dict[str, Any], answers: dict[str, str]) -> dict[str, Any]:
        questions = quiz.get("questions", [])
        graded: list[dict[str, Any]] = []
        correct_count = 0
        weak_topics: list[str] = []
        strong_topics: list[str] = []

        for question in questions:
            question_id = str(question.get("id"))
            given = str(answers.get(question_id, "")).strip()
            expected = str(question.get("answer", "")).strip()
            is_correct = self._is_answer_correct(given, expected, question.get("type", ""))
            if is_correct:
                correct_count += 1
                strong_topics.extend(question.get("topics", []))
            else:
                weak_topics.extend(question.get("topics", []))
            graded.append(
                {
                    "id": question.get("id"),
                    "question": question.get("question"),
                    "student_answer": given,
                    "correct_answer": expected,
                    "correct": is_correct,
                    "explanation": question.get("explanation", ""),
                    "topics": question.get("topics", []),
                    "source_references": question.get("source_references", []),
                }
            )

        total = len(questions)
        score = round((correct_count / total) * 100, 1) if total else 0.0
        return {
            "score_percentage": score,
            "correct_count": correct_count,
            "total_questions": total,
            "results": graded,
            "weak_topics": self._dedupe(weak_topics),
            "strong_topics": self._dedupe(strong_topics),
        }

    def generate_final_exam(self, index: PDFIndex, sections: list[StudySection], count: int = 12) -> dict[str, Any]:
        if not sections:
            sections = self.create_study_plan(index)
        questions: list[dict[str, Any]] = []
        difficulties = ["Easy", "Medium", "Hard"]
        types = ["multiple_choice", "short_answer", "open_question"]

        for number in range(1, count + 1):
            section = sections[(number - 1) % len(sections)]
            chunks = self.section_chunks(index, section)
            content = self._join_chunks(chunks, max_chars=2500)
            sentences = self._important_sentences(content, limit=3) or [section.summary]
            concepts = section.key_concepts or self._key_concepts(content, limit=4)
            question_type = types[(number - 1) % len(types)]
            difficulty = difficulties[(number - 1) % len(difficulties)]
            question = self._exam_question(number, question_type, difficulty, section, sentences[0], concepts, chunks)
            questions.append(question)

        return {
            "title": f"Practice Final Exam: {index.pdf_name}",
            "pdf_name": index.pdf_name,
            "questions": questions,
        }

    def grade_final_exam(self, exam: dict[str, Any], answers: dict[str, str]) -> dict[str, Any]:
        graded = self.grade_quiz(exam, answers)
        open_feedback = []
        for result in graded["results"]:
            question = next(
                (item for item in exam.get("questions", []) if item.get("id") == result.get("id")),
                {},
            )
            if question.get("type") == "open_question":
                expected_terms = self._keywords(str(question.get("answer", "")))
                given_terms = self._keywords(str(result.get("student_answer", "")))
                overlap = len(expected_terms & given_terms)
                result["correct"] = overlap >= max(1, min(3, len(expected_terms) // 3))
                result["feedback"] = (
                    "Good explanation with relevant PDF-grounded terms."
                    if result["correct"]
                    else "Review the cited section and include more of the expected concepts."
                )
                open_feedback.append({"id": result["id"], "feedback": result["feedback"]})

        correct_count = sum(1 for item in graded["results"] if item["correct"])
        total = len(graded["results"])
        graded["correct_count"] = correct_count
        graded["score_percentage"] = round((correct_count / total) * 100, 1) if total else 0.0
        graded["open_feedback"] = open_feedback
        return graded

    def exam_focus(self, section: StudySection, index: PDFIndex) -> dict[str, Any]:
        chunks = self.section_chunks(index, section)
        content = self._join_chunks(chunks, max_chars=4500)
        concepts = section.key_concepts or self._key_concepts(content, limit=6)
        important = normalize_bullets(self._important_sentences(content, limit=5), limit=5)
        questions = normalize_bullets(self._example_questions(section, concepts), limit=3)
        mistakes = normalize_bullets(
            [
                f"Confusing {concept} with a related idea not supported by this section."
                for concept in concepts[:3]
            ],
            limit=4,
        )
        terms = normalize_bullets(concepts[:8], fallback="Review the key vocabulary in this section.", limit=8)
        return {
            "section_id": section.section_id,
            "important_points": important[:5],
            "possible_exam_questions": questions[:3],
            "common_mistakes": mistakes,
            "key_terms": terms,
            "sources": self._source_refs(chunks[:3]),
        }

    def evaluate_understanding(self, section: StudySection, index: PDFIndex, answer: str) -> dict[str, Any]:
        answer = (answer or "").strip()
        if not answer:
            raise StudyServiceError("Write a short explanation before running the understanding check.")

        chunks = self.section_chunks(index, section)
        content = self._join_chunks(chunks, max_chars=4500)
        concepts = section.key_concepts or self._key_concepts(content, limit=8)
        expected_terms = {concept.lower() for concept in concepts}
        answer_terms = self._keywords(answer)
        matched = [
            concept
            for concept in concepts
            if concept.lower() in answer.lower() or concept.lower() in answer_terms
        ]
        missing = [concept for concept in concepts if concept not in matched]
        coverage = len(set(concept.lower() for concept in matched)) / max(1, len(expected_terms))
        length_bonus = min(20, len(answer.split()) * 1.5)
        score = round(min(100.0, coverage * 80 + length_bonus), 1)
        corrected = self._exam_explanation(
            section,
            self._important_sentences(content, limit=3) or [section.summary],
            concepts,
        )
        return {
            "section_id": section.section_id,
            "score": score,
            "understood_well": matched[:5],
            "missing": missing[:5],
            "review_topics": missing[:5],
            "corrected_explanation": corrected,
            "recommend_review": score < 70,
            "sources": self._source_refs(chunks[:3]),
        }

    def generate_mistake_review(
        self,
        section: StudySection,
        index: PDFIndex,
        grade: dict[str, Any],
    ) -> dict[str, Any]:
        wrong = [item for item in grade.get("results", []) if not item.get("correct")]
        weak_topics = grade.get("weak_topics", [])
        chunks = self.section_chunks(index, section)
        content = self._join_chunks(chunks, max_chars=3500)
        points = self._important_sentences(content, limit=4)
        if weak_topics:
            lesson = (
                f"Review {', '.join(weak_topics[:4])} in {section.title}. "
                f"Use the source section to reconnect each term to these PDF-grounded points: {' '.join(points[:3])}"
            )
        else:
            lesson = "No repeated weak topic was detected in this quiz."
        return {
            "section_id": section.section_id,
            "review_section": section.title,
            "wrong_questions": wrong,
            "weak_topics": weak_topics,
            "review_lesson": lesson,
            "sources": self._source_refs(chunks[:3]),
        }

    def generate_flashcards(self, section: StudySection, index: PDFIndex, count: int = 6) -> list[dict[str, Any]]:
        chunks = self.section_chunks(index, section)
        content = self._join_chunks(chunks, max_chars=4000)
        concepts = section.key_concepts or self._key_concepts(content, limit=count)
        definitions = self._definitions(content, concepts)
        refs = self._source_refs(chunks[:2])
        cards = []
        for item in definitions[:count]:
            cards.append(
                {
                    "front": f"What should you know about {item['term']}?",
                    "back": item["definition"],
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "page_range": section.page_range_label(),
                    "source_references": refs,
                }
            )
        return cards

    def export_study_pack_markdown(
        self,
        index: PDFIndex,
        sections: list[StudySection],
        progress: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        artifacts = artifacts or {}
        lines = [
            f"# Smart Study Pack: {index.pdf_name}",
            "",
            "## Progress",
            f"- Overall progress: {progress.get('total_progress_percentage', 0):.1f}%",
            f"- Average quiz score: {self._format_percent_dict_average(progress.get('quiz_scores', {}))}",
            f"- Average understanding score: {self._format_percent_dict_average(progress.get('understanding_scores', {}))}",
            f"- Final exam score: {progress.get('final_exam_score', 'Not taken')}",
            "",
            "## Weak Topics",
        ]
        weak_topics = progress.get("weak_topics", [])
        lines.extend([f"- {topic}" for topic in weak_topics] or ["- None recorded yet."])
        lines.extend(["", "## Study Plan"])
        for section in sections:
            lines.extend(
                [
                    f"### {section.title}",
                    f"- Source: {section.source_id}, {section.page_range_label()}",
                    f"- Difficulty: {section.difficulty}",
                    f"- Estimated time: {section.estimated_minutes} min",
                    f"- Summary: {section.summary}",
                    f"- Key concepts: {', '.join(section.key_concepts)}",
                    "",
                ]
            )
            focus = artifacts.get("exam_focus", {}).get(section.section_id)
            if focus:
                lines.append("#### Exam Focus")
                lines.extend(f"- {point}" for point in focus.get("important_points", []))
                if focus.get("possible_exam_questions"):
                    lines.append("")
                    lines.append("Possible exam questions:")
                    lines.extend(f"- {question}" for question in focus.get("possible_exam_questions", []))
                if focus.get("common_mistakes"):
                    lines.append("")
                    lines.append("Common mistakes:")
                    lines.extend(f"- {mistake}" for mistake in focus.get("common_mistakes", []))
                if focus.get("key_terms"):
                    lines.append("")
                    lines.append("Key terms to memorize:")
                    lines.extend(f"- {term}" for term in focus.get("key_terms", []))
                lines.append("")

            definitions = self._definitions(
                self._join_chunks(self.section_chunks(index, section), max_chars=3500),
                section.key_concepts,
            )
            if definitions:
                lines.append("#### Key Definitions")
                lines.extend(f"- **{item['term']}**: {item['definition']}" for item in definitions)
                lines.append("")

            flashcards = artifacts.get("flashcards", {}).get(section.section_id, [])
            if flashcards:
                lines.append("#### Flashcards")
                for card in flashcards:
                    lines.extend([f"- Q: {card['front']}", f"  A: {card['back']}"])
                lines.append("")

        lines.extend(["## Quiz Results"])
        quiz_scores = progress.get("quiz_scores", {})
        lines.extend([f"- {section_id}: {score:.1f}%" for section_id, score in quiz_scores.items()] or ["- No quizzes submitted yet."])
        final_exam_grade = artifacts.get("final_exam_grade")
        if final_exam_grade:
            lines.extend(
                [
                    "",
                    "## Final Exam Results",
                    f"- Score: {float(final_exam_grade.get('score_percentage', 0.0)):.1f}%",
                    f"- Correct: {final_exam_grade.get('correct_count', 0)} / {final_exam_grade.get('total_questions', 0)}",
                ]
            )
            weak_topics = final_exam_grade.get("weak_topics", [])
            if weak_topics:
                lines.append("- Weak topics: " + ", ".join(str(topic) for topic in weak_topics))
        if progress.get("mistake_history"):
            lines.extend(["", "## Review Plan"])
            for item in progress["mistake_history"][-10:]:
                sources = item.get("source_references", [])
                source_label = self._format_sources(sources)
                lines.extend(
                    [
                        f"- Section: {item.get('section_id')}",
                        f"  Question: {item.get('question')}",
                        f"  Correct answer: {item.get('correct_answer')}",
                        f"  Explanation: {item.get('explanation')}",
                    ]
                )
                if source_label:
                    lines.append(f"  Source: {source_label}")
        return "\n".join(lines).strip() + "\n"

    def section_chunks(self, index: PDFIndex, section: StudySection) -> list[DocumentChunk]:
        chunk_ids = set(section.chunk_ids)
        return [chunk for chunk in index.chunks if chunk.chunk_id in chunk_ids]

    def sections_to_dicts(self, sections: list[StudySection]) -> list[dict[str, Any]]:
        return [asdict(section) for section in sections]

    def sections_from_dicts(self, payload: list[dict[str, Any]]) -> list[StudySection]:
        return [StudySection(**item) for item in payload]

    def _section_groups(self, index: PDFIndex, target_section_count: int | None) -> list[list[DocumentChunk]]:
        by_page: dict[tuple[str, int], list[DocumentChunk]] = {}
        for chunk in index.chunks:
            by_page.setdefault((chunk.source_id, chunk.page_number), []).append(chunk)

        ordered_pages = [by_page[key] for key in sorted(by_page, key=lambda item: (item[0], item[1]))]
        if not ordered_pages:
            return []

        page_count = len(ordered_pages)
        desired = target_section_count or min(max(3, page_count), 10)
        pages_per_section = max(1, round(page_count / desired))
        groups: list[list[DocumentChunk]] = []
        current: list[DocumentChunk] = []
        for page_chunks in ordered_pages:
            current.extend(page_chunks)
            if len({chunk.page_number for chunk in current}) >= pages_per_section:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return [group for group in groups if group]

    def _build_section(self, index: PDFIndex, chunks: list[DocumentChunk], number: int) -> StudySection:
        content = self._join_chunks(chunks, max_chars=4000)
        pages = [chunk.page_number for chunk in chunks]
        source_id = chunks[0].source_id or index.pdf_name
        concepts = self._key_concepts(content, limit=6)
        title = self._title_from_content(content, concepts, number)
        summary = self._summary(content)
        estimated_minutes = self._estimated_minutes(content)
        difficulty = self._difficulty(content, concepts)
        return StudySection(
            section_id=f"section-{number}",
            title=title,
            page_start=min(pages),
            page_end=max(pages),
            summary=summary,
            key_concepts=concepts,
            estimated_minutes=estimated_minutes,
            difficulty=difficulty,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            content_preview=self._trim(content, 1200),
            source_id=source_id,
        )

    def _quiz_question(
        self,
        number: int,
        question_type: str,
        concept: str,
        sentence: str,
        concepts: list[str],
        chunks: list[DocumentChunk],
    ) -> dict[str, Any]:
        answer = self._trim(sentence, 240)
        source_refs = self._source_refs(chunks[:2])
        if question_type == "multiple_choice":
            distractors = [item for item in concepts if item.lower() != concept.lower()][:3]
            options = [answer, *[f"Only {item}" for item in distractors]]
            while len(options) < 4:
                options.append("Not supported by the uploaded PDF")
            return {
                "id": number,
                "type": question_type,
                "question": f"Which statement is supported by the PDF about {concept}?",
                "options": options[:4],
                "answer": answer,
                "explanation": f"The cited section states: {answer}",
                "topics": [concept],
                "source_references": source_refs,
            }
        if question_type == "true_false":
            return {
                "id": number,
                "type": question_type,
                "question": f"True or False: {answer}",
                "options": ["True", "False"],
                "answer": "True",
                "explanation": "This statement is copied or closely paraphrased from the cited section.",
                "topics": [concept],
                "source_references": source_refs,
            }
        return {
            "id": number,
            "type": question_type,
            "question": f"In one or two sentences, explain {concept} using this section.",
            "options": [],
            "answer": answer,
            "explanation": f"A strong answer should include the PDF idea: {answer}",
            "topics": [concept],
            "source_references": source_refs,
        }

    def _exam_question(
        self,
        number: int,
        question_type: str,
        difficulty: str,
        section: StudySection,
        sentence: str,
        concepts: list[str],
        chunks: list[DocumentChunk],
    ) -> dict[str, Any]:
        concept = concepts[0] if concepts else section.title
        base = self._quiz_question(number, "short_answer", concept, sentence, concepts, chunks)
        base["type"] = question_type
        base["difficulty"] = difficulty
        base["section_id"] = section.section_id
        if question_type == "multiple_choice":
            return self._quiz_question(number, "multiple_choice", concept, sentence, concepts, chunks) | {
                "difficulty": difficulty,
                "section_id": section.section_id,
            }
        if question_type == "open_question":
            base["question"] = f"Explain how {concept} connects to the main ideas in {section.title}."
            base["explanation"] = "Open answers are graded by overlap with PDF-grounded expected concepts."
        return base

    def _beginner_explanation(self, section: StudySection, sentences: list[str]) -> str:
        simple_points = " ".join(self._trim(sentence, 180) for sentence in sentences[:3])
        return f"This section is about {section.title}. In simple terms, focus on these ideas: {simple_points}"

    def _university_explanation(self, section: StudySection, sentences: list[str], concepts: list[str]) -> str:
        return (
            f"{section.title} introduces {', '.join(concepts[:4])}. "
            f"The core argument from the PDF is: {' '.join(sentences[:4])}"
        )

    def _exam_explanation(self, section: StudySection, sentences: list[str], concepts: list[str]) -> str:
        return (
            f"For exam preparation, prioritize {', '.join(concepts[:5])}. "
            f"Be ready to define them, compare them, and apply these cited points: {' '.join(sentences[:4])}"
        )

    def _definitions(self, content: str, concepts: list[str]) -> list[dict[str, str]]:
        sentences = self._sentences(content)
        definitions = []
        for concept in concepts[:6]:
            match = next((sentence for sentence in sentences if concept.lower() in sentence.lower()), "")
            definitions.append(
                {
                    "term": sanitize_visible_text(concept),
                    "definition": self._trim(match or f"Discussed in this section: {concept}.", 220),
                }
            )
        return definitions

    def _example_questions(self, section: StudySection, concepts: list[str]) -> list[str]:
        topic = concepts[0] if concepts else section.title
        return normalize_bullets(
            [
            f"What is the main idea of {section.title}?",
            f"How does the PDF describe {topic}?",
            f"Which details from {section.page_range_label()} would support an exam answer?",
            ],
            limit=3,
        )

    def _title_from_content(self, content: str, concepts: list[str], number: int) -> str:
        content = sanitize_visible_text(content, remove_files=False)
        for line in content.splitlines():
            clean = clean_section_title(" ".join(line.split()).strip(":-"), "")
            if 4 <= len(clean) <= 80 and len(clean.split()) <= 10:
                if not clean.endswith(".") and not clean.lower().startswith(("figure", "table")):
                    return clean

        first_sentence = next(iter(self._sentences(content)), "")
        if first_sentence:
            title_seed = re.split(
                r"\b(?:combines|introduces|explains|describes|defines|covers|examines|uses|is|are|means)\b",
                first_sentence,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            title = clean_section_title(title_seed, "")
            if 4 <= len(title) <= 80 and 2 <= len(title.split()) <= 8:
                return title

        if concepts:
            title = clean_section_title(concepts[0], "")
            if title:
                return title
        return f"Section {number}"

    def _summary(self, content: str) -> str:
        sentences = self._important_sentences(content, limit=2)
        summary = self._trim(" ".join(sentences), 420) if sentences else self._trim(content, 420)
        return sanitize_visible_text(summary)

    def _key_concepts(self, content: str, limit: int = 6) -> list[str]:
        words = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", sanitize_visible_text(content or "", remove_files=False))
            if token.lower() not in self.STOPWORDS
        ]
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return normalize_bullets([word.replace("-", " ").title() for word, _count in ranked[:limit]], limit=limit)

    def _important_sentences(self, content: str, limit: int) -> list[str]:
        sentences = self._sentences(content)
        if not sentences:
            return []
        concepts = set(term.lower() for term in self._key_concepts(content, limit=12))
        scored = []
        for sentence in sentences:
            terms = self._keywords(sentence)
            score = len(terms & concepts) + min(len(sentence) // 120, 3)
            scored.append((score, sanitize_visible_text(sentence)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return normalize_bullets([self._trim(sentence, 260) for _score, sentence in scored[:limit]], limit=limit)

    def _sentences(self, content: str) -> list[str]:
        sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content or ""):
            clean = sanitize_visible_text(" ".join(sentence.split()))
            if len(clean) >= 35:
                sentences.append(clean)
        return sentences

    def _difficulty(self, content: str, concepts: list[str]) -> str:
        average_words = 0
        sentences = self._sentences(content)
        if sentences:
            average_words = round(sum(len(sentence.split()) for sentence in sentences) / len(sentences))
        if len(concepts) >= 6 and average_words >= 24:
            return "Hard"
        if len(concepts) >= 4 or average_words >= 18:
            return "Medium"
        return "Easy"

    def _estimated_minutes(self, content: str) -> int:
        words = len((content or "").split())
        return max(8, min(45, round(words / 140 * 5) * 5))

    def _join_chunks(self, chunks: list[DocumentChunk], max_chars: int) -> str:
        parts = []
        total = 0
        for chunk in chunks:
            block = sanitize_visible_text(chunk.text.strip(), remove_files=False)
            if total + len(block) > max_chars:
                break
            if block:
                parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    def _source_refs(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        refs = []
        seen = set()
        for chunk in chunks:
            key = (chunk.source_id, chunk.page_number, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "pdf_name": chunk.source_id,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "section_title": chunk.metadata.get("section_title", ""),
                    "label": format_source_label(
                        {
                            "page_number": chunk.page_number,
                            "chunk_id": chunk.chunk_id,
                            "section_title": chunk.metadata.get("section_title", ""),
                        }
                    ),
                }
            )
        return refs

    def _is_answer_correct(self, given: str, expected: str, question_type: str) -> bool:
        if not given:
            return False
        if question_type == "true_false":
            return given.strip().lower() == expected.strip().lower()
        expected_terms = self._keywords(expected)
        given_terms = self._keywords(given)
        if not expected_terms:
            return given.strip().lower() == expected.strip().lower()
        return len(expected_terms & given_terms) >= max(1, min(3, len(expected_terms) // 3))

    def _keywords(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text or "")
            if token.lower() not in self.STOPWORDS
        }

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            normalized = value.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(value)
        return result

    @staticmethod
    def _format_percent_dict_average(values: dict[str, Any]) -> str:
        scores = [float(value) for value in values.values()]
        if not scores:
            return "No scores yet"
        return f"{round(sum(scores) / len(scores), 1)}%"

    @staticmethod
    def _format_sources(refs: list[dict[str, Any]]) -> str:
        labels = []
        for ref in refs[:3]:
            label = format_source_label(ref)
            if label:
                labels.append(label)
        return "; ".join(labels)

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        clean = sanitize_visible_text(" ".join((text or "").split()))
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."
