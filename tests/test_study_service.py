import tempfile
import unittest
from pathlib import Path

from core.models import DocumentPage
from services.progress_service import ProgressService
from services.rag_service import PDFRAGService
from services.study_service import StudyService


class StudyServiceTests(unittest.TestCase):
    def _index(self):
        rag = PDFRAGService(
            chunk_size=260,
            chunk_overlap=20,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        return rag.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text=(
                        "Retrieval augmented generation combines document retrieval with grounded answering. "
                        "A vector store compares embedded questions with embedded chunks. "
                        "Source citations help students verify every answer."
                    ),
                    source_id="rag.pdf",
                ),
                DocumentPage(
                    page_number=2,
                    text=(
                        "Evaluation checks whether generated answers are supported by the retrieved evidence. "
                        "Weak topics should be reviewed before the final exam. "
                        "Practice questions can include multiple choice and short answer formats."
                    ),
                    source_id="rag.pdf",
                ),
            ],
            "rag.pdf",
        )

    def test_study_plan_adds_sections_and_chunk_metadata(self):
        index = self._index()
        sections = StudyService().create_study_plan(index, target_section_count=2)

        self.assertGreaterEqual(len(sections), 1)
        self.assertTrue(sections[0].summary)
        self.assertTrue(sections[0].key_concepts)
        self.assertIn(sections[0].difficulty, {"Easy", "Medium", "Hard"})
        self.assertEqual(index.chunks[0].metadata["section_id"], sections[0].section_id)
        self.assertEqual(index.chunks[0].metadata["section_title"], sections[0].title)

    def test_section_quiz_can_be_graded_and_reports_review_topics(self):
        index = self._index()
        service = StudyService()
        section = service.create_study_plan(index, target_section_count=2)[0]
        quiz = service.generate_section_quiz(section, index, count=3)
        answers = {str(question["id"]): "unsupported answer" for question in quiz["questions"]}

        grade = service.grade_quiz(quiz, answers)

        self.assertEqual(grade["total_questions"], 3)
        self.assertLess(grade["score_percentage"], 100)
        self.assertTrue(grade["weak_topics"])
        self.assertIn("label", quiz["questions"][0]["source_references"][0])
        self.assertNotIn("chunk", quiz["questions"][0]["source_references"][0]["label"].lower())

    def test_explanation_uses_default_student_friendly_exam_style(self):
        index = self._index()
        service = StudyService()
        section = service.create_study_plan(index, target_section_count=2)[0]

        explanation = service.explain_section(section, index)

        self.assertEqual(explanation["level"], StudyService.DEFAULT_EXPLANATION_LEVEL)
        self.assertTrue(explanation["explanation"])
        self.assertTrue(explanation["important_points"])
        self.assertTrue(explanation["definitions"])

    def test_final_exam_generation_is_grounded_in_sections(self):
        index = self._index()
        service = StudyService()
        sections = service.create_study_plan(index, target_section_count=2)
        exam = service.generate_final_exam(index, sections, count=6)

        self.assertEqual(len(exam["questions"]), 6)
        self.assertEqual({item["difficulty"] for item in exam["questions"]}, {"Easy", "Medium", "Hard"})
        self.assertTrue(exam["questions"][0]["source_references"])

    def test_understanding_exam_focus_mistake_review_and_export(self):
        index = self._index()
        service = StudyService()
        section = service.create_study_plan(index, target_section_count=2)[0]

        focus = service.exam_focus(section, index)
        evaluation = service.evaluate_understanding(
            section,
            index,
            "Retrieval augmented generation uses a vector store and source citations.",
        )
        quiz = service.generate_section_quiz(section, index, count=3)
        grade = service.grade_quiz(quiz, {str(question["id"]): "wrong" for question in quiz["questions"]})
        review = service.generate_mistake_review(section, index, grade)
        cards = service.generate_flashcards(section, index)
        markdown = service.export_study_pack_markdown(
            index,
            [section],
            {
                "total_progress_percentage": 50,
                "quiz_scores": {section.section_id: 60},
                "understanding_scores": {section.section_id: evaluation["score"]},
                "weak_topics": grade["weak_topics"],
                "mistake_history": review["wrong_questions"],
            },
            {
                "exam_focus": {section.section_id: focus},
                "flashcards": {section.section_id: cards},
                "final_exam_grade": {
                    "score_percentage": 75,
                    "correct_count": 3,
                    "total_questions": 4,
                    "weak_topics": ["Evaluation"],
                },
            },
        )

        self.assertTrue(focus["important_points"])
        self.assertGreaterEqual(evaluation["score"], 0)
        self.assertTrue(review["wrong_questions"])
        self.assertTrue(cards)
        self.assertIn("# Smart Study Pack", markdown)
        self.assertIn("Exam Focus", markdown)
        self.assertIn("Key Definitions", markdown)
        self.assertIn("Final Exam Results", markdown)
        self.assertIn("Source:", markdown)


class ProgressServiceTests(unittest.TestCase):
    def test_progress_persists_document_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProgressService(Path(tmpdir) / "progress.json")
            progress = service.load_document("notes.pdf", section_count=2)
            progress = service.mark_completed(progress, "section-1")
            progress = service.record_quiz(
                progress,
                "section-1",
                {
                    "score_percentage": 80,
                    "weak_topics": ["Vectors"],
                    "strong_topics": ["Retrieval"],
                    "results": [
                        {
                            "correct": False,
                            "question": "What is a vector store?",
                            "student_answer": "wrong",
                            "correct_answer": "A searchable embedding store.",
                            "topics": ["Vectors"],
                            "explanation": "Review the PDF section.",
                            "source_references": [{"page_number": 1, "chunk_id": "chunk-1"}],
                        }
                    ],
                },
            )
            progress = service.record_study_time(progress, "section-1", 125)
            progress = service.record_understanding(
                progress,
                "section-1",
                {"score": 65, "review_topics": ["Vectors"], "understood_well": ["Retrieval"]},
            )

            loaded = service.load_document("notes.pdf", section_count=2)

        self.assertEqual(loaded["completed_sections"], ["section-1"])
        self.assertEqual(loaded["quiz_scores"]["section-1"], 80.0)
        self.assertEqual(loaded["section_time_seconds"]["section-1"], 125.0)
        self.assertEqual(loaded["understanding_scores"]["section-1"], 65.0)
        self.assertIn("Vectors", loaded["weak_topics"])
        self.assertTrue(loaded["mistake_history"])
        self.assertEqual(loaded["mistake_history"][0]["source_references"][0]["page_number"], 1)
        self.assertGreater(loaded["total_progress_percentage"], 0)

    def test_exam_readiness_uses_completion_quiz_understanding_final_and_weak_topics(self):
        progress = {
            "section_count": 2,
            "completed_sections": ["section-1"],
            "quiz_scores": {"section-1": 80},
            "understanding_scores": {"section-1": 70},
            "final_exam_score": 75,
            "weak_topics": ["Vectors"],
            "review_sections": ["section-2"],
        }

        readiness = ProgressService.exam_readiness(progress)
        timing = ProgressService.timing_summary(
            {"section_time_seconds": {"section-1": 900}},
            {"section-1": 10},
        )

        self.assertGreater(readiness, 0)
        self.assertIn(ProgressService.readiness_status(readiness), {"Not ready", "Needs review", "Almost ready", "Ready"})
        self.assertIn("section-2", ProgressService.readiness_action(progress))
        self.assertEqual(timing["longer_than_expected"], ["section-1"])


if __name__ == "__main__":
    unittest.main()
