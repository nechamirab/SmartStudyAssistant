from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from core.models import DocumentPage
from services.exam_grading_service import ExamGradingService
from services.exam_service import ExamOptions, ExamService
from services.persistence_service import PersistenceService
from services.progress_service import ProgressService
from services.quiz_grading_service import QuizGradingService
from services.section_state_service import SectionStateService
from services.study_service import StudyService


class TestMVPServices(unittest.TestCase):
    def test_study_plan_can_be_generated_from_valid_text(self):
        pages = [
            DocumentPage(
                page_number=1,
                text="GRAPH SEARCH BASICS\nBreadth first search explores neighbors level by level in a graph.",
            ),
            DocumentPage(
                page_number=2,
                text="Depth first search follows a path before backtracking to explore alternatives.",
            ),
        ]

        sections = StudyService().generate_study_plan(pages, pages_per_section=1)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].start_page, 1)
        self.assertTrue(sections[0].summary)
        self.assertGreaterEqual(len(sections[0].learning_objectives), 2)

    def test_empty_text_does_not_generate_study_plan(self):
        pages = [DocumentPage(page_number=1, text="   ")]

        sections = StudyService().generate_study_plan(pages)

        self.assertEqual(sections, [])

    def test_session_suggestion_uses_text_size(self):
        self.assertEqual(StudyService.suggest_session_count("word " * 1000), 3)
        self.assertEqual(StudyService.suggest_session_count("word " * 2000), 5)
        self.assertEqual(StudyService.suggest_session_count("word " * 4500), 7)
        self.assertEqual(StudyService.suggest_session_count("word " * 8000), 10)
        self.assertEqual(StudyService.suggest_session_count("word " * 12000), 12)
        self.assertEqual(StudyService.suggest_session_count("word " * 18000), 15)

    def test_empty_text_returns_default_session_suggestion(self):
        self.assertEqual(StudyService.suggest_session_count(""), 5)
        self.assertEqual(StudyService.suggest_session_count([DocumentPage(page_number=1, text="")]), 5)

    def test_manual_session_number_overrides_suggestion(self):
        pages = [
            DocumentPage(page_number=index, text=f"Readable page {index} with enough study text.")
            for index in range(1, 11)
        ]

        sections = StudyService().generate_study_plan_for_sessions(pages, session_count=6)

        self.assertEqual(len(sections), 6)
        self.assertEqual(sections[0].start_page, 1)
        self.assertEqual(sections[-1].end_page, 10)

    def test_session_count_is_capped_by_readable_pages(self):
        pages = [
            DocumentPage(page_number=1, text="Readable page one with enough study text."),
            DocumentPage(page_number=2, text=""),
            DocumentPage(page_number=3, text="Readable page three with enough study text."),
        ]

        sections = StudyService().generate_study_plan_for_sessions(pages, session_count=10)

        self.assertEqual(len(sections), 2)
        self.assertEqual([section.start_page for section in sections], [1, 3])

    def test_section_state_is_independent_per_section(self):
        states = SectionStateService.ensure_states({}, [1, 2])
        first = SectionStateService.get_state(states, 1)
        second = SectionStateService.get_state(states, 2)

        first["explanation"] = "Section one explanation"
        first["quiz_score"] = 90
        second["answer"] = "Section two answer"

        self.assertEqual(states["1"]["explanation"], "Section one explanation")
        self.assertEqual(states["1"]["quiz_score"], 90)
        self.assertEqual(states["2"]["answer"], "Section two answer")
        self.assertEqual(states["2"]["explanation"], "")

    def test_quiz_grading_works(self):
        questions = [
            {"type": "multiple_choice", "answer": "A"},
            {"type": "true_false", "answer": "True"},
        ]
        score, feedback = QuizGradingService.grade(questions, {1: "A", 2: "False"})

        self.assertEqual(score, 50)
        self.assertIn("Correct", feedback[0])
        self.assertIn("Incorrect", feedback[1])

    def test_malformed_ai_output_does_not_crash_exam_generation(self):
        service = ExamService()

        def bad_ai_response(_context, _options):
            return "not json"

        service._call_ai = bad_ai_response
        exam = service.generate_final_exam("Sorting and searching notes.", ExamOptions(question_count=4))

        self.assertTrue(exam["fallback_used"])
        self.assertEqual(len(exam["questions"]), 4)

    def test_progress_updates_correctly(self):
        progress = ProgressService.default_state()

        ProgressService.start_timer(progress, now=10.0)
        ProgressService.finish_section(progress, 2, now=75.0)
        progress.quiz_scores.extend([80, 100])

        self.assertEqual(progress.completed_sections, {2})
        self.assertEqual(progress.actual_study_seconds, 65)
        self.assertFalse(progress.timer_running)
        self.assertEqual(ProgressService.quiz_average(progress), 90)

    def test_timer_can_restart_from_zero(self):
        progress = ProgressService.default_state()
        progress.actual_study_seconds = 120

        ProgressService.restart_timer(progress, now=50.0)

        self.assertTrue(progress.timer_running)
        self.assertEqual(progress.actual_study_seconds, 0)
        self.assertEqual(progress.timer_started_at, 50.0)
        self.assertEqual(ProgressService.elapsed_seconds(progress, now=80.0), 30)

    def test_final_exam_grading_detects_weak_topics_and_sections(self):
        sections = StudyService().generate_study_plan(
            [
                DocumentPage(1, "GRAPH SEARCH\nBreadth first search explores neighbors level by level."),
                DocumentPage(2, "SORTING\nMerge sort divides input and merges sorted halves."),
            ],
            pages_per_section=1,
        )
        exam = {
            "questions": [
                {
                    "id": 1,
                    "type": "multiple_choice",
                    "question": "Which algorithm explores neighbors level by level?",
                    "options": ["Breadth first search", "Merge sort"],
                    "answer": "Breadth first search",
                    "topic": "Graph search",
                },
                {
                    "id": 2,
                    "type": "short_answer",
                    "question": "What does merge sort do?",
                    "answer": "divides input and merges sorted halves",
                    "topic": "Sorting",
                },
            ]
        }

        result = ExamGradingService.grade_exam(exam, {"1": "Merge sort", "2": "divides input"}, sections)

        self.assertEqual(result["score"], 50)
        self.assertEqual(result["wrong_count"], 1)
        self.assertIn("Graph search", result["weak_topics"])
        self.assertTrue(result["weak_sections"])

    def test_persistence_saves_and_restores_progress(self):
        progress = ProgressService.default_state()
        progress.completed_sections.add(1)
        progress.section_quiz_scores[1] = 85
        sections = StudyService().generate_study_plan(
            [DocumentPage(1, "DATABASE INDEXES\nIndexes improve lookup performance.")],
            pages_per_section=1,
        )
        payload = PersistenceService.build_payload(
            pdf_name="notes.pdf",
            pages=[DocumentPage(1, "DATABASE INDEXES\nIndexes improve lookup performance.", "notes.pdf")],
            sections=sections,
            progress=progress,
            section_states=SectionStateService.ensure_states({}, [1]),
            final_exam={"questions": []},
            final_exam_answers={},
            final_exam_result=None,
            current_section_index=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            PersistenceService.save(payload, path)
            restored = PersistenceService.load(path)

        self.assertEqual(restored["pdf"]["name"], "notes.pdf")
        self.assertEqual(ProgressService.load(restored["progress"]).completed_sections, {1})
        self.assertEqual(PersistenceService.sections_from_payload(restored)[0].section_number, 1)


if __name__ == "__main__":
    unittest.main()
