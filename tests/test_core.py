from __future__ import annotations

import json
import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.models import DocumentPage
from services.auth_service import AuthService
from services.context_retrieval_service import ContextRetrievalService
from services.database_service import DatabaseService
from services.exam_grading_service import ExamGradingService
from services.exam_service import ExamOptions, ExamService
from services.persistence_service import PersistenceService
from services.progress_service import ProgressService
from services.quiz_grading_service import QuizGradingService
from services.section_state_service import SectionStateService
from services.study_service import StudySection, StudyService


def import_workflow_with_fake_streamlit():
    fake_streamlit = types.SimpleNamespace(session_state={})
    with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
        import ui.workflow as workflow
    return workflow


def import_state_with_fake_streamlit(session_state):
    fake_streamlit = types.SimpleNamespace(session_state=session_state)
    with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
        import ui.state as state
    state.st = fake_streamlit
    return state


class FakeSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


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

    def test_ai_study_plan_is_used_when_valid(self):
        pages = [
            DocumentPage(1, "GRAPH SEARCH\nBreadth first search explores neighbors level by level."),
            DocumentPage(2, "SORTING\nMerge sort divides input and merges sorted halves."),
        ]
        ai_payload = {
            "sections": [
                {
                    "section_number": 1,
                    "title": "Graph Search",
                    "start_page": 1,
                    "end_page": 1,
                    "estimated_minutes": 18,
                    "difficulty": "Medium",
                    "summary": "Understand breadth first search and graph exploration order.",
                    "key_concepts": ["Breadth First Search", "Graph Exploration"],
                    "learning_objectives": ["Explain level-order graph traversal."],
                },
                {
                    "section_number": 2,
                    "title": "Sorting",
                    "start_page": 2,
                    "end_page": 2,
                    "estimated_minutes": 20,
                    "difficulty": "Easy",
                    "summary": "Understand how merge sort divides and combines sorted data.",
                    "key_concepts": ["Merge Sort"],
                    "learning_objectives": ["Describe divide and merge steps."],
                },
            ]
        }

        with patch("services.study_service.GeneralAIService") as ai_service:
            ai_service.return_value.complete.return_value = {"ok": True, "answer": json.dumps(ai_payload)}
            sections = StudyService().generate_study_plan_for_sessions(pages, session_count=2)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "Section 1: Graph Search")
        self.assertEqual(sections[0].summary, "Understand breadth first search and graph exploration order.")
        self.assertEqual(sections[0].key_concepts, ["Breadth First Search", "Graph Exploration"])
        self.assertEqual(sections[0].difficulty, "Medium")

    def test_malformed_ai_study_plan_falls_back_to_heuristic(self):
        pages = [
            DocumentPage(1, "DYNAMIC PROGRAMMING\nMemoization stores repeated subproblem results."),
            DocumentPage(2, "Tabulation builds solutions from smaller states."),
        ]

        with patch("services.study_service.GeneralAIService") as ai_service:
            ai_service.return_value.complete.return_value = {"ok": True, "answer": "not json"}
            sections = StudyService().generate_study_plan_for_sessions(pages, session_count=2)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].start_page, 1)
        self.assertTrue(sections[0].summary)

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

    def test_retrieve_relevant_chunks_finds_correct_section(self):
        chunks = [
            {
                "section_number": 1,
                "section_title": "Breadth First Search",
                "start_page": 1,
                "end_page": 2,
                "page": 1,
                "text": "BFS explores graph nodes level by level using a queue.",
                "key_concepts": ["BFS", "Queue"],
            },
            {
                "section_number": 2,
                "section_title": "Gradient Descent",
                "start_page": 3,
                "end_page": 4,
                "page": 3,
                "text": "Gradient descent updates model weights using the loss gradient.",
                "key_concepts": ["Optimization", "Gradient"],
            },
        ]

        results = ContextRetrievalService.retrieve_relevant_chunks("How does BFS use a queue?", chunks)

        self.assertEqual(results[0]["section_number"], 1)
        self.assertIn("BFS", results[0]["text"])

    def test_retrieve_relevant_chunks_returns_empty_for_unrelated_question(self):
        chunks = [
            {
                "section_number": 1,
                "section_title": "Breadth First Search",
                "start_page": 1,
                "end_page": 1,
                "page": 1,
                "text": "BFS explores graph nodes level by level.",
                "key_concepts": ["BFS"],
            }
        ]

        results = ContextRetrievalService.retrieve_relevant_chunks("What is photosynthesis?", chunks)

        self.assertEqual(results, [])

    def test_format_chunks_for_prompt_includes_sources(self):
        formatted = ContextRetrievalService.format_chunks_for_prompt(
            [
                {
                    "section_number": 4,
                    "section_title": "Shortest Paths",
                    "start_page": 10,
                    "end_page": 12,
                    "page": 11,
                    "text": "Dijkstra computes shortest paths with non-negative edge weights.",
                    "key_concepts": ["Dijkstra"],
                }
            ]
        )

        self.assertIn("[Section 4 | Shortest Paths | Page 11]", formatted)
        self.assertIn("Dijkstra computes shortest paths", formatted)

    def test_retrieve_exam_context_covers_all_sections(self):
        sections = [
            StudySection(1, "Section 1: BFS", 1, 1, 10, "Easy", "Graph traversal.", ["Trace BFS."], ["BFS"]),
            StudySection(2, "Section 2: Gradient Descent", 2, 2, 10, "Medium", "Optimization.", ["Explain updates."], ["Gradient"]),
        ]
        pages = [
            DocumentPage(1, "BFS uses a queue to visit nodes level by level."),
            DocumentPage(2, "Gradient descent changes weights by following the negative gradient."),
        ]

        context = ContextRetrievalService.retrieve_exam_context(sections, pages, max_chars=5000)

        self.assertIn("Section 1: BFS", context)
        self.assertIn("Section 2: Gradient Descent", context)
        self.assertIn("Representative text:", context)

    def test_question_answering_uses_retrieved_chunks_not_full_pdf(self):
        workflow = import_workflow_with_fake_streamlit()

        sections = [
            StudySection(1, "Section 1: BFS", 1, 1, 10, "Easy", "Graph traversal.", [], ["BFS"]),
            StudySection(2, "Section 2: Gradient Descent", 2, 2, 10, "Medium", "Optimization.", [], ["Gradient"]),
        ]
        pages = [
            DocumentPage(1, "BFS uses a queue to visit graph nodes level by level."),
            DocumentPage(2, "Gradient descent updates model weights from a loss gradient."),
        ]

        class FakeSt:
            session_state = FakeSessionState({
                "pages": pages,
                "sections": sections,
                "language": "en",
            })

        with patch.object(workflow, "st", FakeSt), patch.object(workflow, "has_pdf", return_value=True):
            with patch.object(workflow, "GeneralAIService") as ai_service:
                ai_service.return_value.complete.return_value = {"ok": True, "answer": "BFS uses a queue.\n\nSource: Section 1 - Page 1"}
                answer = workflow.answer_section_question(sections[0], "How does BFS use a queue?")

        prompt = ai_service.return_value.complete.call_args.args[1]
        self.assertIn("BFS uses a queue", prompt)
        self.assertNotIn("Gradient descent updates", prompt)
        self.assertIn("Retrieved sources:", answer)

    def test_explain_section_still_uses_only_current_section(self):
        workflow = import_workflow_with_fake_streamlit()

        section = StudySection(1, "Section 1: BFS", 1, 1, 10, "Easy", "Graph traversal.", [], ["BFS"])
        current_text = "BFS uses a queue to visit graph nodes level by level."

        with patch.object(workflow, "section_context", return_value=current_text):
            with patch.object(workflow, "GeneralAIService") as ai_service:
                ai_service.return_value.ask.return_value = {"ok": True, "answer": "BFS explanation", "provider": "test"}
                answer = workflow.generate_explanation(section)

        prompt = ai_service.return_value.ask.call_args.args[1]
        self.assertIn(current_text, prompt)
        self.assertIn("Section title: Section 1: BFS", prompt)
        self.assertNotIn("Gradient descent", prompt)
        self.assertIn("BFS explanation", answer)

    def test_unsupported_pdf_question_returns_not_enough_information(self):
        workflow = import_workflow_with_fake_streamlit()

        sections = [StudySection(1, "Section 1: BFS", 1, 1, 10, "Easy", "Graph traversal.", [], ["BFS"])]
        pages = [DocumentPage(1, "BFS uses a queue to visit graph nodes level by level.")]

        class FakeSt:
            session_state = FakeSessionState({
                "pages": pages,
                "sections": sections,
                "language": "en",
            })

        with patch.object(workflow, "st", FakeSt), patch.object(workflow, "has_pdf", return_value=True):
            answer = workflow.answer_section_question(sections[0], "What is photosynthesis?")

        self.assertEqual(answer, workflow.NOT_ENOUGH_INFORMATION)

    def test_user_registration_creates_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})

            result = auth.register_user("student", "secret123")

        self.assertTrue(result["ok"])
        self.assertEqual(result["user"]["username"], "student")
        self.assertNotIn("password", result["user"])

    def test_duplicate_username_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})
            self.assertTrue(auth.register_user("student", "secret123")["ok"])

            result = auth.register_user("student", "another123")

        self.assertFalse(result["ok"])
        self.assertIn("exists", result["error"])

    def test_login_works_with_correct_password(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            session_state: dict[str, object] = {}
            auth = AuthService(database=db, session_state=session_state)
            auth.register_user("student", "secret123")
            auth.logout_user()

            result = auth.login_user("student", "secret123")

        self.assertTrue(result["ok"])
        self.assertEqual(result["user"]["username"], "student")

    def test_login_fails_with_wrong_password(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})
            auth.register_user("student", "secret123")

            result = auth.login_user("student", "badpass")

        self.assertFalse(result["ok"])
        self.assertIn("Invalid", result["error"])

    def test_saved_study_session_can_be_created_and_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})
            user = auth.register_user("student", "secret123")["user"]
            pages = [DocumentPage(1, "BFS uses a queue to visit nodes level by level.")]
            sections = [
                StudySection(1, "Section 1: BFS", 1, 1, 20, "Easy", "Graph traversal.", ["Trace BFS."], ["BFS"])
            ]

            _document_id, session_id = db.create_session_from_state(
                user_id=user["id"],
                filename="notes.pdf",
                title="Algorithms",
                language="en",
                pages=pages,
                sections=sections,
            )
            loaded = db.load_study_session(user["id"], session_id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["session"]["filename"], "notes.pdf")
        self.assertEqual(loaded["sections"][0].title, "Section 1: BFS")
        self.assertIn("BFS uses a queue", loaded["pages"][0].text)

    def test_progress_update_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})
            user = auth.register_user("student", "secret123")["user"]
            pages = [DocumentPage(1, "BFS uses a queue to visit nodes level by level.")]
            sections = [
                StudySection(1, "Section 1: BFS", 1, 1, 20, "Easy", "Graph traversal.", ["Trace BFS."], ["BFS"])
            ]
            _document_id, session_id = db.create_session_from_state(
                user_id=user["id"],
                filename="notes.pdf",
                title="Algorithms",
                language="en",
                pages=pages,
                sections=sections,
            )
            progress = ProgressService.default_state()
            progress.completed_sections.add(1)
            states = SectionStateService.ensure_states({}, [1])
            states["1"]["explanation"] = "BFS explanation"

            db.save_runtime_state(
                user_id=user["id"],
                session_id=session_id,
                sections=sections,
                progress=progress,
                section_states=states,
                final_exam=None,
                final_exam_answers={},
                final_exam_result=None,
            )
            loaded = db.load_study_session(user["id"], session_id)

        self.assertIn(1, loaded["progress"].completed_sections)
        self.assertEqual(loaded["section_states"]["1"]["explanation"], "BFS explanation")

    def test_quiz_attempt_save_load_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})
            user = auth.register_user("student", "secret123")["user"]
            document_id = db.create_document(user["id"], "notes.pdf")
            session_id = db.create_study_session(user["id"], document_id, "Algorithms", "en")

            db.save_quiz_attempt(
                user["id"],
                session_id,
                1,
                [{"type": "multiple_choice", "question": "Q?", "answer": "A"}],
                {"1": "A"},
                100,
                ["Correct."],
            )
            attempts = db.load_quiz_attempts(user["id"], session_id)

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["score"], 100)
        self.assertEqual(attempts[0]["answers"], {"1": "A"})

    def test_exam_attempt_save_load_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseService(Path(tmpdir) / "test.db")
            auth = AuthService(database=db, session_state={})
            user = auth.register_user("student", "secret123")["user"]
            document_id = db.create_document(user["id"], "notes.pdf")
            session_id = db.create_study_session(user["id"], document_id, "Algorithms", "en")

            db.save_exam_attempt(
                user["id"],
                session_id,
                {"questions": [{"id": 1, "question": "Explain BFS"}]},
                {"1": "It uses a queue"},
                90,
                ["BFS"],
            )
            attempts = db.load_exam_attempts(user["id"], session_id)

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["score"], 90)
        self.assertEqual(attempts[0]["weak_topics"], ["BFS"])

    def test_logged_in_user_does_not_restore_legacy_json_pdf(self):
        state_module = import_state_with_fake_streamlit(
            FakeSessionState({"auth_user": {"id": 1, "username": "first"}})
        )
        legacy_payload = {
            "pdf": {"name": "legacy.pdf", "page_count": 1},
            "pages": [{"page_number": 1, "text": "Legacy PDF text."}],
            "sections": [
                {
                    "section_number": 1,
                    "title": "Legacy Section",
                    "start_page": 1,
                    "end_page": 1,
                    "estimated_minutes": 20,
                    "difficulty": "Easy",
                    "summary": "Legacy summary.",
                    "learning_objectives": ["Review legacy."],
                    "key_concepts": ["Legacy"],
                }
            ],
        }

        with patch.object(state_module.PersistenceService, "load", return_value=legacy_payload):
            state_module.init_state()

        self.assertEqual(state_module.st.session_state.pdf_name, "")
        self.assertEqual(state_module.st.session_state.pages, [])
        self.assertEqual(state_module.st.session_state.sections, [])

    def test_switching_users_clears_active_pdf_state(self):
        session_state = FakeSessionState(
            {
                "auth_user": {"id": 2, "username": "second"},
                "active_auth_user_id": 1,
                "pdf_name": "first-user.pdf",
                "pages": [DocumentPage(1, "First user data")],
                "sections": [
                    StudySection(1, "First Section", 1, 1, 20, "Easy", "Summary", [], ["First"])
                ],
                "current_db_session_id": 10,
                "persistence_loaded": True,
            }
        )
        state_module = import_state_with_fake_streamlit(session_state)

        with patch.object(state_module.PersistenceService, "load", return_value={}):
            state_module.init_state()

        self.assertEqual(state_module.st.session_state.active_auth_user_id, 2)
        self.assertEqual(state_module.st.session_state.pdf_name, "")
        self.assertEqual(state_module.st.session_state.pages, [])
        self.assertEqual(state_module.st.session_state.current_db_session_id, None)


if __name__ == "__main__":
    unittest.main()
