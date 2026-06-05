import unittest
import urllib.error
from unittest.mock import patch

from core.models import DocumentPage
from services.exam_service import ExamGenerationError, ExamRequest, FullExamService
from services.rag_service import PDFRAGService


class FullExamServiceTests(unittest.TestCase):
    def _index(self):
        rag = PDFRAGService(
            chunk_size=400,
            chunk_overlap=40,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        return rag.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text=(
                        "Retrieval augmented generation retrieves evidence before answering. "
                        "Grounded systems cite source documents and avoid unsupported claims. "
                        "Vector stores compare embedded questions with embedded document chunks."
                    ),
                    source_id="rag.pdf",
                )
            ],
            "rag.pdf",
        )

    @patch("services.exam_service.read_groq_api_key", return_value="")
    def test_missing_groq_key_has_clear_error(self, _mock_key):
        with self.assertRaisesRegex(ExamGenerationError, "Groq API key is missing"):
            FullExamService().generate_exam(self._index(), ExamRequest(number_of_questions=2))

    @patch("services.exam_service.read_groq_api_key", return_value="")
    def test_test_fallback_is_explicitly_marked(self, _mock_key):
        exam = FullExamService(allow_test_fallback=True).generate_exam(
            self._index(),
            ExamRequest(
                number_of_questions=4,
                question_types=["multiple_choice", "open_question", "true_false", "short_answer"],
                difficulty="medium",
                include_answer_key=True,
            ),
        )

        self.assertTrue(exam["fallback_used"])
        self.assertIn("Test fallback used", exam["fallback_note"])
        self.assertEqual(len(exam["questions"]), 4)
        self.assertIn("answer_key", exam)
        self.assertEqual(exam["questions"][0]["source_references"][0]["page_number"], 1)

    def test_prompt_requires_pdf_only_grounding_and_json(self):
        prompt = FullExamService.build_prompt(
            "rag.pdf",
            "[PDF: rag.pdf; page_number: 1; chunk_id: c1]\nGrounded systems cite sources.",
            ExamRequest(number_of_questions=1, question_types=["short_answer"]),
            "easy",
        )

        self.assertIn("ONLY the uploaded PDF context", prompt)
        self.assertIn("Do not use outside knowledge", prompt)
        self.assertIn("Return valid JSON only", prompt)
        self.assertIn("page_number", prompt)
        self.assertIn("chunk_id", prompt)

    @patch("services.exam_service.read_groq_api_key", return_value="test-key")
    def test_groq_generation_returns_structured_json(self, _mock_key):
        content = (
            '{"choices":[{"message":{"content":"{\\"title\\":\\"AI Quiz / Exam\\",'
            '\\"questions\\":[{\\"id\\":1,\\"type\\":\\"short_answer\\",\\"difficulty\\":\\"easy\\",'
            '\\"question\\":\\"What do grounded systems cite?\\",\\"options\\":[],'
            '\\"answer\\":\\"source documents\\",\\"source_references\\":[{\\"page_number\\":1,'
            '\\"chunk_id\\":\\"rag_pdf_page_1_chunk_1\\"}]}],\\"answer_key\\":[{\\"id\\":1,'
            '\\"answer\\":\\"source documents\\",\\"source_references\\":[{\\"page_number\\":1,'
            '\\"chunk_id\\":\\"rag_pdf_page_1_chunk_1\\"}]}]}"}}]}'
        )
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self):
                return content.encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            exam = FullExamService().generate_exam(
                self._index(),
                ExamRequest(number_of_questions=1, question_types=["short_answer"], difficulty="easy"),
            )

        self.assertFalse(exam["fallback_used"])
        self.assertEqual(exam["questions"][0]["type"], "short_answer")
        self.assertEqual(exam["answer_key"][0]["source_references"][0]["page_number"], 1)

    def test_groq_limit_error_has_required_message(self):
        class FakeErrorBody:
            def read(self):
                return b'{"error":{"message":"rate limit"}}'

            def close(self):
                return None

        error = urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=FakeErrorBody(),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                ExamGenerationError,
                "Groq free API limit reached. Please try again later or reduce the number of questions.",
            ):
                FullExamService()._generate_with_groq("prompt", "test-key")

    def test_malformed_ai_final_exam_can_fall_back_without_crashing(self):
        with patch("services.exam_service.read_groq_api_key", return_value="test-key"):
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("bad response")):
                exam = FullExamService().generate_exam_with_fallback(
                    self._index(),
                    ExamRequest(number_of_questions=2, question_types=["short_answer"]),
                )

        self.assertTrue(exam["fallback_used"])
        self.assertEqual(len(exam["questions"]), 2)
        self.assertTrue(exam["questions"][0]["source_references"])

    def test_incomplete_ai_payload_is_normalized(self):
        payload = FullExamService._normalize_payload(
            {
                "questions": [
                    {
                        "question": "What do grounded systems cite?",
                        "source_references": [{"page_number": 1, "chunk_id": "rag_pdf_page_1_chunk_1"}],
                    }
                ]
            }
        )

        self.assertEqual(payload["questions"][0]["id"], 1)
        self.assertEqual(payload["questions"][0]["type"], "short_answer")
        self.assertEqual(payload["questions"][0]["source_references"][0]["label"], "Source: Page 1")


if __name__ == "__main__":
    unittest.main()
