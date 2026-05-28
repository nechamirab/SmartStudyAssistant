import unittest

from generation.answer_generator import AnswerGenerator
from generation.base import RetrievedContext
from generation.citation_formatter import CitationFormatter


class GenerationTests(unittest.TestCase):
    def test_citation_formatting(self):
        context = RetrievedContext(
            chunk_id="chunk-1",
            text="Alpha evidence.",
            score=0.87,
            source="lecture.pdf",
            page_number=4,
        )

        citation = CitationFormatter.citation_for(context, 1)

        self.assertEqual(citation.chunk_id, "chunk-1")
        self.assertIn("lecture.pdf", citation.label)
        self.assertIn("p. 4", citation.label)

    def test_weak_context_behavior(self):
        generator = AnswerGenerator(llm_provider="mock", weak_score_threshold=0.5)
        result = generator.generate(
            question="What is alpha?",
            contexts=[
                RetrievedContext(
                    chunk_id="weak",
                    text="Alpha is described briefly.",
                    score=0.01,
                    source="notes.pdf",
                    page_number=1,
                )
            ],
        )

        self.assertIsNotNone(result.weak_context_warning)
        self.assertNotIn("Warning:", result.answer)
        self.assertLess(result.confidence, 0.5)

    def test_mock_generation_is_deterministic(self):
        context = RetrievedContext(
            chunk_id="a",
            text="Machine learning learns patterns from data.",
            score=0.9,
            source="ml.pdf",
            page_number=2,
        )
        generator = AnswerGenerator(llm_provider="mock", show_citations=True)

        first = generator.generate("What does machine learning learn?", [context])
        second = generator.generate("What does machine learning learn?", [context])

        self.assertEqual(first.answer, second.answer)
        self.assertEqual(first.used_chunk_ids, ["a"])
        self.assertEqual(first.citations[0].chunk_id, "a")

    def test_mock_generation_uses_only_provided_context_when_citations_hidden(self):
        context_text = "Machine learning learns patterns from data."
        generator = AnswerGenerator(llm_provider="mock", show_citations=False)

        result = generator.generate(
            question="What does machine learning learn?",
            contexts=[
                RetrievedContext(
                    chunk_id="a",
                    text=context_text,
                    score=0.9,
                    source="ml.pdf",
                    page_number=2,
                )
            ],
        )

        self.assertEqual(result.answer, context_text)
        self.assertNotIn("outside", result.answer.lower())


if __name__ == "__main__":
    unittest.main()
