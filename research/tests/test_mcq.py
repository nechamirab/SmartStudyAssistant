import unittest

from generation.base import RetrievedContext
from generation.mcq_generator import MCQGenerator


class MCQTests(unittest.TestCase):
    def test_mcq_output_shape(self):
        context = RetrievedContext(
            chunk_id="c1",
            text=(
                "Machine learning systems learn patterns from training data. "
                "Retrieval augmented generation grounds answers in evidence."
            ),
            score=0.9,
            source="notes.pdf",
            page_number=1,
        )

        questions = MCQGenerator().generate([context], count=1, difficulty="easy")

        self.assertEqual(len(questions), 1)
        self.assertEqual(len(questions[0].options), 4)
        self.assertIn(questions[0].correct_answer, questions[0].options)
        self.assertEqual(questions[0].citation["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
