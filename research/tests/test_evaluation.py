import unittest

from main_experiment import define_experiments, format_metric, format_metric_short
from services.evaluation_service import AggregatedMetrics, EvaluationService


class EvaluationTests(unittest.TestCase):
    def test_missing_source_text_makes_retrieval_metrics_unavailable(self):
        result = EvaluationService.evaluate_single_question(
            question="What is alpha?",
            ground_truth_answer="alpha",
            generated_answer="alpha",
            retrieved_chunks=["alpha context"],
            response_time=0.1,
            expected_source_text="",
        )

        self.assertIsNone(result.precision_at_k)
        self.assertIsNone(result.recall_at_k)
        aggregate = AggregatedMetrics([result]).to_dict()
        self.assertEqual(aggregate["retrieval_labels_available"], 0)
        self.assertIsNone(aggregate["precision_at_k"])
        self.assertIsNone(aggregate["recall_at_k"])

    def test_grounding_score_uses_context_tokens(self):
        self.assertEqual(
            EvaluationService.calculate_grounding_score("alpha beta", ["alpha beta gamma"]),
            1.0,
        )

    def test_define_experiments_accepts_explicit_overlap(self):
        configs = define_experiments(chunk_size=500, overlap=50, top_k=3)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].chunk_size, 500)
        self.assertEqual(configs[0].chunk_overlap, 50)

    def test_optional_metric_formatting_does_not_force_zero(self):
        self.assertEqual(format_metric(None), "not_available")
        self.assertEqual(format_metric_short(None), "n/a")


if __name__ == "__main__":
    unittest.main()
