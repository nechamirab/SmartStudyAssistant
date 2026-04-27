"""
Evaluation Service: Metrics for assessing RAG system quality.

Why each metric matters:
1. Accuracy (token F1-score): Did we generate the right answer?
2. Precision@K: Did retrieval find the right source chunks?
3. Grounding Score: Is the answer actually grounded in retrieved chunks?
4. Response Time: Is the system fast enough for real use?

These metrics help us understand which design choices matter most
for the final system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass(frozen=True)
class MetricResult:
    """Single metric measurement."""
    metric_name: str
    value: float
    details: Optional[dict] = None


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation for one question."""
    question: str
    accuracy: float
    precision_at_k: float
    grounding_score: float
    response_time: float
    retrieved_chunks: List[str]
    generated_answer: str
    ground_truth_answer: str
    success: bool  # Did all metrics compute without error?


class EvaluationService:
    """
    Computes quality metrics for RAG system outputs.
    """

    @staticmethod
    def calculate_accuracy_token_f1(
        predicted_answer: str,
        ground_truth_answer: str,
    ) -> float:
        """
        Calculate token-level F1-score between predicted and ground truth.

        Why F1 instead of exact match?
        - Exact match is too strict (answer phrasing varies)
        - F1-score rewards partial correctness
        - Captures semantic overlap

        Scale: 0.0 (completely wrong) to 1.0 (perfect match)

        Args:
            predicted_answer: Generated answer text
            ground_truth_answer: Expected answer text

        Returns:
            F1 score between 0 and 1
        """
        if not predicted_answer or not ground_truth_answer:
            return 0.0

        # Tokenize: split on whitespace and convert to lowercase
        pred_tokens = set(predicted_answer.lower().split())
        truth_tokens = set(ground_truth_answer.lower().split())

        if not truth_tokens:
            return 0.0

        # Calculate precision, recall, F1
        intersection = pred_tokens & truth_tokens
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(truth_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return min(f1, 1.0)  # Ensure in range [0, 1]

    @staticmethod
    def calculate_precision_at_k(
        retrieved_chunks: List[str],
        expected_source_text: str,
        k: Optional[int] = None,
    ) -> float:
        """
        Precision@K: What fraction of top-K retrieved chunks contain the answer source?

        Why this matters:
        - Even if we have good chunks, do they actually help answer the question?
        - Measures retrieval quality for the specific question
        - High Precision@K = retrieval is focused and relevant

        Implementation:
        - Check if expected_source_text appears in any of the top-K chunks
        - Return fraction of chunks that contain the source

        Args:
            retrieved_chunks: List of chunk texts retrieved from vector store
            expected_source_text: Source text that should be in a good chunk
            k: Only count first k chunks (None = use all)

        Returns:
            Precision value between 0 and 1
        """
        if not retrieved_chunks or not expected_source_text:
            return 0.0

        # Use all chunks if k not specified
        if k is None:
            k = len(retrieved_chunks)

        # Take only top-k chunks
        top_k_chunks = retrieved_chunks[:k]

        # Count how many chunks contain the source text
        # (case-insensitive substring matching)
        source_lower = expected_source_text.lower()
        matches = sum(
            1 for chunk in top_k_chunks
            if source_lower in chunk.lower()
        )

        # Precision = matches / total chunks retrieved
        precision = matches / len(top_k_chunks) if top_k_chunks else 0.0
        return min(precision, 1.0)

    @staticmethod
    def calculate_grounding_score(
        answer: str,
        retrieved_context: List[str],
    ) -> float:
        """
        Grounding Score: What percentage of the answer is grounded in retrieved chunks?

        Why this matters:
        - Prevents hallucinations (answer making up facts)
        - Ensures answer is based on actual document content
        - Higher score = more trustworthy answer

        Implementation:
        - Tokenize answer and context
        - Count how many answer tokens appear in context
        - Grounding = (grounded_tokens / total_answer_tokens)

        Scale: 0.0 (hallucinated) to 1.0 (fully grounded)

        Args:
            answer: Generated answer text
            retrieved_context: List of chunk texts used for answer

        Returns:
            Grounding score between 0 and 1
        """
        if not answer or not retrieved_context:
            return 0.0

        # Tokenize answer into words
        answer_tokens = set(answer.lower().split())

        # Combine all context into one string
        context_text = " ".join(retrieved_context).lower()
        context_tokens = set(context_text.split())

        if not answer_tokens:
            return 1.0  # Empty answer is trivially grounded

        # Count how many answer tokens appear in context
        grounded_tokens = answer_tokens & context_tokens
        grounding_score = len(grounded_tokens) / len(answer_tokens)

        return min(grounding_score, 1.0)

    @staticmethod
    def calculate_response_time(
        start_time: float,
        end_time: float,
    ) -> float:
        """
        Response Time: How long did it take to generate an answer?

        Why this matters:
        - System must be fast for interactive use
        - Measures efficiency of entire pipeline
        - Affects student experience

        Args:
            start_time: Time when query started (from time.time())
            end_time: Time when answer was generated (from time.time())

        Returns:
            Time in seconds (float)
        """
        return max(0.0, end_time - start_time)

    @staticmethod
    def evaluate_single_question(
        question: str,
        ground_truth_answer: str,
        generated_answer: str,
        retrieved_chunks: List[str],
        response_time: float,
        expected_source_text: str,
    ) -> EvaluationResult:
        """
        Evaluate a single Q&A pair using all metrics.

        Args:
            question: The question asked
            ground_truth_answer: Expected correct answer
            generated_answer: Answer generated by system
            retrieved_chunks: Chunks retrieved for this question
            response_time: Time to generate answer
            expected_source_text: Source text that should be in chunks

        Returns:
            EvaluationResult with all metrics
        """
        try:
            accuracy = EvaluationService.calculate_accuracy_token_f1(
                generated_answer, ground_truth_answer
            )
            precision_at_k = EvaluationService.calculate_precision_at_k(
                retrieved_chunks, expected_source_text
            )
            grounding = EvaluationService.calculate_grounding_score(
                generated_answer, retrieved_chunks
            )

            return EvaluationResult(
                question=question,
                accuracy=accuracy,
                precision_at_k=precision_at_k,
                grounding_score=grounding,
                response_time=response_time,
                retrieved_chunks=retrieved_chunks,
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth_answer,
                success=True,
            )
        except Exception as e:
            print(f"  ⚠️  Error evaluating question: {e}")
            return EvaluationResult(
                question=question,
                accuracy=0.0,
                precision_at_k=0.0,
                grounding_score=0.0,
                response_time=response_time,
                retrieved_chunks=[],
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth_answer,
                success=False,
            )


class AggregatedMetrics:
    """Aggregate metrics across multiple questions."""

    def __init__(self, results: List[EvaluationResult]):
        """
        Aggregate evaluation results.

        Args:
            results: List of EvaluationResult objects
        """
        self.results = results
        self.count = len(results)
        self.successful = sum(1 for r in results if r.success)

    @property
    def accuracy(self) -> float:
        """Average token F1-score."""
        if not self.results:
            return 0.0
        return sum(r.accuracy for r in self.results) / self.count

    @property
    def precision_at_k(self) -> float:
        """Average Precision@K."""
        if not self.results:
            return 0.0
        return sum(r.precision_at_k for r in self.results) / self.count

    @property
    def grounding_score(self) -> float:
        """Average grounding score."""
        if not self.results:
            return 0.0
        return sum(r.grounding_score for r in self.results) / self.count

    @property
    def avg_response_time(self) -> float:
        """Average response time (seconds)."""
        if not self.results:
            return 0.0
        return sum(r.response_time for r in self.results) / self.count

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "grounding_score": round(self.grounding_score, 4),
            "avg_response_time_sec": round(self.avg_response_time, 3),
            "questions_successful": f"{self.successful}/{self.count}",
        }
