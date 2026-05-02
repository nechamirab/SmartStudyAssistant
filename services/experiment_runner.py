"""
Experiment Runner: Orchestrates RAG experiments with different configurations.

How it works:
1. Load PDF and evaluation dataset
2. For each configuration (chunk_size, overlap, top_k, embedding_type):
   - Process PDF with those settings
   - Run evaluation dataset
   - Collect metrics
3. Compare results and generate report

This lets us measure exactly which design choices affect system quality.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.models import DocumentChunk
from services.pdf_service import PdfService
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.retrieval_service import RetrievalService
from services.qa_service import QAService
from services.evaluation_service import EvaluationService, AggregatedMetrics, EvaluationResult
from services.baseline_retriever import RetrievalBaselines

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_provider: str = "mock"
    embedding_model: str = "text-embedding-3-small"
    answer_mode: str = "retrieved_chunks"  # Options: retrieved_chunks, llm (future), baseline
    baseline_method: Optional[str] = None  # If answer_mode=baseline

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable config description."""
        return (
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"top_k={self.top_k}, "
            f"provider={self.embedding_provider}"
        )


@dataclass
class ExperimentResult:
    """Results from running one configuration."""
    config: ExperimentConfig
    num_chunks: int
    num_questions_attempted: int
    evaluation_results: List[EvaluationResult]
    aggregated_metrics: AggregatedMetrics
    error_count: int
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        metrics_dict = self.aggregated_metrics.to_dict()
        return {
            **self.config.to_dict(),
            "num_chunks": self.num_chunks,
            "questions_attempted": self.num_questions_attempted,
            "errors": self.error_count,
            **metrics_dict,
        }


class ExperimentRunner:
    """
    Runs RAG experiments with different configurations.
    """

    def __init__(
        self,
        pdf_path: str,
        eval_dataset_path: str,
    ):
        """
        Initialize experiment runner.

        Args:
            pdf_path: Path to PDF file
            eval_dataset_path: Path to evaluation dataset JSON
        """
        self.pdf_path = Path(pdf_path)
        self.eval_dataset_path = Path(eval_dataset_path)

        # Load evaluation dataset
        with open(self.eval_dataset_path) as f:
            self.eval_dataset = json.load(f)

        print(f"✓ Loaded {len(self.eval_dataset)} evaluation questions")
        print(f"✓ PDF: {self.pdf_path}")

    def run_experiments(
        self,
        configs: List[ExperimentConfig],
        verbose: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run experiments for multiple configurations.

        Args:
            configs: List of ExperimentConfig objects
            verbose: Print progress

        Returns:
            List of ExperimentResult objects
        """
        results = []

        for i, config in enumerate(configs, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Experiment {i}/{len(configs)}: {config}")
                print(f"{'='*60}")

            result = self.run_single_config(config, verbose=verbose)
            results.append(result)

        return results

    def run_single_config(
        self,
        config: ExperimentConfig,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Run experiment for a single configuration.

        Steps:
        1. Extract PDF
        2. Chunk with config settings
        3. Create embeddings
        4. Build vector store
        5. For each question:
           - Retrieve chunks
           - Generate answer
           - Calculate metrics

        Args:
            config: Configuration for this experiment
            verbose: Print progress

        Returns:
            ExperimentResult with all metrics
        """
        try:
            # Step 1: Extract PDF
            if verbose:
                print("  1. Extracting PDF...")
            pdf_service = PdfService()
            pages = pdf_service.extract_pages(self.pdf_path)
            if verbose:
                print(f"     ✓ Extracted {len(pages)} pages")

            # Step 2: Chunk with config settings
            if verbose:
                print(f"  2. Chunking (size={config.chunk_size}, overlap={config.chunk_overlap})...")
            chunk_service = ChunkService(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            chunks = chunk_service.chunk_pages(pages)
            if verbose:
                print(f"     ✓ Created {len(chunks)} chunks")

            # Step 3: Embed
            if verbose:
                print(f"  3. Embedding with {config.embedding_provider}...")
            embedding_service = EmbeddingService(
                provider=config.embedding_provider,
                model=config.embedding_model,
            )
            embeddings = embedding_service.embed_texts(chunks)
            if verbose:
                print(f"     ✓ Created {len(embeddings)} embeddings")

            # Step 4: Build vector store
            if verbose:
                print("  4. Building vector store...")
            vector_store = VectorStoreService()
            vector_store.add(chunks, embeddings)

            # Step 5: Setup retrieval
            retrieval_service = RetrievalService(
                embedding_service=embedding_service,
                vector_store=vector_store,
            )

            # Also setup baselines if needed
            baselines = RetrievalBaselines(chunks)

            # Step 6: Evaluate all questions
            if verbose:
                print(f"  5. Evaluating {len(self.eval_dataset)} questions...")

            evaluation_results = []
            error_count = 0

            for q_idx, q_data in enumerate(self.eval_dataset, 1):
                try:
                    if verbose and q_idx % max(1, len(self.eval_dataset) // 5) == 0:
                        print(f"     Processing question {q_idx}/{len(self.eval_dataset)}...")

                    result = self._evaluate_question(
                        q_data,
                        config,
                        retrieval_service,
                        baselines,
                        chunks,
                    )
                    evaluation_results.append(result)

                except Exception as e:
                    error_count += 1
                    if verbose:
                        print(f"     ⚠️  Error on question {q_idx}: {str(e)[:50]}")

            # Step 7: Aggregate results
            aggregated = AggregatedMetrics(evaluation_results)

            if verbose:
                print(f"\n  Results:")
                print(f"    Accuracy: {aggregated.accuracy:.3f}")
                print(f"    Precision@K: {aggregated.precision_at_k:.3f}")
                print(f"    Grounding Score: {aggregated.grounding_score:.3f}")
                print(f"    Avg Response Time: {aggregated.avg_response_time:.3f}s")

            return ExperimentResult(
                config=config,
                num_chunks=len(chunks),
                num_questions_attempted=len(evaluation_results),
                evaluation_results=evaluation_results,
                aggregated_metrics=aggregated,
                error_count=error_count,
                notes=f"Successfully evaluated {len(evaluation_results)} questions",
            )

        except Exception as e:
            if verbose:
                print(f"  ✗ Experiment failed: {e}")

            # Return empty result on failure
            return ExperimentResult(
                config=config,
                num_chunks=0,
                num_questions_attempted=0,
                evaluation_results=[],
                aggregated_metrics=AggregatedMetrics([]),
                error_count=1,
                notes=f"Experiment failed: {str(e)[:100]}",
            )

    def _evaluate_question(
        self,
        q_data: dict,
        config: ExperimentConfig,
        retrieval_service: RetrievalService,
        baselines: RetrievalBaselines,
        chunks: List[DocumentChunk],
    ) -> EvaluationResult:
        """
        Evaluate a single question.

        Args:
            q_data: Question data from eval_dataset
            config: Current configuration
            retrieval_service: FAISS retrieval service
            baselines: Baseline retrievers
            chunks: All chunks for this experiment

        Returns:
            EvaluationResult with all metrics
        """
        question = q_data["question"]
        ground_truth_answer = q_data["answer"]
        source_text = q_data["source_text"]

        # Step 1: Retrieve chunks
        start_time = time.time()

        if config.answer_mode == "baseline" and config.baseline_method:
            # Use baseline retrieval
            baseline_results = baselines.retrieve_all_baselines(
                question,
                top_k=config.top_k,
            )
            retrieved_chunks = RetrievalBaselines.chunks_to_text(
                baseline_results[config.baseline_method]
            )

        elif config.answer_mode in ("retrieved_chunks", "llm"):
            retrieval_response = retrieval_service.retrieve(

                question,

                top_k=config.top_k,

            )

            retrieved_chunks = [r.chunk.text for r in retrieval_response.results]
            retrieved_pages = [r.chunk.page_number for r in retrieval_response.results]

        else:
            retrieved_chunks = []
            retrieved_pages = []

        # Step 2: Generate answer
        if config.answer_mode == "llm":
            qa_service = QAService(retrieval_service)
            prompt = qa_service._build_prompt(question, retrieved_chunks, retrieved_pages)
            generated_answer = qa_service.llm_service.generate(prompt)

        elif retrieved_chunks:
            generated_answer = "\n".join(retrieved_chunks)

        else:
            generated_answer = ""

        response_time = time.time() - start_time

        # Step 3: Evaluate
        return EvaluationService.evaluate_single_question(
            question=question,
            ground_truth_answer=ground_truth_answer,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            response_time=response_time,
            expected_source_text=source_text,
        )


class ExperimentComparator:
    """Compare results across experiments."""

    def __init__(self, results: List[ExperimentResult]):
        self.results = results

    def find_best_config_by_accuracy(self) -> Optional[ExperimentResult]:
        """Find configuration with highest accuracy."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.aggregated_metrics.accuracy)

    def find_best_config_by_grounding(self) -> Optional[ExperimentResult]:
        """Find configuration with highest grounding score."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.aggregated_metrics.grounding_score)

    def find_best_config_by_speed(self) -> Optional[ExperimentResult]:
        """Find configuration with lowest response time."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.aggregated_metrics.avg_response_time)

    def generate_summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("EXPERIMENT SUMMARY")
        lines.append("="*70)

        lines.append(f"\nTotal experiments: {len(self.results)}")

        best_accuracy = self.find_best_config_by_accuracy()
        if best_accuracy:
            lines.append(
                f"\nBest Accuracy ({best_accuracy.aggregated_metrics.accuracy:.3f}): "
                f"{best_accuracy.config}"
            )

        best_grounding = self.find_best_config_by_grounding()
        if best_grounding:
            lines.append(
                f"Best Grounding ({best_grounding.aggregated_metrics.grounding_score:.3f}): "
                f"{best_grounding.config}"
            )

        best_speed = self.find_best_config_by_speed()
        if best_speed:
            lines.append(
                f"Best Speed ({best_speed.aggregated_metrics.avg_response_time:.3f}s): "
                f"{best_speed.config}"
            )

        lines.append("\n" + "="*70)
        return "\n".join(lines)
