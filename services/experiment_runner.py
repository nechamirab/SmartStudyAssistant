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

import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

from core.models import DocumentChunk
from services.chunk_service import ChunkService
from services.dataset_loader import DatasetLoader, ExperimentDataset
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService, AggregatedMetrics, EvaluationResult
from services.baseline_retriever import RetrievalBaselines
from reranking.rerankers import HeuristicReranker
from vectorstores.factory import VectorStoreFactory
from generation.answer_generator import AnswerGenerator


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_provider: str = "mock"
    embedding_model: str = ""
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    embedding_dimension: int | None = None
    chunking_strategy: str = "recursive"
    retrieval_mode: str = "semantic"  # Options: semantic, bm25, hybrid
    reranker: Optional[str] = None  # Options: heuristic
    vector_store: str = "memory"  # Options: memory, faiss, chroma, qdrant
    vector_store_path: Optional[str] = None
    generation_mode: str = "retrieved_chunks"  # Options: retrieved_chunks, grounded
    llm_provider: str = "mock"  # Options: mock, openai
    show_citations: bool = False
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
            f"provider={self.embedding_provider}, "
            f"model={self.embedding_model or 'default'}, "
            f"chunking={self.chunking_strategy}, "
            f"retrieval={self.retrieval_mode}, "
            f"vector_store={self.vector_store}, "
            f"generation={self.generation_mode}"
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

    def retrieval_method(self) -> str:
        """Label the retrieval method used in reports."""
        if self.config.baseline_method:
            return f"baseline:{self.config.baseline_method}"
        if self.config.reranker:
            return f"{self.config.retrieval_mode}+rerank:{self.config.reranker}"
        return self.config.retrieval_mode


class ExperimentRunner:
    """
    Runs RAG experiments with different configurations.
    """

    def __init__(
        self,
        pdf_path: str,
        eval_dataset_path: str,
        dataset: ExperimentDataset | None = None,
    ):
        """
        Initialize experiment runner.

        Args:
            pdf_path: Path to PDF file
            eval_dataset_path: Path to evaluation dataset JSON
            dataset: Optional pre-loaded dataset. If omitted, load a local PDF dataset.
        """
        if dataset is None:
            dataset = DatasetLoader.load_pdf_dataset(pdf_path, eval_dataset_path)

        self.dataset = dataset
        self.pdf_path = Path(pdf_path)
        self.eval_dataset_path = Path(eval_dataset_path)
        self.eval_dataset = dataset.eval_questions

        print(f"✓ Loaded dataset: {self.dataset.name}")
        print(f"✓ Loaded {len(self.eval_dataset)} evaluation questions")
        print(f"✓ Source: {self.dataset.source_path}")

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
            # Step 1: Load dataset pages
            if verbose:
                print("  1. Loading dataset pages...")
            pages = self.dataset.pages
            if verbose:
                print(f"     ✓ Loaded {len(pages)} pages/sections")

            # Step 2: Chunk with config settings
            if verbose:
                print(f"  2. Chunking (size={config.chunk_size}, overlap={config.chunk_overlap})...")
            chunk_service = ChunkService(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                strategy=config.chunking_strategy,
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
                batch_size=config.embedding_batch_size,
                normalize_embeddings=config.normalize_embeddings,
            )
            embeddings = embedding_service.embed_texts(chunks)
            config.embedding_provider = embedding_service.provider
            config.embedding_model = embedding_service.model
            config.embedding_dimension = embedding_service.embedding_dimension
            if verbose:
                print(
                    f"     ✓ Created {len(embeddings)} embeddings "
                    f"(dim={config.embedding_dimension or 'unknown'})"
                )

            # Step 4: Build vector store
            if verbose:
                print(f"  4. Building vector store ({config.vector_store})...")
            vector_store = VectorStoreFactory.create(
                backend=config.vector_store,
                collection_name=f"{self.dataset.name}_{config.chunking_strategy}",
                persist_path=config.vector_store_path,
            )
            vector_store.add(chunks, embeddings)
            if config.vector_store_path:
                vector_store.save(config.vector_store_path)

            # Step 5: Setup retrieval
            reranker = HeuristicReranker() if config.reranker == "heuristic" else None
            retrieval_service = RetrievalService(
                embedding_service=embedding_service,
                vector_store=vector_store,
                chunks=chunks,
                retrieval_mode=config.retrieval_mode,
                reranker=reranker,
            )
            answer_generator = AnswerGenerator(
                llm_provider=config.llm_provider,
                show_citations=config.show_citations,
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
                        answer_generator,
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
                precision = aggregated.optional_average("precision_at_k")
                precision_label = f"{precision:.3f}" if precision is not None else "n/a"
                print(f"    Precision@K: {precision_label}")
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
        answer_generator: AnswerGenerator,
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
        question = str(q_data.get("question", "") or "").strip()
        ground_truth_answer = str(q_data.get("answer", "") or "").strip()
        source_text = str(q_data.get("source_text", "") or "").strip()

        if not question or not ground_truth_answer:
            raise ValueError("Evaluation record is missing question or answer.")

        # Step 1: Retrieve chunks
        start_time = time.time()

        if config.answer_mode == "retrieved_chunks":
            # Use standard retrieval
            retrieval_response = retrieval_service.retrieve(
                question,
                top_k=config.top_k,
            )
            retrieved_chunks = [r.chunk.text for r in retrieval_response.results]
            used_chunk_ids: list[str] = []
            cited_chunk_ids: list[str] = []

        elif config.answer_mode == "baseline" and config.baseline_method:
            # Use baseline retrieval
            baseline_results = baselines.retrieve_all_baselines(
                question,
                top_k=config.top_k,
            )
            retrieved_chunks = RetrievalBaselines.chunks_to_text(
                baseline_results[config.baseline_method]
            )
            used_chunk_ids = []
            cited_chunk_ids = []
        else:
            retrieved_chunks = []
            used_chunk_ids = []
            cited_chunk_ids = []

        # Step 2: Generate answer
        if config.answer_mode == "retrieved_chunks" and config.generation_mode in {
            "grounded",
            "grounded_mock",
            "llm",
        }:
            generation_contexts = AnswerGenerator.contexts_from_search_results(
                retrieval_response.results
            )
            generation_result = answer_generator.generate(
                question=question,
                contexts=generation_contexts,
            )
            generated_answer = generation_result.answer
            used_chunk_ids = generation_result.used_chunk_ids
            cited_chunk_ids = [citation.chunk_id for citation in generation_result.citations]
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
            used_chunk_ids=used_chunk_ids,
            cited_chunk_ids=cited_chunk_ids,
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

    def find_worst_config_by_accuracy(self) -> Optional[ExperimentResult]:
        """Find configuration with lowest accuracy."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.aggregated_metrics.accuracy)

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
