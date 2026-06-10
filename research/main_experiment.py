"""
Benchmark runner for Smart Study Assistant.

Usage:
    python main_experiment.py --dataset local
    python main_experiment.py --dataset local --rag-backend langchain
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

from rag import LangChainDependencyError, LangChainPipelineError, LangChainRAGPipeline
from services.evaluation_service import AggregatedMetrics, EvaluationService
from services.experiment_runner import (
    ExperimentComparator,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)


def define_experiments(
    provider: str = "mock",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    overlap: int | None = None,
    top_k: int = 3,
) -> List[ExperimentConfig]:
    """Define a simple set of benchmark configurations."""
    if overlap is not None:
        chunk_overlap = overlap
    return [
        ExperimentConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            embedding_provider=provider,
            embedding_model=embedding_model,
            answer_mode="retrieved_chunks",
        ),
    ]


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not_available"
    return f"{value:.{digits}f}"


def format_metric_short(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def save_results_to_csv(results: List[ExperimentResult], output_path: Path) -> None:
    """Save experiment results to CSV."""
    if not results:
        return

    fieldnames = list(results[0].to_dict().keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())


def save_results_to_markdown(
    results: List[ExperimentResult],
    comparator: ExperimentComparator,
    output_path: Path,
    rag_backend: str,
) -> None:
    """Save a human-readable benchmark summary."""
    lines: List[str] = [
        "# Smart Study Assistant Benchmark Results\n",
        "A clean local benchmark report for the RAG platform.\n",
        "## Configuration\n",
        "- PDF: data/example.pdf\n",
        "- Evaluation dataset: data/evaluation/eval_dataset.json\n",
        f"- RAG backend: {rag_backend}\n",
        "- Retrieval: semantic vector search\n",
        "## Results\n",
        "| Chunk Size | Overlap | Top K | Provider | Accuracy | Precision@K | Grounding | Avg Latency (ms) | Chunks |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]

    for result in results:
        config = result.config
        metrics = result.aggregated_metrics
        lines.append(
            f"| {config.chunk_size} | {config.chunk_overlap} | {config.top_k} | "
            f"{config.embedding_provider} | {metrics.accuracy:.3f} | "
            f"{metrics.precision_at_k:.3f} | {metrics.grounding_score:.3f} | "
            f"{metrics.avg_response_time * 1000:.1f} | {result.num_chunks} |\n"
        )

    best_accuracy = comparator.find_best_config_by_accuracy()
    if best_accuracy:
        lines.append("\n## Best Configuration\n")
        lines.append(
            f"- Best accuracy: {best_accuracy.config} ({best_accuracy.aggregated_metrics.accuracy:.3f})\n"
        )

    output_path.write_text("".join(lines), encoding="utf-8")


def run_langchain_benchmark(
    pdf_path: Path,
    eval_dataset_path: Path,
    config: ExperimentConfig,
    verbose: bool = True,
) -> ExperimentResult:
    """Run a simplified benchmark using the LangChain pipeline used by the UI."""
    embedding_model = config.embedding_model
    if config.embedding_provider == "mock":
        embedding_model = "mock"

    pipeline = LangChainRAGPipeline(
        embedding_model_name=embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
    )

    stats = pipeline.process_pdf(str(pdf_path))
    with eval_dataset_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    evaluation_results = []
    error_count = 0

    for index, row in enumerate(dataset, start=1):
        if verbose and index % max(1, len(dataset) // 5) == 0:
            print(f"     Processing question {index}/{len(dataset)}...")

        try:
            start_time = time.time()
            answer_payload = pipeline.answer_question(row["question"])
            end_time = time.time()
            retrieved_chunks = [chunk["text"] for chunk in answer_payload["retrieved_chunks"]]
            evaluation_results.append(
                EvaluationService.evaluate_single_question(
                    question=row["question"],
                    ground_truth_answer=row["answer"],
                    generated_answer=answer_payload["answer"],
                    retrieved_chunks=retrieved_chunks,
                    response_time=end_time - start_time,
                    expected_source_text=row["source_text"],
                )
            )
        except Exception as exc:
            error_count += 1
            if verbose:
                print(f"     Error on question {index}: {exc}")

    aggregated = AggregatedMetrics(evaluation_results)
    return ExperimentResult(
        config=config,
        num_chunks=stats["chunks"],
        num_questions_attempted=len(evaluation_results),
        evaluation_results=evaluation_results,
        aggregated_metrics=aggregated,
        error_count=error_count,
        notes="LangChain pipeline benchmark",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark for Smart Study Assistant.")
    parser.add_argument(
        "--dataset",
        choices=["local"],
        default="local",
        help="Dataset to use for the benchmark.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "sentence-transformers", "openai"],
        default="mock",
        help="Legacy embedding provider for the old benchmark path.",
    )
    parser.add_argument(
        "--rag-backend",
        choices=["legacy", "langchain"],
        default="legacy",
        help="Select the benchmark backend.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for the LangChain backend.",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks.")
    args = parser.parse_args()

    pdf_path = Path("data/example.pdf")
    eval_dataset_path = Path("data/evaluation/eval_dataset.json")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not eval_dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_dataset_path}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    provider = args.provider
    if args.rag_backend == "langchain":
        provider = "mock" if args.embedding_model.strip().lower() == "mock" else "sentence-transformers"

    configs = define_experiments(
        provider=provider,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        top_k=args.top_k,
    )

    try:
        if args.rag_backend == "langchain":
            results = [run_langchain_benchmark(pdf_path, eval_dataset_path, configs[0], verbose=True)]
        else:
            runner = ExperimentRunner(str(pdf_path), str(eval_dataset_path))
            results = runner.run_experiments(configs, verbose=True)
    except (LangChainDependencyError, LangChainPipelineError) as exc:
        raise SystemExit(f"LangChain benchmark failed: {exc}") from exc

    comparator = ExperimentComparator(results)
    csv_path = results_dir / "benchmark_results.csv"
    md_path = results_dir / "benchmark_summary.md"

    save_results_to_csv(results, csv_path)
    save_results_to_markdown(results, comparator, md_path, args.rag_backend)

    print("\nBenchmark complete!")
    print(f"Results saved to {results_dir}")
    print(comparator.generate_summary())


if __name__ == "__main__":
    main()
