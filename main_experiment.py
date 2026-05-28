"""
Main Experiment Runner
======================

This script runs experiments comparing different RAG configurations.

Usage:
    python main_experiment.py

What it does:
1. Loads evaluation dataset (10 game theory questions)
2. Tests multiple configurations (chunk size, overlap, top-k)
3. Measures: accuracy, precision@K, grounding, speed
4. Saves results to CSV and markdown
5. Prints summary table

Why different chunk sizes matter:
- Small chunks (300): More precise retrieval, but context lost
- Medium chunks (500): Balance of context and precision
- Large chunks (800): More context, but noise added

Why different top-K values matter:
- top_k=3: Fast, focused results
- top_k=5: More information, slower
- top_k=7: Too much context, diminishing returns

This lets you understand the trade-offs and write informed analysis in your report.
"""

import csv
import argparse
from pathlib import Path
from typing import List

from services.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentComparator,
)
from services.dataset_loader import DatasetLoader, DatasetLoadError
from services.ragbench_loader import RAGBenchLoader


def define_experiments(
    embedding_provider: str = "mock",
    chunk_size: int | None = None,
    top_k: int | None = None,
    chunking_strategy: str = "recursive",
    retrieval_mode: str = "semantic",
    reranker: str | None = None,
) -> List[ExperimentConfig]:
    """
    Define configurations to test.

    These are carefully chosen to test realistic trade-offs:
    - Small chunks vs large chunks
    - Shallow retrieval vs deep retrieval
    - Cost vs quality
    """
    if chunk_size is not None or top_k is not None:
        selected_chunk_size = chunk_size or 500
        selected_top_k = top_k or 3
        return [
            ExperimentConfig(
                chunk_size=selected_chunk_size,
                chunk_overlap=max(0, selected_chunk_size // 10),
                top_k=selected_top_k,
                embedding_provider=embedding_provider,
                chunking_strategy=chunking_strategy,
                retrieval_mode=retrieval_mode,
                reranker=reranker,
                answer_mode="retrieved_chunks",
            )
        ]

    configs = [
        # ===== BASELINE: Current default settings =====
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),

        # ===== LLM ANSWER GENERATION =====
        # Same retrieval settings as baseline, but answer is generated via QAService / LLM
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider="mock",
            answer_mode="llm",
        ),

        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider="openai",
            answer_mode="llm"
        ),

        # ===== CHUNK SIZE EXPERIMENTS =====
        # Small chunks: More focused retrieval
        ExperimentConfig(
            chunk_size=300,
            chunk_overlap=30,
            top_k=3,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),
        # Large chunks: More context
        ExperimentConfig(
            chunk_size=800,
            chunk_overlap=80,
            top_k=3,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),

        # ===== TOP-K EXPERIMENTS =====
        # Shallow retrieval
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=1,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),
        # Deep retrieval
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=5,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),

        # ===== COMBINED EXPERIMENTS =====
        # Small chunks + deep retrieval (most precise)
        ExperimentConfig(
            chunk_size=300,
            chunk_overlap=30,
            top_k=5,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),
        # Large chunks + shallow retrieval (fastest)
        ExperimentConfig(
            chunk_size=800,
            chunk_overlap=80,
            top_k=1,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            reranker=reranker,
            answer_mode="retrieved_chunks",
        ),

        # ===== BASELINE COMPARISON =====
        # Keyword search baseline
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            answer_mode="baseline",
            baseline_method="keyword_overlap",
        ),
        # Random baseline
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider=embedding_provider,
            chunking_strategy=chunking_strategy,
            retrieval_mode=retrieval_mode,
            answer_mode="baseline",
            baseline_method="random",
        ),
    ]
    return configs


def save_results_to_csv(
    results: List,
    output_path: Path,
    dataset_name: str,
) -> None:
    """
    Save experiment results to CSV for analysis.

    Each row = one configuration
    Columns = metrics

    Args:
        results: List of ExperimentResult objects
        output_path: Where to save CSV
    """
    if not results:
        print("No results to save")
        return

    fieldnames = [
        "dataset_name",
        "question",
        "expected_answer",
        "generated_answer",
        "precision_at_k",
        "accuracy",
        "grounding_score",
        "response_time",
        "chunk_size",
        "top_k",
        "embedding_provider",
        "chunking_strategy",
        "retrieval_mode",
        "reranker",
        "retrieval_method",
        "chunk_overlap",
        "num_chunks",
        "success",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            config = result.config
            for evaluation in result.evaluation_results:
                writer.writerow(
                    {
                        "dataset_name": dataset_name,
                        "question": evaluation.question,
                        "expected_answer": evaluation.ground_truth_answer,
                        "generated_answer": evaluation.generated_answer,
                        "precision_at_k": round(evaluation.precision_at_k, 4),
                        "accuracy": round(evaluation.accuracy, 4),
                        "grounding_score": round(evaluation.grounding_score, 4),
                        "response_time": round(evaluation.response_time, 4),
                        "chunk_size": config.chunk_size,
                        "top_k": config.top_k,
                        "embedding_provider": config.embedding_provider,
                        "chunking_strategy": config.chunking_strategy,
                        "retrieval_mode": config.retrieval_mode,
                        "reranker": config.reranker or "",
                        "retrieval_method": result.retrieval_method(),
                        "chunk_overlap": config.chunk_overlap,
                        "num_chunks": result.num_chunks,
                        "success": evaluation.success,
                    }
                )

    print(f"✓ Saved results to {output_path}")


def save_results_to_markdown(
    results: List,
    comparator,
    output_path: Path,
    dataset_name: str = "local-pdf",
    question_count: int | None = None,
) -> None:
    """
    Save experiment results as markdown report.

    This generates a report suitable for inclusion in your project writeup.

    Args:
        results: List of ExperimentResult objects
        comparator: ExperimentComparator instance
        output_path: Where to save markdown
    """
    lines = []

    # Header
    lines.append("# RAG Experimentation Results\n")
    lines.append("## Experiment Summary\n")

    lines.append(f"- **Total configurations tested**: {len(results)}\n")
    lines.append(f"- **Dataset**: {dataset_name}\n")
    if question_count is not None:
        lines.append(f"- **Evaluation questions**: {question_count}\n")
    lines.append("- **Vector store**: in-memory cosine similarity\n")
    lines.append("- **Retrieval options**: semantic, BM25, hybrid fusion, optional reranking\n")

    # Results table
    lines.append("\n## Results by Configuration\n")
    lines.append("| Chunk Size | Strategy | Retrieval | Top-K | Accuracy | Precision@K | Recall@K | MRR | NDCG | Grounding | Hallucination | Resp. Time (ms) | Num Chunks |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    for result in results:
        config = result.config
        metrics = result.aggregated_metrics

        lines.append(
            f"| {config.chunk_size} | {config.chunking_strategy} | {result.retrieval_method()} | {config.top_k} | "
            f"{metrics.accuracy:.3f} | {metrics.precision_at_k:.3f} | "
            f"{metrics.recall_at_k:.3f} | {metrics.mrr:.3f} | {metrics.ndcg:.3f} | "
            f"{metrics.grounding_score:.3f} | {metrics.hallucination_rate:.3f} | "
            f"{metrics.avg_response_time*1000:.1f} | "
            f"{result.num_chunks} |\n"
        )

    if results:
        total_attempted = sum(result.num_questions_attempted for result in results)
        total_errors = sum(result.error_count for result in results)
        overall_accuracy = sum(
            result.aggregated_metrics.accuracy for result in results
        ) / len(results)
        overall_precision = sum(
            result.aggregated_metrics.precision_at_k for result in results
        ) / len(results)
        overall_recall = sum(
            result.aggregated_metrics.recall_at_k for result in results
        ) / len(results)
        overall_grounding = sum(
            result.aggregated_metrics.grounding_score for result in results
        ) / len(results)
        overall_hallucination = sum(
            result.aggregated_metrics.hallucination_rate for result in results
        ) / len(results)
        overall_time = sum(
            result.aggregated_metrics.avg_response_time for result in results
        ) / len(results)

        lines.append("\n## Aggregate Metrics\n")
        lines.append(f"- **Evaluated question/configuration pairs**: {total_attempted}\n")
        lines.append(f"- **Average accuracy**: {overall_accuracy:.3f}\n")
        lines.append(f"- **Average Precision@K**: {overall_precision:.3f}\n")
        lines.append(f"- **Average Recall@K**: {overall_recall:.3f}\n")
        lines.append(f"- **Average grounding score**: {overall_grounding:.3f}\n")
        lines.append(f"- **Average hallucination rate**: {overall_hallucination:.3f}\n")
        lines.append(f"- **Average response time**: {overall_time*1000:.1f} ms\n")
        lines.append(f"- **Question-level errors**: {total_errors}\n")

    # Key findings
    lines.append("\n## Key Findings\n")

    best_accuracy = comparator.find_best_config_by_accuracy()
    if best_accuracy:
        lines.append(
            f"**Best Accuracy**: {best_accuracy.config}\n"
            f"- Accuracy: {best_accuracy.aggregated_metrics.accuracy:.3f}\n"
        )

    best_grounding = comparator.find_best_config_by_grounding()
    if best_grounding:
        lines.append(
            f"\n**Best Grounding**: {best_grounding.config}\n"
            f"- Grounding Score: {best_grounding.aggregated_metrics.grounding_score:.3f}\n"
        )

    best_speed = comparator.find_best_config_by_speed()
    if best_speed:
        lines.append(
            f"\n**Fastest**: {best_speed.config}\n"
            f"- Response Time: {best_speed.aggregated_metrics.avg_response_time*1000:.1f}ms\n"
        )

    worst_accuracy = comparator.find_worst_config_by_accuracy()
    if worst_accuracy and worst_accuracy is not best_accuracy:
        lines.append(
            f"\n**Lowest Accuracy**: {worst_accuracy.config}\n"
            f"- Accuracy: {worst_accuracy.aggregated_metrics.accuracy:.3f}\n"
        )

    # Analysis
    lines.append("\n## Analysis & Recommendations\n")

    lines.append("### Chunk Size Impact\n")
    lines.append("- Smaller chunks enable more precise retrieval\n")
    lines.append("- Larger chunks provide more context\n")
    lines.append("- Trade-off between specificity and information richness\n")

    lines.append("\n### Top-K Impact\n")
    lines.append("- Higher top-k values improve coverage but reduce speed\n")
    lines.append("- Baseline performance depends on document structure\n")

    lines.append("\n### Baseline Comparison\n")
    lines.append("- Semantic retrieval should outperform keyword search when embeddings are meaningful\n")
    lines.append("- Large improvements over random suggest embedding quality\n")

    lines.append("\n### Failures & Limitations\n")
    lines.append("- Mock embeddings are deterministic but not semantically strong\n")
    lines.append("- Current generated answers are retrieved chunks, not polished LLM responses\n")
    lines.append("- RAGBench includes multimodal references; this prototype currently evaluates text/table text only\n")

    lines.append("\n## Recommended Configuration\n")
    if best_accuracy:
        lines.append(
            f"Based on accuracy and grounding, recommend:\n"
            f"- Chunk Size: {best_accuracy.config.chunk_size}\n"
            f"- Overlap: {best_accuracy.config.chunk_overlap}\n"
            f"- Top-K: {best_accuracy.config.top_k}\n"
        )

    # Save
    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"✓ Saved markdown report to {output_path}")


def print_results_table(results: List) -> None:
    """
    Print a compact results table to console.

    Args:
        results: List of ExperimentResult objects
    """
    print("\n" + "="*120)
    print("EXPERIMENT RESULTS TABLE")
    print("="*120)

    print(
        f"{'Chunk':<8} {'Overlap':<8} {'Top-K':<6} {'Accuracy':<10} "
        f"{'Prec@K':<10} {'Ground':<10} {'Time(ms)':<10} {'Chunks':<8} {'Method':<20}"
    )
    print("-"*120)

    for result in results:
        config = result.config
        metrics = result.aggregated_metrics

        method = config.answer_mode
        if config.baseline_method:
            method = f"baseline:{config.baseline_method}"

        print(
            f"{config.chunk_size:<8} {config.chunk_overlap:<8} {config.top_k:<6} "
            f"{metrics.accuracy:<10.3f} {metrics.precision_at_k:<10.3f} "
            f"{metrics.grounding_score:<10.3f} {metrics.avg_response_time*1000:<10.1f} "
            f"{result.num_chunks:<8} {method:<20}"
        )

    print("="*120 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Smart Study Assistant RAG experiments.")
    parser.add_argument(
        "--dataset",
        choices=["local", "ragbench", "local-pdf", "open-rag-bench"],
        default="local",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--pdf-path",
        default="data/example.pdf",
        help="PDF path for --dataset local-pdf.",
    )
    parser.add_argument(
        "--eval-dataset-path",
        default="data/evaluation/eval_dataset.json",
        help="Evaluation JSON path for --dataset local-pdf.",
    )
    parser.add_argument(
        "--open-rag-bench-path",
        default="data/open-rag-bench",
        help=(
            "Path to vectara/open_ragbench data. Can point at the dataset root "
            "or the official/pdf/arxiv directory."
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=100,
        help="Maximum Open RAG Bench questions to load. Use 0 for all questions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum RAGBench questions to load. Alias for --max-questions.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Run one configuration with this chunk size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Run one configuration with this retrieval depth.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["mock", "openai", "sentence-transformers", "huggingface", "bge", "e5"],
        default="mock",
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--chunking-strategy",
        choices=["recursive", "sentence", "token", "sliding_window", "semantic", "parent_child"],
        default="recursive",
        help="Chunking strategy to evaluate.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["semantic", "bm25", "hybrid"],
        default="semantic",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "heuristic"],
        default="none",
        help="Optional reranker for hybrid retrieval experiments.",
    )
    return parser.parse_args()


def main():
    """Run the full experiment pipeline."""
    args = parse_args()

    print("\n" + "="*70)
    print("SMART STUDY ASSISTANT - EXPERIMENTATION FRAMEWORK")
    print("="*70)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    try:
        if args.dataset in {"ragbench", "open-rag-bench"}:
            raw_limit = args.limit if args.limit is not None else args.max_questions
            limit = None if raw_limit == 0 else raw_limit

            local_ragbench_path = Path(args.open_rag_bench_path)
            if local_ragbench_path.exists():
                dataset = RAGBenchLoader.load_from_directory(
                    local_ragbench_path,
                    limit=limit,
                )
            else:
                dataset = RAGBenchLoader.load(limit=limit)
        else:
            dataset = DatasetLoader.load_pdf_dataset(
                args.pdf_path,
                args.eval_dataset_path,
                name="local",
            )
    except DatasetLoadError as e:
        print(f"✗ Dataset load failed: {e}")
        if args.dataset in {"ragbench", "open-rag-bench"}:
            print(
                "  If internet access is unavailable, download vectara/open_ragbench "
                "and point --open-rag-bench-path at official/pdf/arxiv."
            )
        return

    # Define experiments
    print("\n📋 Defining experiment configurations...")
    configs = define_experiments(
        embedding_provider=args.embedding_provider,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        chunking_strategy=args.chunking_strategy,
        retrieval_mode=args.retrieval_mode,
        reranker=None if args.reranker == "none" else args.reranker,
    )
    print(f"   {len(configs)} configurations defined")

    # Run experiments
    print("\n🔬 Running experiments...")
    runner = ExperimentRunner(
        args.pdf_path,
        args.eval_dataset_path,
        dataset=dataset,
    )
    results = runner.run_experiments(configs, verbose=True)

    # Print results table
    print_results_table(results)

    # Compare and analyze
    print("📊 Comparing results...")
    comparator = ExperimentComparator(results)

    # Save results
    if args.dataset in {"ragbench", "open-rag-bench"}:
        csv_path = results_dir / "ragbench_results.csv"
        md_path = results_dir / "ragbench_summary.md"
    else:
        csv_path = results_dir / "experiment_results.csv"
        md_path = results_dir / "experiment_summary.md"

    save_results_to_csv(results, csv_path, dataset_name=dataset.name)
    save_results_to_markdown(
        results,
        comparator,
        md_path,
        dataset_name=dataset.name,
        question_count=len(dataset.eval_questions),
    )

    # Print summary
    print(comparator.generate_summary())

    print("\n✓ Experiment complete!")
    print(f"  Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
