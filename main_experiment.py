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

import json
import csv
from pathlib import Path
from typing import List

from services.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentComparator,
)


def define_experiments() -> List[ExperimentConfig]:
    """
    Define configurations to test.

    These are carefully chosen to test realistic trade-offs:
    - Small chunks vs large chunks
    - Shallow retrieval vs deep retrieval
    - Cost vs quality
    """
    configs = [
        # ===== BASELINE: Current default settings =====
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),

        # ===== CHUNK SIZE EXPERIMENTS =====
        # Small chunks: More focused retrieval
        ExperimentConfig(
            chunk_size=300,
            chunk_overlap=30,
            top_k=3,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),
        # Large chunks: More context
        ExperimentConfig(
            chunk_size=800,
            chunk_overlap=80,
            top_k=3,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),

        # ===== TOP-K EXPERIMENTS =====
        # Shallow retrieval
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=1,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),
        # Deep retrieval
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=5,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),

        # ===== COMBINED EXPERIMENTS =====
        # Small chunks + deep retrieval (most precise)
        ExperimentConfig(
            chunk_size=300,
            chunk_overlap=30,
            top_k=5,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),
        # Large chunks + shallow retrieval (fastest)
        ExperimentConfig(
            chunk_size=800,
            chunk_overlap=80,
            top_k=1,
            embedding_provider="mock",
            answer_mode="retrieved_chunks",
        ),

        # ===== BASELINE COMPARISON =====
        # Keyword search baseline
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider="mock",
            answer_mode="baseline",
            baseline_method="keyword_overlap",
        ),
        # Random baseline
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider="mock",
            answer_mode="baseline",
            baseline_method="random",
        ),
    ]
    return configs


def save_results_to_csv(
    results: List,
    output_path: Path,
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

    # Get fieldnames from first result
    fieldnames = list(results[0].to_dict().keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    print(f"✓ Saved results to {output_path}")


def save_results_to_markdown(
    results: List,
    comparator,
    output_path: Path,
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
    lines.append(f"- **Evaluation questions**: 10 game theory questions\n")
    lines.append(f"- **Embedding model**: mock (deterministic for reproducibility)\n")
    lines.append(f"- **Vector store**: FAISS in-memory\n")

    # Results table
    lines.append("\n## Results by Configuration\n")
    lines.append("| Chunk Size | Overlap | Top-K | Accuracy | Precision@K | Grounding | Resp. Time (ms) | Num Chunks |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")

    for result in results:
        config = result.config
        metrics = result.aggregated_metrics

        lines.append(
            f"| {config.chunk_size} | {config.chunk_overlap} | {config.top_k} | "
            f"{metrics.accuracy:.3f} | {metrics.precision_at_k:.3f} | "
            f"{metrics.grounding_score:.3f} | {metrics.avg_response_time*1000:.1f} | "
            f"{result.num_chunks} |\n"
        )

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
    lines.append("- FAISS semantic retrieval should outperform keyword search\n")
    lines.append("- Large improvements over random suggest embedding quality\n")

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


def main():
    """Run the full experiment pipeline."""
    print("\n" + "="*70)
    print("SMART STUDY ASSISTANT - EXPERIMENTATION FRAMEWORK")
    print("="*70)

    # Define paths
    pdf_path = "data/example.pdf"
    eval_dataset_path = "data/evaluation/eval_dataset.json"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Check if files exist
    if not Path(pdf_path).exists():
        print(f"✗ PDF not found: {pdf_path}")
        return

    if not Path(eval_dataset_path).exists():
        print(f"✗ Evaluation dataset not found: {eval_dataset_path}")
        return

    # Define experiments
    print("\n📋 Defining experiment configurations...")
    configs = define_experiments()
    print(f"   {len(configs)} configurations defined")

    # Run experiments
    print("\n🔬 Running experiments...")
    runner = ExperimentRunner(pdf_path, eval_dataset_path)
    results = runner.run_experiments(configs, verbose=True)

    # Print results table
    print_results_table(results)

    # Compare and analyze
    print("📊 Comparing results...")
    comparator = ExperimentComparator(results)

    # Save results
    csv_path = results_dir / "experiment_results.csv"
    md_path = results_dir / "experiment_summary.md"

    save_results_to_csv(results, csv_path)
    save_results_to_markdown(results, comparator, md_path)

    # Print summary
    print(comparator.generate_summary())

    print("\n✓ Experiment complete!")
    print(f"  Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
