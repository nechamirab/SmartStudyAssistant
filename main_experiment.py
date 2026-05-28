"""
Benchmark runner for Smart Study Assistant.

Usage:
    python main_experiment.py --dataset local

This script runs a stable local benchmark using the example PDF and evaluation dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from services.experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentComparator,
)


def define_experiments(provider: str = "mock") -> List[ExperimentConfig]:
    """Define a simple set of benchmark configurations."""
    return [
        ExperimentConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3,
            embedding_provider=provider,
            answer_mode="retrieved_chunks",
        ),
    ]


def save_results_to_csv(results: List, output_path: Path) -> None:
    """Save experiment results to CSV."""
    if not results:
        return

    fieldnames = list(results[0].to_dict().keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())


def save_results_to_markdown(results: List, comparator: ExperimentComparator, output_path: Path) -> None:
    """Save a human-readable benchmark summary."""
    lines: List[str] = [
        "# Smart Study Assistant Benchmark Results\n",
        "A clean local benchmark report for the final RAG platform.\n",
        "## Configuration\n",
        f"- PDF: data/example.pdf\n",
        f"- Evaluation dataset: data/evaluation/eval_dataset.json\n",
        "- Retrieval: semantic vector search\n",
        "- Answer mode: retrieved chunks\n",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean benchmark for Smart Study Assistant.")
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
        help="Embedding provider to use for the benchmark.",
    )
    args = parser.parse_args()

    pdf_path = Path("data/example.pdf")
    eval_dataset_path = Path("data/evaluation/eval_dataset.json")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not eval_dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_dataset_path}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = define_experiments(provider=args.provider)
    runner = ExperimentRunner(str(pdf_path), str(eval_dataset_path))
    results = runner.run_experiments(configs, verbose=True)

    comparator = ExperimentComparator(results)
    csv_path = results_dir / "benchmark_results.csv"
    md_path = results_dir / "benchmark_summary.md"

    save_results_to_csv(results, csv_path)
    save_results_to_markdown(results, comparator, md_path)

    print("\nBenchmark complete!")
    print(f"Results saved to {results_dir}")
    print(comparator.generate_summary())


if __name__ == "__main__":
    main()
