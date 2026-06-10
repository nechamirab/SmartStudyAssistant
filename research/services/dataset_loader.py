from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import DocumentPage
from services.pdf_service import PdfService


class DatasetLoadError(Exception):
    """Raised when an experiment dataset cannot be loaded."""
    pass


@dataclass(frozen=True)
class ExperimentDataset:
    """Loaded document pages plus evaluation questions."""
    name: str
    source_path: Path
    pages: list[DocumentPage]
    eval_questions: list[dict[str, Any]]


class DatasetLoader:
    """Load supported experiment datasets into the runner's common format."""

    @staticmethod
    def load_pdf_dataset(
        pdf_path: str | Path,
        eval_dataset_path: str | Path,
        name: str = "local-pdf",
    ) -> ExperimentDataset:
        pdf_path = Path(pdf_path)
        eval_dataset_path = Path(eval_dataset_path)

        if not pdf_path.exists():
            raise DatasetLoadError(f"PDF not found: {pdf_path}")
        if not eval_dataset_path.exists():
            raise DatasetLoadError(f"Evaluation dataset not found: {eval_dataset_path}")

        try:
            with open(eval_dataset_path) as f:
                raw_questions = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetLoadError(
                f"Evaluation dataset is not valid JSON: {eval_dataset_path}"
            ) from e

        if not isinstance(raw_questions, list):
            raise DatasetLoadError("Evaluation dataset must be a JSON list.")

        eval_questions = DatasetLoader.normalize_eval_records(
            raw_questions,
            dataset_name=name,
            expected_pdf=pdf_path.name,
        )
        if not eval_questions:
            raise DatasetLoadError(
                f"No valid evaluation records found in {eval_dataset_path}"
            )

        pages = PdfService().extract_pages(pdf_path)
        return ExperimentDataset(
            name=name,
            source_path=pdf_path,
            pages=pages,
            eval_questions=eval_questions,
        )

    @staticmethod
    def normalize_eval_records(
        records: list[Any],
        dataset_name: str,
        expected_pdf: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Normalize a hand-written evaluation JSON file.

        Academic projects often grow evaluation files by hand, so this function
        is intentionally defensive: nested lists are flattened, malformed rows
        are skipped, and optional fields such as source_text are filled safely.
        """
        normalized: list[dict[str, Any]] = []

        for index, record in enumerate(DatasetLoader._flatten_records(records), 1):
            if not isinstance(record, dict):
                print(f"⚠️  Skipping eval record {index}: expected object, got {type(record).__name__}")
                continue

            question = str(record.get("question", "") or "").strip()
            answer = str(record.get("answer", "") or "").strip()
            if not question or not answer:
                print(f"⚠️  Skipping eval record {index}: missing question or answer")
                continue

            pdf_name = str(record.get("pdf", "") or "").strip()
            if expected_pdf and pdf_name and pdf_name != expected_pdf:
                print(
                    f"⚠️  Skipping eval record {index}: pdf '{pdf_name}' "
                    f"does not match '{expected_pdf}'"
                )
                continue

            source_text = str(record.get("source_text", "") or "").strip()
            if not source_text:
                print(
                    f"⚠️  Eval record {index} has no source_text; "
                    "retrieval metrics will be marked unavailable"
                )

            metadata = dict(record.get("metadata", {}) or {})
            for key, value in record.items():
                if key not in {"question", "answer", "source_text", "pdf", "page", "metadata"}:
                    metadata[key] = value

            normalized.append(
                {
                    "question": question,
                    "answer": answer,
                    "source_text": source_text,
                    "pdf": pdf_name or expected_pdf or "",
                    "page": record.get("page"),
                    "metadata": metadata,
                    "dataset": dataset_name,
                }
            )

        return normalized

    @staticmethod
    def _flatten_records(records: list[Any]) -> list[Any]:
        """Flatten nested lists while preserving record order."""
        flattened: list[Any] = []
        for record in records:
            if isinstance(record, list):
                flattened.extend(DatasetLoader._flatten_records(record))
            else:
                flattened.append(record)
        return flattened

    @staticmethod
    def load_open_rag_bench(
        dataset_root: str | Path,
        max_questions: int | None = 100,
        source_text_chars: int = 160,
    ) -> ExperimentDataset:
        """Backward-compatible local directory loader for Open RAG Bench."""
        from services.ragbench_loader import RAGBenchLoader

        return RAGBenchLoader.load_from_directory(
            dataset_root,
            limit=max_questions,
            source_text_chars=source_text_chars,
        )
