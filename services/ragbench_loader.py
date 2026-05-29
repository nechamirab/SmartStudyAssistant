from __future__ import annotations

import json
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from core.models import DocumentPage
from services.chunk_service import normalize_text
from services.dataset_loader import DatasetLoadError, ExperimentDataset


RAGBENCH_DATASET_ID = "vectara/open_ragbench"


class RAGBenchLoader:
    """Load Vectara Open RAG Bench into SmartStudyAssistant's eval format."""

    @staticmethod
    def load(
        limit: int | None = 50,
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ExperimentDataset:
        """
        Load and normalize the Hugging Face dataset.

        The official dataset is stored as BEIR-like JSON files. We try the
        standard datasets API first because it is the public Hugging Face entry
        point, then fall back to snapshot_download because the dataset viewer can
        have trouble inferring mixed JSON schemas.
        """
        allow_download = allow_download or os.getenv("SMARTSTUDY_ALLOW_RAGBENCH_DOWNLOAD") == "1"
        if not allow_download:
            raise DatasetLoadError(
                "RAGBench local files were not found and online download is disabled. "
                "Download vectara/open_ragbench locally and pass --open-rag-bench-path, "
                "or rerun with --download-ragbench / SMARTSTUDY_ALLOW_RAGBENCH_DOWNLOAD=1."
            )

        try:
            return RAGBenchLoader._load_with_datasets(limit=limit)
        except Exception as datasets_error:
            print(
                "⚠️  Hugging Face datasets loader could not read "
                f"{RAGBENCH_DATASET_ID}: {datasets_error}"
            )
            print("   Falling back to the raw repository snapshot.")

        try:
            return RAGBenchLoader._load_from_hub_snapshot(
                limit=limit,
                cache_dir=cache_dir,
            )
        except Exception as snapshot_error:
            raise DatasetLoadError(
                "Unable to load vectara/open_ragbench. Check internet access, "
                "Hugging Face availability, and the installed datasets package. "
                f"Details: {snapshot_error}"
            ) from snapshot_error

    @staticmethod
    def _load_with_datasets(limit: int | None) -> ExperimentDataset:
        load_dataset = RAGBenchLoader._import_huggingface_load_dataset()

        dataset = load_dataset(RAGBENCH_DATASET_ID, split="train")
        rows = list(dataset.take(limit)) if limit is not None else list(dataset)
        if not rows:
            raise DatasetLoadError("datasets.load_dataset returned no RAGBench rows.")

        eval_questions: list[dict[str, Any]] = []
        pages: list[DocumentPage] = []

        for index, row in enumerate(rows, 1):
            normalized = RAGBenchLoader._normalize_generic_row(row, index)
            if normalized is None:
                continue
            eval_questions.append(normalized)

            source_text = normalized.get("source_text", "")
            if source_text:
                pages.append(DocumentPage(page_number=index, text=source_text))

        if not eval_questions or not pages:
            raise DatasetLoadError(
                "datasets.load_dataset did not expose enough question/context fields."
            )

        return ExperimentDataset(
            name="ragbench",
            source_path=Path(RAGBENCH_DATASET_ID),
            pages=pages,
            eval_questions=eval_questions,
        )

    @staticmethod
    def _import_huggingface_load_dataset():
        """
        Import Hugging Face's datasets package even though this repo has a
        local datasets/ facade package. Python resolves the local package first
        when running from the project root, so we temporarily remove the project
        root from sys.path for this one third-party import.
        """
        project_root = Path(__file__).resolve().parents[1]
        original_path = list(sys.path)
        local_module = sys.modules.get("datasets")

        try:
            if local_module is not None:
                module_file = Path(getattr(local_module, "__file__", "") or "")
                if module_file.is_relative_to(project_root):
                    del sys.modules["datasets"]

            sys.path = [
                entry
                for entry in sys.path
                if entry
                and Path(entry).resolve() != project_root
            ]
            module = importlib.import_module("datasets")
            return module.load_dataset
        finally:
            sys.path = original_path
            # Keep the Hugging Face package in sys.modules. Its load_dataset
            # implementation performs follow-up absolute imports from
            # datasets.*, which would break if we restored the local facade.

    @staticmethod
    def _load_from_hub_snapshot(
        limit: int | None,
        cache_dir: str | Path | None,
    ) -> ExperimentDataset:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=RAGBENCH_DATASET_ID,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        root = RAGBenchLoader._resolve_beir_root(snapshot_path)
        return RAGBenchLoader.load_from_directory(root, limit=limit)

    @staticmethod
    def load_from_directory(
        dataset_root: str | Path,
        limit: int | None = 50,
        source_text_chars: int = 180,
    ) -> ExperimentDataset:
        """Load an already-downloaded Open RAG Bench BEIR-style directory."""
        root = RAGBenchLoader._resolve_beir_root(Path(dataset_root))
        queries = RAGBenchLoader._load_json_object(root / "queries.json")
        qrels = RAGBenchLoader._load_json_object(root / "qrels.json")
        answers = RAGBenchLoader._load_json_object(root / "answers.json")
        pdf_urls = RAGBenchLoader._load_optional_json_object(root / "pdf_urls.json")
        corpus_dir = root / "corpus"

        if not corpus_dir.exists():
            raise DatasetLoadError(f"RAGBench corpus directory not found: {corpus_dir}")

        eval_questions: list[dict[str, Any]] = []
        needed_doc_ids: set[str] = set()

        for query_index, (query_id, query_meta) in enumerate(queries.items(), 1):
            if limit is not None and len(eval_questions) >= limit:
                break
            if not isinstance(query_meta, dict):
                print(f"⚠️  Skipping RAGBench query {query_id}: metadata is malformed")
                continue

            qrel = qrels.get(query_id)
            answer = answers.get(query_id)
            if not isinstance(qrel, dict) or answer is None:
                print(f"⚠️  Skipping RAGBench query {query_id}: missing qrel or answer")
                continue

            doc_id = str(qrel.get("doc_id", "") or "").strip()
            question = str(query_meta.get("query", "") or "").strip()
            if not doc_id or not question:
                print(f"⚠️  Skipping RAGBench query {query_id}: missing doc_id or question")
                continue

            try:
                section_id = int(qrel.get("section_id", 0) or 0)
            except (TypeError, ValueError):
                print(f"⚠️  RAGBench query {query_id}: invalid section_id; using 0")
                section_id = 0

            section_text = RAGBenchLoader._load_section_text(corpus_dir, doc_id, section_id)
            if not section_text:
                print(f"⚠️  RAGBench query {query_id}: missing source section text")

            source_text = RAGBenchLoader._source_text_snippet(
                section_text,
                max_chars=source_text_chars,
            )

            eval_questions.append(
                {
                    "question": question,
                    "answer": str(answer or "").strip(),
                    "source_text": source_text,
                    "pdf": pdf_urls.get(doc_id, doc_id),
                    "page": None,
                    "metadata": {
                        "dataset_id": RAGBENCH_DATASET_ID,
                        "query_id": query_id,
                        "query_index": query_index,
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "query_type": query_meta.get("type"),
                        "query_source": query_meta.get("source"),
                    },
                    "dataset": "ragbench",
                }
            )
            needed_doc_ids.add(doc_id)

        if not eval_questions:
            raise DatasetLoadError("No valid RAGBench records were loaded.")

        pages = RAGBenchLoader._load_pages(corpus_dir, needed_doc_ids)
        return ExperimentDataset(
            name="ragbench",
            source_path=root,
            pages=pages,
            eval_questions=eval_questions,
        )

    @staticmethod
    def _normalize_generic_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
        """Best-effort normalizer for any future flattened Hugging Face schema."""
        question = RAGBenchLoader._first_text(row, ["question", "query", "queries"])
        answer = RAGBenchLoader._first_text(row, ["answer", "answers", "expected_answer"])
        source_text = RAGBenchLoader._first_text(
            row,
            ["source_text", "context", "contexts", "text", "section_text"],
        )

        if not question or not answer:
            print(f"⚠️  Skipping RAGBench row {index}: missing question or answer")
            return None
        if not source_text:
            print(f"⚠️  RAGBench row {index}: missing source text")

        return {
            "question": question,
            "answer": answer,
            "source_text": source_text,
            "pdf": RAGBenchLoader._first_text(row, ["pdf", "doc_id", "document_id"]),
            "page": None,
            "metadata": {"row_index": index, "raw_keys": sorted(row.keys())},
            "dataset": "ragbench",
        }

    @staticmethod
    def _resolve_beir_root(path: Path) -> Path:
        required = ["answers.json", "queries.json", "qrels.json"]
        if all((path / filename).exists() for filename in required):
            return path

        for nested in (path / "official" / "pdf" / "arxiv", path / "pdf" / "arxiv"):
            if all((nested / filename).exists() for filename in required):
                return nested

        matches = list(path.glob("**/official/pdf/arxiv")) + list(path.glob("**/pdf/arxiv"))
        for match in matches:
            if all((match / filename).exists() for filename in required):
                return match

        raise DatasetLoadError(
            "Could not find RAGBench files. Expected answers.json, queries.json, "
            "qrels.json, and corpus/ under pdf/arxiv or official/pdf/arxiv."
        )

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise DatasetLoadError(f"Required RAGBench file not found: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetLoadError(f"Invalid JSON in {path}") from e
        if not isinstance(data, dict):
            raise DatasetLoadError(f"Expected JSON object in {path}")
        return data

    @staticmethod
    def _load_optional_json_object(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return RAGBenchLoader._load_json_object(path)
        except DatasetLoadError as e:
            print(f"⚠️  Could not read optional RAGBench file {path}: {e}")
            return {}

    @staticmethod
    def _load_section_text(corpus_dir: Path, doc_id: str, section_id: int) -> str:
        document_path = corpus_dir / f"{doc_id}.json"
        try:
            document = RAGBenchLoader._load_json_object(document_path)
        except DatasetLoadError as e:
            print(f"⚠️  Could not read corpus document {doc_id}: {e}")
            return ""

        sections = document.get("sections", [])
        if not isinstance(sections, list) or section_id < 0 or section_id >= len(sections):
            return ""

        return RAGBenchLoader._section_to_text(sections[section_id])

    @staticmethod
    def _load_pages(corpus_dir: Path, doc_ids: set[str]) -> list[DocumentPage]:
        """Turn RAGBench document sections into pseudo-pages for existing chunking."""
        pages: list[DocumentPage] = []
        page_number = 0

        for doc_id in sorted(doc_ids):
            try:
                document = RAGBenchLoader._load_json_object(corpus_dir / f"{doc_id}.json")
            except DatasetLoadError as e:
                print(f"⚠️  Skipping corpus document {doc_id}: {e}")
                continue

            title = normalize_text(str(document.get("title", "") or ""))
            abstract = normalize_text(str(document.get("abstract", "") or ""))
            sections = document.get("sections", [])
            if not isinstance(sections, list):
                print(f"⚠️  Skipping corpus document {doc_id}: sections is not a list")
                continue

            for section_index, section in enumerate(sections):
                section_text = RAGBenchLoader._section_to_text(section)
                if not section_text:
                    continue

                page_number += 1
                parts = [
                    f"Document ID: {doc_id}",
                    f"Title: {title}" if title else "",
                    f"Abstract: {abstract}" if abstract else "",
                    f"Section {section_index}: {section_text}",
                ]
                pages.append(
                    DocumentPage(
                        page_number=page_number,
                        text="\n".join(part for part in parts if part),
                    )
                )

        if not pages:
            raise DatasetLoadError("No RAGBench corpus text was available for retrieval.")
        return pages

    @staticmethod
    def _section_to_text(section: Any) -> str:
        if not isinstance(section, dict):
            return normalize_text(str(section or ""))

        parts = [normalize_text(str(section.get("text", "") or ""))]
        tables = section.get("tables", {})
        if isinstance(tables, dict):
            parts.extend(normalize_text(str(table)) for table in tables.values())
        images = section.get("images", {})
        if isinstance(images, dict) and images:
            parts.append(" ".join(f"[image:{image_id}]" for image_id in images.keys()))
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _source_text_snippet(text: str, max_chars: int) -> str:
        words = normalize_text(text).split()
        snippet = ""
        for word in words:
            candidate = f"{snippet} {word}".strip()
            if len(candidate) > max_chars:
                break
            snippet = candidate
        return snippet

    @staticmethod
    def _first_text(row: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                text = " ".join(str(item) for item in value if item)
                if text.strip():
                    return text.strip()
            if isinstance(value, dict):
                text = " ".join(str(item) for item in value.values() if item)
                if text.strip():
                    return text.strip()
        return ""
