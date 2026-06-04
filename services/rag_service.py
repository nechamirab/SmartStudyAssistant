from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    MIN_RETRIEVAL_SCORE,
    NOT_FOUND_ANSWER,
    RETRIEVAL_TOP_K,
    VECTOR_STORE_BACKEND,
)
from core.models import DocumentChunk, DocumentPage
from services.chunk_service import ChunkService, ChunkingError
from services.embedding_service import EmbeddingError, EmbeddingService
from services.pdf_service import OCRMode, PdfExtractionError, PdfService
from services.retrieval_service import RetrievalError, RetrievalService
from services.vector_store_service import SearchResult
from vectorstores.base import VectorStoreError
from vectorstores.factory import VectorStoreFactory


class RAGPipelineError(Exception):
    """Raised when PDF-based RAG processing or answering fails."""


@dataclass(frozen=True)
class SourceReference:
    pdf_name: str
    page_number: int
    chunk_id: str
    score: float
    text: str

    def to_dict(self) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 4),
            "text": self.text,
        }


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[SourceReference]
    found: bool
    confidence: float

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "found": self.found,
            "confidence": self.confidence,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass
class PDFIndex:
    pdf_name: str
    pages: list[DocumentPage]
    chunks: list[DocumentChunk]
    embedding_service: EmbeddingService
    vector_store: object
    retrieval_service: RetrievalService

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def to_summary(self) -> dict:
        extraction = self.extraction_summary()
        return {
            "pdf_name": self.pdf_name,
            "pages": self.page_count,
            "chunks": self.chunk_count,
            "extraction": extraction,
            "embedding_provider": self.embedding_service.provider,
            "embedding_model": self.embedding_service.model,
            "vector_store": self.vector_store.backend_name,
            "vector_store_note": "In-memory only; upload again after restarting the app.",
        }

    def extraction_summary(self) -> dict:
        normal_pages = 0
        ocr_pages = 0
        total_characters = 0
        ocr_mode = "auto"
        for page in self.pages:
            text = page.text or ""
            total_characters += len(text)
            metadata = page.metadata or {}
            ocr_mode = str(metadata.get("ocr_mode", ocr_mode))
            if metadata.get("extraction_method") == "ocr" and text.strip():
                ocr_pages += 1
            elif text.strip():
                normal_pages += 1

        return {
            "ocr_mode": ocr_mode,
            "pages_processed": self.page_count,
            "pages_using_normal_text": normal_pages,
            "pages_using_ocr": ocr_pages,
            "total_characters_extracted": total_characters,
        }


class PDFRAGService:
    """Small, focused RAG pipeline for uploaded PDF documents."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = RETRIEVAL_TOP_K,
        min_score: float = MIN_RETRIEVAL_SCORE,
        embedding_provider: str = EMBEDDING_PROVIDER,
        embedding_model: str = EMBEDDING_MODEL,
        vector_store_backend: str = VECTOR_STORE_BACKEND,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_score = min_score
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.vector_store_backend = vector_store_backend

    def build_index_from_upload(self, pdf_bytes: bytes, filename: str, ocr_mode: OCRMode = "auto") -> PDFIndex:
        if not pdf_bytes:
            raise RAGPipelineError("Upload a non-empty PDF file.")
        if not filename.lower().endswith(".pdf"):
            raise RAGPipelineError("Only PDF files are supported.")

        safe_name = Path(filename).name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = Path(tmpdir) / safe_name
                pdf_path.write_bytes(pdf_bytes)
                pages = PdfService().extract_pages(pdf_path, ocr_mode=ocr_mode)
            return self.build_index(pages, safe_name)
        except PdfExtractionError as exc:
            raise RAGPipelineError(f"PDF text extraction failed: {exc}") from exc
        except Exception as exc:
            raise RAGPipelineError(f"Could not process PDF: {exc}") from exc

    def build_index_from_uploads(self, uploads: list[tuple[str, bytes]], ocr_mode: OCRMode = "auto") -> PDFIndex:
        if not uploads:
            raise RAGPipelineError("Upload at least one PDF file.")

        all_pages: list[DocumentPage] = []
        safe_names: list[str] = []
        seen_names: dict[str, int] = {}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                for filename, pdf_bytes in uploads:
                    if not pdf_bytes:
                        raise RAGPipelineError(f"Upload a non-empty PDF file: {filename or 'unnamed file'}.")
                    if not filename.lower().endswith(".pdf"):
                        raise RAGPipelineError(f"Only PDF files are supported: {filename}.")

                    safe_name = Path(filename).name
                    seen_names[safe_name] = seen_names.get(safe_name, 0) + 1
                    if seen_names[safe_name] > 1:
                        stem = Path(safe_name).stem
                        suffix = Path(safe_name).suffix
                        safe_name = f"{stem}-{seen_names[safe_name]}{suffix}"

                    pdf_path = tmp_path / safe_name
                    pdf_path.write_bytes(pdf_bytes)
                    pages = PdfService().extract_pages(pdf_path, ocr_mode=ocr_mode)
                    all_pages.extend(pages)
                    safe_names.append(safe_name)

            return self.build_index(all_pages, self._combined_pdf_name(safe_names))
        except PdfExtractionError as exc:
            raise RAGPipelineError(f"PDF text extraction failed: {exc}") from exc
        except RAGPipelineError:
            raise
        except Exception as exc:
            raise RAGPipelineError(f"Could not process PDFs: {exc}") from exc

    def build_index(self, pages: list[DocumentPage], pdf_name: str) -> PDFIndex:
        if not pages:
            raise RAGPipelineError("No pages were extracted from the PDF.")
        if not any((page.text or "").strip() for page in pages):
            raise RAGPipelineError("The PDF does not contain extractable text.")

        try:
            chunks = ChunkService(self.chunk_size, self.chunk_overlap).chunk_pages(pages)
        except ChunkingError as exc:
            raise RAGPipelineError(f"Chunking failed: {exc}") from exc

        if not chunks:
            raise RAGPipelineError("No text chunks could be created from the PDF.")

        try:
            embedding_service = EmbeddingService(
                provider=self.embedding_provider,
                model=self.embedding_model,
                cache_enabled=True,
                fallback_to_mock=True,
            )
            embeddings = embedding_service.embed_texts(chunks)
            if not embeddings:
                raise RAGPipelineError("No embeddings were created for the PDF chunks.")
            vector_store = VectorStoreFactory.create(self.vector_store_backend, collection_name=pdf_name)
            vector_store.add(chunks, embeddings)
        except (EmbeddingError, VectorStoreError) as exc:
            raise RAGPipelineError(f"Embedding or vector storage failed: {exc}") from exc

        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            chunks=chunks,
        )
        return PDFIndex(
            pdf_name=pdf_name,
            pages=pages,
            chunks=chunks,
            embedding_service=embedding_service,
            vector_store=vector_store,
            retrieval_service=retrieval_service,
        )

    def answer(self, index: PDFIndex, question: str, top_k: int | None = None) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            raise RAGPipelineError("Enter a question to ask the PDF.")

        try:
            response = index.retrieval_service.retrieve(question, top_k=top_k or self.top_k)
        except RetrievalError as exc:
            raise RAGPipelineError(f"Retrieval failed: {exc}") from exc

        relevant = self._relevant_results(question, response.results)
        sources = [self._source_from_result(index.pdf_name, result) for result in relevant]
        if not relevant:
            return RAGAnswer(
                question=question,
                answer=NOT_FOUND_ANSWER,
                sources=[],
                found=False,
                confidence=0.0,
            )

        answer = self._generate_grounded_answer(question, sources)
        confidence = max(source.score for source in sources) if sources else 0.0
        return RAGAnswer(
            question=question,
            answer=answer,
            sources=sources,
            found=True,
            confidence=round(float(confidence), 4),
        )

    def _relevant_results(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        terms = self._keywords(question)
        relevant: list[SearchResult] = []
        for result in results:
            score_ok = result.score >= self.min_score
            lexical_ok = not terms or bool(terms & self._keywords(result.chunk.text))
            if score_ok and lexical_ok:
                relevant.append(result)
        return relevant

    def _generate_grounded_answer(self, question: str, sources: list[SourceReference]) -> str:
        return self._extractive_answer(question, sources)

    @staticmethod
    def build_answer_prompt(question: str, sources: list[SourceReference]) -> str:
        context = "\n\n".join(
            (
                f"[{index}] PDF: {source.pdf_name}; page {source.page_number}; "
                f"chunk {source.chunk_id}; score {source.score:.4f}\n{source.text}"
            )
            for index, source in enumerate(sources, 1)
        )
        return (
            "Use only the PDF context below. Do not use general knowledge.\n"
            f"If the answer is not present, say exactly: {NOT_FOUND_ANSWER}\n"
            "Cite page numbers in the answer when possible.\n\n"
            f"PDF context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Grounded answer:"
        )

    def _extractive_answer(self, question: str, sources: list[SourceReference]) -> str:
        terms = self._keywords(question)
        sentences: list[tuple[int, SourceReference, str]] = []
        for source in sources:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", source.text):
                clean = " ".join(sentence.split())
                if len(clean) < 30:
                    continue
                overlap = len(terms & self._keywords(clean))
                if overlap:
                    sentences.append((overlap, source, clean))
        sentences.sort(key=lambda item: item[0], reverse=True)
        selected = sentences[:3]
        if not selected:
            selected = [(0, source, self._trim(source.text, 260)) for source in sources[:2]]

        parts = [
            f"{self._trim(sentence, 320)} ({source.pdf_name}, page {source.page_number})"
            for _score, source, sentence in selected
        ]
        return " ".join(parts)

    @classmethod
    def _keywords(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
            if token not in cls.STOPWORDS
        }

    @staticmethod
    def _source_from_result(pdf_name: str, result: SearchResult) -> SourceReference:
        return SourceReference(
            pdf_name=result.chunk.source_id or pdf_name,
            page_number=result.chunk.page_number,
            chunk_id=result.chunk.chunk_id,
            score=float(result.score),
            text=result.chunk.text,
        )

    @staticmethod
    def _combined_pdf_name(pdf_names: list[str]) -> str:
        if len(pdf_names) == 1:
            return pdf_names[0]
        preview = ", ".join(pdf_names[:3])
        if len(pdf_names) > 3:
            preview += f", +{len(pdf_names) - 3} more"
        return f"{len(pdf_names)} PDFs: {preview}"

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."
