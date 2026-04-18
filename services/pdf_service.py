from __future__ import annotations

import logging
from pathlib import Path

from core.models import DocumentPage

logger = logging.getLogger(__name__)


class PdfExtractionError(Exception):
    """Raised when PDF extraction fails or the given PDF path is invalid."""
    pass


class PdfService:
    """
    Service responsible for extracting text from PDF files.

    The service attempts extraction using PyMuPDF first,
    and falls back to pypdf if extraction fails.
    """
    def extract_pages(self, pdf_path: str | Path) -> list[DocumentPage]:
        """
        Extract text from a PDF file and return pages.
        Args:
            pdf_path: Path to the PDF file.
        Returns:
            List of DocumentPage objects.
        Raises:
            PdfExtractionError: If extraction fails or file is invalid.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise PdfExtractionError("Invalid PDF path.")

        try:
            pages = self._extract_with_pymupdf(pdf_path)
            if self._is_effectively_empty(pages):
                raise PdfExtractionError("Empty text extracted with PyMuPDF.")
            return pages
        except Exception as e:
            logger.warning("PyMuPDF extraction failed, falling back to pypdf: %s", e)

        pages = self._extract_with_pypdf(pdf_path)
        if self._is_effectively_empty(pages):
            raise PdfExtractionError("Empty text extracted from PDF.")
        return pages

    @staticmethod
    def _extract_with_pymupdf(pdf_path: Path) -> list[DocumentPage]:
        """
        Extract pages using PyMuPDF.
        """
        import fitz

        pages: list[DocumentPage] = []
        with fitz.open(pdf_path) as doc:
            for idx in range(doc.page_count):
                page = doc.load_page(idx)
                text = page.get_text("text") or ""
                pages.append(DocumentPage(page_number=idx + 1, text=text))
        return pages

    @staticmethod
    def _extract_with_pypdf(pdf_path: Path) -> list[DocumentPage]:
        """
        Extract pages using pypdf (fallback).
        """
        from pypdf import PdfReader

        pages: list[DocumentPage] = []
        reader = PdfReader(str(pdf_path))
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(DocumentPage(page_number=idx + 1, text=text))
        return pages

    @staticmethod
    def _is_effectively_empty(pages: list[DocumentPage]) -> bool:
        return not any((page.text or "").strip() for page in pages)