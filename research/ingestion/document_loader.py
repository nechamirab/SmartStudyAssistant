from __future__ import annotations

from pathlib import Path

from core.models import DocumentPage
from ingestion.ocr_loader import OCRLoader
from services.pdf_service import PdfService


class DocumentLoader:
    """Load PDFs or images, optionally using OCR when text extraction is empty."""

    def load(self, path: str | Path, use_ocr: bool = False) -> list[DocumentPage]:
        source = Path(path)
        if source.suffix.lower() == ".pdf" and not use_ocr:
            return PdfService().extract_pages(source)
        if source.suffix.lower() == ".pdf" and use_ocr:
            try:
                pages = PdfService().extract_pages(source)
                if any(page.text.strip() for page in pages):
                    return pages
            except Exception:
                pass
        return OCRLoader().extract(source)
