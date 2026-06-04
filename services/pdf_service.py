from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from core.models import DocumentPage

logger = logging.getLogger(__name__)

OCRMode = Literal["off", "auto", "force"]
MIN_OCR_TRIGGER_CHARS = 25


class PdfExtractionError(Exception):
    """Raised when PDF extraction fails or the given PDF path is invalid."""
    pass


class PdfService:
    """
    Service responsible for extracting text from PDF files.

    The service attempts extraction using PyMuPDF first,
    and falls back to pypdf if extraction fails.
    """
    def extract_pages(self, pdf_path: str | Path, ocr_mode: OCRMode = "auto") -> list[DocumentPage]:
        """
        Extract text from a PDF file and return pages.
        Args:
            pdf_path: Path to the PDF file.
            ocr_mode: "off", "auto", or "force". Auto OCRs pages with little or no text.
        Returns:
            List of DocumentPage objects.
        Raises:
            PdfExtractionError: If extraction fails or file is invalid.
        """
        pdf_path = Path(pdf_path)
        ocr_mode = self._normalize_ocr_mode(ocr_mode)

        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise PdfExtractionError("Invalid PDF path.")

        try:
            pages = self._extract_with_pymupdf(pdf_path, ocr_mode)
            if self._is_effectively_empty(pages):
                raise PdfExtractionError(self._empty_text_error(ocr_mode))
            return pages
        except Exception as e:
            if ocr_mode != "off":
                raise self._as_extraction_error(e) from e
            logger.warning("PyMuPDF extraction failed, falling back to pypdf: %s", e)

        pages = self._extract_with_pypdf(pdf_path)
        if self._is_effectively_empty(pages):
            raise PdfExtractionError(self._empty_text_error(ocr_mode))
        return pages

    @staticmethod
    def _extract_with_pymupdf(pdf_path: Path, ocr_mode: OCRMode) -> list[DocumentPage]:
        """
        Extract pages using PyMuPDF.
        """
        import fitz

        pages: list[DocumentPage] = []
        with fitz.open(pdf_path) as doc:
            for idx in range(doc.page_count):
                page = doc.load_page(idx)
                normal_text = page.get_text("text") or ""
                normal_text = normal_text.strip()
                should_ocr = PdfService._should_ocr_page(normal_text, ocr_mode)
                ocr_text = ""
                ocr_error = ""
                if should_ocr:
                    try:
                        ocr_text = PdfService._ocr_page(page).strip()
                    except PdfExtractionError as exc:
                        raise exc
                    except Exception as exc:
                        ocr_error = str(exc)
                        logger.warning("OCR failed for %s page %s: %s", pdf_path.name, idx + 1, exc)

                if ocr_text:
                    text = ocr_text
                    extraction_method = "ocr"
                else:
                    text = normal_text
                    extraction_method = "normal"

                metadata = {
                    "source": str(pdf_path),
                    "extraction_method": extraction_method,
                    "ocr_attempted": should_ocr,
                    "ocr_mode": ocr_mode,
                }
                if ocr_error:
                    metadata["ocr_error"] = ocr_error

                pages.append(
                    DocumentPage(
                        page_number=idx + 1,
                        text=text,
                        source_id=pdf_path.name,
                        metadata=metadata,
                    )
                )
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
            pages.append(
                DocumentPage(
                    page_number=idx + 1,
                    text=text,
                    source_id=pdf_path.name,
                    metadata={
                        "source": str(pdf_path),
                        "extraction_method": "normal",
                        "ocr_attempted": False,
                        "ocr_mode": "off",
                    },
                )
            )
        return pages

    @staticmethod
    def _is_effectively_empty(pages: list[DocumentPage]) -> bool:
        return not any((page.text or "").strip() for page in pages)

    @staticmethod
    def _normalize_ocr_mode(ocr_mode: str) -> OCRMode:
        normalized = (ocr_mode or "auto").strip().lower()
        if normalized not in {"off", "auto", "force"}:
            raise PdfExtractionError("OCR mode must be one of: off, auto, force.")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _should_ocr_page(text: str, ocr_mode: OCRMode) -> bool:
        if ocr_mode == "off":
            return False
        if ocr_mode == "force":
            return True
        return len((text or "").strip()) < MIN_OCR_TRIGGER_CHARS

    @staticmethod
    def _ocr_page(page) -> str:
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise PdfExtractionError(
                "OCR dependency missing. Install pytesseract and Pillow, and install the local Tesseract binary."
            ) from exc

        from io import BytesIO

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(BytesIO(pix.tobytes("png")))
        try:
            text = pytesseract.image_to_string(image) or ""
        except pytesseract.pytesseract.TesseractNotFoundError as exc:
            raise PdfExtractionError(
                "OCR dependency missing. Install the local Tesseract binary and ensure it is on PATH."
            ) from exc
        if len(text.strip()) >= MIN_OCR_TRIGGER_CHARS:
            return text

        easyocr_text = PdfService._try_easyocr(image)
        return easyocr_text or text

    @staticmethod
    def _try_easyocr(image) -> str:
        try:
            import easyocr
            import numpy as np
        except ImportError:
            return ""

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(np.array(image), detail=0, paragraph=True)
        return "\n".join(str(item) for item in results if str(item).strip())

    @staticmethod
    def _empty_text_error(ocr_mode: OCRMode) -> str:
        if ocr_mode == "off":
            return "Empty text extracted from PDF. Try OCR mode auto or force for scanned PDFs."
        return "No text found after PDF extraction and OCR."

    @staticmethod
    def _as_extraction_error(exc: Exception) -> PdfExtractionError:
        if isinstance(exc, PdfExtractionError):
            return exc
        return PdfExtractionError(f"Unreadable PDF or OCR extraction failed: {exc}")
