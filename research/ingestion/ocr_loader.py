from __future__ import annotations

from pathlib import Path

from core.models import DocumentPage


class OCRDependencyError(Exception):
    """Raised when OCR is requested but optional OCR dependencies are missing."""


class OCRLoader:
    """Optional OCR extraction for scanned PDFs and images."""

    def extract(self, path: str | Path, dpi: int = 200) -> list[DocumentPage]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"OCR source not found: {source}")
        if source.suffix.lower() == ".pdf":
            return self._extract_pdf(source, dpi=dpi)
        return [self._extract_image(source, page_number=1)]

    def _extract_pdf(self, path: Path, dpi: int) -> list[DocumentPage]:
        try:
            import fitz
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise OCRDependencyError(
                "OCR requires optional packages: PyMuPDF, Pillow, and pytesseract. "
                "Install system Tesseract separately, then run pip install Pillow pytesseract."
            ) from exc

        pages: list[DocumentPage] = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(path) as document:
            for index, page in enumerate(document, 1):
                pixmap = page.get_pixmap(matrix=matrix)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                text = pytesseract.image_to_string(image) or ""
                pages.append(
                    DocumentPage(
                        page_number=index,
                        text=text,
                        source_id=path.name,
                        metadata={"source": str(path), "loader": "ocr"},
                    )
                )
        return pages

    def _extract_image(self, path: Path, page_number: int) -> DocumentPage:
        try:
            from PIL import Image, ImageOps
            import pytesseract
        except ImportError as exc:
            raise OCRDependencyError(
                "Image OCR requires Pillow and pytesseract. "
                "Install system Tesseract separately, then run pip install Pillow pytesseract."
            ) from exc

        with Image.open(path) as image:
            processed = ImageOps.grayscale(image)
            text = pytesseract.image_to_string(processed) or ""
        return DocumentPage(
            page_number=page_number,
            text=text,
            source_id=path.name,
            metadata={"source": str(path), "loader": "ocr"},
        )
