from __future__ import annotations


class PdfSectionError(Exception):
    """Raised when a section PDF cannot be created."""


class PdfSectionService:
    """Create downloadable PDFs for a selected 1-based page range."""

    @staticmethod
    def extract_section_pdf(pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
        if not pdf_bytes:
            raise PdfSectionError("PDF bytes are empty.")
        if start_page < 1 or end_page < start_page:
            raise PdfSectionError("Invalid page range.")

        try:
            import fitz
        except ImportError as exc:
            raise PdfSectionError("PyMuPDF is required to create section PDFs.") from exc

        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as source:
                if end_page > source.page_count:
                    raise PdfSectionError("Page range is outside the PDF.")
                target = fitz.open()
                target.insert_pdf(source, from_page=start_page - 1, to_page=end_page - 1)
                data = target.tobytes()
                target.close()
                return data
        except PdfSectionError:
            raise
        except Exception as exc:
            raise PdfSectionError(f"Could not create section PDF: {exc}") from exc
