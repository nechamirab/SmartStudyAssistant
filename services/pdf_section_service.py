from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from core.config import ROOT_DIR


SECTION_PDF_CACHE_DIR = ROOT_DIR / ".cache" / "section_pdfs"


def create_section_pdf(pdf_bytes: bytes, start_page: int, end_page: int, cache_key: str) -> bytes:
    """Return a PDF containing only the requested 1-based page range.

    Invalid ranges or extraction failures return the original PDF bytes so Study
    Mode can still display/download the source document.
    """
    if not pdf_bytes:
        return b""

    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        return pdf_bytes

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        if page_count <= 0:
            return pdf_bytes

        start = max(1, int(start_page or 1))
        end = max(1, int(end_page or start))
        if start > end:
            start, end = end, start
        if start > page_count:
            return pdf_bytes
        end = min(end, page_count)

        digest = hashlib.sha256(
            f"{cache_key}:{len(pdf_bytes)}:{start}:{end}".encode("utf-8")
        ).hexdigest()
        cache_path = SECTION_PDF_CACHE_DIR / f"{digest}.pdf"
        if cache_path.exists():
            return cache_path.read_bytes()

        writer = PdfWriter()
        for page_index in range(start - 1, end):
            writer.add_page(reader.pages[page_index])

        output = BytesIO()
        writer.write(output)
        section_bytes = output.getvalue()
        if not section_bytes.startswith(b"%PDF"):
            return pdf_bytes

        SECTION_PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(section_bytes)
        return section_bytes
    except Exception:
        return pdf_bytes
