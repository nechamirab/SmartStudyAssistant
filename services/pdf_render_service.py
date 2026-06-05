from __future__ import annotations


class PdfRenderService:
    """Render selected PDF pages to PNG image bytes for Streamlit display."""

    @staticmethod
    def render_pages(pdf_bytes: bytes, start_page: int, end_page: int, zoom: float = 1.6) -> list[bytes]:
        if not pdf_bytes or start_page < 1 or end_page < start_page:
            return []
        try:
            import fitz
        except ImportError:
            return []

        images: list[bytes] = []
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                if start_page > doc.page_count:
                    return []
                last_page = min(end_page, doc.page_count)
                matrix = fitz.Matrix(zoom, zoom)
                for page_number in range(start_page, last_page + 1):
                    page = doc.load_page(page_number - 1)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    images.append(pix.tobytes("png"))
        except Exception:
            return []
        return images
