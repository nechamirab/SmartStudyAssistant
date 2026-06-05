from __future__ import annotations


def render_pdf_pages_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """Render PDF pages to PNG bytes with PyMuPDF.

    Returns an empty list on invalid input or render failure so the UI can fall
    back to extracted text without crashing.
    """
    if not pdf_bytes:
        return []

    try:
        import fitz
    except Exception:
        return []

    try:
        zoom = max(72, int(dpi or 150)) / 72
        matrix = fitz.Matrix(zoom, zoom)
        images: list[bytes] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(pixmap.tobytes("png"))
        return images
    except Exception:
        return []
