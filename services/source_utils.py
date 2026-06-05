from __future__ import annotations

import re
from typing import Any


SOURCE_LABEL_RE = re.compile(
    r"\[\s*(?:source|section|pdf|file)\s*:[^\]]*(?:\][^\[]*)?\]",
    re.IGNORECASE,
)
CHUNK_ID_RE = re.compile(
    r"\b(?:chunk[_\s-]*id|chunk)\s*[:=#-]?\s*[A-Za-z0-9_.-]+\b",
    re.IGNORECASE,
)
FILE_NAME_RE = re.compile(r"\b[\w .()~-]+\.(?:pdf|docx?|pptx?|txt)\b", re.IGNORECASE)
PAGE_METADATA_RE = re.compile(r"\b(?:page_number|page number)\s*[:=#-]?\s*\d+\b", re.IGNORECASE)
BAD_LABEL_RE = re.compile(r"^\s*\[?\s*(?:source|section|pdf|file|chunk)\s*:", re.IGNORECASE)


def format_source_label(metadata: dict[str, Any] | None, debug: bool = False) -> str:
    """Return a clean, user-facing source label.

    Internal chunk IDs are useful for retrieval debugging but should not leak
    into normal study, quiz, exam, or export screens.
    """
    data = metadata or {}
    page = data.get("page_number") or data.get("page")
    page_start = data.get("page_start")
    page_end = data.get("page_end")
    chunk_id = str(data.get("chunk_id") or "").strip()

    if page_start and page_end and int(page_start) != int(page_end):
        page_label = f"Pages {page_start}-{page_end}"
    elif page_start:
        page_label = f"Page {page_start}"
    elif page:
        page_label = f"Page {page}"
    else:
        page_label = "Uploaded Material"

    if page:
        label = f"Source: Page {page}"
    else:
        label = f"Source: {page_label}"

    if debug and chunk_id:
        label = f"{label} (chunk: {chunk_id})"
    return label


def format_page_range(page_start: int | None, page_end: int | None) -> str:
    start = max(1, int(page_start or 1))
    end = max(start, int(page_end or start))
    if start == end:
        return f"page {start}"
    return f"pages {start}-{end}"


def pdf_page_anchor(page_start: int | None, zoom: str = "page-width") -> str:
    page = max(1, int(page_start or 1))
    return f"#page={page}&zoom={zoom}"


def source_metadata_from_ref(ref: Any) -> dict[str, Any]:
    if hasattr(ref, "to_dict"):
        ref = ref.to_dict()
    if not isinstance(ref, dict):
        return {}
    return {
        "pdf_name": ref.get("pdf_name"),
        "page_number": ref.get("page_number"),
        "chunk_id": ref.get("chunk_id"),
        "section_title": ref.get("section_title"),
        "score": ref.get("score"),
    }


def remove_source_labels(text: Any) -> str:
    """Remove raw nested source labels from visible study content."""
    clean = str(text or "")
    previous = None
    while previous != clean:
        previous = clean
        clean = SOURCE_LABEL_RE.sub(" ", clean)
    return clean


def remove_chunk_ids(text: Any) -> str:
    """Remove internal chunk identifiers from user-facing text."""
    return CHUNK_ID_RE.sub(" ", str(text or ""))


def remove_file_names(text: Any) -> str:
    """Remove uploaded file names from generated content."""
    return FILE_NAME_RE.sub(" ", str(text or ""))


def clean_repeated_metadata(text: Any) -> str:
    """Clean common retrieval metadata that can leak into AI-visible text."""
    clean = PAGE_METADATA_RE.sub(" ", str(text or ""))
    clean = re.sub(r"\bscore\s*[:=#-]?\s*\d+(?:\.\d+)?\b", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bpdf\s*set\s*:\s*", " ", clean, flags=re.IGNORECASE)
    return clean


def sanitize_visible_text(text: Any, remove_files: bool = True) -> str:
    """Normalize text shown in the app or used to generate study artifacts."""
    clean = remove_source_labels(text)
    clean = remove_chunk_ids(clean)
    clean = clean_repeated_metadata(clean)
    if remove_files:
        clean = remove_file_names(clean)
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean.strip(" \t\r\n-:;,.")


def normalize_bullets(items: Any, fallback: str = "Review the cited section.", limit: int | None = None) -> list[str]:
    """Return clean, non-empty bullet strings without raw metadata."""
    if items is None:
        values: list[Any] = []
    elif isinstance(items, str):
        values = re.split(r"(?:\n+|^\s*[-*]\s+)", items)
    else:
        values = list(items)

    bullets: list[str] = []
    seen: set[str] = set()
    for item in values:
        clean = sanitize_visible_text(item)
        clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", clean).strip()
        if not clean or BAD_LABEL_RE.match(clean):
            continue
        normalized = clean.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        bullets.append(clean)
        if limit and len(bullets) >= limit:
            break
    return bullets or [fallback]


def clean_section_title(title: Any, fallback: str = "Section") -> str:
    """Return a human-readable section title, never a raw source label."""
    clean = sanitize_visible_text(title)
    clean = re.sub(r"^\s*(?:section|source|page)\s+\d+\s*[:.-]?\s*", "", clean, flags=re.IGNORECASE)
    clean = clean.strip(" []-:;,.")
    if not clean or BAD_LABEL_RE.match(clean) or re.fullmatch(r"(?:page|source)\s*\d*", clean, re.IGNORECASE):
        return fallback
    words = clean.split()
    if len(words) > 10:
        clean = " ".join(words[:10])
    return clean.title()


def clean_citation_section(title: Any) -> str:
    """Clean a section label for source captions without hiding valid Section N labels."""
    clean = sanitize_visible_text(title)
    clean = clean.strip(" []-:;,.")
    if not clean or BAD_LABEL_RE.match(clean) or re.fullmatch(r"(?:page|source)\s*\d*", clean, re.IGNORECASE):
        return ""
    words = clean.split()
    if len(words) > 10:
        clean = " ".join(words[:10])
    return clean.title()
