from __future__ import annotations

NAV_ITEMS = ("Upload", "Study Plan", "Study Mode", "AI Tutor", "Final Exam", "Dashboard")
DEFAULT_CURRENT_PAGE = "Upload"


def normalize_current_page(value: str | None) -> str:
    return value if value in NAV_ITEMS else DEFAULT_CURRENT_PAGE
