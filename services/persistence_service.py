from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.models import DocumentPage
from services.progress_service import ProgressService, ProgressState
from services.section_state_service import SectionStateService
from services.study_service import StudySection


class PersistenceService:
    """Lightweight local JSON persistence for the Streamlit MVP."""

    DEFAULT_PATH = Path(os.getenv("SMARTSTUDY_PROGRESS_PATH", ".smartstudy_progress.json"))

    @classmethod
    def load(cls, path: str | Path | None = None) -> dict[str, Any]:
        file_path = Path(path or cls.DEFAULT_PATH)
        if not file_path.exists():
            return {}
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @classmethod
    def save(cls, payload: dict[str, Any], path: str | Path | None = None) -> None:
        file_path = Path(path or cls.DEFAULT_PATH)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def build_payload(
        *,
        pdf_name: str,
        pages: list[DocumentPage],
        sections: list[StudySection],
        progress: ProgressState,
        section_states: dict[str, Any],
        final_exam: dict[str, Any] | None,
        final_exam_answers: dict[str, Any],
        final_exam_result: dict[str, Any] | None,
        current_section_index: int,
    ) -> dict[str, Any]:
        return {
            "pdf": {
                "name": pdf_name,
                "page_count": len(pages),
            },
            "pages": [asdict(page) for page in pages],
            "sections": [asdict(section) for section in sections],
            "progress": ProgressService.dump(progress),
            "section_states": SectionStateService.serialize(section_states),
            "final_exam": final_exam,
            "final_exam_answers": dict(final_exam_answers or {}),
            "final_exam_result": final_exam_result,
            "current_section_index": int(current_section_index or 0),
        }

    @staticmethod
    def pages_from_payload(payload: dict[str, Any]) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        for item in payload.get("pages", []):
            if not isinstance(item, dict):
                continue
            pages.append(
                DocumentPage(
                    page_number=int(item.get("page_number", 0)),
                    text=str(item.get("text", "")),
                    source_id=str(item.get("source_id", "")),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
        return pages

    @staticmethod
    def sections_from_payload(payload: dict[str, Any]) -> list[StudySection]:
        sections: list[StudySection] = []
        for item in payload.get("sections", []):
            if not isinstance(item, dict):
                continue
            sections.append(
                StudySection(
                    section_number=int(item.get("section_number", 0)),
                    title=str(item.get("title", "")),
                    start_page=int(item.get("start_page", 0)),
                    end_page=int(item.get("end_page", 0)),
                    estimated_minutes=int(item.get("estimated_minutes", 0)),
                    difficulty=str(item.get("difficulty", "Easy")),
                    summary=str(item.get("summary", "")),
                    learning_objectives=list(item.get("learning_objectives") or []),
                    key_concepts=list(item.get("key_concepts") or []),
                )
            )
        return sections
