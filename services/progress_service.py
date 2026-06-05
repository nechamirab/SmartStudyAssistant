from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import ROOT_DIR


class ProgressService:
    """Persist study progress locally for Streamlit sessions."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else ROOT_DIR / ".cache" / "progress.json"

    def load_document(self, document_name: str, section_count: int = 0) -> dict[str, Any]:
        store = self._load_store()
        document = store.get(document_name)
        if not document:
            document = self._empty_progress(document_name, section_count)
            store[document_name] = document
            self._save_store(store)
        else:
            document.setdefault("document_name", document_name)
            document.setdefault("completed_sections", [])
            document.setdefault("quiz_scores", {})
            document.setdefault("final_exam_score", None)
            document.setdefault("weak_topics", [])
            document.setdefault("strong_topics", [])
            document.setdefault("last_studied_section", None)
            document.setdefault("section_time_seconds", {})
            document.setdefault("understanding_scores", {})
            document.setdefault("mistake_history", [])
            document.setdefault("review_sections", [])
            document["section_count"] = max(section_count, int(document.get("section_count", 0) or 0))
            document["total_progress_percentage"] = self.progress_percentage(document)
        return document

    def save_document(self, progress: dict[str, Any]) -> dict[str, Any]:
        progress = dict(progress)
        progress["total_progress_percentage"] = self.progress_percentage(progress)
        store = self._load_store()
        store[str(progress["document_name"])] = progress
        self._save_store(store)
        return progress

    def mark_completed(self, progress: dict[str, Any], section_id: str) -> dict[str, Any]:
        completed = list(progress.get("completed_sections", []))
        if section_id not in completed:
            completed.append(section_id)
        progress["completed_sections"] = completed
        progress["last_studied_section"] = section_id
        return self.save_document(progress)

    def record_quiz(self, progress: dict[str, Any], section_id: str, grade: dict[str, Any]) -> dict[str, Any]:
        scores = dict(progress.get("quiz_scores", {}))
        scores[section_id] = float(grade.get("score_percentage", 0.0))
        progress["quiz_scores"] = scores
        progress["weak_topics"] = self._merge_topics(progress.get("weak_topics", []), grade.get("weak_topics", []))
        progress["strong_topics"] = self._merge_topics(progress.get("strong_topics", []), grade.get("strong_topics", []))
        progress["mistake_history"] = self._merge_mistakes(
            progress.get("mistake_history", []),
            section_id,
            grade.get("results", []),
        )
        if grade.get("weak_topics"):
            progress["review_sections"] = self._merge_topics(progress.get("review_sections", []), [section_id])
        progress["last_studied_section"] = section_id
        return self.save_document(progress)

    def record_final_exam(self, progress: dict[str, Any], grade: dict[str, Any]) -> dict[str, Any]:
        progress["final_exam_score"] = float(grade.get("score_percentage", 0.0))
        progress["weak_topics"] = self._merge_topics(progress.get("weak_topics", []), grade.get("weak_topics", []))
        progress["strong_topics"] = self._merge_topics(progress.get("strong_topics", []), grade.get("strong_topics", []))
        return self.save_document(progress)

    def record_study_time(self, progress: dict[str, Any], section_id: str, seconds: float) -> dict[str, Any]:
        times = dict(progress.get("section_time_seconds", {}))
        times[section_id] = round(float(times.get(section_id, 0.0)) + max(0.0, float(seconds)), 1)
        progress["section_time_seconds"] = times
        progress["last_studied_section"] = section_id
        return self.save_document(progress)

    def record_understanding(
        self,
        progress: dict[str, Any],
        section_id: str,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        scores = dict(progress.get("understanding_scores", {}))
        scores[section_id] = float(evaluation.get("score", 0.0))
        progress["understanding_scores"] = scores
        progress["weak_topics"] = self._merge_topics(progress.get("weak_topics", []), evaluation.get("review_topics", []))
        progress["strong_topics"] = self._merge_topics(progress.get("strong_topics", []), evaluation.get("understood_well", []))
        if float(evaluation.get("score", 0.0)) < 70:
            progress["review_sections"] = self._merge_topics(progress.get("review_sections", []), [section_id])
        progress["last_studied_section"] = section_id
        return self.save_document(progress)

    @staticmethod
    def progress_percentage(progress: dict[str, Any]) -> float:
        section_count = int(progress.get("section_count", 0) or 0)
        completed = len(set(progress.get("completed_sections", [])))
        if section_count <= 0:
            return 0.0
        section_progress = completed / section_count
        quiz_scores = progress.get("quiz_scores", {})
        quiz_completion = min(1.0, len(quiz_scores) / section_count)
        exam_bonus = 1.0 if progress.get("final_exam_score") is not None else 0.0
        return round((section_progress * 0.6 + quiz_completion * 0.25 + exam_bonus * 0.15) * 100, 1)

    @staticmethod
    def average_quiz_score(progress: dict[str, Any]) -> float:
        scores = [float(value) for value in progress.get("quiz_scores", {}).values()]
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    @staticmethod
    def exam_readiness(progress: dict[str, Any]) -> float:
        section_count = int(progress.get("section_count", 0) or 0)
        if section_count <= 0:
            return 0.0
        completion = (len(set(progress.get("completed_sections", []))) / section_count) * 100
        quiz = ProgressService.average_quiz_score(progress)
        understanding = ProgressService.average_understanding_score(progress)
        final = float(progress.get("final_exam_score") or 0.0)
        weak_penalty = min(20.0, len(progress.get("weak_topics", [])) * 2.0)
        score = completion * 0.30 + quiz * 0.25 + understanding * 0.25 + final * 0.20 - weak_penalty
        return round(max(0.0, min(100.0, score)), 1)

    @staticmethod
    def average_understanding_score(progress: dict[str, Any]) -> float:
        scores = [float(value) for value in progress.get("understanding_scores", {}).values()]
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    @staticmethod
    def readiness_status(score: float) -> str:
        if score < 45:
            return "Not ready"
        if score < 65:
            return "Needs review"
        if score < 85:
            return "Almost ready"
        return "Ready"

    @staticmethod
    def readiness_action(progress: dict[str, Any]) -> str:
        review_sections = progress.get("review_sections", [])
        weak_topics = progress.get("weak_topics", [])
        if review_sections:
            return "Review " + ", ".join(str(section) for section in review_sections[:3]) + " before taking the final exam again."
        if weak_topics:
            return "Review weak topics: " + ", ".join(str(topic) for topic in weak_topics[:4]) + "."
        if progress.get("final_exam_score") is None:
            return "Take the final practice exam after completing the remaining sections."
        return "Keep practicing with section quizzes and re-check any missed questions."

    @staticmethod
    def timing_summary(progress: dict[str, Any], section_estimates: dict[str, int]) -> dict[str, Any]:
        times = {key: float(value) for key, value in progress.get("section_time_seconds", {}).items()}
        total_seconds = sum(times.values())
        section_count = len(times)
        estimated_seconds = sum(float(section_estimates.get(section_id, 0)) * 60 for section_id in times)
        longer = [
            section_id
            for section_id, seconds in times.items()
            if section_estimates.get(section_id) and seconds > float(section_estimates[section_id]) * 60
        ]
        return {
            "total_seconds": round(total_seconds, 1),
            "average_seconds": round(total_seconds / section_count, 1) if section_count else 0.0,
            "estimated_seconds": round(estimated_seconds, 1),
            "difference_seconds": round(total_seconds - estimated_seconds, 1),
            "longer_than_expected": longer,
        }

    def _load_store(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    def _save_store(self, store: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(store, indent=2), encoding="utf-8")

    @staticmethod
    def _empty_progress(document_name: str, section_count: int) -> dict[str, Any]:
        return {
            "document_name": document_name,
            "section_count": section_count,
            "completed_sections": [],
            "quiz_scores": {},
            "final_exam_score": None,
            "weak_topics": [],
            "strong_topics": [],
            "last_studied_section": None,
            "section_time_seconds": {},
            "understanding_scores": {},
            "mistake_history": [],
            "review_sections": [],
            "total_progress_percentage": 0.0,
        }

    @staticmethod
    def _merge_topics(existing: list[str], incoming: list[str]) -> list[str]:
        seen = set()
        merged = []
        for value in [*existing, *incoming]:
            normalized = str(value).strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(str(value))
        return merged[:20]

    @staticmethod
    def _merge_mistakes(existing: list[dict[str, Any]], section_id: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        mistakes = list(existing)
        for item in results:
            if item.get("correct"):
                continue
            mistakes.append(
                {
                    "section_id": section_id,
                    "question": item.get("question", ""),
                    "student_answer": item.get("student_answer", ""),
                    "correct_answer": item.get("correct_answer", ""),
                    "topics": item.get("topics", []),
                    "explanation": item.get("explanation", ""),
                    "source_references": item.get("source_references", []),
                }
            )
        return mistakes[-50:]
