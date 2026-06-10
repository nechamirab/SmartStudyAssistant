from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressState:
    completed_sections: set[int] = field(default_factory=set)
    quiz_scores: list[float] = field(default_factory=list)
    section_quiz_scores: dict[int, float] = field(default_factory=dict)
    actual_study_seconds: int = 0
    weak_topics: list[str] = field(default_factory=list)
    weak_sections: list[str] = field(default_factory=list)
    final_exam_score: float | None = None
    timer_running: bool = False
    timer_started_at: float | None = None


class ProgressService:
    """Manage study progress in a cache-safe serializable shape."""

    @staticmethod
    def default_state() -> ProgressState:
        return ProgressState()

    @staticmethod
    def load(raw: Any) -> ProgressState:
        if isinstance(raw, ProgressState):
            return raw
        try:
            data = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            return ProgressState(
                completed_sections={int(value) for value in data.get("completed_sections", [])},
                quiz_scores=[float(value) for value in data.get("quiz_scores", [])],
                section_quiz_scores={
                    int(key): float(value)
                    for key, value in dict(data.get("section_quiz_scores", {}) or {}).items()
                },
                actual_study_seconds=max(0, int(data.get("actual_study_seconds", 0))),
                weak_topics=[str(value) for value in data.get("weak_topics", [])],
                weak_sections=[str(value) for value in data.get("weak_sections", [])],
                final_exam_score=(
                    None if data.get("final_exam_score") is None else float(data.get("final_exam_score"))
                ),
                timer_running=bool(data.get("timer_running", False)),
                timer_started_at=(
                    None if data.get("timer_started_at") is None else float(data.get("timer_started_at"))
                ),
            )
        except Exception:
            return ProgressService.default_state()

    @staticmethod
    def dump(progress: ProgressState) -> dict[str, Any]:
        return {
            "completed_sections": sorted(progress.completed_sections),
            "quiz_scores": progress.quiz_scores,
            "section_quiz_scores": {str(key): value for key, value in sorted(progress.section_quiz_scores.items())},
            "actual_study_seconds": progress.actual_study_seconds,
            "weak_topics": progress.weak_topics,
            "weak_sections": progress.weak_sections,
            "final_exam_score": progress.final_exam_score,
            "timer_running": progress.timer_running,
            "timer_started_at": progress.timer_started_at,
        }

    @staticmethod
    def start_timer(progress: ProgressState, now: float | None = None) -> ProgressState:
        if not progress.timer_running:
            progress.timer_running = True
            progress.timer_started_at = now if now is not None else time.time()
        return progress

    @staticmethod
    def restart_timer(progress: ProgressState, now: float | None = None) -> ProgressState:
        progress.actual_study_seconds = 0
        progress.timer_running = True
        progress.timer_started_at = now if now is not None else time.time()
        return progress

    @staticmethod
    def pause_timer(progress: ProgressState, now: float | None = None) -> ProgressState:
        if progress.timer_running and progress.timer_started_at is not None:
            current = now if now is not None else time.time()
            progress.actual_study_seconds += max(0, int(current - progress.timer_started_at))
        progress.timer_running = False
        progress.timer_started_at = None
        return progress

    @staticmethod
    def finish_section(progress: ProgressState, section_number: int, now: float | None = None) -> ProgressState:
        ProgressService.pause_timer(progress, now=now)
        progress.completed_sections.add(int(section_number))
        return progress

    @staticmethod
    def elapsed_seconds(progress: ProgressState, now: float | None = None) -> int:
        if progress.timer_running and progress.timer_started_at is not None:
            current = now if now is not None else time.time()
            return progress.actual_study_seconds + max(0, int(current - progress.timer_started_at))
        return progress.actual_study_seconds

    @staticmethod
    def quiz_average(progress: ProgressState) -> float:
        scores = list(progress.section_quiz_scores.values()) or progress.quiz_scores
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
