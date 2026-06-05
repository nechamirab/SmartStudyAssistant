from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressState:
    completed_sections: set[int] = field(default_factory=set)
    quiz_scores: list[float] = field(default_factory=list)
    actual_study_seconds: int = 0
    weak_topics: list[str] = field(default_factory=list)
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
                actual_study_seconds=max(0, int(data.get("actual_study_seconds", 0))),
                weak_topics=[str(value) for value in data.get("weak_topics", [])],
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
            "actual_study_seconds": progress.actual_study_seconds,
            "weak_topics": progress.weak_topics,
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
    def quiz_average(progress: ProgressState) -> float:
        if not progress.quiz_scores:
            return 0.0
        return sum(progress.quiz_scores) / len(progress.quiz_scores)
