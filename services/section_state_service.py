from __future__ import annotations

from copy import deepcopy
from typing import Any


class SectionStateService:
    """Manage per-section UI state without tying it to Streamlit."""

    @staticmethod
    def default_state() -> dict[str, Any]:
        return {
            "explanation": "",
            "quiz": [],
            "quiz_answers": {},
            "quiz_score": None,
            "quiz_feedback": [],
            "question": "",
            "answer": "",
        }

    @classmethod
    def ensure_states(cls, raw: Any, section_numbers: list[int]) -> dict[str, dict[str, Any]]:
        states = dict(raw or {}) if isinstance(raw, dict) else {}
        normalized: dict[str, dict[str, Any]] = {}
        for section_number in section_numbers:
            key = str(section_number)
            section_state = cls.default_state()
            if isinstance(states.get(key), dict):
                section_state.update(states[key])
            normalized[key] = section_state
        return normalized

    @classmethod
    def get_state(cls, states: dict[str, Any], section_number: int) -> dict[str, Any]:
        key = str(section_number)
        if key not in states or not isinstance(states[key], dict):
            states[key] = cls.default_state()
        return states[key]

    @classmethod
    def reset_interaction_state(cls, states: dict[str, Any], section_number: int) -> dict[str, Any]:
        key = str(section_number)
        current = cls.get_state(states, section_number)
        preserved_score = current.get("quiz_score")
        preserved_feedback = deepcopy(current.get("quiz_feedback", []))
        states[key] = cls.default_state()
        states[key]["quiz_score"] = preserved_score
        states[key]["quiz_feedback"] = preserved_feedback
        return states[key]

    @staticmethod
    def serialize(states: dict[str, Any]) -> dict[str, Any]:
        return {str(key): value for key, value in dict(states or {}).items() if isinstance(value, dict)}
