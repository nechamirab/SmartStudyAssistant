from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.models import DocumentPage
from services.progress_service import ProgressService, ProgressState
from services.section_state_service import SectionStateService
from services.study_service import StudySection


class DatabaseService:
    """SQLite persistence for authenticated Smart Study Assistant sessions."""

    DEFAULT_PATH = Path(".smartstudy.db")

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or self.DEFAULT_PATH)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    document_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS study_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    section_number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    start_page INTEGER NOT NULL,
                    end_page INTEGER NOT NULL,
                    estimated_minutes INTEGER NOT NULL,
                    difficulty TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_concepts TEXT NOT NULL DEFAULT '[]',
                    learning_objectives TEXT NOT NULL DEFAULT '[]',
                    section_text TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES study_sessions(id) ON DELETE CASCADE,
                    UNIQUE (session_id, section_number)
                );

                CREATE TABLE IF NOT EXISTS section_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    section_number INTEGER NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    quiz_score REAL,
                    explanation_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES study_sessions(id) ON DELETE CASCADE,
                    UNIQUE (user_id, session_id, section_number)
                );

                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    section_number INTEGER NOT NULL,
                    questions TEXT NOT NULL DEFAULT '[]',
                    answers TEXT NOT NULL DEFAULT '{}',
                    score REAL,
                    feedback TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES study_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS exam_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER NOT NULL,
                    exam_payload TEXT NOT NULL DEFAULT '{}',
                    user_answers TEXT NOT NULL DEFAULT '{}',
                    score REAL,
                    weak_topics TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES study_sessions(id) ON DELETE CASCADE
                );
                """
            )

    def create_user(self, username: str, password_hash: str) -> dict[str, Any]:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
            return self.get_user_by_id(int(cursor.lastrowid), conn=conn) or {}

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return self._row_dict(row)

    def get_user_by_id(self, user_id: int, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if conn is not None:
            return self._row_dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
        with self.connect() as owned_conn:
            return self.get_user_by_id(user_id, conn=owned_conn)

    def create_document(self, user_id: int, filename: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (user_id, filename) VALUES (?, ?)",
                (user_id, filename),
            )
            return int(cursor.lastrowid)

    def create_study_session(self, user_id: int, document_id: int, title: str, language: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO study_sessions (user_id, document_id, title, language) VALUES (?, ?, ?, ?)",
                (user_id, document_id, title, language),
            )
            return int(cursor.lastrowid)

    def create_session_from_state(
        self,
        *,
        user_id: int,
        filename: str,
        title: str,
        language: str,
        pages: list[DocumentPage],
        sections: list[StudySection],
    ) -> tuple[int, int]:
        document_id = self.create_document(user_id, filename)
        session_id = self.create_study_session(user_id, document_id, title, language)
        self.save_study_sections(session_id, sections, pages)
        return document_id, session_id

    def save_study_sections(self, session_id: int, sections: list[StudySection], pages: list[DocumentPage]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM study_sections WHERE session_id = ?", (session_id,))
            for section in sections:
                section_text = self._section_text(pages, section)
                conn.execute(
                    """
                    INSERT INTO study_sections (
                        session_id, section_number, title, start_page, end_page,
                        estimated_minutes, difficulty, summary, key_concepts,
                        learning_objectives, section_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        section.section_number,
                        section.title,
                        section.start_page,
                        section.end_page,
                        section.estimated_minutes,
                        section.difficulty,
                        section.summary,
                        self._json(section.key_concepts),
                        self._json(section.learning_objectives),
                        section_text,
                    ),
                )
            self.touch_session(conn, session_id)

    def list_study_sessions(self, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ss.*,
                    d.filename,
                    COUNT(study_sections.id) AS section_count,
                    COALESCE(SUM(CASE WHEN section_progress.completed = 1 THEN 1 ELSE 0 END), 0) AS completed_count
                FROM study_sessions ss
                JOIN documents d ON d.id = ss.document_id
                LEFT JOIN study_sections ON study_sections.session_id = ss.id
                LEFT JOIN section_progress
                    ON section_progress.session_id = ss.id
                    AND section_progress.section_number = study_sections.section_number
                    AND section_progress.user_id = ss.user_id
                WHERE ss.user_id = ?
                GROUP BY ss.id
                ORDER BY ss.updated_at DESC, ss.created_at DESC
                """,
                (user_id,),
            ).fetchall()
            sessions = []
            for row in rows:
                item = dict(row)
                total = max(1, int(item.get("section_count") or 0))
                item["progress_percent"] = round(float(item.get("completed_count") or 0) / total * 100)
                sessions.append(item)
            return sessions

    def load_study_session(self, user_id: int, session_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            session = conn.execute(
                """
                SELECT ss.*, d.filename
                FROM study_sessions ss
                JOIN documents d ON d.id = ss.document_id
                WHERE ss.id = ? AND ss.user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
            if session is None:
                return None

            section_rows = conn.execute(
                "SELECT * FROM study_sections WHERE session_id = ? ORDER BY section_number",
                (session_id,),
            ).fetchall()
            sections = [self._section_from_row(row) for row in section_rows]
            pages = [
                DocumentPage(
                    page_number=int(row["start_page"]),
                    text=str(row["section_text"] or ""),
                    source_id=str(session["filename"]),
                    metadata={"source": str(session["filename"]), "restored_from": "sqlite"},
                )
                for row in section_rows
            ]
            progress, section_states = self._load_progress_and_states(conn, user_id, session_id, sections)
            exam = self._load_latest_exam(conn, user_id, session_id)

            return {
                "session": dict(session),
                "pages": pages,
                "sections": sections,
                "progress": progress,
                "section_states": section_states,
                "final_exam": exam.get("exam_payload") if exam else None,
                "final_exam_answers": exam.get("user_answers") if exam else {},
                "final_exam_result": exam.get("result") if exam else None,
            }

    def save_runtime_state(
        self,
        *,
        user_id: int,
        session_id: int,
        sections: list[StudySection],
        progress: ProgressState,
        section_states: dict[str, Any],
        final_exam: dict[str, Any] | None,
        final_exam_answers: dict[str, Any],
        final_exam_result: dict[str, Any] | None,
    ) -> None:
        with self.connect() as conn:
            loaded_progress = ProgressService.load(progress)
            for section in sections:
                state = SectionStateService.get_state(section_states, section.section_number)
                quiz_score = state.get("quiz_score")
                conn.execute(
                    """
                    INSERT INTO section_progress (
                        user_id, session_id, section_number, completed,
                        quiz_score, explanation_text, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, session_id, section_number) DO UPDATE SET
                        completed = excluded.completed,
                        quiz_score = excluded.quiz_score,
                        explanation_text = excluded.explanation_text,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        user_id,
                        session_id,
                        section.section_number,
                        1 if section.section_number in loaded_progress.completed_sections else 0,
                        None if quiz_score is None else float(quiz_score),
                        str(state.get("explanation", "") or ""),
                    ),
                )
                if state.get("quiz"):
                    self._replace_quiz_attempt(conn, user_id, session_id, section.section_number, state)

            if final_exam:
                self._replace_exam_attempt(
                    conn,
                    user_id,
                    session_id,
                    final_exam,
                    final_exam_answers or {},
                    final_exam_result or {},
                )
            self.touch_session(conn, session_id)

    def save_quiz_attempt(
        self,
        user_id: int,
        session_id: int,
        section_number: int,
        questions: list[dict[str, Any]],
        answers: dict[str, Any],
        score: float | None,
        feedback: list[str],
    ) -> None:
        with self.connect() as conn:
            self._replace_quiz_attempt(
                conn,
                user_id,
                session_id,
                section_number,
                {"quiz": questions, "quiz_answers": answers, "quiz_score": score, "quiz_feedback": feedback},
            )
            self.touch_session(conn, session_id)

    def load_quiz_attempts(self, user_id: int, session_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM quiz_attempts WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
                (user_id, session_id),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "questions": self._loads(row["questions"], []),
                    "answers": self._loads(row["answers"], {}),
                    "feedback": self._loads(row["feedback"], []),
                }
                for row in rows
            ]

    def save_exam_attempt(
        self,
        user_id: int,
        session_id: int,
        exam_payload: dict[str, Any],
        user_answers: dict[str, Any],
        score: float | None,
        weak_topics: list[str],
    ) -> None:
        with self.connect() as conn:
            self._replace_exam_attempt(
                conn,
                user_id,
                session_id,
                exam_payload,
                user_answers,
                {"score": score, "weak_topics": weak_topics},
            )
            self.touch_session(conn, session_id)

    def load_exam_attempts(self, user_id: int, session_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM exam_attempts WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC",
                (user_id, session_id),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "exam_payload": self._loads(row["exam_payload"], {}),
                    "user_answers": self._loads(row["user_answers"], {}),
                    "weak_topics": self._loads(row["weak_topics"], []),
                }
                for row in rows
            ]

    @staticmethod
    def touch_session(conn: sqlite3.Connection, session_id: int) -> None:
        conn.execute("UPDATE study_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        conn.execute(
            """
            UPDATE documents
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = (SELECT document_id FROM study_sessions WHERE id = ?)
            """,
            (session_id,),
        )

    def _replace_quiz_attempt(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        session_id: int,
        section_number: int,
        state: dict[str, Any],
    ) -> None:
        conn.execute(
            "DELETE FROM quiz_attempts WHERE user_id = ? AND session_id = ? AND section_number = ?",
            (user_id, session_id, section_number),
        )
        conn.execute(
            """
            INSERT INTO quiz_attempts (
                user_id, session_id, section_number, questions, answers, score, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                section_number,
                self._json(state.get("quiz", [])),
                self._json(state.get("quiz_answers", {})),
                state.get("quiz_score"),
                self._json(state.get("quiz_feedback", [])),
            ),
        )

    def _replace_exam_attempt(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        session_id: int,
        final_exam: dict[str, Any],
        final_exam_answers: dict[str, Any],
        final_exam_result: dict[str, Any],
    ) -> None:
        conn.execute("DELETE FROM exam_attempts WHERE user_id = ? AND session_id = ?", (user_id, session_id))
        conn.execute(
            """
            INSERT INTO exam_attempts (
                user_id, session_id, exam_payload, user_answers, score, weak_topics
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                self._json(final_exam),
                self._json(final_exam_answers),
                final_exam_result.get("score"),
                self._json(final_exam_result.get("weak_topics", [])),
            ),
        )

    def _load_progress_and_states(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        session_id: int,
        sections: list[StudySection],
    ) -> tuple[ProgressState, dict[str, Any]]:
        progress = ProgressService.default_state()
        section_states = SectionStateService.ensure_states({}, [section.section_number for section in sections])
        rows = conn.execute(
            "SELECT * FROM section_progress WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        for row in rows:
            section_number = int(row["section_number"])
            state = SectionStateService.get_state(section_states, section_number)
            if row["completed"]:
                progress.completed_sections.add(section_number)
            if row["quiz_score"] is not None:
                progress.section_quiz_scores[section_number] = float(row["quiz_score"])
                state["quiz_score"] = float(row["quiz_score"])
            state["explanation"] = str(row["explanation_text"] or "")

        for attempt in self.load_quiz_attempts(user_id, session_id):
            section_number = int(attempt["section_number"])
            state = SectionStateService.get_state(section_states, section_number)
            state["quiz"] = attempt["questions"]
            state["quiz_answers"] = attempt["answers"]
            state["quiz_score"] = attempt["score"]
            state["quiz_feedback"] = attempt["feedback"]
        return progress, section_states

    def _load_latest_exam(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        session_id: int,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM exam_attempts WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id, session_id),
        ).fetchone()
        if row is None:
            return None
        score = row["score"]
        weak_topics = self._loads(row["weak_topics"], [])
        result = None
        if score is not None:
            result = {
                "score": score,
                "correct_count": 0,
                "wrong_count": 0,
                "total": 0,
                "weak_topics": weak_topics,
                "weak_sections": [],
                "results": [],
                "recommendation": "Review your saved final exam answers.",
            }
        return {
            "exam_payload": self._loads(row["exam_payload"], {}),
            "user_answers": self._loads(row["user_answers"], {}),
            "result": result,
        }

    def _section_from_row(self, row: sqlite3.Row) -> StudySection:
        return StudySection(
            section_number=int(row["section_number"]),
            title=str(row["title"]),
            start_page=int(row["start_page"]),
            end_page=int(row["end_page"]),
            estimated_minutes=int(row["estimated_minutes"]),
            difficulty=str(row["difficulty"]),
            summary=str(row["summary"]),
            key_concepts=self._loads(row["key_concepts"], []),
            learning_objectives=self._loads(row["learning_objectives"], []),
        )

    @staticmethod
    def _section_text(pages: list[DocumentPage], section: StudySection) -> str:
        return "\n\n".join(
            (page.text or "").strip()
            for page in pages
            if section.start_page <= int(page.page_number) <= section.end_page and (page.text or "").strip()
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _loads(value: Any, default: Any) -> Any:
        try:
            return json.loads(value or "")
        except Exception:
            return default

    @staticmethod
    def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return None if row is None else dict(row)
