from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sqlite3
from typing import Any

from services.database_service import DatabaseService

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - lets service tests run without Streamlit.
    st = None

try:
    from werkzeug.security import check_password_hash as werkzeug_check_password_hash
    from werkzeug.security import generate_password_hash as werkzeug_generate_password_hash
except Exception:  # pragma: no cover - exercised only when Werkzeug is unavailable.
    werkzeug_check_password_hash = None
    werkzeug_generate_password_hash = None


_FALLBACK_SESSION_STATE: dict[str, Any] = {}


class AuthService:
    """Simple username/password authentication backed by SQLite."""

    SESSION_KEY = "auth_user"

    def __init__(
        self,
        database: DatabaseService | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> None:
        self.database = database or DatabaseService()
        self.session_state = session_state if session_state is not None else self._session_state()

    def register_user(self, username: str, password: str) -> dict[str, Any]:
        username = self._normalize_username(username)
        validation_error = self._validate_credentials(username, password)
        if validation_error:
            return {"ok": False, "error": validation_error}

        if self.database.get_user_by_username(username):
            return {"ok": False, "error": "Username already exists."}

        try:
            user = self.database.create_user(username, hash_password(password))
        except sqlite3.IntegrityError:
            return {"ok": False, "error": "Username already exists."}
        except Exception as exc:
            return {"ok": False, "error": f"Could not create user: {exc}"}

        self.session_state[self.SESSION_KEY] = self._public_user(user)
        return {"ok": True, "user": self._public_user(user)}

    def login_user(self, username: str, password: str) -> dict[str, Any]:
        username = self._normalize_username(username)
        if not username or not password:
            return {"ok": False, "error": "Username and password are required."}

        try:
            user = self.database.get_user_by_username(username)
        except Exception as exc:
            return {"ok": False, "error": f"Could not log in: {exc}"}

        if not user or not verify_password(user.get("password_hash", ""), password):
            return {"ok": False, "error": "Invalid username or password."}

        self.session_state[self.SESSION_KEY] = self._public_user(user)
        return {"ok": True, "user": self._public_user(user)}

    def logout_user(self) -> None:
        self.session_state.pop(self.SESSION_KEY, None)

    def current_user(self) -> dict[str, Any] | None:
        user = self.session_state.get(self.SESSION_KEY)
        return dict(user) if isinstance(user, dict) and user.get("id") else None

    def require_login(self) -> dict[str, Any] | None:
        return self.current_user()

    @staticmethod
    def _normalize_username(username: str) -> str:
        return (username or "").strip()

    @staticmethod
    def _validate_credentials(username: str, password: str) -> str:
        if not username:
            return "Username is required."
        if len(username) < 3:
            return "Username must be at least 3 characters."
        if not password:
            return "Password is required."
        if len(password) < 6:
            return "Password must be at least 6 characters."
        return ""

    @staticmethod
    def _public_user(user: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(user["id"]),
            "username": str(user["username"]),
            "created_at": str(user.get("created_at", "")),
        }

    @staticmethod
    def _session_state() -> dict[str, Any]:
        if st is None:
            return _FALLBACK_SESSION_STATE
        return st.session_state


def hash_password(password: str) -> str:
    if werkzeug_generate_password_hash is not None:
        return werkzeug_generate_password_hash(password)

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260000)
    return "pbkdf2_sha256$" + base64.b64encode(salt).decode("ascii") + "$" + base64.b64encode(digest).decode("ascii")


def verify_password(password_hash: str, password: str) -> bool:
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, salt_raw, digest_raw = password_hash.split("$", 2)
            salt = base64.b64decode(salt_raw.encode("ascii"))
            expected = base64.b64decode(digest_raw.encode("ascii"))
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260000)
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    if werkzeug_check_password_hash is None:
        return False
    return bool(werkzeug_check_password_hash(password_hash, password))


def register_user(username: str, password: str) -> dict[str, Any]:
    return AuthService().register_user(username, password)


def login_user(username: str, password: str) -> dict[str, Any]:
    return AuthService().login_user(username, password)


def logout_user() -> None:
    AuthService().logout_user()


def current_user() -> dict[str, Any] | None:
    return AuthService().current_user()


def require_login() -> dict[str, Any] | None:
    return AuthService().require_login()
