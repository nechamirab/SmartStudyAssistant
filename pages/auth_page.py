from __future__ import annotations

import streamlit as st

from services.auth_service import AuthService


def render_auth_page() -> None:
    st.title("Smart Study Assistant")
    st.caption("Log in or register to save and continue your study sessions.")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login-form"):
            username = st.text_input("Username", key="login-username")
            password = st.text_input("Password", type="password", key="login-password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            result = AuthService().login_user(username, password)
            if result["ok"]:
                st.success(f"Welcome back, {result['user']['username']}.")
                st.rerun()
            else:
                st.error(result["error"])

    with register_tab:
        with st.form("register-form"):
            username = st.text_input("Choose username", key="register-username")
            password = st.text_input("Choose password", type="password", key="register-password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            result = AuthService().register_user(username, password)
            if result["ok"]:
                st.success(f"Account created for {result['user']['username']}.")
                st.rerun()
            else:
                st.error(result["error"])


def render_auth_sidebar() -> None:
    user = AuthService().current_user()
    if not user:
        return
    with st.sidebar:
        st.caption("Signed in")
        st.write(f"**{user['username']}**")
        if st.button("Logout"):
            AuthService().logout_user()
            st.rerun()
