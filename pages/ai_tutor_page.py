from __future__ import annotations

import html

import streamlit as st

from translations import t
from ui.state import current_section, has_pdf, persist_current_state, section_state
from ui.workflow import answer_ai_tutor, answer_section_question


def render_ai_tutor() -> None:
    st.subheader(t("ai_tutor"))
    mode_options = ["general", "section"]
    mode = st.radio(
        "Mode",
        mode_options,
        format_func=lambda value: t("ai_tutor_mode_general") if value == "general" else t("ai_tutor_mode_section"),
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "general":
        render_general_tutor()
    else:
        render_current_section_tutor()


def render_general_tutor() -> None:
    st.caption(t("ai_tutor_caption"))
    prompts = [
        t("prompt_summarize_material"),
        t("prompt_explain_concept"),
        t("prompt_study_plan"),
        t("prompt_practice_questions"),
    ]
    prompt_cols = st.columns(4)
    for index, prompt_text in enumerate(prompts):
        with prompt_cols[index]:
            st.markdown(f"<div class='prompt-card'>{html.escape(prompt_text)}</div>", unsafe_allow_html=True)

    use_pdf_context = st.checkbox(t("use_pdf_context"), value=has_pdf(), disabled=not has_pdf())

    if st.button(t("clear_chat")):
        st.session_state.ai_tutor_history = []
        persist_current_state()
        st.rerun()

    for message in st.session_state.ai_tutor_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(t("ask_general_question"))
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        result = answer_ai_tutor(prompt, use_pdf_context=use_pdf_context)
        with st.chat_message("assistant"):
            st.write(result["answer"])
            if result["provider"] != "none":
                provider_label = str(result["provider"])
                if result.get("context") == "pdf":
                    provider_label = f"{provider_label} with PDF context"
                st.caption(f"{t('provider')}: {provider_label}")
        st.session_state.ai_tutor_history.extend(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": result["answer"]}]
        )
        persist_current_state()


def render_current_section_tutor() -> None:
    section = current_section()
    if section is None:
        st.info(t("generate_plan_first"))
        return

    state = section_state(section)
    st.caption(t("current_section_label", title=section.title))
    question_key = f"ai-tutor-section-question-{section.section_number}"
    st.session_state.setdefault(question_key, state.get("question", ""))
    with st.form(key=f"ai-tutor-section-form-{section.section_number}"):
        question = st.text_area(t("question"), key=question_key, height=100)
        submitted = st.form_submit_button(t("ask_about_section"), type="primary")

    if submitted:
        if not question.strip():
            st.warning(t("enter_question_first"))
        else:
            with st.spinner(t("reading_section")):
                state["question"] = question
                state["answer"] = answer_section_question(section, question)
                persist_current_state()

    if state.get("answer"):
        st.markdown(state["answer"])
