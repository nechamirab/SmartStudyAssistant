from __future__ import annotations

import html

import streamlit as st

from ui.state import current_section, has_pdf, persist_current_state, section_state
from ui.workflow import answer_ai_tutor, answer_section_question


def render_ai_tutor() -> None:
    st.subheader("AI Tutor")
    mode = st.radio(
        "Mode",
        ["General AI Tutor", "Ask about current section"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "General AI Tutor":
        render_general_tutor()
    else:
        render_current_section_tutor()


def render_general_tutor() -> None:
    st.caption("Ask general study questions, request examples, or get help understanding a topic.")
    prompts = [
        "Explain recursion with an example",
        "Help me prepare for an algorithms exam",
        "Explain Big O notation simply",
        "Create practice questions",
    ]
    prompt_cols = st.columns(4)
    for index, prompt_text in enumerate(prompts):
        with prompt_cols[index]:
            st.markdown(f"<div class='prompt-card'>{html.escape(prompt_text)}</div>", unsafe_allow_html=True)

    use_pdf_context = st.checkbox("Use uploaded PDF context", value=has_pdf(), disabled=not has_pdf())

    if st.button("Clear Chat"):
        st.session_state.ai_tutor_history = []
        persist_current_state()
        st.rerun()

    for message in st.session_state.ai_tutor_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a general study question")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        result = answer_ai_tutor(prompt, use_pdf_context=use_pdf_context)
        with st.chat_message("assistant"):
            st.write(result["answer"])
            if result["provider"] != "none":
                st.caption(f"Provider: {result['provider']}")
        st.session_state.ai_tutor_history.extend(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": result["answer"]}]
        )
        persist_current_state()


def render_current_section_tutor() -> None:
    section = current_section()
    if section is None:
        st.info("Generate a study plan first.")
        return

    state = section_state(section)
    st.caption(f"Current section: {section.title}")
    question_key = f"ai-tutor-section-question-{section.section_number}"
    st.session_state.setdefault(question_key, state.get("question", ""))
    with st.form(key=f"ai-tutor-section-form-{section.section_number}"):
        question = st.text_area("Question", key=question_key, height=100)
        submitted = st.form_submit_button("Ask About This Section", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Reading the current section..."):
                state["question"] = question
                state["answer"] = answer_section_question(section, question)
                persist_current_state()

    if state.get("answer"):
        st.markdown(state["answer"])
