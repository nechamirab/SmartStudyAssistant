from __future__ import annotations

import html
from typing import Any

import streamlit as st

from services.exam_grading_service import ExamGradingService
from ui.components import card
from ui.state import has_pdf, persist_current_state
from ui.workflow import generate_final_exam


def render_final_exam() -> None:
    st.subheader("Final Exam")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    cols = st.columns(2)
    question_count = cols[0].number_input("Questions", min_value=3, max_value=25, value=10)
    difficulty = cols[1].selectbox("Difficulty", ["mixed", "easy", "medium", "hard"])
    if st.button("Generate AI final exam", type="primary"):
        with st.spinner("Generating final exam..."):
            st.session_state.final_exam = generate_final_exam(int(question_count), difficulty)
            st.session_state.final_exam_answers = {}
            st.session_state.final_exam_result = None
            persist_current_state()
        st.success("Final exam generated.")

    exam = st.session_state.final_exam
    if not exam:
        st.info("Generate a final exam when you finish reviewing the study plan.")
        return

    if exam.get("fallback_used"):
        st.warning(exam.get("fallback_note", "Fallback exam was used."))

    st.markdown(f"### {exam.get('title', 'AI Final Exam')}")
    render_exam_form(exam)
    render_exam_result()


def render_exam_form(exam: dict[str, Any]) -> None:
    with st.form("final-exam-form"):
        answers: dict[str, Any] = {}
        for question in exam.get("questions", []):
            question_id = str(question.get("id", len(answers) + 1))
            question_type = str(question.get("type", "short_answer"))
            answer_key = f"final-exam-answer-{question_id}"
            if question_id in st.session_state.final_exam_answers:
                st.session_state.setdefault(answer_key, st.session_state.final_exam_answers[question_id])
            st.markdown(f"**{question_id}. {question.get('question', '')}**")

            if question_type in {"multiple_choice", "true_false"} and (question.get("options") or question_type == "true_false"):
                options = question.get("options") or (["True", "False"] if question_type == "true_false" else [])
                answers[question_id] = st.radio(
                    "Answer",
                    options,
                    key=answer_key,
                    label_visibility="collapsed",
                    index=None,
                )
            else:
                answers[question_id] = st.text_area(
                    "Short answer",
                    key=answer_key,
                    label_visibility="collapsed",
                    height=90,
                )
            st.caption(f"Topic: {question.get('topic', 'Review')}")

        submitted = st.form_submit_button("Submit Exam", type="primary")

    if submitted:
        st.session_state.final_exam_answers = {key: value for key, value in answers.items() if value is not None}
        result = ExamGradingService.grade_exam(
            exam,
            st.session_state.final_exam_answers,
            st.session_state.sections,
        )
        st.session_state.final_exam_result = result
        st.session_state.progress.final_exam_score = float(result["score"])
        st.session_state.progress.weak_topics = result["weak_topics"]
        st.session_state.progress.weak_sections = result["weak_sections"]
        persist_current_state()
        st.success("Final exam submitted.")


def render_exam_result() -> None:
    result = st.session_state.final_exam_result
    if not result:
        return

    cols = st.columns(3)
    cols[0].metric("Score", f"{result['score']}%")
    cols[1].metric("Correct answers", result["correct_count"])
    cols[2].metric("Wrong answers", result["wrong_count"])

    weak_sections = result.get("weak_sections", [])
    weak_topics = result.get("weak_topics", [])
    if weak_sections:
        card("Related weak sections", html.escape(", ".join(weak_sections)))
    elif weak_topics:
        card("Weak topics", html.escape(", ".join(weak_topics)))

    card("Recommendation", html.escape(result.get("recommendation", "Review your missed questions.")))

    missed = [item for item in result.get("results", []) if not item.get("is_correct")]
    if not missed:
        st.success("All answers were correct.")
        return

    st.markdown("**Review missed answers**")
    for item in missed:
        related = f" Related section: {item['related_section']}." if item.get("related_section") else ""
        st.write(
            f"Q{item['id']}: Correct answer: {item['expected_answer']}.{related}"
        )
        if item.get("related_section"):
            if st.button("Review section", key=f"review-section-{item['id']}"):
                go_to_section(item["related_section"])


def go_to_section(section_title: str) -> None:
    for index, section in enumerate(st.session_state.sections):
        if section.title == section_title:
            st.session_state.current_section_index = index
            st.session_state.current_page = "Study Mode"
            persist_current_state()
            st.rerun()
