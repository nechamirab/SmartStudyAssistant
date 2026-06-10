from __future__ import annotations

import html

import streamlit as st

from services.progress_service import ProgressService
from ui.components import card
from ui.state import format_seconds, has_pdf, overall_progress
from ui.workflow import build_weak_topic_review, next_recommended_section, recommended_review_sections


def render_dashboard() -> None:
    st.subheader("Dashboard")
    if not has_pdf():
        st.info("Upload and process a PDF first to see progress.")
        return

    progress = st.session_state.progress
    cols = st.columns(4)
    cols[0].metric("Learning Progress", f"{overall_progress():.0f}%")
    cols[1].metric("Completed Sections", f"{len(progress.completed_sections)}/{len(st.session_state.sections)}")
    cols[2].metric("Quiz Average", f"{ProgressService.quiz_average(progress):.0f}%")
    readiness = progress.final_exam_score if progress.final_exam_score is not None else overall_progress()
    cols[3].metric("Exam Readiness", f"{readiness:.0f}%")
    st.progress(overall_progress() / 100)

    total_sections = max(1, len(st.session_state.sections))
    average_seconds = progress.actual_study_seconds // total_sections
    with st.expander("Study time"):
        cols = st.columns(2)
        cols[0].metric("Total study time", format_seconds(progress.actual_study_seconds))
        cols[1].metric("Average per section", format_seconds(average_seconds))

    if not progress.completed_sections and not progress.section_quiz_scores and progress.final_exam_score is None:
        st.info("No progress yet. Start a section, take a quiz, or submit the final exam.")

    review = progress.weak_sections or recommended_review_sections()
    next_section = next_recommended_section()
    recommendation = []
    if review:
        recommendation.append(f"Review {review[0]} before taking the final exam again.")
    if next_section:
        recommendation.append(f"Recommended next section: {next_section.title}.")
    card("Recommendations", html.escape(" ".join(recommendation) if recommendation else "Keep reviewing completed sections."))

    if progress.weak_topics:
        card("Weak topics", html.escape(", ".join(progress.weak_topics)))
    if progress.weak_sections:
        card("Weak sections", html.escape(", ".join(progress.weak_sections)))

    if st.button("Review Weak Topics", type="primary"):
        st.session_state.weak_topic_review = build_weak_topic_review()
    if st.session_state.weak_topic_review:
        st.markdown(st.session_state.weak_topic_review)
