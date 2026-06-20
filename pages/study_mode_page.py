from __future__ import annotations

import re
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from services.general_ai_service import GeneralAIService
from services.pdf_render_service import PdfRenderService
from services.pdf_section_service import PdfSectionError, PdfSectionService
from services.progress_service import ProgressService
from services.quiz_grading_service import QuizGradingService
from services.study_service import StudyService
from translations import current_language, t
from ui.components import badge, card
from ui.state import (
    current_section,
    format_seconds,
    has_pdf,
    page_label,
    persist_current_state,
    section_context,
    section_state,
    source_label,
)
from ui.workflow import answer_section_question, build_section_quiz, generate_explanation
from services.ai_answer_grading_service import AIAnswerGradingService


PDF_PREVIEW_SCROLL_HEIGHT = 720


def render_pdf_pages(section) -> None:
    st.markdown(f"### {t('pdf_section')}")
    st.markdown(f"**{t('now_studying', page_label=page_label(section))}**")
    images = PdfRenderService.render_pages(st.session_state.pdf_bytes, section.start_page, section.end_page)
    if images:
        with st.container(height=PDF_PREVIEW_SCROLL_HEIGHT, border=True):
            for offset, image in enumerate(images, start=section.start_page):
                st.image(image, caption=source_label(section, offset), width="stretch")
    else:
        st.info(t("page_images_unavailable"))

    try:
        section_pdf = PdfSectionService.extract_section_pdf(
            st.session_state.pdf_bytes,
            section.start_page,
            section.end_page,
        )
        st.download_button(
            t("download_section_pdf"),
            data=section_pdf,
            file_name=f"section-{section.section_number}.pdf",
            mime="application/pdf",
        )
    except PdfSectionError:
        st.caption(t("section_pdf_unavailable"))

    with st.expander(t("extracted_text_fallback")):
        st.text_area(t("section_text"), value=section_context(section), height=300, label_visibility="collapsed")


def render_study_mode() -> None:
    st.subheader(t("study_mode"))
    if not has_pdf():
        st.info(t("upload_pdf_first"))
        return

    section = current_section()
    if section is None:
        st.info(t("generate_plan_first"))
        return

    left, right = st.columns([0.65, 0.35])
    with left:
        render_pdf_pages(section)

    with right:
        card(
            section.title,
            f"{badge(t('section_count', number=section.section_number, total=len(st.session_state.sections)), 'primary')}"
            f"{badge(page_label(section), 'accent')}"
            f"{badge(t('difficulty_' + section.difficulty.lower()) if section.difficulty.lower() in {'easy', 'medium', 'hard'} else section.difficulty, 'warning')}"
            f"<p class='muted'>{t('estimated')}: {section.estimated_minutes} {t('minutes')}</p>",
        )
        next_index = StudyService.next_section_index(
            st.session_state.current_section_index,
            len(st.session_state.sections),
        )
        completed = section.section_number in st.session_state.progress.completed_sections
        next_section = st.session_state.sections[next_index] if next_index != st.session_state.current_section_index else None
        st.caption(t("completed") if completed else t("current_section"))
        if next_section:
            st.caption(t("next_section_title", title=next_section.title))
        st.progress((section.section_number - 1) / max(1, len(st.session_state.sections)))
        render_timer_display()
        if st.session_state.pdf_name and not st.session_state.pdf_bytes:
            st.caption(t("restored_pdf_notice"))

        timer_cols = st.columns([1, 1, 1])
        is_running = st.session_state.progress.timer_running
        has_started = st.session_state.progress.actual_study_seconds > 0

        if not is_running:
            button_label = t("resume_session") if has_started else t("start_session")
            if timer_cols[0].button(button_label, type="primary", use_container_width=True):
                st.session_state.progress = ProgressService.start_timer(st.session_state.progress)
                persist_current_state()
                st.rerun()
        else:
            if timer_cols[0].button(t("pause"), use_container_width=True):
                st.session_state.progress = ProgressService.pause_timer(st.session_state.progress)
                persist_current_state()
                st.rerun()
            timer_cols[1].markdown(
                f"<div style='margin-top: 5px; color: #0EA5A4; font-weight: bold; text-align: center;'>{t('running')}</div>",
                unsafe_allow_html=True,
            )

        if timer_cols[1].button(t("reset"), use_container_width=True):
            st.session_state.progress = ProgressService.restart_timer(st.session_state.progress)
            persist_current_state()
            st.rerun()

        if timer_cols[2].button(t("finish_section"), use_container_width=True):
            st.session_state.progress = ProgressService.finish_section(
                st.session_state.progress,
                section.section_number,
            )
            persist_current_state()
            st.toast(t("section_completed"))
            st.rerun()

        state = section_state(section)
        if st.button(t("explain_section"), use_container_width=True):
            with st.spinner(t("generating_explanation")):
                state["explanation"] = generate_explanation(section)
                persist_current_state()
        if state.get("explanation"):
            with st.expander(t("explanation"), expanded=True):
                st.markdown(state["explanation"])

        with st.expander(t("quiz"), expanded=bool(state.get("quiz"))):
            if st.button(t("generate_quiz"), use_container_width=True):
                with st.spinner(t("generating_quiz")):
                    state["quiz"] = build_section_quiz(section)
                    state["quiz_answers"] = {}
                    state["quiz_score"] = None
                    state["quiz_feedback"] = []
                    persist_current_state()

            if state.get("quiz"):
                render_section_quiz(section, state)

        with st.expander(t("ask_section_question")):
            question_key = f"section-question-{section.section_number}"
            st.session_state.setdefault(question_key, state.get("question", ""))
            with st.form(key=f"section-question-form-{section.section_number}"):
                question = st.text_area(t("question"), key=question_key, height=90)
                submitted = st.form_submit_button(t("ask_about_section"))
            if submitted:
                if not question.strip():
                    st.warning(t("enter_question_first"))
                else:
                    with st.spinner(t("finding_answer")):
                        state["question"] = question
                        state["answer"] = answer_section_question(section, question)
                        persist_current_state()
            if state.get("answer"):
                st.markdown(state["answer"])

        if should_show_next_section_button(
            st.session_state.current_section_index,
            len(st.session_state.sections),
        ) and st.button(t("next_section"), use_container_width=True):
            st.session_state.current_section_index = StudyService.next_section_index(
                st.session_state.current_section_index,
                len(st.session_state.sections),
            )
            persist_current_state()
            st.rerun()


def render_section_quiz(section, state: dict[str, Any]) -> None:
    for index, question in enumerate(state["quiz"], start=1):
        st.markdown(f"**{index}. {question['question']}**")
        key = f"quiz-answer-{section.section_number}-{index}"

        if question["type"] in {"multiple_choice", "true_false"}:
            answer = st.radio(t("answer"), question["options"], key=key, label_visibility="collapsed", index=None)
            state["quiz_answers"][index] = answer
        else:
            answer = st.text_area(t("short_answer"), key=key, height=80)
            state["quiz_answers"][index] = answer

        st.caption(source_label(section, question.get("source_page")))

    if st.button(t("submit_quiz")):
        with st.spinner(t("grading")):
            state["quiz_score"], state["quiz_feedback"] = grade_section_quiz(
                state["quiz"],
                state["quiz_answers"],
                section,
            )
            st.session_state.progress.section_quiz_scores[section.section_number] = float(state["quiz_score"])
            if state["quiz_score"] < 80 and section.title not in st.session_state.progress.weak_sections:
                st.session_state.progress.weak_sections.append(section.title)
            persist_current_state()
            st.success(t("quiz_submitted"))

    if state.get("quiz_score") is not None:
        st.success(f"{t('final_score')}: {state['quiz_score']}%")
        st.markdown(f"**{t('detailed_review')}**")
        for feedback in state.get("quiz_feedback", []):
            st.write(feedback)


def grade_section_quiz(questions: list[dict[str, Any]], answers: dict[int, Any], section) -> tuple[int, list[str]]:
    return QuizGradingService.grade(
        questions,
        answers,
        short_answer_evaluator=lambda question, answer: evaluate_short_answer(section, question, answer),
        language=current_language(),
    )


def should_show_next_section_button(current_index: int, total_sections: int) -> bool:
    return total_sections > 0 and current_index < total_sections - 1


def evaluate_short_answer(section, question: dict[str, Any], user_answer: Any) -> dict[str, Any]:
    return AIAnswerGradingService.grade_short_answer(
        question=question["question"],
        expected_answer=question["answer"],
        user_answer=user_answer,
        context=section_context(section),
        language=current_language(),
    )


def render_timer_display() -> None:
    progress = st.session_state.progress
    if not progress.timer_running or progress.timer_started_at is None:
        st.metric(t("actual_study_time"), format_seconds(progress.actual_study_seconds))
        return

    components.html(
        f"""
        <div style="font-family: sans-serif; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.55rem 0.75rem;">
            <div style="font-size: 0.85rem; color: #64748B;">{t("actual_study_time")}</div>
            <div id="study-timer" style="font-size: 1.55rem; font-weight: 700; color: #0F172A;"></div>
        </div>
        <script>
        const baseSeconds = {int(progress.actual_study_seconds)};
        const startedAt = {float(progress.timer_started_at)};
        const target = document.getElementById("study-timer");
        function formatTime(total) {{
            total = Math.max(0, Math.floor(total));
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const seconds = total % 60;
            if (hours > 0) return `${{hours}}h ${{minutes}}m`;
            return `${{minutes}}m ${{seconds}}s`;
        }}
        function tick() {{
            const now = Date.now() / 1000;
            target.textContent = formatTime(baseSeconds + now - startedAt);
        }}
        tick();
        setInterval(tick, 1000);
        </script>
        """,
        height=78,
    )
