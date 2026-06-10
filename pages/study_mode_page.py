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
from ui.components import badge, card
from ui.state import (
    current_section,
    format_seconds,
    has_pdf,
    persist_current_state,
    section_context,
    section_state,
    source_label,
)
from ui.workflow import answer_section_question, build_section_quiz, generate_explanation


def render_pdf_pages(section) -> None:
    st.markdown("### PDF Section")
    st.markdown(f"**Now studying {section.page_label}**")
    images = PdfRenderService.render_pages(st.session_state.pdf_bytes, section.start_page, section.end_page)
    if images:
        for offset, image in enumerate(images, start=section.start_page):
            st.image(image, caption=source_label(section, offset), use_container_width=True)
    else:
        st.info("Page images are unavailable. Use the extracted text fallback below.")

    try:
        section_pdf = PdfSectionService.extract_section_pdf(
            st.session_state.pdf_bytes,
            section.start_page,
            section.end_page,
        )
        st.download_button(
            "Download section PDF",
            data=section_pdf,
            file_name=f"section-{section.section_number}.pdf",
            mime="application/pdf",
        )
    except PdfSectionError:
        st.caption("Section PDF download is unavailable for this page range.")

    with st.expander("Extracted text fallback"):
        st.text_area("Section text", value=section_context(section), height=300, label_visibility="collapsed")


def render_study_mode() -> None:
    st.subheader("Study Mode")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    section = current_section()
    if section is None:
        st.info("Generate a study plan first.")
        return

    left, right = st.columns([0.65, 0.35])
    with left:
        render_pdf_pages(section)

    with right:
        card(
            section.title,
            f"{badge(f'Section {section.section_number} of {len(st.session_state.sections)}', 'primary')}"
            f"{badge(section.page_label, 'accent')}"
            f"{badge(section.difficulty, 'warning')}"
            f"<p class='muted'>Estimated: {section.estimated_minutes} minutes</p>",
        )
        next_index = StudyService.next_section_index(
            st.session_state.current_section_index,
            len(st.session_state.sections),
        )
        completed = section.section_number in st.session_state.progress.completed_sections
        next_section = st.session_state.sections[next_index] if next_index != st.session_state.current_section_index else None
        st.caption("Completed" if completed else "Current section")
        if next_section:
            st.caption(f"Next: {next_section.title}")
        st.progress((section.section_number - 1) / max(1, len(st.session_state.sections)))
        render_timer_display()
        if st.session_state.pdf_name and not st.session_state.pdf_bytes:
            st.caption("Restored progress. PDF preview/download returns after uploading the PDF again.")

        timer_cols = st.columns([1, 1, 1])
        is_running = st.session_state.progress.timer_running
        has_started = st.session_state.progress.actual_study_seconds > 0

        if not is_running:
            button_label = "Resume Session" if has_started else "Start Session"
            if timer_cols[0].button(button_label, type="primary", use_container_width=True):
                st.session_state.progress = ProgressService.start_timer(st.session_state.progress)
                persist_current_state()
                st.rerun()
        else:
            if timer_cols[0].button("Pause", use_container_width=True):
                st.session_state.progress = ProgressService.pause_timer(st.session_state.progress)
                persist_current_state()
                st.rerun()
            timer_cols[1].markdown(
                "<div style='margin-top: 5px; color: #0EA5A4; font-weight: bold; text-align: center;'>Running</div>",
                unsafe_allow_html=True,
            )

        if timer_cols[1].button("Reset", use_container_width=True):
            st.session_state.progress = ProgressService.restart_timer(st.session_state.progress)
            persist_current_state()
            st.rerun()

        if timer_cols[2].button("Finish Section", use_container_width=True):
            st.session_state.progress = ProgressService.finish_section(
                st.session_state.progress,
                section.section_number,
            )
            persist_current_state()
            st.toast("Section completed.")
            st.rerun()

        state = section_state(section)
        if st.button("Explain This Section", use_container_width=True):
            state["explanation"] = generate_explanation(section)
            persist_current_state()
        if state.get("explanation"):
            with st.expander("Explanation", expanded=True):
                st.markdown(state["explanation"])

        with st.expander("Quiz", expanded=bool(state.get("quiz"))):
            if st.button("Generate Quiz", use_container_width=True):
                state["quiz"] = build_section_quiz(section)
                state["quiz_answers"] = {}
                state["quiz_score"] = None
                state["quiz_feedback"] = []
                persist_current_state()
            if state.get("quiz"):
                render_section_quiz(section, state)

        with st.expander("Ask a question about this section"):
            question_key = f"section-question-{section.section_number}"
            st.session_state.setdefault(question_key, state.get("question", ""))
            with st.form(key=f"section-question-form-{section.section_number}"):
                question = st.text_area("Question", key=question_key, height=90)
                submitted = st.form_submit_button("Ask About This Section")
            if submitted:
                if not question.strip():
                    st.warning("Enter a question first.")
                else:
                    with st.spinner("Finding an answer from this section..."):
                        state["question"] = question
                        state["answer"] = answer_section_question(section, question)
                        persist_current_state()
            if state.get("answer"):
                st.markdown(state["answer"])

        if should_show_next_section_button(
            st.session_state.current_section_index,
            len(st.session_state.sections),
        ) and st.button("Next Section", use_container_width=True):
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
            answer = st.radio("Answer", question["options"], key=key, label_visibility="collapsed", index=None)
            state["quiz_answers"][index] = answer
        else:
            answer = st.text_area("Short answer", key=key, height=80)
            state["quiz_answers"][index] = answer

        st.caption(source_label(section, question.get("source_page")))

    if st.button("Submit quiz"):
        with st.spinner("Grading..."):
            state["quiz_score"], state["quiz_feedback"] = grade_section_quiz(
                state["quiz"],
                state["quiz_answers"],
                section,
            )
            st.session_state.progress.section_quiz_scores[section.section_number] = float(state["quiz_score"])
            if state["quiz_score"] < 80 and section.title not in st.session_state.progress.weak_sections:
                st.session_state.progress.weak_sections.append(section.title)
            persist_current_state()
            st.success("Quiz submitted.")

    if state.get("quiz_score") is not None:
        st.success(f"Final Score: {state['quiz_score']}%")
        st.markdown("**Detailed Review**")
        for feedback in state.get("quiz_feedback", []):
            st.write(feedback)


def grade_section_quiz(questions: list[dict[str, Any]], answers: dict[int, Any], section) -> tuple[int, list[str]]:
    return QuizGradingService.grade(
        questions,
        answers,
        short_answer_evaluator=lambda question, answer: evaluate_short_answer(section, question, answer),
    )


def should_show_next_section_button(current_index: int, total_sections: int) -> bool:
    return total_sections > 0 and current_index < total_sections - 1


def evaluate_short_answer(section, question: dict[str, Any], user_answer: Any) -> dict[str, Any]:
    prompt = (
        f"Study section text: {section_context(section)}\n\n"
        f"Question: {question['question']}\n"
        f"Expected answer: {question['answer']}\n"
        f"User's answer: {user_answer}\n\n"
        "Evaluate the answer strictly based on the provided text. "
        "Provide a score from 0 to 100 and short feedback. Format: Score: [0-100] | Feedback: [Explanation]"
    )
    response = GeneralAIService().ask([], prompt)
    if not response["ok"]:
        return {"score": 0, "feedback": "Could not grade."}

    score_match = re.search(r"Score:\s*(\d+)", response["answer"])
    score = int(score_match.group(1)) if score_match else 0
    return {"score": max(0, min(score, 100)), "feedback": response["answer"]}


def render_timer_display() -> None:
    progress = st.session_state.progress
    if not progress.timer_running or progress.timer_started_at is None:
        st.metric("Actual study time", format_seconds(progress.actual_study_seconds))
        return

    components.html(
        f"""
        <div style="font-family: sans-serif; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.55rem 0.75rem;">
            <div style="font-size: 0.85rem; color: #64748B;">Actual study time</div>
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
