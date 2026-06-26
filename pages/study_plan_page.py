from __future__ import annotations

import html

import streamlit as st

from translations import t
from ui.components import badge, roadmap_card
from ui.state import current_section, has_pdf, overall_progress, page_label, persist_current_state


def section_status(section) -> str:
    if section.section_number in st.session_state.progress.completed_sections:
        return "Completed"
    active = current_section()
    if active and active.section_number == section.section_number:
        return "In Progress"
    if active and section.section_number == active.section_number + 1:
        return "Next"
    return "Not Started"


def render_study_plan() -> None:
    st.subheader(t("study_plan"))
    if not has_pdf():
        st.info(t("upload_pdf_first"))
        return

    total_time = sum(section.estimated_minutes for section in st.session_state.sections)
    cols = st.columns(3)
    cols[0].metric(t("total_sessions"), len(st.session_state.sections))
    cols[1].metric(t("total_estimated_time"), f"{total_time} {t('minutes')}")
    cols[2].metric(t("completed_sections"), len(st.session_state.progress.completed_sections))
    st.progress(overall_progress() / 100)
    st.caption(f"{overall_progress():.0f}%")

    for index, section in enumerate(st.session_state.sections):
        status = section_status(section)
        status_label = {
            "Completed": t("completed"),
            "In Progress": t("in_progress"),
            "Next": t("next"),
            "Not Started": t("not_started"),
        }.get(status, status)
        status_kind = {
            "Completed": "success",
            "In Progress": "current",
            "Next": "next",
        }.get(status, "secondary")
        concepts = "".join(
            f'<span class="tag">{html.escape(tag)}</span>' for tag in getattr(section, "key_concepts", [])
        )
        objectives = "".join(
            f"<li>{html.escape(objective)}</li>" for objective in getattr(section, "learning_objectives", [])[:5]
        )
        time_explanation = getattr(section, "time_explanation", "") or "Based on reading time, concepts, and practice."
        body = (
            f"{badge(page_label(section), 'accent')}"
            f"{badge(t('difficulty_' + section.difficulty.lower()) if section.difficulty.lower() in {'easy', 'medium', 'hard'} else section.difficulty, 'warning')}"
            f"{badge(status_label, status_kind)}"
            f"<p>{html.escape(section.summary)}</p>"
            f"<div class='muted'><strong>{html.escape(t('learning_objectives'))}</strong></div>"
            f"<ul class='objective-list'>{objectives}</ul>"
            f"<div>{concepts}</div>"
            f"<p class='muted'>{html.escape(t('estimated_time'))}: {section.estimated_minutes} {html.escape(t('minutes'))}</p>"
            f"<p class='muted'>{html.escape(time_explanation)}</p>"
        )
        roadmap_card(section, body)
        if st.button(t("start_studying"), key=f"start-section-{section.section_number}"):
            st.session_state.current_section_index = index
            st.session_state.current_page = "Study Mode"
            persist_current_state()
            st.rerun()
