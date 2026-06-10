from __future__ import annotations

import html

import streamlit as st

from ui.components import badge, roadmap_card
from ui.state import current_section, has_pdf, overall_progress, persist_current_state


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
    st.subheader("Study Plan")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    total_time = sum(section.estimated_minutes for section in st.session_state.sections)
    cols = st.columns(3)
    cols[0].metric("Total sessions", len(st.session_state.sections))
    cols[1].metric("Total estimated time", f"{total_time} min")
    cols[2].metric("Completed sections", len(st.session_state.progress.completed_sections))
    st.progress(overall_progress() / 100)
    st.caption(f"{overall_progress():.0f}%")

    for index, section in enumerate(st.session_state.sections):
        status = section_status(section)
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
        body = (
            f"{badge(section.page_label, 'accent')}"
            f"{badge(section.difficulty, 'warning')}"
            f"{badge(status, status_kind)}"
            f"<p>{html.escape(section.summary)}</p>"
            f"<div class='muted'><strong>Learning objectives</strong></div>"
            f"<ul class='objective-list'>{objectives}</ul>"
            f"<div>{concepts}</div>"
            f"<p class='muted'>Estimated time: {section.estimated_minutes} minutes</p>"
        )
        roadmap_card(section, body)
        if st.button("Start Studying", key=f"start-section-{section.section_number}"):
            st.session_state.current_section_index = index
            st.session_state.current_page = "Study Mode"
            persist_current_state()
            st.rerun()
