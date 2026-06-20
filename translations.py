from __future__ import annotations

from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - lets service tests run without Streamlit installed.
    st = None


DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {
    "en": "English 🇺🇸",
    "he": "עברית 🇮🇱",
}
RTL_LANGUAGES = {"he"}
_FALLBACK_SESSION_STATE: dict[str, Any] = {}


TRANSLATIONS = {
    "en": {
        "app_title": "Smart Study Assistant",
        "language": "Language",
        "upload_pdf": "Upload PDF",
        "upload": "Upload",
        "study_plan": "Study Plan",
        "study_mode": "Study Mode",
        "ai_tutor": "AI Tutor",
        "final_exam": "Final Exam",
        "dashboard": "Dashboard",
        "generate": "Generate",
        "submit": "Submit",
        "next": "Next",
        "back": "Back",
        "success": "Success",
        "error": "Error",
        "current_pdf": "Current PDF",
        "pages": "Pages",
        "sessions": "Sessions",
        "progress": "Progress",
        "no_pdf_loaded": "No PDF loaded yet - upload a document to begin.",
        "change_pdf": "Change PDF",
        "choose_another_pdf": "Choose another PDF",
        "upload_hero_title": "Turn PDFs into guided study sessions, quizzes, and exam practice.",
        "upload_hero_body": "Upload course material, study section by section, and ask the AI Tutor for help.",
        "pdf_source": "PDF source",
        "upload_file": "Upload file",
        "upload_folder": "Upload folder",
        "choose_pdf": "Choose a PDF",
        "extracting_pdf": "Extracting text from your PDF...",
        "pdf_extraction_failed": "PDF extraction failed",
        "pdf_process_failed": "Could not process PDF safely",
        "reprocess_pdf": "Reprocess PDF",
        "choose_pdf_folder": "Choose a folder that contains PDFs",
        "folder_upload_help": "Upload a folder, then choose one PDF from the uploaded files.",
        "choose_pdf_from_folder": "Choose a PDF from the uploaded folder",
        "pdfs_uploaded_from_folder": "{count} PDF file(s) uploaded from the folder.",
        "process_selected_pdf": "Process selected PDF",
        "reprocess_selected_pdf": "Reprocess selected PDF",
        "number_of_study_sessions": "Number of Study Sessions",
        "session_count_help": "Suggested based on PDF length. You can change it.",
        "pages_processed": "Pages processed",
        "suggested_sessions": "Suggested sessions",
        "estimated_study_time": "Estimated study time",
        "study_plan_session_caption": "Suggested sessions: {suggested}. Your study plan will use {count} study sessions.",
        "generate_study_plan": "Generate Study Plan",
        "processed_pdf_ready": "Processed {pdf_name}. Ready to generate a study plan.",
        "generated_sessions": "Generated {count} study sessions.",
        "no_readable_sessions": "No readable study sessions could be created from this PDF.",
        "upload_pdf_first": "Upload and process a PDF first.",
        "total_sessions": "Total sessions",
        "total_estimated_time": "Total estimated time",
        "completed_sections": "Completed sections",
        "completed": "Completed",
        "in_progress": "In Progress",
        "not_started": "Not Started",
        "current_section": "Current section",
        "learning_objectives": "Learning objectives",
        "estimated_time": "Estimated time",
        "minutes": "minutes",
        "start_studying": "Start Studying",
        "generate_plan_first": "Generate a study plan first.",
        "pdf_section": "PDF Section",
        "now_studying": "Now studying {page_label}",
        "page_images_unavailable": "Page images are unavailable. Use the extracted text fallback below.",
        "download_section_pdf": "Download section PDF",
        "section_pdf_unavailable": "Section PDF download is unavailable for this page range.",
        "extracted_text_fallback": "Extracted text fallback",
        "section_text": "Section text",
        "section_count": "Section {number} of {total}",
        "estimated": "Estimated",
        "next_section_title": "Next: {title}",
        "restored_pdf_notice": "Restored progress. PDF preview/download returns after uploading the PDF again.",
        "resume_session": "Resume Session",
        "start_session": "Start Session",
        "pause": "Pause",
        "running": "Running",
        "reset": "Reset",
        "finish_section": "Finish Section",
        "section_completed": "Section completed.",
        "explain_section": "Explain This Section",
        "generating_explanation": "Generating AI explanation...",
        "explanation": "Explanation",
        "quiz": "Quiz",
        "generate_quiz": "Generate Quiz",
        "generating_quiz": "Generating AI quiz questions...",
        "ask_section_question": "Ask a question about this section",
        "question": "Question",
        "ask_about_section": "Ask About This Section",
        "enter_question_first": "Enter a question first.",
        "finding_answer": "Finding an answer from this section...",
        "next_section": "Next Section",
        "answer": "Answer",
        "short_answer": "Short answer",
        "submit_quiz": "Submit quiz",
        "grading": "Grading...",
        "quiz_submitted": "Quiz submitted.",
        "final_score": "Final Score",
        "detailed_review": "Detailed Review",
        "actual_study_time": "Actual study time",
        "ai_tutor_mode_general": "General AI Tutor",
        "ai_tutor_mode_section": "Ask about current section",
        "ai_tutor_caption": "Ask general study questions, request examples, or get help understanding a topic.",
        "prompt_summarize_material": "Summarize this study material",
        "prompt_explain_concept": "Explain a difficult concept simply",
        "prompt_study_plan": "Help me build a study plan",
        "prompt_practice_questions": "Create practice questions from my material",
        "use_pdf_context": "Use uploaded PDF context",
        "clear_chat": "Clear Chat",
        "ask_general_question": "Ask a general study question",
        "provider": "Provider",
        "current_section_label": "Current section: {title}",
        "reading_section": "Reading the current section...",
        "questions": "Questions",
        "difficulty": "Difficulty",
        "difficulty_mixed": "mixed",
        "difficulty_easy": "easy",
        "difficulty_medium": "medium",
        "difficulty_hard": "hard",
        "generate_final_exam": "Generate AI final exam",
        "generating_final_exam": "Generating final exam...",
        "final_exam_generated": "Final exam generated.",
        "generate_exam_hint": "Generate a final exam when you finish reviewing the study plan.",
        "fallback_exam_used": "Fallback exam was used.",
        "ai_final_exam": "AI Final Exam",
        "topic": "Topic",
        "review": "Review",
        "submit_exam": "Submit Exam",
        "final_exam_submitted": "Final exam submitted.",
        "score": "Score",
        "correct_answers": "Correct answers",
        "wrong_answers": "Wrong answers",
        "related_weak_sections": "Related weak sections",
        "weak_topics": "Weak topics",
        "weak_sections": "Weak sections",
        "recommendation": "Recommendation",
        "review_missed_answers": "Review missed answers",
        "related_section": "Related section",
        "your_answer": "Your answer:",
        "correct_answer": "Correct answer:",
        "feedback": "Feedback:",
        "review_section": "Review section",
        "no_answer_provided": "No answer provided.",
        "no_expected_answer": "No expected answer available.",
        "all_answers_correct": "All answers were correct.",
        "review_missed_default": "Review your missed questions.",
        "learning_progress": "Learning Progress",
        "completed_sessions": "Completed Sessions",
        "quiz_scores": "Quiz Scores",
        "quiz_average": "Quiz Average",
        "exam_readiness": "Exam Readiness",
        "study_time": "Study Time",
        "total_study_time": "Total study time",
        "average_per_section": "Average per section",
        "final_exam_score": "Final Exam Score",
        "no_progress_yet": "No progress yet. Start a section, take a quiz, or submit the final exam.",
        "recommendations": "Recommendations",
        "keep_reviewing": "Keep reviewing completed sections.",
        "review_before_exam": "Review {title} before taking the final exam again.",
        "recommended_next_section": "Recommended next section: {title}.",
        "review_weak_topics": "Review Weak Topics",
        "weak_review_complete": "All sections are complete. Revisit the final exam answers and retake any quiz below 80%.",
        "weak_review_title": "Weak Topic Review Plan",
        "weak_review_item": "Review {topic} - {reason} in Section {number}. Re-read {page_label} and retake the section quiz.",
        "reason_low_quiz": "your quiz average is low",
        "reason_not_completed": "this section is not completed yet",
        "source_section_page": "Source: Section {section}, Page {page}",
        "source_page": "Source: Page {page}",
        "source_pages": "Source: Pages {start}-{end}",
        "page_label_one": "Page {page}",
        "page_label_range": "Pages {start}-{end}",
        "true": "True",
        "false": "False",
        "fallback_core_idea": "Core idea",
        "fallback_example": "Example",
        "fallback_review_point": "Review point",
    },
    "he": {
        "app_title": "עוזר לימוד חכם",
        "language": "שפה",
        "upload_pdf": "העלאת PDF",
        "upload": "העלאה",
        "study_plan": "תוכנית לימוד",
        "study_mode": "מצב לימוד",
        "ai_tutor": "מורה AI",
        "final_exam": "מבחן מסכם",
        "dashboard": "לוח בקרה",
        "generate": "צור",
        "submit": "שלח",
        "next": "הבא",
        "back": "חזור",
        "success": "הצלחה",
        "error": "שגיאה",
        "current_pdf": "PDF נוכחי",
        "pages": "עמודים",
        "sessions": "מפגשים",
        "progress": "התקדמות",
        "no_pdf_loaded": "עדיין לא נטען PDF - העלו מסמך כדי להתחיל.",
        "change_pdf": "החלפת PDF",
        "choose_another_pdf": "בחרו PDF אחר",
        "upload_hero_title": "הפכו קבצי PDF למפגשי לימוד, שאלונים ותרגול למבחן.",
        "upload_hero_body": "העלו חומר קורס, למדו לפי חלקים, ושאלו את מורה ה-AI כשצריך עזרה.",
        "pdf_source": "מקור PDF",
        "upload_file": "העלאת קובץ",
        "upload_folder": "העלאת תיקייה",
        "choose_pdf": "בחרו PDF",
        "extracting_pdf": "מחלץ טקסט מה-PDF...",
        "pdf_extraction_failed": "חילוץ ה-PDF נכשל",
        "pdf_process_failed": "לא ניתן היה לעבד את ה-PDF בבטחה",
        "reprocess_pdf": "עבד PDF מחדש",
        "choose_pdf_folder": "בחרו תיקייה שמכילה קבצי PDF",
        "folder_upload_help": "העלו תיקייה ואז בחרו PDF אחד מתוך הקבצים.",
        "choose_pdf_from_folder": "בחרו PDF מתוך התיקייה שהועלתה",
        "pdfs_uploaded_from_folder": "{count} קבצי PDF הועלו מהתיקייה.",
        "process_selected_pdf": "עבד את ה-PDF שנבחר",
        "reprocess_selected_pdf": "עבד מחדש את ה-PDF שנבחר",
        "number_of_study_sessions": "מספר מפגשי לימוד",
        "session_count_help": "ההצעה מבוססת על אורך ה-PDF. ניתן לשנות אותה.",
        "pages_processed": "עמודים שעובדו",
        "suggested_sessions": "מפגשים מומלצים",
        "estimated_study_time": "זמן לימוד משוער",
        "study_plan_session_caption": "מפגשים מומלצים: {suggested}. תוכנית הלימוד תכלול {count} מפגשים.",
        "generate_study_plan": "צור תוכנית לימוד",
        "processed_pdf_ready": "הקובץ {pdf_name} עובד. אפשר ליצור תוכנית לימוד.",
        "generated_sessions": "נוצרו {count} מפגשי לימוד.",
        "no_readable_sessions": "לא ניתן ליצור מפגשי לימוד קריאים מתוך ה-PDF.",
        "upload_pdf_first": "העלו ועבדו PDF קודם.",
        "total_sessions": "סך כל המפגשים",
        "total_estimated_time": "זמן משוער כולל",
        "completed_sections": "חלקים שהושלמו",
        "completed": "הושלם",
        "in_progress": "בתהליך",
        "not_started": "טרם התחיל",
        "current_section": "החלק הנוכחי",
        "learning_objectives": "מטרות לימוד",
        "estimated_time": "זמן משוער",
        "minutes": "דקות",
        "start_studying": "התחל ללמוד",
        "generate_plan_first": "צרו קודם תוכנית לימוד.",
        "pdf_section": "חלק PDF",
        "now_studying": "לומדים עכשיו {page_label}",
        "page_images_unavailable": "תמונות העמודים אינן זמינות. השתמשו בטקסט המחולץ למטה.",
        "download_section_pdf": "הורד PDF של החלק",
        "section_pdf_unavailable": "הורדת PDF לחלק אינה זמינה לטווח העמודים הזה.",
        "extracted_text_fallback": "טקסט מחולץ כגיבוי",
        "section_text": "טקסט החלק",
        "section_count": "חלק {number} מתוך {total}",
        "estimated": "משוער",
        "next_section_title": "הבא: {title}",
        "restored_pdf_notice": "ההתקדמות שוחזרה. תצוגת PDF והורדה יחזרו אחרי העלאת ה-PDF שוב.",
        "resume_session": "המשך מפגש",
        "start_session": "התחל מפגש",
        "pause": "השהה",
        "running": "פעיל",
        "reset": "אפס",
        "finish_section": "סיים חלק",
        "section_completed": "החלק הושלם.",
        "explain_section": "הסבר את החלק הזה",
        "generating_explanation": "יוצר הסבר AI...",
        "explanation": "הסבר",
        "quiz": "שאלון",
        "generate_quiz": "צור שאלון",
        "generating_quiz": "יוצר שאלות AI...",
        "ask_section_question": "שאלו שאלה על החלק הזה",
        "question": "שאלה",
        "ask_about_section": "שאל על החלק הזה",
        "enter_question_first": "הכניסו שאלה קודם.",
        "finding_answer": "מחפש תשובה מתוך החלק...",
        "next_section": "החלק הבא",
        "answer": "תשובה",
        "short_answer": "תשובה קצרה",
        "submit_quiz": "שלח שאלון",
        "grading": "בודק...",
        "quiz_submitted": "השאלון נשלח.",
        "final_score": "ציון סופי",
        "detailed_review": "סקירה מפורטת",
        "actual_study_time": "זמן לימוד בפועל",
        "ai_tutor_mode_general": "מורה AI כללי",
        "ai_tutor_mode_section": "שאלה על החלק הנוכחי",
        "ai_tutor_caption": "שאלו שאלות לימוד כלליות, בקשו דוגמאות או עזרה בהבנת נושא.",
        "prompt_summarize_material": "סכם את חומר הלימוד הזה",
        "prompt_explain_concept": "הסבר מושג קשה בפשטות",
        "prompt_study_plan": "עזור לי לבנות תוכנית לימוד",
        "prompt_practice_questions": "צור שאלות תרגול מהחומר שלי",
        "use_pdf_context": "השתמש בהקשר מה-PDF שהועלה",
        "clear_chat": "נקה צ'אט",
        "ask_general_question": "שאלו שאלת לימוד כללית",
        "provider": "ספק",
        "current_section_label": "החלק הנוכחי: {title}",
        "reading_section": "קורא את החלק הנוכחי...",
        "questions": "שאלות",
        "difficulty": "רמת קושי",
        "difficulty_mixed": "מעורב",
        "difficulty_easy": "קל",
        "difficulty_medium": "בינוני",
        "difficulty_hard": "קשה",
        "generate_final_exam": "צור מבחן מסכם AI",
        "generating_final_exam": "יוצר מבחן מסכם...",
        "final_exam_generated": "המבחן המסכם נוצר.",
        "generate_exam_hint": "צרו מבחן מסכם לאחר שסיימתם לחזור על תוכנית הלימוד.",
        "fallback_exam_used": "נעשה שימוש במבחן גיבוי.",
        "ai_final_exam": "מבחן AI מסכם",
        "topic": "נושא",
        "review": "חזרה",
        "submit_exam": "שלח מבחן",
        "final_exam_submitted": "המבחן המסכם נשלח.",
        "score": "ציון",
        "correct_answers": "תשובות נכונות",
        "wrong_answers": "תשובות שגויות",
        "related_weak_sections": "חלקים חלשים קשורים",
        "weak_topics": "נושאים חלשים",
        "weak_sections": "חלקים חלשים",
        "recommendation": "המלצה",
        "review_missed_answers": "סקירת תשובות שגויות",
        "related_section": "חלק קשור",
        "your_answer": "התשובה שלך:",
        "correct_answer": "התשובה הנכונה:",
        "feedback": "משוב:",
        "review_section": "חזור על החלק",
        "no_answer_provided": "לא ניתנה תשובה.",
        "no_expected_answer": "לא קיימת תשובה צפויה.",
        "all_answers_correct": "כל התשובות נכונות.",
        "review_missed_default": "חזרו על השאלות שפספסתם.",
        "learning_progress": "התקדמות לימודית",
        "completed_sessions": "מפגשים שהושלמו",
        "quiz_scores": "ציוני שאלונים",
        "quiz_average": "ממוצע שאלונים",
        "exam_readiness": "מוכנות למבחן",
        "study_time": "זמן לימוד",
        "total_study_time": "זמן לימוד כולל",
        "average_per_section": "ממוצע לכל חלק",
        "final_exam_score": "ציון מבחן מסכם",
        "no_progress_yet": "אין עדיין התקדמות. התחילו חלק, פתרו שאלון או שלחו את המבחן המסכם.",
        "recommendations": "המלצות",
        "keep_reviewing": "המשיכו לחזור על החלקים שהושלמו.",
        "review_before_exam": "חזרו על {title} לפני ניסיון נוסף במבחן המסכם.",
        "recommended_next_section": "החלק הבא המומלץ: {title}.",
        "review_weak_topics": "חזור על נושאים חלשים",
        "weak_review_complete": "כל החלקים הושלמו. חזרו על תשובות המבחן ופתרו שוב כל שאלון מתחת ל-80%.",
        "weak_review_title": "תוכנית חזרה לנושאים חלשים",
        "weak_review_item": "חזרה על {topic} - {reason} בחלק {number}. קראו שוב את {page_label} ופתרו מחדש את שאלון החלק.",
        "reason_low_quiz": "ממוצע השאלונים נמוך",
        "reason_not_completed": "החלק הזה עדיין לא הושלם",
        "source_section_page": "מקור: חלק {section}, עמוד {page}",
        "source_page": "מקור: עמוד {page}",
        "source_pages": "מקור: עמודים {start}-{end}",
        "page_label_one": "עמוד {page}",
        "page_label_range": "עמודים {start}-{end}",
        "true": "נכון",
        "false": "לא נכון",
        "fallback_core_idea": "רעיון מרכזי",
        "fallback_example": "דוגמה",
        "fallback_review_point": "נקודת חזרה",
    },
}


def _session_state() -> Any:
    if st is None:
        return _FALLBACK_SESSION_STATE
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return _FALLBACK_SESSION_STATE
    except Exception:
        return _FALLBACK_SESSION_STATE
    return st.session_state


def normalize_language(language: str | None) -> str:
    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def current_language() -> str:
    return normalize_language(_session_state().get("language", DEFAULT_LANGUAGE))


def set_language(language: str) -> str:
    normalized = normalize_language(language)
    _session_state()["language"] = normalized
    return normalized


def translate(key: str, language: str | None = None, **kwargs: Any) -> str:
    lang = normalize_language(language or current_language())
    value = TRANSLATIONS.get(lang, {}).get(key)
    if value is None:
        value = TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
    if kwargs:
        return value.format(**kwargs)
    return value


def t(key: str, **kwargs: Any) -> str:
    return translate(key, **kwargs)


def is_rtl(language: str | None = None) -> bool:
    return normalize_language(language or current_language()) in RTL_LANGUAGES


def language_direction(language: str | None = None) -> str:
    return "rtl" if is_rtl(language) else "ltr"


def text_align(language: str | None = None) -> str:
    return "right" if is_rtl(language) else "left"


def rtl_css(language: str | None = None) -> str:
    direction = language_direction(language)
    align = text_align(language)
    return f"""
        html, body, [class*="css"] {{
            direction: {direction};
            text-align: {align};
        }}
        .stApp {{
            direction: {direction};
            text-align: {align};
        }}
    """


def study_plan_language_instruction(language: str | None = None) -> str:
    if normalize_language(language or current_language()) == "he":
        return "צור תוכנית לימוד מסודרת בעברית.\nהשתמש בעברית תקינה וברורה."
    return "Generate a structured study plan in English."


def tutor_language_instruction(language: str | None = None) -> str:
    if normalize_language(language or current_language()) == "he":
        return "ענה בעברית בלבד.\nהשתמש בשפה ברורה ופשוטה.\nאם המידע אינו מופיע במסמך ציין זאת."
    return "Answer in English."


def quiz_language_instruction(language: str | None = None) -> str:
    if normalize_language(language or current_language()) == "he":
        return "Generate quiz questions in Hebrew. Questions, options, correct answers, and explanations must be in Hebrew."
    return "Generate quiz questions in English."


def exam_language_instruction(language: str | None = None) -> str:
    if normalize_language(language or current_language()) == "he":
        return "כתוב את כל המבחן בעברית: שאלות, אפשרויות, תשובות, נושאים והסברים."
    return "Write the full exam in English: questions, options, answers, topics, and explanations."
