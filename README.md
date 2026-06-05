# Smart Study Assistant

Smart Study Assistant is a Streamlit MVP for studying from uploaded PDFs. It focuses on a stable student workflow instead of experimental RAG controls.

## MVP Features
- Upload a PDF and extract text safely with PyMuPDF plus pypdf fallback.
- Generate a study plan with section cards, page ranges, estimated time, difficulty, summaries, concepts, and progress status.
- Study section by section while viewing rendered PDF pages.
- Download the current section PDF.
- Generate a section explanation and quiz.
- Ask a general AI tutor question outside the PDF context.
- Generate an AI final exam with safe fallback behavior.
- View dashboard progress, quiz average, actual study time, weak topics, and final exam score.

## Navigation
The Streamlit app uses these pages:

`Upload -> Study Plan -> Study Mode -> Ask AI -> Final Exam -> Dashboard`

The old RAG Check, separate Quiz tab, OCR tab, benchmark Results tab, and explanation level controls are intentionally not part of this MVP.

## Installation
```bash
pip install -r requirements.txt
```

## Running The App
```bash
streamlit run ui/streamlit_app.py
```

## Optional AI Keys
General Ask AI and Final Exam generation select providers in this order:

1. `OPENAI_API_KEY`
2. `GROQ_API_KEY`
3. Clear setup/fallback message when no key exists

No real API calls are made by the unit tests.

## Verification
```bash
python -m py_compile $(find . -name '*.py' -not -path './.git/*')
python -m unittest discover -s tests
```

## Notes
- Chunk IDs remain internal and are not shown in the normal UI.
- The older benchmark and LangChain modules remain in the repo for compatibility with existing tests and experiments.
- Text-based PDFs work best. Scanned PDFs may need separate OCR preprocessing.
