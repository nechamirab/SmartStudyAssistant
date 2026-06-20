# AI Prompting and Data Flow Documentation

## Purpose and Scope

This document explains how AI prompting is implemented in the Smart Study Assistant project and what data is sent to external AI providers. It is intended for the final academic submission for the course "מעבדת תכנות מתקדמת ב-AI".

The main application is a Streamlit-based PDF study assistant. A student uploads a PDF, the system extracts text, creates a study plan, supports section-based studying, generates quizzes and a final exam, and optionally uses an AI tutor. The production app uses hosted chat-completion providers through `OPENAI_API_KEY` or `GROQ_API_KEY`. If those keys are missing, or if a provider response fails, several features fall back to deterministic local behavior.

The repository also contains a `research/` area for RAG, embeddings, benchmarking, and experimentation. Those files are documented separately in this file when they contain prompt construction or external AI calls.

## AI Provider Layer

| File | Function / class | Role |
| --- | --- | --- |
| `services/general_ai_service.py` | `GeneralAIService.select_provider()` | Loads local environment values and chooses OpenAI first, then Groq. |
| `services/general_ai_service.py` | `GeneralAIService.complete()` | Sends a system message and one user prompt. Used when the caller provides a custom system prompt, mainly study plan generation. |
| `services/general_ai_service.py` | `GeneralAIService.ask()` | Sends the general AI tutor system message, up to the last 12 supplied conversation messages, and the current user question. Used by tutoring, section explanations, quiz generation, and answer grading. |
| `services/general_ai_service.py` | `_ask_openai()` | Calls `client.chat.completions.create()` with model, messages, temperature `0.3`, and optional `response_format`. |
| `services/general_ai_service.py` | `_ask_groq()` | Calls Groq's OpenAI-compatible chat completions endpoint with the same message shape. |
| `services/exam_service.py` | `ExamService._call_ai()` | Calls OpenAI or Groq directly for final exam generation using temperature `0.7` and JSON response format. |

Provider selection:

```python
openai_key = os.getenv("OPENAI_API_KEY", "").strip()
if openai_key:
    return AIProvider("openai", openai_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
groq_key = os.getenv("GROQ_API_KEY", "").strip()
if groq_key:
    return AIProvider("groq", groq_key, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
```

Default models in the current code:

| Provider | Default model |
| --- | --- |
| OpenAI | `gpt-4o-mini` |
| Groq | `llama-3.1-8b-instant` |

## Where Prompting Happens

### Study Plan Generation

| Item | Implementation |
| --- | --- |
| Feature | Generate coherent study sessions from PDF pages. |
| Main file | `services/study_service.py` |
| Main functions | `generate_study_plan()`, `generate_study_plan_for_sessions()`, `_generate_ai_study_plan_for_sessions()` |
| Provider wrapper | `GeneralAIService.complete()` |
| UI caller | `ui/workflow.py`: `set_pending_pdf()` and `generate_study_plan_from_pending()` |

Prompt purpose: ask the AI to group readable PDF pages into a fixed number of ordered, non-overlapping study sections.

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | Indirectly | The selected session count is sent as `target_count`. |
| Extracted PDF text | Yes | `_ai_page_context()` sends readable pages, truncated to a maximum of 14,000 characters total. Each page is also limited by `per_page = max(120, min(1200, max_chars // page_count))`. |
| Section/session content | Not yet | This call creates sections, so no prior sections are sent. |
| Metadata | Yes | Readable page numbers, requested section count, language, page labels embedded as `Page {number}`. |
| Conversation history | No | Uses `complete()` with only one system message and one user prompt. |
| Full PDF text | No | Only selected/truncated page text is sent. |
| Private data | Possibly | Any text extracted from the uploaded PDF may be sent to the provider. |

System instruction:

```text
You create exam-focused study plans from extracted PDF text.
Return only valid JSON that follows the requested schema.
Do not include markdown, commentary, citations, or extra keys.
```

Required output format:

```json
{
  "sections": [
    {
      "section_number": 1,
      "title": "short topic title",
      "start_page": 1,
      "end_page": 2,
      "estimated_minutes": 25,
      "difficulty": "Easy|Medium|Hard",
      "summary": "2-3 sentence student-facing summary",
      "key_concepts": ["concept 1", "concept 2"],
      "learning_objectives": ["objective 1", "objective 2"]
    }
  ]
}
```

Validation and fallback:

- The response is requested with `response_format={"type": "json_object"}`.
- `_parse_ai_json()` removes Markdown fences if needed and attempts to parse JSON.
- `_sections_from_ai_payload()` verifies section count, page ranges, ordering, coverage of all readable pages, and field types.
- If the AI fails, returns invalid JSON, returns invalid ranges, or omits coverage, the service creates a heuristic local study plan from page groups, extracted headings, key terms, estimated minutes, and difficulty.

### Section Explanation

| Item | Implementation |
| --- | --- |
| Feature | Explain the current study section for exam preparation. |
| Main file | `ui/workflow.py` |
| Function | `generate_explanation(section)` |
| Provider wrapper | `GeneralAIService.ask()` for non-PDF general chat; `GeneralAIService.complete()` for grounded PDF-context answers. |
| UI caller | `pages/study_mode_page.py` |

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | No direct free-text input | The user clicks "Explain This Section". |
| Extracted PDF text | Yes | Current section text only, truncated to 6,000 characters. |
| Section/session content | Yes | Section title, page label, key concepts, and section text. |
| Metadata | Yes | Section title, page range, key concepts, active language. |
| Conversation history | No | `ask([], prompt, language=...)` sends no history. |
| Full PDF text | No | Only the current section is sent. |
| Private data | Possibly | The current section text from the uploaded PDF leaves the local app. |

Prompt constraints:

```text
Explain this study section for a student preparing for an exam.
Use the provided section text only.
Structure the answer with these headings:
Summary, Key Ideas, Important Definitions, Exam Tips.
Keep it clear and practical.
```

Fallback:

- If the provider call fails, the app builds a local explanation from the first useful section sentences, key concepts, and fixed exam tips.
- If no key concepts are available, it uses fallback terms from translations such as "Core idea", "Example", and "Review point".

### Current Section Question Answering

| Item | Implementation |
| --- | --- |
| Feature | Ask a grounded question about the uploaded PDF from the current section UI. |
| Main file | `ui/workflow.py` |
| Function | `answer_section_question(section, question)` |
| Retrieval service | `services/context_retrieval_service.py`: `ContextRetrievalService` |
| Provider wrapper | `GeneralAIService.complete()` |
| UI callers | `pages/study_mode_page.py`, `pages/ai_tutor_page.py` |

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | Yes | The student's question is sent as the current user message. |
| Extracted PDF text | Yes, selected locally first | The app builds local chunks from all extracted pages and study sections, retrieves the top relevant chunks, and sends only those chunks. |
| Section/session content | Yes, only for retrieved chunks | Each sent chunk includes section number, section title, page, text, and key concepts. |
| Metadata | Yes | Prompt context includes `[Section X | Title | Page Y]` before each chunk. The UI also appends a retrieved-source list. |
| Conversation history | No | Grounded PDF Q&A does not send previous chat history. |
| Full PDF text | No | The whole PDF is searched locally, but only top relevant chunks are sent, capped by `format_chunks_for_prompt(..., max_chars=7000)`. |
| Private data | Possibly | The student's question and selected PDF chunks leave the local app when a provider is configured. |

Current prompt shape:

```text
You are a grounded PDF study assistant.
Answer ONLY using the provided PDF context.
Do not use outside knowledge.
Do not guess.
If the answer is not supported by the provided PDF context, reply exactly:
"The uploaded PDF does not contain enough information to answer this question."
When you answer, include a short source line using the provided section/page metadata.

Provided PDF context:
[Section 4 | Shortest Paths | Page 11]
...

Question:
{question}
```

Fallback:

- If local retrieval finds no relevant chunks, the app returns exactly: `The uploaded PDF does not contain enough information to answer this question.`
- If a provider fails after chunks are retrieved, the app builds a local answer only from retrieved chunk sentences and appends retrieved source metadata.
- The app no longer asks the model to use the current section "when helpful"; the prompt requires using only retrieved PDF context.

### General AI Tutor

| Item | Implementation |
| --- | --- |
| Feature | General chat-style study assistant, optionally using uploaded PDF context. |
| Main file | `ui/workflow.py` |
| Function | `answer_ai_tutor(question, use_pdf_context=False)` |
| Provider wrapper | `GeneralAIService.ask()` |
| UI caller | `pages/ai_tutor_page.py` |

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | Yes | The chat input question is sent. |
| Extracted PDF text | Optional | If "Use uploaded PDF context" is checked and a PDF exists, `retrieve_ai_tutor_pdf_chunks()` first retrieves top relevant chunks locally. For broad requests such as summaries, main ideas, study plans, or practice questions, it falls back to representative chunks across sections. |
| Section/session content | Optional | Grounded mode sends retrieved or representative section/page chunks with source metadata. General mode sends no PDF context. |
| Metadata | Yes, if context enabled | Retrieved chunks include section number, section title, page, and text. |
| Conversation history | General mode only | General tutor mode sends recent chat history through `GeneralAIService.ask()`. Grounded PDF mode does not send history. |
| Full PDF text | No | The whole PDF is searched locally; the AI receives only selected chunks. Broad PDF-level questions receive representative chunks from multiple sections, not the full PDF. |
| Private data | Possibly | Chat history, user question, and selected PDF context may leave the local app. |

General system instruction:

```text
You are a clear, supportive AI tutor. Answer general study questions without PDF citations.
Answer in English.
```

For Hebrew mode, `tutor_language_instruction()` changes the instruction to answer only in Hebrew and state when information is not in the document.

Fallback:

- If no provider is configured or the request fails, the UI returns a setup message explaining that `OPENAI_API_KEY` or `GROQ_API_KEY` is required.
- If PDF context is enabled and a broad PDF-level question has representative chunks, provider failure falls back to a local section-based overview built from selected PDF chunks.
- If PDF context is enabled but no PDF text exists and no representative chunks can be built, the app returns the not-enough-information message.

### Quiz Generation

| Item | Implementation |
| --- | --- |
| Feature | Generate section quiz questions. |
| Main file | `services/quiz_service.py` |
| Main functions | `_generate_ai_questions()`, `generate_from_documents()` |
| UI helper | `ui/workflow.py`: `build_section_quiz(section)` |
| UI caller | `pages/study_mode_page.py` |

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | No direct free-text input | The user clicks "Generate Quiz". |
| Extracted PDF text | Yes | Current section text is sent through `documents`, limited to 6,000 characters by `_documents_to_context()`. |
| Section/session content | Yes | The current section context is provided as one document. |
| Metadata | Yes | Source label uses `st.session_state.pdf_name`; page is `section.start_page`; requested number of questions; language; random variation seed. |
| Conversation history | No | `ask([], prompt, language=...)` sends no prior history. |
| Full PDF text | No | Only the current section context is sent. |
| Private data | Possibly | The section text and PDF filename may leave the local app. |

Prompt constraints:

```text
Create multiple-choice quiz questions from the study material below.
Use only the provided material.
Create different questions each time, using the variation seed.
Return only valid JSON, without markdown.
The JSON must be a list of objects.
Each object must contain: prompt, options, answer, explanation, citation.
Each question must have exactly 4 options.
The answer must exactly match one of the options.
Avoid simple fill-in-the-blank questions.
Prefer conceptual understanding questions.
```

Expected JSON:

```json
[
  {
    "prompt": "Which statement best explains ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "explanation": "A is correct because ...",
    "citation": "Uploaded PDF p.3"
  }
]
```

Validation and fallback:

- The service parses the model output with `json.loads(raw_answer)`.
- Invalid JSON returns no AI questions.
- Invalid question objects are skipped unless they have `prompt`, exactly 4 options, and an answer that exactly matches one option.
- If no valid AI questions are available, `generate_from_documents()` builds deterministic fill-in-the-blank questions from local section sentences and keywords.
- `build_section_quiz()` uses the first AI or fallback MCQ, then adds a deterministic true/false question and a deterministic short-answer question.

### Section Quiz Short-Answer Evaluation

| Item | Implementation |
| --- | --- |
| Feature | Grade a student's short-answer response in a section quiz. |
| Main file | `services/ai_answer_grading_service.py` |
| Function | `AIAnswerGradingService.grade_short_answer()` |
| UI caller | `pages/study_mode_page.py`: `evaluate_short_answer()` |

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | Yes | The student's answer is sent. |
| Extracted PDF text | Yes for section quiz grading | The current section text is passed as `context=section_context(section)`. |
| Section/session content | Yes | The section context, quiz question, expected answer, and user answer are sent. |
| Metadata | Yes | Active language. |
| Conversation history | No | `ask([], prompt, language=...)` sends no history. |
| Full PDF text | No | Only the current section text is sent. |
| Private data | Possibly | Student answer and section text leave the local app. |

Prompt:

```text
Study context: {context}

Question: {question}
Expected answer: {expected_answer}
User's answer: {user_answer}

Write the score and feedback in English.
Evaluate the answer strictly based on the provided context and expected answer.
Use semantic meaning, not exact wording.
Provide a score from 0 to 100 and short feedback.
Format: Score: [0-100] | Feedback: [Explanation]
```

Parsing and fallback:

- If the user answer is empty, the service returns score `0` locally.
- If the provider fails, it returns score `0` and "Could not grade."
- It extracts the numeric score using `re.search(r"Score:\s*(\d+)")`.
- If the model does not follow the expected format, the parsed score becomes `0`, but the raw AI feedback is still returned.

### Final Exam Generation

| Item | Implementation |
| --- | --- |
| Feature | Generate a final exam from all study sections. |
| Main file | `services/exam_service.py` |
| Main functions | `generate_final_exam()`, `build_exam_prompt()`, `_call_ai()` |
| UI helper | `ui/workflow.py`: `generate_final_exam(question_count, difficulty)` |
| UI caller | `pages/final_exam_page.py` |

Data sent:

| Data type | Sent? | Details |
| --- | --- | --- |
| User input | Yes | User-selected question count and difficulty are sent. |
| Extracted PDF text | Yes, summarized and sampled | `ContextRetrievalService.retrieve_exam_context()` builds broad context from section summaries, key concepts, page ranges, and representative text excerpts. |
| Section/session content | Yes | The context tries to include every study section when the 12,000-character budget allows. |
| Metadata | Yes | Question count, difficulty, language, random variation seed. |
| Conversation history | No | Uses one system message and one user prompt. |
| Full PDF text | No | The final exam context is a coverage-oriented section summary plus representative excerpts, not a blind full-PDF concatenation. |
| Private data | Possibly | Extracted PDF text and exam generation options leave the local app. |

System instruction:

```text
Return valid JSON only. Do not wrap output in Markdown.
```

Prompt:

```text
Create a final exam from this study material. Return JSON only with keys title, questions, answer_key.
Each question needs id, type, question, options, answer, topic.

Write the full exam in English: questions, options, answers, topics, and explanations.
Use only the provided PDF context. Do not create questions from outside knowledge.
Use only these question types: multiple_choice, true_false, short_answer.
Create a different version each time while staying based only on the material.
Variation seed: {seed}
Question count: {question_count}
Difficulty: {difficulty}
Provided PDF context:
{context[:12000]}
```

Expected JSON:

```json
{
  "title": "AI Final Exam",
  "questions": [
    {
      "id": 1,
      "type": "multiple_choice",
      "question": "Which concept is most important for ...?",
      "options": ["A", "B", "C", "D"],
      "answer": "A",
      "topic": "Topic name"
    }
  ],
  "answer_key": [{"id": 1, "answer": "A"}]
}
```

Validation and fallback:

- `parse_json()` removes Markdown fences and requires a JSON object.
- `normalize_payload()` requires a list of question objects and keeps only supported types: `multiple_choice`, `true_false`, and `short_answer`.
- If no valid questions remain, the service raises an error and returns a local fallback exam.
- The fallback exam contains review questions, simple true/false and multiple-choice options, answer keys, and `fallback_used=True`.

### Final Exam Grading

| Item | Implementation |
| --- | --- |
| Feature | Grade submitted final exam answers and identify weak topics/sections. |
| Main file | `services/exam_grading_service.py` |
| Main functions | `grade_exam()`, `_is_correct()` |
| AI sub-call | `AIAnswerGradingService.grade_short_answer()` for short-answer questions only |
| UI caller | `pages/final_exam_page.py` |

Multiple-choice and true/false answers are graded locally. Short-answer grading first checks simple normalized string containment. If that local check fails, it calls `AIAnswerGradingService.grade_short_answer()`.

Important limitation: for final exam short-answer grading, the current code does not pass PDF context into `AIAnswerGradingService.grade_short_answer()`. The AI receives the question, expected answer, user answer, and an empty `Study context:` field. Passing the related section text would improve grounding.

### Research RAG and Experimentation Prompts

These paths are in `research/` and are not the primary Streamlit user workflow.

| File | Function / class | Data sent |
| --- | --- | --- |
| `research/generation/prompt_builder.py` | `PromptBuilder.build()` | Builds a grounded answer prompt from a question and retrieved source texts. |
| `research/generation/openai_llm.py` | `OpenAILLM.generate()` | Sends system instruction "Answer only from provided context..." and the prompt to OpenAI. |
| `research/generation/answer_generator.py` | `AnswerGenerator.generate()` | Uses OpenAI only when `llm_provider="openai"` and an API key exists; otherwise uses deterministic local answers. |
| `research/services/llm_service.py` | `LLMService.generate()` | Intended to send a single user prompt to OpenAI when `LLM_PROVIDER` is configured as `openai`; however, the referenced `LLM_*` config constants are not currently defined in `core/config.py`, so this path needs configuration cleanup before use. |
| `research/services/embedding_service.py` | `_openai_embed()` | Sends chunk text or query text to OpenAI embeddings when `EMBEDDING_PROVIDER="openai"`. |
| `research/rag/langchain_pipeline.py` | `answer_question()` | The current implementation retrieves chunks and composes an answer locally; it does not call an LLM in this method. |

Research prompt example:

```text
You are a grounded study assistant. Answer only from the retrieved sources.
If the sources are weak or insufficient, say so. Include citation markers like [1] beside claims.

Question: {question}

Retrieved sources:
Source 1: {citation}
Text: {retrieved_text}

Grounded answer:
```

## End-to-End AI Flow

1. User uploads a PDF in the Streamlit UI.
2. `ui/workflow.py:extract_pdf()` writes the uploaded bytes to a temporary `.pdf` file.
3. `PdfService.extract_pages()` extracts per-page text with PyMuPDF and falls back to `pypdf` if needed.
4. The temporary file is deleted after extraction.
5. Extracted pages are stored in `st.session_state.pending_pages`.
6. `StudyService.suggest_session_count()` estimates a session count from readable text size.
7. `StudyService.generate_study_plan_for_sessions()` constructs a study-plan prompt from truncated page text.
8. `GeneralAIService.complete()` sends the study-plan system prompt and user prompt to OpenAI or Groq.
9. The response is parsed and validated as JSON. If invalid, a local heuristic study plan is generated.
10. When the user confirms the study plan, pages, sections, progress, and interaction state are moved into active session state and persisted locally.
11. In Study Mode, the user can request an explanation, quiz, or section Q&A.
12. Each feature builds a feature-specific prompt using the current section text and metadata.
13. `GeneralAIService.ask()` or `ExamService._call_ai()` sends the prompt to the selected provider.
14. The app parses JSON outputs for study plans, quizzes, and final exams, or displays text outputs for explanations and tutor answers.
15. Fallback logic generates local content when the API key is missing, the API fails, or the model output is malformed.
16. UI state, generated content, scores, answers, and progress are saved by `PersistenceService` to `.smartstudy_progress.json`.

## Privacy and Data Handling

### What Leaves the Local App

When AI features are used with `OPENAI_API_KEY` or `GROQ_API_KEY`, the following information may be sent to the external provider:

- Extracted PDF text, either as page excerpts, section text, or combined study context.
- PDF filename in quiz context labels.
- Section titles, page labels, key concepts, selected difficulty, selected question count, and language.
- User questions in the AI tutor and section Q&A.
- Chat history for the general AI tutor, limited to the last 12 supplied messages by `GeneralAIService.ask()`. Grounded PDF mode does not send chat history.
- Student short-answer responses when AI grading is used.
- Research mode may send chunk text or query text to OpenAI embeddings if configured.

### Local Storage

| Data | Stored locally? | Current behavior |
| --- | --- | --- |
| Uploaded PDF bytes | Runtime only | Stored in Streamlit session state as `pdf_bytes` or `pending_pdf_bytes`. Not written by `PersistenceService`. |
| Temporary uploaded PDF file | Temporary only | Written to a temp path for extraction and deleted in `extract_pdf()`. |
| Extracted PDF text | Yes | Persisted in `.smartstudy_progress.json` through `PersistenceService.build_payload()`. |
| Study sections | Yes | Persisted in `.smartstudy_progress.json`. |
| Section explanations, questions, answers, quizzes, quiz scores, quiz feedback | Yes | Persisted as `section_states`. |
| Final exam, submitted final exam answers, final exam result | Yes | Persisted in `.smartstudy_progress.json`. |
| General AI tutor history | Not currently persisted | Stored in Streamlit session state during runtime, but not included in `PersistenceService.build_payload()`. |
| API keys | Environment only | Loaded from environment variables or `ui/.streamlit/_env`; not written by persistence code. |

### Risks and Limitations

- Uploaded course material may contain private, copyrighted, or personally identifiable information. Extracted text can be sent to external providers when AI features are used.
- The app stores extracted PDF text and student answers in a local JSON file without encryption.
- The app does not currently redact names, IDs, emails, or other sensitive fields before prompting.
- The app does not currently display a per-call consent notice describing exactly which text will be sent.
- Token counting is based on character truncation, not model token estimation.
- Grounded PDF Q&A now uses local retrieval and strict "PDF context only" prompting, but provider output is still not independently fact-checked after generation.
- Prompt and response logging for debugging is not currently implemented.

### Recommended Privacy Improvements

- Add a privacy notice before the first AI call.
- Add an option to disable external AI calls and run only local fallbacks.
- Redact common sensitive patterns before prompts are sent.
- Encrypt or disable `.smartstudy_progress.json` for sensitive PDFs.
- Add configurable retention and a "Delete local progress" button.
- Log only hashes or summaries by default, with a separate explicit debug mode for full prompts.

## Prompt Examples

The following examples are representative and based on the current prompt templates. PDF text is shortened for readability.

### Study Plan Example

Example input:

```text
Language: English
Requested sessions: 3
Readable page numbers: 1, 2, 3
Page 1 text: "Introduction to graph algorithms..."
Page 2 text: "Breadth-first search explores neighbors..."
Page 3 text: "Dijkstra's algorithm computes shortest paths..."
```

Constructed prompt excerpt:

```text
Create a study plan by grouping the PDF pages into coherent study sessions.
Generate a structured study plan in English.
Return exactly 3 sections.
Write titles, summaries, key concepts, and learning objectives in English.
Use only the provided page text.
Prefer topic boundaries over equal page counts, but keep page ranges ordered and non-overlapping.
Return a JSON object with one top-level key named sections.

Readable page numbers: 1, 2, 3

PDF page text:
Page 1
Introduction to graph algorithms...
```

Expected response:

```json
{
  "sections": [
    {
      "section_number": 1,
      "title": "Graph Algorithm Foundations",
      "start_page": 1,
      "end_page": 1,
      "estimated_minutes": 20,
      "difficulty": "Medium",
      "summary": "This section introduces graph algorithm terminology and goals.",
      "key_concepts": ["Graphs", "Vertices", "Edges"],
      "learning_objectives": ["Explain graph terminology.", "Identify common graph problems."]
    }
  ]
}
```

### Section Explanation Example

Example input:

```text
Section title: Section 2: Breadth-First Search
Pages: Pages 2-3
Key concepts: BFS, Queue, Shortest Path
Section text: "Breadth-first search explores a graph level by level..."
```

Constructed prompt excerpt:

```text
Explain this study section for a student preparing for an exam.
Answer in English.
Use the provided section text only.
Structure the answer with these headings:
Summary, Key Ideas, Important Definitions, Exam Tips.
Keep it clear and practical.

Section title: Section 2: Breadth-First Search
Pages: Pages 2-3
Key concepts: BFS, Queue, Shortest Path

Section text:
Breadth-first search explores a graph level by level...
```

Expected response:

```text
Summary
Breadth-first search visits nodes in layers and is useful for unweighted shortest paths.

Key Ideas
- BFS uses a queue.
- Nodes are explored by distance from the start node.

Important Definitions
- Queue: ...

Exam Tips
- Be able to trace BFS order on a small graph.
```

### Section Question Example

Example input:

```text
Student question: "Why does BFS use a queue?"
Retrieved chunk:
[Section 2 | Breadth-First Search | Page 3]
Breadth-first search uses a queue to process nodes in first-in, first-out order...
```

Constructed message excerpt:

```text
Answer ONLY using the provided PDF context.
Do not use outside knowledge.
Do not guess.

Provided PDF context:
[Section 2 | Breadth-First Search | Page 3]
Breadth-first search uses a queue to process nodes in first-in, first-out order...

Question:
Why does BFS use a queue?
```

Expected response:

```text
BFS uses a queue because first-in, first-out processing keeps nodes in level order.
That is what allows BFS to explore all nodes at distance 1 before nodes at distance 2.

Source: Pages 2-3

Retrieved sources:
- Section 2, Page 3: Breadth-First Search
```

### General AI Tutor Example

Example input:

```text
User question: "Help me build a study plan."
Use uploaded PDF context: true
Locally retrieved chunks:
[Section 1 | Core Concepts | Page 1]
[Section 2 | Examples and Practice | Page 4]
```

Constructed prompt excerpt:

```text
System:
You are a grounded PDF study assistant.

User:
Provided PDF context:
[Section 1 | Core Concepts | Page 1]
...
[Section 2 | Examples and Practice | Page 4]
...

Question:
Help me build a study plan.
```

Expected response:

```text
Start by reviewing the main sections, then divide the material into short daily study blocks.
For each block, write a short summary, review key terms, and answer practice questions.
```

### Quiz Generation Example

Example input:

```text
Number of questions: 3
Variation seed: 482931
Study material:
[lecture.pdf p.2]
Breadth-first search explores a graph level by level using a queue...
```

Constructed prompt excerpt:

```text
Create multiple-choice quiz questions from the study material below.
Generate quiz questions in English.
Use only the provided material.
Create different questions each time, using the variation seed.
Return only valid JSON, without markdown.
The JSON must be a list of objects.
Each object must contain: prompt, options, answer, explanation, citation.
Each question must have exactly 4 options.
The answer must exactly match one of the options.
Avoid simple fill-in-the-blank questions.
Prefer conceptual understanding questions.
```

Expected response:

```json
[
  {
    "prompt": "Why is a queue important in breadth-first search?",
    "options": [
      "It preserves level-order traversal",
      "It sorts nodes by weight",
      "It stores only visited nodes",
      "It prevents all repeated edges"
    ],
    "answer": "It preserves level-order traversal",
    "explanation": "The queue processes nodes in first-in, first-out order, keeping BFS level-based.",
    "citation": "lecture.pdf p.2"
  }
]
```

### Short-Answer Grading Example

Example input:

```text
Study context: "BFS uses a queue to explore nodes level by level."
Question: "Why does BFS use a queue?"
Expected answer: "A queue preserves first-in, first-out level-order traversal."
User's answer: "It keeps the nodes in the order they were discovered."
```

Constructed prompt:

```text
Evaluate the answer strictly based on the provided context and expected answer.
Use semantic meaning, not exact wording.
Provide a score from 0 to 100 and short feedback.
Format: Score: [0-100] | Feedback: [Explanation]
```

Expected response:

```text
Score: 85 | Feedback: The answer captures the main idea that discovery order matters, but it should explicitly mention FIFO or level-order traversal.
```

### Final Exam Example

Example input:

```text
Question count: 10
Difficulty: mixed
Language: English
Provided PDF context: section summaries, key concepts, page ranges, and representative excerpts from all sections when possible
```

Constructed prompt excerpt:

```text
Create a final exam from this study material. Return JSON only with keys title, questions, answer_key.
Each question needs id, type, question, options, answer, topic.

Write the full exam in English: questions, options, answers, topics, and explanations.
Use only the provided PDF context. Do not create questions from outside knowledge.
Use only these question types: multiple_choice, true_false, short_answer.
Create a different version each time while staying based only on the material.
Variation seed: 812322
Question count: 10
Difficulty: mixed
Provided PDF context:
Section 1: Complexity
...
```

Expected response:

```json
{
  "title": "Algorithms Final Exam",
  "questions": [
    {
      "id": 1,
      "type": "short_answer",
      "question": "Explain why BFS can find shortest paths in an unweighted graph.",
      "options": [],
      "answer": "BFS explores nodes by distance from the start, so the first time it reaches a node is via the shortest number of edges.",
      "topic": "Breadth-First Search"
    }
  ],
  "answer_key": [
    {
      "id": 1,
      "answer": "BFS explores nodes by distance from the start, so the first time it reaches a node is via the shortest number of edges."
    }
  ]
}
```

## Error Handling and Fallbacks

| Failure case | Current behavior |
| --- | --- |
| No API key | `GeneralAIService` returns `ok=False` with a setup message. Study plans, quizzes, explanations, and final exams use local fallback logic where implemented. |
| OpenAI/Groq request exception | Error is caught and returned as `ok=False`; caller decides fallback. Final exam generation catches exceptions and produces a fallback exam. |
| Empty prompt/question | `GeneralAIService` returns the translated "Enter a question first" message. |
| PDF extraction fails | `PdfService` tries PyMuPDF first, then pypdf. If both produce no readable text, `PdfExtractionError` is raised. |
| PDF has too little or no readable text | Study plan creation returns no sections, and the UI raises a "No readable study sessions" error. |
| Invalid study-plan JSON | `_parse_ai_json()` attempts cleanup; if parsing or validation fails, heuristic study sections are generated locally. |
| Invalid quiz JSON | Quiz AI generation returns no AI questions and deterministic local keyword questions are used. |
| Invalid final exam JSON | `parse_json()` or `normalize_payload()` raises, and `_fallback_exam()` is returned. |
| AI answer is not grounded enough | Grounded PDF Q&A retrieves local chunks first and prompts the model to answer only from those chunks. If no chunks are found, it returns the not-enough-information message before calling AI. Provider output is not yet automatically verified sentence-by-sentence. |
| Short-answer grading response lacks `Score:` | Regex parsing returns score `0`, while raw feedback is preserved. |

## Evaluation of Prompting Quality

The following evaluation plan can be used for academic testing and debugging:

1. Prepare at least 20 test sessions with different PDFs, including short lecture notes, long course chapters, Hebrew material, English material, and PDFs with weak extraction quality.
2. For each PDF, record the generated study plan and verify that sections are ordered, non-overlapping, complete, and aligned with real topic boundaries.
3. Compare AI tutor answers against PDF ground truth. Mark each answer as supported, partially supported, unsupported, or hallucinated.
4. Check whether section Q&A responses match the current section and whether they should have answered "not found."
5. Validate every study-plan, quiz, and final-exam AI response as JSON before using it in the UI.
6. Check quiz relevance by verifying that each question can be answered from the cited section or page.
7. Check final exam coverage by mapping each generated question to a section, page, or key concept.
8. Track hallucination rate: count claims that do not appear in the PDF context sent to the model.
9. Log prompt input, model output, parsing result, fallback usage, provider, model, language, and final UI result in a debug mode.
10. Compare fallback content against AI-generated content to ensure the app remains usable without API keys.

Suggested metrics:

| Metric | How to measure |
| --- | --- |
| JSON validity | Percentage of model responses parsed successfully without cleanup. |
| Grounding accuracy | Percentage of answers fully supported by sent PDF context. |
| Citation/page match | Percentage of quiz or answer citations that point to a relevant section/page. |
| Relevance | Human rating from 1 to 5 for study-plan sections, quiz questions, and exam questions. |
| Fallback rate | Percentage of AI calls that use deterministic fallback. |
| Language compliance | Percentage of Hebrew-mode outputs fully written in Hebrew and English-mode outputs fully written in English. |

## Suggested Improvements

| Improvement | Current status | Suggested implementation |
| --- | --- | --- |
| Central prompt manager | Not currently implemented | Create `services/prompt_manager.py` or `services/prompts.py` with named prompt builders for study plan, tutor, quiz, grading, and exam prompts. |
| Prompt version numbers | Not currently implemented | Add constants such as `STUDY_PLAN_PROMPT_VERSION = "2026-06-19.v1"` and persist version in generated artifacts. |
| Prompt/debug logging | Not currently implemented | Add optional debug logging controlled by an environment variable. Log provider, prompt version, token estimate, response, parser result, and fallback reason. |
| Token length limits | Partially implemented | Current code uses character truncation. Add tokenizer-aware limits per model. |
| Chunk selection before prompting | Implemented for PDF Q&A; partially implemented for final exams | Q&A sends top relevant chunks. Final exams use section summaries and representative excerpts. A future version could use semantic retrieval or per-topic sampling. |
| Stricter JSON schema validation | Partially implemented | Replace manual checks with JSON Schema or Pydantic models for study plans, quizzes, and exams. |
| Citations to PDF section/page | Partially implemented | Quiz prompts request citations; section Q&A appends source labels. Add page-specific citations to all generated answers and exam questions. |
| Tests for prompt construction | Partially implemented | Current tests cover retrieval ranking, prompt source formatting, exam context coverage, section explanation scope, grounded Q&A chunk selection, and unsupported-question fallback. Add more tests for token limits and multilingual prompts. |
| Stronger grounding for section Q&A | Implemented | The app now searches the whole PDF locally, sends only retrieved chunks, and requires the model to answer only from provided PDF context. |
| Privacy redaction | Not currently implemented | Add preprocessing that redacts emails, phone numbers, IDs, and names when configured. |
| Encrypted local persistence | Not currently implemented | Encrypt `.smartstudy_progress.json` or store only derived progress unless the user opts into saving extracted text. |

## Summary

Documenting prompts and sent data is essential for reliability, privacy, debugging, and academic evaluation. It makes clear which parts of the system depend on external AI providers, what PDF and user data leaves the local application, how responses are constrained and validated, and how the app behaves when AI output fails. This transparency helps evaluate whether the system is genuinely grounded in uploaded learning material, protects student data responsibly, and supports reproducible testing for the final project submission.
