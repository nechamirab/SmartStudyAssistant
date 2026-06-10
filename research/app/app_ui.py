import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from services.pdf_service import PdfService
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.retrieval_service import RetrievalService
from services.qa_service import QAService
from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from services.quiz_service import QuizService

st.title("Smart Study Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process PDF"):
        st.write("Processing...")

        pdf_service = PdfService()
        pages = pdf_service.extract_pages("temp.pdf")

        chunk_service = ChunkService(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = chunk_service.chunk_pages(pages)

        embedding_service = EmbeddingService()
        embeddings = embedding_service.embed_texts(chunks)

        vector_store = VectorStoreService()
        vector_store.add(chunks, embeddings)

        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        st.session_state.retrieval_service = retrieval_service
        st.session_state.qa_service = QAService(retrieval_service)

        st.success("PDF processed!")

# questions
if "qa_service" in st.session_state:
    with st.form("question_form"):
        query = st.text_input("Ask a question")
        submitted = st.form_submit_button("Ask")

    if submitted:
        with st.spinner("Thinking..."):
            qa_response = st.session_state.qa_service.answer(query)

        st.subheader("Answer")
        st.write(qa_response.answer)

        st.subheader("Sources")

        unique_pages = sorted(set(qa_response.sources))

        if unique_pages:
            for page in unique_pages:
                st.markdown(f"Page {page}")
        else:
            st.write("No sources found.")

if "retrieval_service" in st.session_state:
    st.subheader("Generate Quiz")

    quiz_topic = st.text_input("Quiz topic", value="Sequential games")
    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("Generate Quiz"):
        quiz_service = QuizService(st.session_state.retrieval_service)

        with st.spinner("Generating quiz..."):
            st.session_state.quiz_questions = quiz_service.generate_quiz(
                topic=quiz_topic,
                num_questions=num_questions,
            )
            st.session_state.quiz_index = 0
            st.session_state.show_answer = False

    if "quiz_questions" in st.session_state:
        quiz_questions = st.session_state.quiz_questions
        quiz_index = st.session_state.quiz_index
        current_question = quiz_questions[quiz_index]

        st.divider()
        st.subheader(f"Question {quiz_index + 1} of {len(quiz_questions)}")
        st.markdown(f"**{current_question.question}**")

        if st.button("Show answer"):
            st.session_state.show_answer = True

        if st.session_state.show_answer:
            st.markdown(f"**Answer:** {current_question.answer}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Previous", disabled=quiz_index == 0):
                st.session_state.quiz_index -= 1
                st.session_state.show_answer = False
                st.rerun()

        with col2:
            if st.button("Next", disabled=quiz_index == len(quiz_questions) - 1):
                st.session_state.quiz_index += 1
                st.session_state.show_answer = False
                st.rerun()