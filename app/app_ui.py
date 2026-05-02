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