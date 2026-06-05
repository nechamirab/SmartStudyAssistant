# Global configuration values

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Supported providers: mock, sentence-transformers, openai
EMBEDDING_PROVIDER = "mock"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MOCK_EMBEDDING_DIM = 128

# Optional hosted AI providers for the general tutor and final exam.
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama-3.1-8b-instant"
