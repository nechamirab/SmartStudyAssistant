# Global configuration values

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_PROVIDER = "openai"   # options: "mock", "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
MOCK_EMBEDDING_DIM = 128

LLM_PROVIDER = "openai"  # options: "mock", "openai"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 500