from pathlib import Path

BOOKS_DIR = Path("./books")
ARTIFACTS_DIR = Path("./artifacts")

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250

TOP_K_SEARCH = 5
TOP_K_QA = 5

MIN_RELEVANCE_SCORE = 0.28
MIN_QA_SCORE = 0.22

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OLLAMA_MODEL_NAME = "gemma3"

AI_PROMPT_BASE = """
Ответь на вопрос строго по цитатам.
Если ответа нет — скажи, что информации недостаточно.

ЦИТАТЫ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
"""
