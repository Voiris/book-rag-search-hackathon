from typing import List, Dict, Any

from config import (
    TOP_K_SEARCH,
    TOP_K_QA,
    MIN_RELEVANCE_SCORE,
    MIN_QA_SCORE, AI_PROMPT_BASE,
)

from data import load_books, build_chunks
from retrieval import (
    load_model,
    embed_texts,
    build_index,
    build_tfidf,
    hybrid_search,
)

from qa.answer import generate_grounded_answer
from qa.ollama_client import generate_answer


def _build_prompt(question: str, context: str) -> str:
    return AI_PROMPT_BASE.format(
        question=question,
        context=context
    )


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []

    for c in chunks[:5]:
        text = c["text"][:500]
        parts.append(
            f"{text}\n(Источник: {c['title']}, фрагмент {c['chunk_index']})"
        )

    return "\n\n".join(parts)


class RAGService:
    def __init__(self, books_dir, embedding_model_name):
        self.books_dir = books_dir
        self.embedding_model_name = embedding_model_name

        self.books: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []

        self.model = None
        self.embeddings = None
        self.index = None

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    # --- init pipeline ---

    def initialize(self):
        print("Загрузка книг...")
        self.books = load_books(self.books_dir)

        if not self.books:
            print("⚠️ Книги не найдены. Добавьте .txt файлы в папку " + self.books_dir)
            self.chunks = []
            self.embeddings = None
            self.index = None
            return

        print("Чанкинг...")
        self.chunks = build_chunks(self.books)

        print("Загрузка embedding модели...")
        self.model = load_model(self.embedding_model_name)

        print("Создание эмбеддингов...")
        texts = [c["text"] for c in self.chunks]
        self.embeddings = embed_texts(texts, self.model)

        print("FAISS индекс...")
        self.index = build_index(self.embeddings)

        print("TF-IDF...")
        self.tfidf_vectorizer, self.tfidf_matrix = build_tfidf(self.chunks)

        print(f"Готово: {len(self.books)} книг, {len(self.chunks)} чанков")

    # --- API ---

    def get_books(self):
        return self.books

    # --- search ---

    def search(self, query: str, top_k: int = TOP_K_SEARCH) -> Dict[str, Any]:
        results = hybrid_search(
            query=query,
            chunks=self.chunks,
            embedding_model=self.model,
            faiss_index=self.index,
            tfidf_vectorizer=self.tfidf_vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            top_k=top_k
        )

        if not results:
            return {
                "found": False,
                "message": "Ничего не найдено.",
                "results": []
            }

        if results[0]["hybrid_score"] < MIN_RELEVANCE_SCORE:
            return {
                "found": False,
                "message": "Недостаточно релевантных результатов.",
                "results": []
            }

        return {
            "found": True,
            "message": f"Найдено {len(results)} фрагментов",
            "results": results
        }

    # --- answer ---

    def answer(self, question: str, top_k: int = TOP_K_QA) -> Dict[str, Any]:
        retrieved = hybrid_search(
            query=question,
            chunks=self.chunks,
            embedding_model=self.model,
            faiss_index=self.index,
            tfidf_vectorizer=self.tfidf_vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            top_k=top_k
        )

        if not retrieved:
            return {
                "found": False,
                "answer": "Не удалось найти информацию.",
                "quotes": []
            }

        grounded = generate_grounded_answer(
            question,
            retrieved,
            min_score=MIN_QA_SCORE
        )

        if grounded["found"]:
            try:
                context = _build_context(retrieved)

                llm_answer = generate_answer(
                    _build_prompt(question, context)
                )

                return {
                    "found": True,
                    "answer": llm_answer,
                    "quotes": grounded["quotes"],
                    "retrieved": retrieved
                }

            except Exception:
                return {
                    "found": True,
                    "answer": grounded["answer"],
                    "quotes": grounded["quotes"],
                    "retrieved": retrieved
                }

        return {
            "found": False,
            "answer": grounded["answer"],
            "quotes": grounded["quotes"],
            "retrieved": retrieved
        }
