import os
import re
import json
import math
import pickle
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from ollama import chat, ChatResponse

BOOKS_DIR = Path("./books")
ARTIFACTS_DIR = Path("./artifacts")

BOOKS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Параметры чанкинга
CHUNK_SIZE = 1400          # размер чанка в символах
CHUNK_OVERLAP = 250        # overlap в символах

# Retrieval
TOP_K_SEARCH = 5
TOP_K_QA = 5
MIN_RELEVANCE_SCORE = 0.28   # нижний порог уверенности retrieval
MIN_QA_SCORE = 0.22          # нижний порог для ответа на вопрос

# Модель эмбеддингов
# Работает и для русского, и для английского
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Пути для сохранения
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf.pkl"

print("Папка книг:", BOOKS_DIR.resolve())
print("Папка артефактов:", ARTIFACTS_DIR.resolve())

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_for_lexical_search(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9\s.,!?;:\-\"'()]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    # Простое, но рабочее разбиение на предложения
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def safe_read_text(file_path: Path) -> str:
    encodings = ["utf-8", "cp1251", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except Exception:
            continue
    # Последняя попытка
    with open(file_path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")


def truncate_text(text: str, max_len: int = 900) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def load_books_from_folder(folder: Path) -> List[Dict[str, Any]]:
    books = []

    for file_path in sorted(folder.glob("*.txt")):
        raw_text = safe_read_text(file_path)
        text = normalize_text(raw_text)

        books.append({
            "book_id": file_path.stem,
            "title": file_path.stem,
            "path": str(file_path),
            "text": text
        })

    return books


books = load_books_from_folder(BOOKS_DIR)

print(f"Загружено книг: {len(books)}")
for b in books:
    print(f"- {b['title']}: {len(b['text'])} символов")

if len(books) == 0:
    print("\nВНИМАНИЕ: положи .txt книги в папку ./books и заново запусти ячейки.")

SPLIT_RULES = (
    (0, lambda c, n: c == "\n" and n == "\n", 1),   # двойной перенос → приоритет 0
    (1, lambda c, n: c == "\n", 0),                 # один перенос
    (2, lambda c, n: c in ".!?", 0),               # конец предложения
    (3, lambda c, n: c == " ", 0),                 # пробел
)

class TextState:
    def __init__(self):
        self.quote_depth = 0  # считаем вложенность кавычек

    @property
    def is_can_split(self):
        return self.quote_depth == 0

    def update_state(self, char):
        if char in ('"', '«'):
            self.quote_depth += 1
        elif char in ('"', '»'):
            self.quote_depth = max(0, self.quote_depth - 1)

def smart_text_split(text: str, max_len: int) -> List[str]:
    if len(text) <= max_len:
        return [text]

    text_state = TextState()
    split_index = max_len - 1
    used_priority = None

    n = len(text)
    limit = min(n - 1, max_len - 1)

    for i in range(limit):
        char = text[i]
        next_char = text[i + 1] if i + 1 < n else None

        text_state.update_state(char)

        if text_state.is_can_split:
            for priority, rule, offset in SPLIT_RULES:
                if rule(char, next_char) and (used_priority is None or priority < used_priority):
                    used_priority = priority
                    split_index = i + offset

    return [
        text[:split_index+1].rstrip(),
        text[split_index+1:].rstrip()
    ]

def split_text_into_chunks(text: str, chunk_size: int = 1400, overlap: int = 250) -> List[Tuple[int, int, str]]:
    """
    Возвращает список чанков: (start_char, end_char, chunk_text)
    """
    text = normalize_text(text)
    chunks = []

    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start:end]

        parts = smart_text_split(chunk_text, chunk_size)
        chunk_text = parts[0]

        actual_end = start + len(chunk_text)
        if chunk_text:
            chunks.append((start, actual_end, chunk_text.strip()))

        if actual_end >= n:
            break

        start = max(actual_end - overlap, start + 1)

    return chunks


def build_chunks(books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks = []

    for book in books:
        text_chunks = split_text_into_chunks(book["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, (start_char, end_char, chunk_text) in enumerate(text_chunks):
            chapter_match = re.search(r"(глава\s+[^\n]+|chapter\s+[^\n]+)", chunk_text[:300], flags=re.IGNORECASE)
            chapter = chapter_match.group(1).strip() if chapter_match else None

            all_chunks.append({
                "chunk_id": f"{book['book_id']}_chunk_{idx}",
                "book_id": book["book_id"],
                "title": book["title"],
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
                "chapter": chapter,
                "text": chunk_text,
                "search_text": clean_for_lexical_search(chunk_text)
            })

    return all_chunks


chunks = build_chunks(books)

print("Всего чанков:", len(chunks))
if chunks:
    print("\nПример чанка:")
    print("title:", chunks[0]["title"])
    print("chunk_index:", chunks[0]["chunk_index"])
    print(chunks[0]["text"][:700])

def build_tfidf(chunks: List[Dict[str, Any]]) -> Tuple[TfidfVectorizer, Any]:
    corpus = [c["search_text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


tfidf_vectorizer, tfidf_matrix = build_tfidf(chunks)
print("TF-IDF матрица:", tfidf_matrix.shape)


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded:", EMBEDDING_MODEL_NAME)

def embed_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


chunk_texts_for_embedding = [c["text"] for c in chunks]
chunk_embeddings = embed_texts(chunk_texts_for_embedding, embedding_model)
print("Embeddings shape:", chunk_embeddings.shape)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


faiss_index = build_faiss_index(chunk_embeddings)
print("FAISS index size:", faiss_index.ntotal)


with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

with open(TFIDF_PATH, "wb") as f:
    pickle.dump({
        "vectorizer": tfidf_vectorizer,
        "matrix": tfidf_matrix
    }, f)

np.save(EMBEDDINGS_PATH, chunk_embeddings)
faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

print("Артефакты сохранены:")
print("-", CHUNKS_PATH)
print("-", TFIDF_PATH)
print("-", EMBEDDINGS_PATH)
print("-", FAISS_INDEX_PATH)

def load_artifacts():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_loaded = json.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    embeddings_loaded = np.load(EMBEDDINGS_PATH)
    index_loaded = faiss.read_index(str(FAISS_INDEX_PATH))

    return (
        chunks_loaded,
        tfidf_obj["vectorizer"],
        tfidf_obj["matrix"],
        embeddings_loaded,
        index_loaded
    )

def vector_search(query: str,
                  embedding_model: SentenceTransformer,
                  faiss_index: faiss.IndexFlatIP,
                  chunks: List[Dict[str, Any]],
                  top_k: int = 5) -> List[Dict[str, Any]]:
    query_emb = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(query_emb, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        item = dict(chunks[idx])
        item["vector_score"] = float(score)
        results.append(item)

    return results

def lexical_search(query: str,
                   tfidf_vectorizer: TfidfVectorizer,
                   tfidf_matrix,
                   chunks: List[Dict[str, Any]],
                   top_k: int = 5) -> List[Dict[str, Any]]:
    query_clean = clean_for_lexical_search(query)
    query_vec = tfidf_vectorizer.transform([query_clean])

    # cosine для l2-нормированных sparse можно считать через dot, но оставим проще
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        item = dict(chunks[idx])
        item["lexical_score"] = score
        results.append(item)

    return results

def hybrid_search(query: str,
                  chunks: List[Dict[str, Any]],
                  embedding_model: SentenceTransformer,
                  faiss_index: faiss.IndexFlatIP,
                  tfidf_vectorizer: TfidfVectorizer,
                  tfidf_matrix,
                  top_k: int = 5,
                  candidate_multiplier: int = 4) -> List[Dict[str, Any]]:
    """
    Гибрид:
    - забираем кандидатов из vector search
    - забираем кандидатов из lexical search
    - объединяем
    - считаем hybrid_score
    """
    vector_candidates = vector_search(
        query, embedding_model, faiss_index, chunks, top_k=top_k * candidate_multiplier
    )
    lexical_candidates = lexical_search(
        query, tfidf_vectorizer, tfidf_matrix, chunks, top_k=top_k * candidate_multiplier
    )

    merged = {}

    for item in vector_candidates:
        cid = item["chunk_id"]
        if cid not in merged:
            merged[cid] = dict(item)
        else:
            merged[cid].update(item)

    for item in lexical_candidates:
        cid = item["chunk_id"]
        if cid not in merged:
            merged[cid] = dict(item)
        else:
            merged[cid].update(item)

    # Нормализация внутри набора кандидатов
    v_scores = [merged[c].get("vector_score", 0.0) for c in merged]
    l_scores = [merged[c].get("lexical_score", 0.0) for c in merged]

    v_max = max(v_scores) if v_scores else 1.0
    l_max = max(l_scores) if l_scores else 1.0

    for cid, item in merged.items():
        v = item.get("vector_score", 0.0) / (v_max + 1e-9)
        l = item.get("lexical_score", 0.0) / (l_max + 1e-9)

        # Чуть сильнее вес semantic retrieval
        hybrid_score = 0.65 * v + 0.35 * l
        item["hybrid_score"] = float(hybrid_score)

    results = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
    return results


def search_fragments(query: str, top_k: int = TOP_K_SEARCH) -> Dict[str, Any]:
    results = hybrid_search(
        query=query,
        chunks=chunks,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=top_k
    )

    if not results:
        return {
            "found": False,
            "message": "Ничего не найдено в загруженных книгах.",
            "results": []
        }

    # если лучший результат слишком слабый — честный отказ
    if results[0]["hybrid_score"] < MIN_RELEVANCE_SCORE:
        return {
            "found": False,
            "message": "Не удалось найти достаточно релевантный фрагмент в загруженных книгах.",
            "results": []
        }

    return {
        "found": True,
        "message": f"Найдено {len(results)} релевантных фрагментов.",
        "results": results
    }

STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "он", "она", "оно", "они", "на", "я", "мы", "ты", "вы",
    "по", "с", "со", "к", "ко", "от", "до", "за", "из", "у", "о", "об", "про", "а", "но",
    "как", "для", "это", "то", "же", "ли", "или", "бы", "был", "была", "были", "быть",
    "его", "ее", "их", "ему", "ей", "им", "этот", "эта", "эти", "тот", "та", "те"
}

def tokenize_simple(text: str) -> List[str]:
    text = text.lower().replace("ё", "е")
    tokens = re.findall(r"[a-zа-я0-9]+", text, flags=re.IGNORECASE)
    return tokens


def keyword_set(text: str) -> set:
    return {t for t in tokenize_simple(text) if t not in STOPWORDS_RU and len(t) >= 3}


def sentence_relevance(sentence: str, question_keywords: set) -> float:
    sent_tokens = keyword_set(sentence)
    if not sent_tokens:
        return 0.0
    overlap = len(sent_tokens & question_keywords)
    return overlap / (len(question_keywords) + 1e-9)


def select_support_sentences(question: str, retrieved_chunks: List[Dict[str, Any]], max_sentences: int = 5) -> List[Dict[str, Any]]:
    q_keywords = keyword_set(question)
    candidates = []

    for chunk in retrieved_chunks:
        for sent in split_into_sentences(chunk["text"]):
            score = sentence_relevance(sent, q_keywords)
            if score > 0:
                candidates.append({
                    "sentence": sent,
                    "score": score,
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"]
                })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Убираем почти дубли
    selected = []
    seen = set()

    for c in candidates:
        key = c["sentence"][:160].lower()
        if key in seen:
            continue
        selected.append(c)
        seen.add(key)
        if len(selected) >= max_sentences:
            break

    return selected


def generate_grounded_answer(question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Формирует ответ только по retrieved chunks.
    Если уверенность низкая — честно отказывает.
    """
    if not retrieved_chunks:
        return {
            "found": False,
            "answer": "Не удалось найти ответ в загруженных книгах.",
            "quotes": []
        }

    best_score = retrieved_chunks[0].get("hybrid_score", 0.0)
    if best_score < MIN_QA_SCORE:
        return {
            "found": False,
            "answer": "В загруженных книгах нет достаточно надёжного контекста для ответа на этот вопрос.",
            "quotes": []
        }

    support_sentences = select_support_sentences(question, retrieved_chunks, max_sentences=5)

    # Если нет релевантных предложений, fallback на сами чанки
    if not support_sentences:
        quotes = []
        for item in retrieved_chunks[:3]:
            quotes.append({
                "title": item["title"],
                "chunk_index": item["chunk_index"],
                "score": round(item.get("hybrid_score", 0.0), 4),
                "text": truncate_text(item["text"], 500)
            })

        answer = (
            "Удалось найти близкие фрагменты, но точный ответ автоматически выделить не получилось. "
            "Ниже приведены наиболее релевантные цитаты."
        )

        return {
            "found": True,
            "answer": answer,
            "quotes": quotes
        }

    # Генерация ответа по опорным предложениям
    top_sentences = [s["sentence"] for s in support_sentences[:3]]
    answer = "Судя по найденным фрагментам: " + " ".join(top_sentences)

    quotes = []
    used_keys = set()
    for s in support_sentences[:5]:
        key = (s["title"], s["chunk_index"], s["sentence"][:120])
        if key in used_keys:
            continue
        used_keys.add(key)
        quotes.append({
            "title": s["title"],
            "chunk_index": s["chunk_index"],
            "score": round(s["score"], 4),
            "text": s["sentence"]
        })

    return {
        "found": True,
        "answer": answer,
        "quotes": quotes
    }


def rag_search(query: str, top_k: int = TOP_K_SEARCH) -> Dict[str, Any]:
    return search_fragments(query=query, top_k=top_k)


def rag_answer_with_ollama(question: str, top_k: int = TOP_K_QA) -> Dict[str, Any]:
    # Сначала retrieval
    retrieval = hybrid_search(
        query=question,
        chunks=chunks,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=top_k
    )

    # Генерация ответа через Ollama
    answer_obj = generate_answer_with_ollama(question, retrieval)

    return {
        "question": question,
        "retrieved": retrieval,
        "found": answer_obj["found"],
        "answer": answer_obj["answer"],
        "quotes": answer_obj["quotes"]
    }

def generate_answer_with_ollama(question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not retrieved_chunks:
        return {
            "found": False,
            "answer": "Не удалось найти контекст для ответа на этот вопрос.",
            "quotes": []
        }

    # Собираем цитаты
    context_texts = []
    for item in retrieved_chunks[:5]:  # берём топ-5
        text_snip = truncate_text(item["text"], 500)
        context_texts.append(f"- {text_snip} (Источник: {item['title']}, фрагмент {item['chunk_index']})")

    context = "\n".join(context_texts)

    prompt = f"""
Ты эксперт по литературе. Тебе надо сформулировать ответ используя информацию из цитат.
Не придумывай ничего, если информации недостаточно — честно скажи.
В ответе не упоминай, что ты ИИ, не используй фразы типа "на основе предоставленных данных" и т.п. Просто дай ответ по существу вопроса.

ЦИТАТЫ:
{context}

ВОПРОС:
{question}

ПОЛНЫЙ ОТВЕТ:
"""

    # Генерируем ответ через Ollama
    response: ChatResponse = chat(model="gemma3", messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])
    answer_text = response.message.content

    return {
        "found": True,
        "answer": answer_text,
        "quotes": retrieved_chunks[:5]
    }

def print_search_results(result_obj: Dict[str, Any]):
    print("=" * 110)
    print(result_obj["message"])
    print()

    if not result_obj["results"]:
        return

    for i, item in enumerate(result_obj["results"], 1):
        print("-" * 110)
        print(f"#{i}")
        print(f"Книга: {item['title']}")
        print(f"Фрагмент: {item['chunk_index']}")
        print(f"Глава: {item['chapter']}")
        print(f"Символы: {item['start_char']} - {item['end_char']}")
        print(f"Vector score: {item.get('vector_score', 0.0):.4f}")
        print(f"Lexical score: {item.get('lexical_score', 0.0):.4f}")
        print(f"Hybrid score: {item.get('hybrid_score', 0.0):.4f}")
        print()
        print(truncate_text(item["text"], 1100))
        print()


def print_qa_result(result_obj: Dict[str, Any]):
    print("=" * 110)
    print("ВОПРОС:")
    print(result_obj["question"])
    print()
    print("ОТВЕТ:")
    print(result_obj["answer"])
    print()

    print("ЦИТАТЫ:")
    if not result_obj["quotes"]:
        print("Нет цитат.")
        return

    for i, q in enumerate(result_obj["quotes"], 1):
        print("-" * 110)
        print(f"#{i} | Книга: {q['title']} | Фрагмент: {q['chunk_index']} | score: {q['score']}")
        print(q["text"])
        print()

# Если книги загружены, можно раскомментировать:
if len(chunks) > 0:
    test_query = "Найди, где говорится про любовь"
    search_result = rag_search(test_query, top_k=5)
    print_search_results(search_result)

if len(chunks) > 0:
    test_question = "Что произошло с главными героями?"
    qa_result = rag_answer_with_ollama(test_question, top_k=5)
    print(qa_result)


import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

def ui_show_search():
    query = query_entry.get().strip()
    if not query:
        messagebox.showwarning("Предупреждение", "Введите поисковый запрос.")
        return

    output_box.delete("1.0", tk.END)

    try:
        result = rag_search(query, top_k=5)

        output_box.insert(tk.END, f"{result['message']}\n\n")

        if not result["results"]:
            return

        for i, item in enumerate(result["results"], 1):
            output_box.insert(tk.END, "=" * 100 + "\n")
            output_box.insert(tk.END, f"#{i}\n")
            output_box.insert(tk.END, f"Книга: {item['title']}\n")
            output_box.insert(tk.END, f"Фрагмент: {item['chunk_index']}\n")
            output_box.insert(tk.END, f"Глава: {item['chapter']}\n")
            output_box.insert(tk.END, f"Hybrid score: {item.get('hybrid_score', 0.0):.4f}\n")
            output_box.insert(tk.END, f"Символы: {item['start_char']} - {item['end_char']}\n\n")
            output_box.insert(tk.END, truncate_text(item["text"], 1500) + "\n\n")

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


def ui_show_answer():
    question = query_entry.get().strip()
    if not question:
        messagebox.showwarning("Предупреждение", "Введите вопрос.")
        return

    output_box.delete("1.0", tk.END)

    try:
        # Используем уже существующую функцию
        result = rag_answer_with_ollama(question, top_k=5)

        # Вопрос
        output_box.insert(tk.END, "ВОПРОС:\n")
        output_box.insert(tk.END, question + "\n\n")

        # Ответ
        output_box.insert(tk.END, "ОТВЕТ:\n")
        output_box.insert(tk.END, result["answer"] + "\n\n")

        # Цитаты
        output_box.insert(tk.END, "ЦИТАТЫ:\n\n")
        if not result["quotes"]:
            output_box.insert(tk.END, "Нет цитат.\n\n")
        else:
            for i, q in enumerate(result["quotes"], 1):
                score_val = q.get("score", q.get("hybrid_score", 0.0))
                output_box.insert(
                    tk.END,
                    "-"*100 + "\n" +
                    f"#{i} | Книга: {q['title']} | Фрагмент: {q['chunk_index']} | score: {score_val:.4f}\n" +
                    q["text"] + "\n\n"
                )

        # Retrieved chunks
        output_box.insert(tk.END, "RETRIEVED CHUNKS:\n\n")
        for i, item in enumerate(result["retrieved"], 1):
            score_val = item.get("hybrid_score", 0.0)
            output_box.insert(
                tk.END,
                "="*100 + "\n" +
                f"#{i} | Книга: {item['title']} | Фрагмент: {item['chunk_index']} | hybrid_score: {score_val:.4f}\n" +
                truncate_text(item["text"], 1200) + "\n\n"
            )

    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка при обработке запроса:\n{str(e)}")

def ui_show_books():
    output_box.delete("1.0", tk.END)

    if not books:
        output_box.insert(tk.END, "Книги не загружены. Положите .txt файлы в папку ./books\n")
        return

    output_box.insert(tk.END, f"Загружено книг: {len(books)}\n\n")
    for i, b in enumerate(books, 1):
        output_box.insert(tk.END, f"{i}. {b['title']}\n")
        output_box.insert(tk.END, f"   Путь: {b['path']}\n")
        output_box.insert(tk.END, f"   Размер текста: {len(b['text'])} символов\n\n")


def ui_insert_example_search():
    query_entry.delete(0, tk.END)
    query_entry.insert(0, "Найди, где говорится про любовь")


def ui_insert_example_qa():
    query_entry.delete(0, tk.END)
    query_entry.insert(0, "Что произошло с главными героями в финале?")


def launch_desktop_app():
    global query_entry, output_box

    root = tk.Tk()
    root.title("RAG по книгам — тестовый desktop UI")
    root.geometry("1200x800")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    top_frame = ttk.Frame(root, padding=10)
    top_frame.grid(row=0, column=0, sticky="ew")

    top_frame.columnconfigure(0, weight=1)

    ttk.Label(top_frame, text="Запрос или вопрос:").grid(row=0, column=0, sticky="w")

    query_entry = ttk.Entry(top_frame, width=120)
    query_entry.grid(row=1, column=0, sticky="ew", pady=6)

    buttons_frame = ttk.Frame(top_frame)
    buttons_frame.grid(row=2, column=0, sticky="w", pady=6)

    ttk.Button(buttons_frame, text="Показать книги", command=ui_show_books).grid(row=0, column=0, padx=5)
    ttk.Button(buttons_frame, text="Найти фрагменты", command=ui_show_search).grid(row=0, column=1, padx=5)
    ttk.Button(buttons_frame, text="Ответить на вопрос", command=ui_show_answer).grid(row=0, column=2, padx=5)
    ttk.Button(buttons_frame, text="Пример поиска", command=ui_insert_example_search).grid(row=0, column=3, padx=5)
    ttk.Button(buttons_frame, text="Пример QA", command=ui_insert_example_qa).grid(row=0, column=4, padx=5)

    output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    output_frame.grid(row=1, column=0, sticky="nsew")
    output_frame.rowconfigure(0, weight=1)
    output_frame.columnconfigure(0, weight=1)

    output_box = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Arial", 11))
    output_box.grid(row=0, column=0, sticky="nsew")

    output_box.insert(
        tk.END,
        "Тестовый интерфейс запущен.\n\n"
        "1. Положите .txt книги в папку ./books\n"
        "2. Запустите ячейки построения индекса\n"
        "3. Используйте кнопки поиска и вопросов\n"
    )

    root.mainloop()

def rebuild_pipeline():
    global books, chunks
    global tfidf_vectorizer, tfidf_matrix
    global embedding_model, chunk_embeddings, faiss_index

    books = load_books_from_folder(BOOKS_DIR)
    chunks = build_chunks(books)

    tfidf_vectorizer, tfidf_matrix = build_tfidf(chunks)

    # embedding_model уже загружена
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = embed_texts(chunk_texts, embedding_model)
    faiss_index = build_faiss_index(chunk_embeddings)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": tfidf_vectorizer,
            "matrix": tfidf_matrix
        }, f)

    np.save(EMBEDDINGS_PATH, chunk_embeddings)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

    print("Пайплайн пересобран.")
    print("Книг:", len(books))
    print("Чанков:", len(chunks))

launch_desktop_app()
