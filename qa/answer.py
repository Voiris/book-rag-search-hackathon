import re
from typing import List, Dict, Any

from utils import split_into_sentences

STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "он", "она", "оно", "они", "на",
    "я", "мы", "ты", "вы", "по", "с", "со", "к", "ко", "от", "до",
    "за", "из", "у", "о", "об", "про", "а", "но", "как", "для",
    "это", "то", "же", "ли", "или", "бы", "был", "была", "были",
    "быть", "его", "ее", "их", "ему", "ей", "им"
}


# --- basic utils ---

def tokenize(text: str) -> List[str]:
    text = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", text)


def keyword_set(text: str) -> set:
    return {
        t for t in tokenize(text)
        if t not in STOPWORDS_RU and len(t) >= 3
    }


def sentence_score(sentence: str, query_keywords: set) -> float:
    tokens = keyword_set(sentence)
    if not tokens:
        return 0.0
    return len(tokens & query_keywords) / (len(query_keywords) + 1e-9)


# --- core logic ---

def select_support_sentences(
    question: str,
    chunks: List[Dict[str, Any]],
    max_sentences: int = 5
) -> List[Dict[str, Any]]:

    q_keywords = keyword_set(question)
    candidates = []

    for chunk in chunks:
        for sent in split_into_sentences(chunk["text"]):
            score = sentence_score(sent, q_keywords)
            if score > 0:
                candidates.append({
                    "sentence": sent.strip(),
                    "score": score,
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"],
                })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # удаляем дубли
    seen = set()
    result = []

    for c in candidates:
        key = c["sentence"][:120].lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(c)

        if len(result) >= max_sentences:
            break

    return result


def truncate(text: str, max_len: int = 500) -> str:
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    min_score: float = 0.22
) -> Dict[str, Any]:

    if not retrieved_chunks:
        return {
            "found": False,
            "answer": "Не удалось найти контекст.",
            "quotes": []
        }

    best_score = retrieved_chunks[0].get("hybrid_score", 0.0)
    if best_score < min_score:
        return {
            "found": False,
            "answer": "Недостаточно релевантной информации.",
            "quotes": []
        }

    support = select_support_sentences(question, retrieved_chunks)

    # fallback
    if not support:
        return {
            "found": True,
            "answer": "Найдены близкие фрагменты, но точный ответ выделить не удалось.",
            "quotes": [
                {
                    "title": c["title"],
                    "chunk_index": c["chunk_index"],
                    "text": truncate(c["text"])
                }
                for c in retrieved_chunks[:3]
            ]
        }

    # собираем ответ
    answer = " ".join(s["sentence"] for s in support[:3])

    quotes = [
        {
            "title": s["title"],
            "chunk_index": s["chunk_index"],
            "score": round(s["score"], 4),
            "text": s["sentence"]
        }
        for s in support
    ]

    return {
        "found": True,
        "answer": answer,
        "quotes": quotes
    }
