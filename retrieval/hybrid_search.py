from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from .vector_search import vector_search
from .lexical_search import lexical_search

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
