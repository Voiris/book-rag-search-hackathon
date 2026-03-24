import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text import clean_for_lexical_search

def build_tfidf(chunks):
    corpus = [c["search_text"] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2))
    return vec, vec.fit_transform(corpus)


def lexical_search(query, vectorizer, matrix, chunks, top_k):
    q = vectorizer.transform([clean_for_lexical_search(query)])
    scores = (matrix @ q.T).toarray().ravel()

    top = np.argsort(scores)[::-1][:top_k]

    return [
        {**chunks[i], "lexical_score": float(scores[i])}
        for i in top if scores[i] > 0
    ]
