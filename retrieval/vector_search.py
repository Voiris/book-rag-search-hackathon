import faiss

def build_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def vector_search(query, model, index, chunks, top_k):
    emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(emb, top_k)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        item = dict(chunks[i])
        item["vector_score"] = float(s)
        results.append(item)

    return results
