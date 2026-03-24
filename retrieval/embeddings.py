import numpy as np
from sentence_transformers import SentenceTransformer

def load_model(name: str):
    return SentenceTransformer(name)


def embed_texts(texts, model):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
