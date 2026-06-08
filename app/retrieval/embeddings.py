from sentence_transformers import SentenceTransformer
import numpy as np


class BGEEmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return np.array(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embedding, dtype="float32")