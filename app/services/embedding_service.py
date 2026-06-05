from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np
from typing import List

class EmbeddingService:
    """
    Handles text embedding using BAAI/bge-m3.
    BGE-M3 supports dense, sparse, and multi-vector retrieval.
    For FAISS we use dense embeddings with normalized vectors.
    """
    def __init__(self):
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE
        )
        self.dimension = 1024  # BGE-M3 output dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns normalized float32 numpy array."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # Required for cosine similarity with BGE
            batch_size=32,
            show_progress_bar=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)

embedding_service = EmbeddingService()
