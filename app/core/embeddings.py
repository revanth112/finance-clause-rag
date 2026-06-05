"""BGE-M3 embedding engine using BAAI/bge-m3 via sentence-transformers."""
import os
import numpy as np
from typing import List, Union, Optional


class BGEEmbedder:
    """Embedding service using BAAI/bge-m3 model."""

    MODEL_NAME = "BAAI/bge-m3"
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", self.MODEL_NAME)
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _load_model(self):
        """Lazy-load the model to avoid startup overhead."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                print(f"Loaded embedding model: {self.model_name} on {self.device}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install: pip install sentence-transformers"
                )
        return self._model

    def embed_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed
            is_query: If True, prepend query instruction for BGE models

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        model = self._load_model()

        if is_query:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query], is_query=True)[0]

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed a list of documents (no query instruction)."""
        return self.embed_texts(documents, is_query=False)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if self.normalize:
            return float(np.dot(vec1, vec2))
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / norm) if norm > 0 else 0.0
