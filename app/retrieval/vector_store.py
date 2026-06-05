"""FAISS vector store with metadata filtering for finance clause retrieval."""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


class FAISSVectorStore:
    """FAISS-based vector store with persist/load and metadata filtering."""

    def __init__(
        self,
        embedding_dim: int = 1024,  # BGE-M3 default dim
        index_type: str = "flat_ip",  # flat inner product (cosine after normalize)
        store_dir: str = "data/vectorstore",
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._index = None
        self._metadata_store: List[Dict[str, Any]] = []
        self._texts: List[str] = []

    def _create_index(self):
        """Create a new FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu required. Install: pip install faiss-cpu")

        if self.index_type == "flat_ip":
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "flat_l2":
            self._index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            import faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)

        return self._index

    def add_chunks(self, chunks_with_embeddings: List[Tuple[str, np.ndarray, Dict]]):
        """Add chunks to the vector store.

        Args:
            chunks_with_embeddings: List of (text, embedding, metadata) tuples
        """
        if self._index is None:
            self._create_index()

        texts, embeddings, metadatas = zip(*chunks_with_embeddings)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity with flat_ip index
        if self.index_type == "flat_ip":
            import faiss
            faiss.normalize_L2(embeddings_array)

        # Train IVF index if needed
        import faiss
        if isinstance(self._index, faiss.IndexIVFFlat) and not self._index.is_trained:
            self._index.train(embeddings_array)

        self._index.add(embeddings_array)
        self._texts.extend(texts)
        self._metadata_store.extend(metadatas)

        print(f"Added {len(texts)} chunks. Total: {self._index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks with optional metadata filtering.

        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            metadata_filter: Dict of {field: value} to filter results
            score_threshold: Minimum similarity score

        Returns:
            List of result dicts with text, metadata, and score
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        if self.index_type == "flat_ip":
            import faiss
            faiss.normalize_L2(query)

        # Retrieve extra candidates if filtering
        fetch_k = top_k * 5 if metadata_filter else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            if score < score_threshold:
                continue

            metadata = self._metadata_store[idx]

            # Apply metadata filter
            if metadata_filter:
                match = all(
                    str(metadata.get(k, "")).lower() == str(v).lower()
                    for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            results.append({
                "text": self._texts[idx],
                "metadata": metadata,
                "score": float(score),
                "index": int(idx),
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, name: str = "finance_index"):
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            raise ValueError("No index to save")

        import faiss
        index_path = self.store_dir / f"{name}.faiss"
        meta_path = self.store_dir / f"{name}_meta.pkl"

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"texts": self._texts, "metadata": self._metadata_store}, f)

        print(f"Saved index ({self._index.ntotal} vectors) to {index_path}")

    def load(self, name: str = "finance_index"):
        """Load a persisted FAISS index and metadata."""
        import faiss
        index_path = self.store_dir / f"{name}.faiss"
        meta_path = self.store_dir / f"{name}_meta.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self._texts = data["texts"]
            self._metadata_store = data["metadata"]

        print(f"Loaded index with {self._index.ntotal} vectors from {index_path}")

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
