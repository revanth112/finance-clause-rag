import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional
from app.core.config import settings

class FAISSService:
    """
    Manages FAISS index for dense vector search.
    Stores chunk metadata separately and maps back via FAISS IDs.
    Pre-filtering and post-filtering both supported via metadata store.
    """
    def __init__(self):
        self.index = None
        self.chunk_metadata: Dict[int, Dict] = {}  # faiss_id -> metadata dict
        self.dimension = 1024  # BGE-M3 output dimension

    def build_index(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """Build FAISS IndexFlatIP (inner product = cosine for normalized vecs)."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunk_metadata = {i: meta for i, meta in enumerate(metadata_list)}
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = 30) -> List[Dict]:
        """Search FAISS for top_k candidates. Returns list of chunk dicts with scores."""
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunk_metadata.get(int(idx), {})
            results.append({**meta, "faiss_score": float(score)})
        return results

    def filter_by_metadata(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """
        Post-filter retrieved chunks by metadata fields.
        filters: dict of field -> value, e.g. {"clause_type": "termination"}
        """
        filtered = []
        for chunk in results:
            match = all(
                chunk.get(k) == v
                for k, v in filters.items()
                if v is not None
            )
            if match:
                filtered.append(chunk)
        return filtered

    def save_index(self):
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        with open(settings.CHUNK_METADATA_PATH, "w") as f:
            json.dump(self.chunk_metadata, f)
        print("FAISS index saved.")

    def load_index(self):
        self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        with open(settings.CHUNK_METADATA_PATH, "r") as f:
            raw = json.load(f)
            self.chunk_metadata = {int(k): v for k, v in raw.items()}
        print(f"FAISS index loaded: {self.index.ntotal} vectors.")

faiss_service = FAISSService()
