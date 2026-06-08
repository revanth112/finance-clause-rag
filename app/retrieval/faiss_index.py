import faiss
import numpy as np
from pathlib import Path


class FAISSIndexService:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add_embeddings(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]

    def save(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(index_path: str):
        return faiss.read_index(index_path)