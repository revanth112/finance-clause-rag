from sentence_transformers import CrossEncoder
from app.core.config import settings
from typing import List, Dict

class RerankerService:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
    Takes a query and a list of candidate chunks, scores each (query, chunk) pair,
    and returns re-sorted chunks with updated scores.
    This is Stage 2 of the two-stage retrieval pipeline:
      Stage 1: FAISS fast dense retrieval (top 30-50 candidates)
      Stage 2: Cross-encoder reranking (narrow to top 5)
    """
    def __init__(self):
        print(f"Loading reranker model: {settings.RERANKER_MODEL}")
        self.model = CrossEncoder(
            settings.RERANKER_MODEL,
            max_length=512
        )

    def rerank(self, query: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank chunks by cross-encoder score.
        Returns sorted list, highest score first.
        """
        if not chunks:
            return []

        top_k = top_k or settings.TOP_K_RERANK
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

reranker_service = RerankerService()
