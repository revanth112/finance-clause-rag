"""Cross-encoder reranker for finance clause retrieval results."""
import os
from typing import List, Dict, Any, Optional


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model."""

    # Best cross-encoder for finance/general use
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FINANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = -10.0,
    ):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", self.DEFAULT_MODEL)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                print(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """Rerank a list of candidate documents using cross-encoder scores.

        Args:
            query: The user query string
            candidates: List of dicts with at least a `text` field
            top_k: Number of top results to return (defaults to self.top_k)
            text_key: Key to use for text content in candidate dicts

        Returns:
            Reranked list with cross_encoder_score added to each dict
        """
        if not candidates:
            return []

        model = self._load_model()
        k = top_k or self.top_k

        # Build query-document pairs for cross-encoder
        pairs = [(query, c[text_key]) for c in candidates]

        # Get relevance scores
        scores = model.predict(pairs)

        # Attach scores and sort
        scored = []
        for candidate, score in zip(candidates, scores):
            if float(score) >= self.score_threshold:
                enriched = {**candidate, "cross_encoder_score": float(score)}
                scored.append(enriched)

        # Sort by cross-encoder score descending
        scored.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        return scored[:k]

    def rerank_with_fusion(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "text",
        retrieval_weight: float = 0.3,
        rerank_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Rerank using a weighted fusion of retrieval score + cross-encoder score.

        Args:
            query: The user query
            candidates: Retrieved chunks with `score` field
            top_k: Number of final results
            text_key: Key for text content
            retrieval_weight: Weight for original retrieval score
            rerank_weight: Weight for cross-encoder score

        Returns:
            Fused and reranked results with `fusion_score`
        """
        if not candidates:
            return []

        model = self._load_model()
        k = top_k or self.top_k

        pairs = [(query, c[text_key]) for c in candidates]
        ce_scores = model.predict(pairs)

        # Normalize retrieval scores to [0,1]
        ret_scores = [float(c.get("score", 0)) for c in candidates]
        max_ret = max(ret_scores) if ret_scores else 1.0
        min_ret = min(ret_scores) if ret_scores else 0.0
        ret_range = max_ret - min_ret or 1.0

        # Normalize CE scores using sigmoid
        import math
        def sigmoid(x): return 1 / (1 + math.exp(-x))

        scored = []
        for candidate, ce_score, ret_score in zip(candidates, ce_scores, ret_scores):
            norm_ret = (ret_score - min_ret) / ret_range
            norm_ce = sigmoid(float(ce_score))
            fusion = retrieval_weight * norm_ret + rerank_weight * norm_ce
            enriched = {
                **candidate,
                "cross_encoder_score": float(ce_score),
                "fusion_score": fusion,
            }
            scored.append(enriched)

        scored.sort(key=lambda x: x["fusion_score"], reverse=True)
        return scored[:k]
