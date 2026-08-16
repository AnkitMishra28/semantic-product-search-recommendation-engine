"""Content-based recommendation baseline constructing user profile preference embeddings."""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from backend.app.models.product import Product
from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate
from backend.app.retrieval.faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Content-Based Recommender building aggregated semantic user profile embeddings.
    
    User preference vector:
        u = sum_{i in H_u} w(i) * e_i / || sum_{i in H_u} w(i) * e_i ||_2
        
    Where:
        e_i = 384-dimensional dense embedding of interacted item i
        w(i) = recency_decay(i) * (rating_i / 5.0)
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
        product_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        faiss_retriever: Optional[FaissRetriever] = None,
        half_life_days: float = 180.0,
    ) -> None:
        self.catalog = product_catalog or {}
        self.faiss_retriever = faiss_retriever
        self.half_life_days = half_life_days

        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.doc_to_idx: Dict[str, int] = {}

        if embeddings is not None and doc_ids is not None:
            self.set_embeddings(embeddings, doc_ids)

    def set_embeddings(self, embeddings: np.ndarray, doc_ids: List[str]) -> None:
        """Register precomputed product embeddings and doc_id mapping."""
        if len(embeddings) != len(doc_ids):
            raise ValueError(f"Embeddings shape {embeddings.shape} does not match doc_ids length {len(doc_ids)}")

        # Ensure unit normalization for fast inner product cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)

        self.doc_ids = [str(d) for d in doc_ids]
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        logger.info(f"ContentBasedRecommender loaded {len(self.doc_ids)} product embeddings ({self.embeddings.shape[1]}-dim).")

    def build_user_vector(
        self,
        history_items: Sequence[Tuple[str, Optional[float], Optional[int]]],
        current_time_ms: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Construct weighted normalized user profile embedding vector.
        
        Args:
            history_items: Sequence of (asin, rating, timestamp_ms) tuples.
            current_time_ms: Optional reference timestamp for recency exponential decay.
            
        Returns:
            Normalized 1D preference vector, or None if no valid history vectors exist.
        """
        if self.embeddings is None or not history_items:
            return None

        weighted_vecs: List[np.ndarray] = []
        weights: List[float] = []

        for item in history_items:
            asin = str(item[0])
            rating = item[1] if len(item) > 1 and item[1] is not None else 5.0
            ts_ms = item[2] if len(item) > 2 and item[2] is not None else None

            if asin not in self.doc_to_idx:
                continue

            idx = self.doc_to_idx[asin]
            vec = self.embeddings[idx]

            # 1. Rating weight: normalize [1..5] to [0.2..1.0]
            rating_weight = max(0.1, float(rating) / 5.0)

            # 2. Recency decay weight: 2^(-delta_days / half_life)
            recency_weight = 1.0
            if ts_ms is not None and current_time_ms is not None and current_time_ms >= ts_ms:
                delta_days = (current_time_ms - ts_ms) / (1000.0 * 86400.0)
                recency_weight = math.pow(0.5, delta_days / max(1.0, self.half_life_days))

            w = rating_weight * recency_weight
            weighted_vecs.append(vec * w)
            weights.append(w)

        if not weighted_vecs:
            return None

        # Aggregate and normalize
        user_vec = np.sum(weighted_vecs, axis=0)
        norm = np.linalg.norm(user_vec)
        if norm > 1e-8:
            user_vec = user_vec / norm
        return user_vec.astype(np.float32)

    def _matches_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if an item's metadata satisfies hard constraints."""
        if not filters:
            return True

        if "category" in filters and filters["category"]:
            target_cat = str(filters["category"]).strip().lower()
            cats = [str(c).strip().lower() for c in metadata.get("categories", [])]
            if not any(target_cat in c or c in target_cat for c in cats):
                return False

        if "brand" in filters and filters["brand"]:
            target_brand = str(filters["brand"]).strip().lower()
            item_brand = str(metadata.get("brand", "")).strip().lower()
            if target_brand != item_brand and target_brand not in item_brand:
                return False

        price = metadata.get("price")
        if "price_min" in filters and filters["price_min"] is not None:
            if price is None or float(price) < float(filters["price_min"]):
                return False

        if "price_max" in filters and filters["price_max"] is not None:
            if price is None or float(price) > float(filters["price_max"]):
                return False

        return True

    def recommend(
        self,
        user_id: Optional[str] = None,
        history_asins: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_consumed: bool = True,
        history_tuples: Optional[List[Tuple[str, Optional[float], Optional[int]]]] = None,
    ) -> List[RecommendationCandidate]:
        """Generate content-based recommendations from user preference embedding vector."""
        if self.embeddings is None:
            logger.warning("ContentBasedRecommender has no embeddings registered.")
            return []

        # Prepare history items
        seen_asins: Set[str] = set(history_asins or [])
        if history_tuples:
            items_to_use = history_tuples
            seen_asins.update(t[0] for t in history_tuples)
        elif history_asins:
            items_to_use = [(asin, 5.0, None) for asin in history_asins]
        else:
            # Cold-start / empty history
            return []

        user_vec = self.build_user_vector(items_to_use)
        if user_vec is None:
            return []

        # Candidate retrieval via FAISS HNSW or exact dot product
        results: List[RecommendationCandidate] = []
        target_k = min(len(self.doc_ids), max(top_k * 5, 200))

        if self.faiss_retriever is not None and self.faiss_retriever._is_trained:
            raw_cands = self.faiss_retriever.search_vector(
                query_vector=user_vec,
                top_k=target_k,
                filters=filters,
            )
            for c in raw_cands:
                asin = str(c.doc_id)
                if exclude_consumed and asin in seen_asins:
                    continue

                meta = self.catalog.get(asin, c.metadata or {})
                if filters and not self._matches_filters(meta, filters):
                    continue

                # Cosine similarity in [-1, 1] normalized to [0, 1]
                sim_score = float(max(0.0, min(1.0, (c.score + 1.0) / 2.0 if c.score <= 1.0 else c.score)))

                cand = RecommendationCandidate(
                    product_id=asin,
                    score=sim_score,
                    rank=len(results) + 1,
                    recommendation_type="content_based",
                    signals={
                        "content": sim_score,
                        "raw_cosine": float(c.score),
                        "rating": float(meta.get("rating", 4.0) or 4.0) / 5.0,
                    },
                    reasons=[
                        f"Matches semantic preferences derived from your past product views ({sim_score:.2f} similarity)."
                    ],
                    metadata=meta,
                )
                results.append(cand)
                if len(results) >= top_k:
                    break
        else:
            # Vectorized exact cosine search
            sims = np.dot(self.embeddings, user_vec)
            # Find top indices
            top_indices = np.argsort(sims)[::-1]

            for idx in top_indices:
                asin = self.doc_ids[idx]
                if exclude_consumed and asin in seen_asins:
                    continue

                meta = self.catalog.get(asin, {})
                if filters and not self._matches_filters(meta, filters):
                    continue

                raw_cos = float(sims[idx])
                sim_score = float(max(0.0, min(1.0, (raw_cos + 1.0) / 2.0)))

                cand = RecommendationCandidate(
                    product_id=asin,
                    score=sim_score,
                    rank=len(results) + 1,
                    recommendation_type="content_based",
                    signals={
                        "content": sim_score,
                        "raw_cosine": raw_cos,
                        "rating": float(meta.get("rating", 4.0) or 4.0) / 5.0,
                    },
                    reasons=[
                        f"Matches semantic preferences derived from your past product views ({sim_score:.2f} similarity)."
                    ],
                    metadata=meta,
                )
                results.append(cand)
                if len(results) >= top_k:
                    break

        return results

    def recommend_for_item(
        self,
        anchor_asin: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RecommendationCandidate]:
        """Recommend items semantically most similar to anchor item."""
        if anchor_asin not in self.doc_to_idx:
            logger.warning(f"Anchor ASIN '{anchor_asin}' not in embeddings vocabulary.")
            return []

        return self.recommend(
            history_asins=[anchor_asin],
            top_k=top_k,
            filters=filters,
            exclude_consumed=True,
        )
