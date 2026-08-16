"""Hybrid personalized recommendation engine combining semantic embeddings, collaborative filtering, and popularity priors."""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np

from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate
from backend.app.recommendation.collaborative import CollaborativeRecommender
from backend.app.recommendation.content_based import ContentBasedRecommender
from backend.app.recommendation.diversity import MMRReranker
from backend.app.recommendation.popularity import PopularityRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """Hybrid recommendation engine uniting multiple behavioral and semantic signals.
    
    Scoring model:
        S_{hybrid}(u, d) = w_content * S_content(u, d)
                         + w_collab  * S_collab(u, d)
                         + w_pop     * S_pop(d)
                         + w_rating  * S_rating(d)
                         
    Where all component scores are normalized to [0.0, 1.0] and weights sum to 1.0.
    
    Cold-start behavior:
        If user history is empty, falls back gracefully to Bayesian popularity
        and rating quality priors with category diversification.
    """

    def __init__(
        self,
        popularity_recommender: PopularityRecommender,
        content_recommender: Optional[ContentBasedRecommender] = None,
        collaborative_recommender: Optional[CollaborativeRecommender] = None,
        diversity_reranker: Optional[MMRReranker] = None,
        content_weight: float = 0.40,
        collaborative_weight: float = 0.30,
        popularity_weight: float = 0.15,
        rating_weight: float = 0.15,
        candidate_pool_size: int = 100,
        product_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.popularity_rec = popularity_recommender
        self.content_rec = content_recommender
        self.collaborative_rec = collaborative_recommender
        self.diversity_reranker = diversity_reranker
        self.candidate_pool_size = candidate_pool_size
        self.catalog = product_catalog or {}

        # Set normalized weights
        self.set_weights(
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
            popularity_weight=popularity_weight,
            rating_weight=rating_weight,
        )

    def set_weights(
        self,
        content_weight: float,
        collaborative_weight: float,
        popularity_weight: float,
        rating_weight: float,
    ) -> None:
        """Update and normalize hybrid combination weights."""
        raw_weights = {
            "content": max(0.0, float(content_weight)),
            "collaborative": max(0.0, float(collaborative_weight)),
            "popularity": max(0.0, float(popularity_weight)),
            "rating": max(0.0, float(rating_weight)),
        }
        total = sum(raw_weights.values())
        if total <= 0:
            total = 1.0
            raw_weights = {"content": 0.25, "collaborative": 0.25, "popularity": 0.25, "rating": 0.25}

        self.weights = {k: v / total for k, v in raw_weights.items()}

    def _generate_explanation_reasons(
        self,
        signals: Dict[str, float],
        meta: Dict[str, Any],
        is_cold_start: bool = False,
    ) -> List[str]:
        """Generate structured human-interpretable rationale for recommendation."""
        reasons: List[str] = []
        brand = meta.get("brand") or ""
        cat = meta.get("categories", ["Electronics"])
        cat_name = cat[0] if cat else "Electronics"
        rating = float(meta.get("rating", 4.0) or 4.0)
        num_reviews = int(meta.get("rating_number", 0) or 0)

        if is_cold_start:
            reasons.append(
                f"Highly rated ({rating:.1f}★) popular choice in {cat_name} with {num_reviews} customer reviews."
            )
            return reasons

        content_sig = signals.get("content", 0.0)
        collab_sig = signals.get("collaborative", 0.0)
        pop_sig = signals.get("popularity", 0.0)

        if content_sig > 0.6:
            reasons.append(
                f"Matches semantic specifications and categories ({cat_name}) from your recent browsing history."
            )
        if collab_sig > 0.4:
            reasons.append(
                "Frequently co-interacted or purchased alongside products in your past history."
            )
        if pop_sig > 0.5:
            reasons.append(
                f"Top-trending {brand or ''} product in {cat_name} with strong customer satisfaction ({rating:.1f}★)."
            )

        if not reasons:
            reasons.append(
                f"Recommended based on hybrid relevance across catalog popularity and category affinity ({cat_name})."
            )

        return reasons

    def recommend(
        self,
        user_id: Optional[str] = None,
        history_asins: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_consumed: bool = True,
        use_mmr: bool = False,
        lambda_diversity: Optional[float] = None,
    ) -> List[RecommendationCandidate]:
        """Generate multi-stage hybrid recommendations with provenance and optional MMR diversity."""
        consumed_asins: Set[str] = set(history_asins or [])

        # Check for cold start
        is_cold_start = len(consumed_asins) == 0 and (
            user_id is None or (self.collaborative_rec and user_id not in self.collaborative_rec.user_history_map)
        )

        if is_cold_start:
            # 1. Cold-start fallback: Popularity + Rating Diversification
            pop_cands = self.popularity_rec.recommend(
                user_id=user_id,
                history_asins=history_asins,
                top_k=max(top_k * 3, self.candidate_pool_size),
                filters=filters,
                exclude_consumed=exclude_consumed,
            )
            for c in pop_cands:
                c.recommendation_type = "cold_start_popularity"
                c.reasons = self._generate_explanation_reasons(c.signals, c.metadata, is_cold_start=True)

            if use_mmr and self.diversity_reranker is not None:
                return self.diversity_reranker.rerank(pop_cands, top_k=top_k, lambda_param=lambda_diversity)
            return pop_cands[:top_k]

        # 2. Multi-channel candidate pool generation
        pool_k = max(self.candidate_pool_size, top_k * 5)
        candidates_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "content": 0.0,
            "collaborative": 0.0,
            "popularity": 0.0,
            "rating": 0.0,
            "metadata": {},
        })

        # Channel A: Content-Based
        if self.content_rec is not None:
            content_results = self.content_rec.recommend(
                user_id=user_id,
                history_asins=history_asins,
                top_k=pool_k,
                filters=filters,
                exclude_consumed=exclude_consumed,
            )
            for c in content_results:
                candidates_map[c.product_id]["content"] = c.score
                candidates_map[c.product_id]["rating"] = c.signals.get("rating", 0.8)
                if not candidates_map[c.product_id]["metadata"]:
                    candidates_map[c.product_id]["metadata"] = c.metadata or self.catalog.get(c.product_id, {})

        # Channel B: Collaborative
        if self.collaborative_rec is not None:
            collab_results = self.collaborative_rec.recommend(
                user_id=user_id,
                history_asins=history_asins,
                top_k=pool_k,
                filters=filters,
                exclude_consumed=exclude_consumed,
            )
            for c in collab_results:
                candidates_map[c.product_id]["collaborative"] = c.score
                candidates_map[c.product_id]["rating"] = c.signals.get("rating", 0.8)
                if not candidates_map[c.product_id]["metadata"]:
                    candidates_map[c.product_id]["metadata"] = c.metadata or self.catalog.get(c.product_id, {})

        # Channel C: Popularity
        pop_results = self.popularity_rec.recommend(
            user_id=user_id,
            history_asins=history_asins,
            top_k=pool_k,
            filters=filters,
            exclude_consumed=exclude_consumed,
        )
        for c in pop_results:
            candidates_map[c.product_id]["popularity"] = c.score
            candidates_map[c.product_id]["rating"] = c.signals.get("rating", 0.8)
            if not candidates_map[c.product_id]["metadata"]:
                candidates_map[c.product_id]["metadata"] = c.metadata or self.catalog.get(c.product_id, {})

        # 3. Fuse scores
        w_cont = self.weights["content"]
        w_collab = self.weights["collaborative"]
        w_pop = self.weights["popularity"]
        w_rating = self.weights["rating"]

        scored_candidates: List[RecommendationCandidate] = []

        for asin, item_data in candidates_map.items():
            if exclude_consumed and asin in consumed_asins:
                continue

            s_cont = item_data["content"]
            s_collab = item_data["collaborative"]
            s_pop = item_data["popularity"]
            s_rating = item_data["rating"]

            # Fallback popularity lookup if missing in dictionary
            if s_pop == 0.0 and asin in self.popularity_rec.item_scores:
                s_pop = self.popularity_rec.item_scores[asin]
                item_data["popularity"] = s_pop

            hybrid_score = (
                (w_cont * s_cont)
                + (w_collab * s_collab)
                + (w_pop * s_pop)
                + (w_rating * s_rating)
            )

            signals = {
                "content": float(s_cont),
                "collaborative": float(s_collab),
                "popularity": float(s_pop),
                "rating": float(s_rating),
                "hybrid_score": float(hybrid_score),
            }

            meta = item_data["metadata"] or self.catalog.get(asin, {})
            reasons = self._generate_explanation_reasons(signals, meta, is_cold_start=False)

            cand = RecommendationCandidate(
                product_id=asin,
                score=float(hybrid_score),
                rank=1,
                recommendation_type="hybrid",
                signals=signals,
                reasons=reasons,
                metadata=meta,
            )
            scored_candidates.append(cand)

        # Sort descending by hybrid score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        for idx, c in enumerate(scored_candidates, start=1):
            c.rank = idx

        # 4. Optional diversity reranking
        if use_mmr and self.diversity_reranker is not None:
            return self.diversity_reranker.rerank(
                scored_candidates[:pool_k],
                top_k=top_k,
                lambda_param=lambda_diversity,
            )

        return scored_candidates[:top_k]

    def recommend_for_item(
        self,
        anchor_asin: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RecommendationCandidate]:
        """Produce item-level recommendations fusing semantic, collaborative, and popularity signals."""
        return self.recommend(
            history_asins=[anchor_asin],
            top_k=top_k,
            filters=filters,
            exclude_consumed=True,
        )
