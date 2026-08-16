"""Unified recommendation service orchestrating candidate models, user history, and response formatting."""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from backend.app.models.product import Product
from backend.app.models.recommendation import (
    ExplanationItem,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate
from backend.app.recommendation.collaborative import CollaborativeRecommender
from backend.app.recommendation.content_based import ContentBasedRecommender
from backend.app.recommendation.diversity import MMRReranker
from backend.app.recommendation.hybrid import HybridRecommender
from backend.app.recommendation.popularity import PopularityRecommender

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service layer coordinating recommendation engines, user profiles, and response formatting."""

    def __init__(
        self,
        popularity_recommender: PopularityRecommender,
        content_recommender: Optional[ContentBasedRecommender] = None,
        collaborative_recommender: Optional[CollaborativeRecommender] = None,
        hybrid_recommender: Optional[HybridRecommender] = None,
        diversity_reranker: Optional[MMRReranker] = None,
        product_catalog: Optional[Dict[str, Product]] = None,
        interactions_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.popularity_rec = popularity_recommender
        self.content_rec = content_recommender
        self.collaborative_rec = collaborative_recommender
        self.hybrid_rec = hybrid_recommender
        self.diversity_reranker = diversity_reranker
        self.catalog = product_catalog or {}
        self.user_history_cache: Dict[str, List[str]] = {}

        if interactions_df is not None and not interactions_df.empty:
            self.set_interactions(interactions_df)

    def set_interactions(self, interactions_df: pd.DataFrame) -> None:
        """Cache historical user interaction sequences."""
        self.user_history_cache = (
            interactions_df.groupby("user_id")["parent_asin"]
            .apply(lambda s: list(s.unique()))
            .to_dict()
        )
        logger.info(f"RecommendationService cached interaction history for {len(self.user_history_cache)} users.")

    def set_catalog(self, catalog: Dict[str, Product]) -> None:
        """Register full product catalog."""
        self.catalog = catalog

    def get_user_history(self, user_id: str) -> List[str]:
        """Fetch historical product ASINs for a user ID."""
        return self.user_history_cache.get(str(user_id), [])

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Execute personalized or item-level recommendation for an API request."""
        start_time = time.perf_counter()

        user_id = request.user_id
        anchor_asin = request.asin
        top_k = request.top_k
        strategy = request.strategy.lower().strip()
        filters = request.filters

        # Determine history items
        history_asins: List[str] = list(request.user_history_asins or [])
        if not history_asins and user_id:
            history_asins = self.get_user_history(user_id)

        # Route to appropriate recommender
        candidates: List[RecommendationCandidate] = []

        if anchor_asin:
            # Item-to-Item mode
            if strategy in ("popularity", "pop"):
                candidates = self.popularity_rec.recommend_for_item(anchor_asin, top_k=top_k, filters=filters)
            elif strategy in ("content", "content_based") and self.content_rec:
                candidates = self.content_rec.recommend_for_item(anchor_asin, top_k=top_k, filters=filters)
            elif strategy in ("collaborative", "collab") and self.collaborative_rec:
                candidates = self.collaborative_rec.recommend_for_item(anchor_asin, top_k=top_k, filters=filters)
            elif self.hybrid_rec:
                use_mmr = "mmr" in strategy
                candidates = self.hybrid_rec.recommend_for_item(anchor_asin, top_k=top_k, filters=filters)
                if use_mmr and self.diversity_reranker:
                    candidates = self.diversity_reranker.rerank(
                        candidates, top_k=top_k, lambda_param=request.lambda_diversity
                    )
        else:
            # User Personalized mode
            if strategy in ("popularity", "pop"):
                candidates = self.popularity_rec.recommend(
                    user_id=user_id, history_asins=history_asins, top_k=top_k, filters=filters
                )
            elif strategy in ("content", "content_based") and self.content_rec:
                candidates = self.content_rec.recommend(
                    user_id=user_id, history_asins=history_asins, top_k=top_k, filters=filters
                )
                if not candidates:
                    # Fallback to popularity on cold start
                    candidates = self.popularity_rec.recommend(
                        user_id=user_id, history_asins=history_asins, top_k=top_k, filters=filters
                    )
            elif strategy in ("collaborative", "collab") and self.collaborative_rec:
                candidates = self.collaborative_rec.recommend(
                    user_id=user_id, history_asins=history_asins, top_k=top_k, filters=filters
                )
                if not candidates:
                    # Fallback to popularity on cold start
                    candidates = self.popularity_rec.recommend(
                        user_id=user_id, history_asins=history_asins, top_k=top_k, filters=filters
                    )
            elif self.hybrid_rec:
                use_mmr = "mmr" in strategy
                candidates = self.hybrid_rec.recommend(
                    user_id=user_id,
                    history_asins=history_asins,
                    top_k=top_k,
                    filters=filters,
                    use_mmr=use_mmr,
                    lambda_diversity=request.lambda_diversity,
                )

        # Convert candidates to API models
        recommendations: List[RecommendationItem] = []
        for c in candidates:
            prod = self.catalog.get(c.product_id)
            rec_item = c.to_recommendation_item(prod)
            if not request.generate_explanations:
                rec_item.explanation = None
            recommendations.append(rec_item)

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        return RecommendResponse(
            user_id=user_id,
            anchor_asin=anchor_asin,
            strategy=strategy,
            total_returned=len(recommendations),
            recommendations=recommendations,
            execution_time_ms=round(exec_time_ms, 3),
        )
