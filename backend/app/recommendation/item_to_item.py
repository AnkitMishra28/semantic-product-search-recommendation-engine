"""Item-to-Item and Hybrid Recommendation Strategy."""

import logging
from typing import Any, Dict, List, Optional
from backend.app.models.product import Product
from backend.app.models.recommendation import ExplanationItem, RecommendationItem
from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate
from backend.app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class ItemToItemRecommender(BaseRecommender):
    """Generates recommendations by fusing dense vector neighborhood with co-purchase graph links."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        product_catalog: Optional[Dict[str, Product]] = None,
    ) -> None:
        self.retriever = retriever
        self.catalog = product_catalog or {}

    def set_catalog(self, catalog: Dict[str, Product]) -> None:
        """Register the product catalog dictionary."""
        self.catalog = catalog

    def recommend_for_item(
        self,
        anchor_asin: str,
        top_k: int = 10,
    ) -> List[RecommendationItem]:
        """Produce complementary and substitute recommendations for an anchor item."""
        if anchor_asin not in self.catalog:
            logger.warning(f"Anchor ASIN '{anchor_asin}' not found in catalog.")
            return []

        anchor_product = self.catalog[anchor_asin]
        recommendations: List[RecommendationItem] = []

        # 1. Co-purchase items (Frequently bought together)
        for bought_asin in anchor_product.bought_together:
            if bought_asin in self.catalog and bought_asin != anchor_asin:
                rec_product = self.catalog[bought_asin]
                recommendations.append(
                    RecommendationItem(
                        product=rec_product,
                        score=0.95,
                        recommendation_type="collaborative_bought_together",
                        explanation=ExplanationItem(
                            summary=f"Customers who bought {anchor_product.title[:30]}... also frequently bought this item.",
                            key_features_matched=list(set(anchor_product.features).intersection(set(rec_product.features))),
                            shared_categories=list(set(anchor_product.categories).intersection(set(rec_product.categories))),
                            confidence=0.95,
                        ),
                    )
                )
                if len(recommendations) >= top_k:
                    return recommendations[:top_k]

        # 2. Vector semantic neighborhood (if retriever is available)
        # Note: Vector search by item embedding will be executed in Phase 2

        return recommendations[:top_k]

    def recommend(
        self,
        user_id: Optional[str] = None,
        history_asins: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_consumed: bool = True,
    ) -> List[RecommendationCandidate]:
        """Generate personalized recommendations based on past user interaction sequence."""
        rec_items = self.recommend_for_user(history_asins=history_asins or [], top_k=top_k)
        candidates = []
        for idx, item in enumerate(rec_items):
            c = RecommendationCandidate(
                product_id=item.product.asin,
                score=item.score,
                rank=idx + 1,
                recommendation_type=item.recommendation_type,
                signals=item.signals,
                reasons=item.reasons,
            )
            candidates.append(c)
        return candidates

    def recommend_for_user(
        self,
        history_asins: List[str],
        top_k: int = 10,
    ) -> List[RecommendationItem]:
        """Aggregate recommendations across user's recently viewed/purchased items."""
        if not history_asins:
            return []

        # Collect candidates from recent history anchors
        seen_asins = set(history_asins)
        candidates: List[RecommendationItem] = []

        for asin in reversed(history_asins[-3:]):
            recs = self.recommend_for_item(asin, top_k=top_k)
            for r in recs:
                if r.product.asin not in seen_asins:
                    candidates.append(r)
                    seen_asins.add(r.product.asin)

        return candidates[:top_k]
