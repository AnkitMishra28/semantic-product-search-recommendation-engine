"""Abstract base interface and data structures for recommendation algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from backend.app.models.product import Product
from backend.app.models.recommendation import ExplanationItem, RecommendationItem


@dataclass
class RecommendationCandidate:
    """Internal candidate representation with component signals and explanation rationale."""

    product_id: str
    score: float
    rank: int = 1
    recommendation_type: str = "hybrid"
    signals: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_recommendation_item(self, product: Optional[Product] = None) -> RecommendationItem:
        """Convert candidate to API response model."""
        if product is None:
            # Construct minimal product from metadata if full model not provided
            product = Product(
                asin=self.product_id,
                parent_asin=self.metadata.get("parent_asin", self.product_id),
                title=self.metadata.get("title", f"Product {self.product_id}"),
                price=float(self.metadata.get("price", 0.0) or 0.0),
                average_rating=float(self.metadata.get("rating", self.metadata.get("average_rating", 0.0)) or 0.0),
                rating_number=int(self.metadata.get("rating_number", 0) or 0),
                features=self.metadata.get("features", []),
                categories=self.metadata.get("categories", []),
                brand=self.metadata.get("brand", "Unknown"),
            )

        explanation = None
        if self.reasons:
            explanation = ExplanationItem(
                summary=self.reasons[0],
                key_features_matched=self.metadata.get("matched_features", []),
                shared_categories=self.metadata.get("categories", []),
                confidence=round(self.score, 4),
            )

        return RecommendationItem(
            product=product,
            score=round(self.score, 4),
            recommendation_type=self.recommendation_type,
            signals={k: round(v, 4) for k, v in self.signals.items()},
            reasons=self.reasons,
            explanation=explanation,
        )


class BaseRecommender(ABC):
    """Abstract interface for product recommendation engines."""

    @abstractmethod
    def recommend(
        self,
        user_id: Optional[str] = None,
        history_asins: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_consumed: bool = True,
    ) -> List[RecommendationCandidate]:
        """Generate personalized recommendations based on past user interaction sequence.
        
        Args:
            user_id: Optional unique user identifier.
            history_asins: Optional sequence of past consumed product ASINs.
            top_k: Number of recommendations to generate.
            filters: Optional dictionary of hard constraints (brand, category, price_min, price_max).
            exclude_consumed: Whether to strictly exclude products in user's historical profile.
            
        Returns:
            List of RecommendationCandidate ranked in descending score order.
        """
        pass

    @abstractmethod
    def recommend_for_item(
        self,
        anchor_asin: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RecommendationCandidate]:
        """Generate recommendations similar to or complementary to an anchor product."""
        pass

    def recommend_for_user(
        self,
        history_asins: List[str],
        top_k: int = 10,
    ) -> List[RecommendationItem]:
        """Backward-compatible user recommendation returning RecommendationItem models."""
        cands = self.recommend(history_asins=history_asins, top_k=top_k)
        return [c.to_recommendation_item() for c in cands]
