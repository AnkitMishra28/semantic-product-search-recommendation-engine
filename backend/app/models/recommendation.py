"""Recommendation API request and response models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.models.explanation import GroundedExplanation
from backend.app.models.product import Product


class ExplanationItem(BaseModel):
    """Structured rationale explaining a recommendation."""

    summary: str = Field(..., description="High-level reason for recommendation")
    key_features_matched: List[str] = Field(default_factory=list, description="Common or matched features")
    shared_categories: List[str] = Field(default_factory=list, description="Common category tags")
    confidence: Optional[float] = Field(default=None, description="Explanation confidence score")


class RecommendationItem(BaseModel):
    """An individual recommended product with similarity score, signal breakdown, and rationale."""

    product: Product
    score: float = Field(..., description="Recommendation affinity / similarity score")
    recommendation_type: str = Field(
        default="hybrid",
        description="Type: popularity | content_based | collaborative | hybrid | hybrid_mmr",
    )
    signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized component score breakdown (content, collaborative, popularity, rating)",
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable rationale items explaining why this product was recommended",
    )
    explanation: Optional[ExplanationItem] = None
    grounded_explanation: Optional[GroundedExplanation] = None


class RecommendRequest(BaseModel):
    """Recommendation endpoint input schema."""

    user_id: Optional[str] = Field(default=None, description="Optional user ID for personalized recommendation")
    asin: Optional[str] = Field(default=None, description="Anchor product ASIN for item-to-item recommendation")
    user_history_asins: Optional[List[str]] = Field(
        default_factory=list,
        description="List of recently viewed or purchased ASINs for personalized recommendation",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    strategy: str = Field(
        default="hybrid",
        description="Recommendation strategy: popularity | content | collaborative | hybrid | hybrid_mmr",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional hard filters (e.g. {'category': 'Headphones', 'brand': 'Sony', 'price_max': 200.0})",
    )
    lambda_diversity: Optional[float] = Field(
        default=None,
        description="MMR diversity weighting parameter (0.0 to 1.0) when using hybrid_mmr strategy",
    )
    generate_explanations: bool = Field(default=True, description="Attach structured explanations")


class RecommendResponse(BaseModel):
    """Recommendation endpoint output schema."""

    user_id: Optional[str] = None
    anchor_asin: Optional[str] = None
    strategy: str
    total_returned: int
    recommendations: List[RecommendationItem]
    execution_time_ms: float

