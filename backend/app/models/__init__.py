"""Domain models and schemas package."""

from backend.app.models.product import Product, ProductFilter
from backend.app.models.interaction import Interaction
from backend.app.models.explanation import (
    ExplanationReason,
    GroundedExplanation,
    ProductEvidence,
)
from backend.app.models.recommendation import (
    ExplanationItem,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from backend.app.models.search import (
    PipelineStageTiming,
    QueryUnderstandingResult,
    RerankSignal,
    RetrievalSignal,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

__all__ = [
    "Product",
    "ProductFilter",
    "Interaction",
    "ProductEvidence",
    "ExplanationReason",
    "GroundedExplanation",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "RetrievalSignal",
    "RerankSignal",
    "PipelineStageTiming",
    "QueryUnderstandingResult",
    "RecommendRequest",
    "RecommendResponse",
    "RecommendationItem",
    "ExplanationItem",
]
