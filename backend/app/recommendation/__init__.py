"""Recommendation engine package exposing baselines, hybrid algorithms, and diversity rerankers."""

from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate
from backend.app.recommendation.collaborative import CollaborativeRecommender
from backend.app.recommendation.content_based import ContentBasedRecommender
from backend.app.recommendation.diversity import MMRReranker
from backend.app.recommendation.hybrid import HybridRecommender
from backend.app.recommendation.item_to_item import ItemToItemRecommender
from backend.app.recommendation.popularity import PopularityRecommender
from backend.app.recommendation.service import RecommendationService

__all__ = [
    "BaseRecommender",
    "RecommendationCandidate",
    "PopularityRecommender",
    "ContentBasedRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "ItemToItemRecommender",
    "MMRReranker",
    "RecommendationService",
]
