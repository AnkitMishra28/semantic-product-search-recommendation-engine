"""Abstract base interface for search and recommendation explanations."""

from abc import ABC, abstractmethod
from typing import List, Optional
from backend.app.models.product import Product
from backend.app.models.recommendation import ExplanationItem


class BaseExplainer(ABC):
    """Abstract interface for explaining search relevance and recommendations."""

    @abstractmethod
    def explain_search_result(
        self,
        query: str,
        product: Product,
        matched_features: Optional[List[str]] = None,
    ) -> str:
        """Generate a concise explanation for why a product matches a user's query."""
        pass

    @abstractmethod
    def explain_recommendation(
        self,
        anchor_product: Product,
        recommended_product: Product,
    ) -> ExplanationItem:
        """Generate structured explanation for why an item was recommended."""
        pass
