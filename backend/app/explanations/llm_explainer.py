"""Explanation generator combining feature extraction with optional LLM reasoning."""

import logging
from typing import List, Optional
from backend.app.explanations.base import BaseExplainer
from backend.app.models.product import Product
from backend.app.models.recommendation import ExplanationItem

logger = logging.getLogger(__name__)


class LLMExplainer(BaseExplainer):
    """Generates natural language and feature-contrast explanations for search and recommendations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        enable_remote_llm: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.enable_remote_llm = enable_remote_llm and bool(api_key)

    def explain_search_result(
        self,
        query: str,
        product: Product,
        matched_features: Optional[List[str]] = None,
    ) -> str:
        """Provide a concise explanation for a search match."""
        # Rule-based fallback explanation
        brand_match = f" by {product.brand}" if product.brand and product.brand.lower() in query.lower() else ""
        rating_highlight = f" Highly rated ({product.average_rating}★ with {product.rating_count:,} reviews)." if product.average_rating and product.average_rating >= 4.3 else ""
        
        feature_snippet = ""
        if product.features:
            feature_snippet = f" Key highlights: {product.features[0]}."

        return f"Matched for query '{query}'{brand_match}.{feature_snippet}{rating_highlight}".strip()

    def explain_recommendation(
        self,
        anchor_product: Product,
        recommended_product: Product,
    ) -> ExplanationItem:
        """Provide structured explanation for recommendation."""
        shared_categories = list(
            set(anchor_product.categories).intersection(set(recommended_product.categories))
        )
        shared_features = list(
            set(anchor_product.features).intersection(set(recommended_product.features))
        )

        summary = f"Complementary to {anchor_product.title[:35]}..."
        if anchor_product.brand and anchor_product.brand == recommended_product.brand:
            summary = f"From the same brand ({anchor_product.brand}) in {shared_categories[0] if shared_categories else 'Electronics'}."

        return ExplanationItem(
            summary=summary,
            key_features_matched=shared_features[:3],
            shared_categories=shared_categories,
            confidence=0.88,
        )
