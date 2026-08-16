"""Grounded explanation API endpoints using Phase 10 GroundedExplainer."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.models.explanation import GroundedExplanation, ProductEvidence
from backend.app.models.product import Product
from backend.app.services.engine import SearchEngine
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/explain", tags=["Grounded Explanations"])


class ExplainRequest(BaseModel):
    """Input payload for generating a factually grounded explanation."""

    query: Optional[str] = Field(default=None, description="Natural language search query")
    product_id: Optional[str] = Field(default=None, description="Target product ASIN to explain")
    asin: Optional[str] = Field(default=None, description="Alias for product_id")
    product: Optional[Product] = Field(default=None, description="Direct product model if provided by caller")
    evidence: Optional[ProductEvidence] = Field(default=None, description="Pre-assembled structured evidence object")
    anchor_product_id: Optional[str] = Field(default=None, description="Anchor ASIN for item-to-item recommendation rationale")
    user_id: Optional[str] = Field(default=None, description="User identifier for personalized recommendation rationale")
    strategy: str = Field(default="hybrid", description="Recommendation strategy")


class SearchExplanationRequest(BaseModel):
    """Backward-compatible simple explanation request."""

    query: str = Field(..., description="The search query")
    asin: str = Field(..., description="The matched product ASIN")


class SimpleExplanationResponse(BaseModel):
    """Backward-compatible simple string response."""

    asin: str
    explanation: str


def get_search_engine() -> SearchEngine:
    registry = ModelRegistry.get_instance()
    return SearchEngine(registry=registry)


@router.post(
    "",
    response_model=GroundedExplanation,
    summary="Generate evidence-grounded explanation for a search result or recommendation",
    responses={
        200: {"description": "Structured grounded explanation returned"},
        400: {"description": "Invalid input: neither query, product_id, nor evidence provided"},
        404: {"description": "Product not found in catalog"},
    },
)
async def generate_explanation(
    request: ExplainRequest,
    engine: SearchEngine = Depends(get_search_engine),
) -> GroundedExplanation:
    """Generate a structured, hallucination-safe explanation grounded strictly in catalog metadata."""
    # 1. Direct pre-assembled evidence object
    if request.evidence is not None:
        return engine.explainer.explain_evidence(request.evidence)

    target_id = request.product_id or request.asin
    registry = ModelRegistry.get_instance()

    # 2. Resolve Product object
    target_product = request.product
    if target_product is None:
        if not target_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'product_id', 'asin', 'product', or 'evidence' must be provided.",
            )
        if target_id in registry.catalog:
            target_product = registry.catalog[target_id]
        else:
            # Linear search
            target_product = next(
                (p for p in registry.catalog.values() if p.asin == target_id or p.parent_asin == target_id),
                None,
            )
            if target_product is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Product with ID '{target_id}' was not found in the catalog.",
                )

    # 3. Search query explanation path
    if request.query:
        qu_result = engine.query_pipeline.process(request.query)
        return engine.explainer.explain_search(
            query=request.query,
            product=target_product,
            query_intent=qu_result,
        )

    # 4. Recommendation explanation path
    anchor_product = None
    if request.anchor_product_id and request.anchor_product_id in registry.catalog:
        anchor_product = registry.catalog[request.anchor_product_id]

    user_history_products = []
    if request.user_id:
        history_source = engine.registry.recommendation_service or engine.recommender
        history_asins = (
            history_source.get_user_history(request.user_id) if hasattr(history_source, "get_user_history") else []
        )
        user_history_products = [registry.catalog[a] for a in history_asins if a in registry.catalog]

    return engine.explainer.explain_recommendation_grounded(
        product=target_product,
        anchor_product=anchor_product,
        user_history_products=user_history_products,
        strategy=request.strategy,
    )


@router.post(
    "/search",
    response_model=SimpleExplanationResponse,
    summary="Legacy string explanation endpoint (backward compatible)",
)
async def explain_search_legacy(
    request: SearchExplanationRequest,
    engine: SearchEngine = Depends(get_search_engine),
) -> SimpleExplanationResponse:
    """Provide feature-contrast or LLM justification for a search match."""
    registry = ModelRegistry.get_instance()
    if request.asin not in registry.catalog:
        dummy_product = Product(
            asin=request.asin,
            title=f"Product {request.asin}",
            features=["High performance", "Durable build"],
            categories=["Electronics"],
        )
        explanation_text = engine.explainer.explain_search_result(request.query, dummy_product)
        return SimpleExplanationResponse(asin=request.asin, explanation=explanation_text)

    product = registry.catalog[request.asin]
    explanation_text = engine.explainer.explain_search_result(request.query, product)
    return SimpleExplanationResponse(asin=request.asin, explanation=explanation_text)
