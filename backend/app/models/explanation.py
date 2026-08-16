"""Domain models and schemas for grounded product explanations."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExplanationReason(BaseModel):
    """An individual structured reason item supporting a product recommendation or search match."""

    type: str = Field(
        ...,
        description="Reason category: constraint_match | semantic_relevance | ranking_strength | personalization | rating_credibility | unsupported_constraint",
    )
    label: str = Field(
        ...,
        description="Short human-readable category label (e.g., 'GPU', 'Budget', 'Brand', 'Category', 'User History', 'Missing Info')",
    )
    text: str = Field(
        ...,
        description="Clear, concise, user-facing sentence explaining why this product satisfies the query or user context",
    )
    evidence: str = Field(
        ...,
        description="Exact snippet or factual token quoted directly from product metadata or query constraints",
    )
    is_matched: bool = Field(
        default=True,
        description="True if the constraint or signal was verified/matched; False if missing or unsupported",
    )


class ProductEvidence(BaseModel):
    """Structured evidence package assembled from upstream pipeline stages for grounded explanation generation."""

    query: Optional[str] = Field(default=None, description="Original natural language user query")
    product_id: str = Field(..., description="Canonical product identifier (ASIN or parent ASIN)")
    product_title: str = Field(..., description="Full product title from catalog")
    price: Optional[float] = Field(default=None, description="Current catalog price in USD")
    currency: str = Field(default="USD", description="Currency code")
    rating: Optional[float] = Field(default=None, description="Average review rating (0.0 - 5.0)")
    rating_count: int = Field(default=0, description="Total number of customer ratings/reviews")
    brand: Optional[str] = Field(default=None, description="Catalog brand name")
    category: Optional[str] = Field(default=None, description="Primary detected or canonical catalog category")
    categories: List[str] = Field(default_factory=list, description="Full hierarchical category path")
    features: List[str] = Field(default_factory=list, description="Extracted bullet points / technical features")
    description: Optional[str] = Field(default="", description="Product description snippet")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted or structured product specifications (e.g. gpu, ram, connectivity, noise_cancelling)",
    )
    query_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured constraints extracted by Query Understanding (category, brand, price_min, price_max, attributes)",
    )
    retrieval_provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-1 retrieval provenance (BM25 rank, Dense FAISS rank, RRF rank, score)",
    )
    cross_encoder_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-2 Cross-Encoder reranking evidence (score, reranked rank, score delta)",
    )
    personalization_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Personalization signals (user history count, shared categories, collaborative co-occurrence, component weights)",
    )
    matched_constraints: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping constraint names to verified match descriptions",
    )
    missing_constraints: List[str] = Field(
        default_factory=list,
        description="List of requested query constraints that have NO evidence in the product catalog",
    )


class GroundedExplanation(BaseModel):
    """Complete structured explanation response strictly grounded in verified pipeline evidence."""

    product_id: str = Field(..., description="Product identifier (ASIN)")
    summary: str = Field(
        ...,
        description="Concise 1-2 sentence overview explaining why this product is suitable for the user/query",
    )
    reasons: List[ExplanationReason] = Field(
        default_factory=list,
        description="Structured list of verified evidence-based reasons",
    )
    semantic_match_score: Optional[float] = Field(
        default=None,
        description="Optional neural cross-attention or semantic similarity score (0.0 to 1.0)",
    )
    grounded: bool = Field(
        default=True,
        description="True indicating all claims are strictly backed by verifiable catalog/pipeline evidence",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Explicit disclaimers regarding requested attributes that were missing from catalog evidence",
    )
    generation_method: str = Field(
        default="deterministic_fallback",
        description="Method used: 'llm' or 'deterministic_fallback'",
    )
