"""Search API request and response schemas including multi-stage ranking signals and query understanding."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.models.explanation import GroundedExplanation
from backend.app.models.product import Product, ProductFilter


class QueryIntent(BaseModel):
    """Structured query intent extracted from natural language input."""

    raw_query: str = Field(..., description="Original raw user query string")
    normalized_query: str = Field(..., description="Normalized, stripped search string")
    intent: str = Field(default="product_search", description="Classified intent type")
    category: Optional[str] = Field(default=None, description="Catalog-canonical category if detected")
    brand: Optional[str] = Field(default=None, description="Catalog-canonical brand if detected")
    price_min: Optional[float] = Field(default=None, description="Extracted minimum price constraint")
    price_max: Optional[float] = Field(default=None, description="Extracted maximum price constraint")
    currency: str = Field(default="USD", description="Currency identifier (default USD, configurable)")
    attributes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted product attributes (e.g. gpu, ram, storage, connectivity, use_case)",
    )
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Grouped entity tokens for backward compatibility and analysis",
    )
    hard_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic constraints to apply before vector search/reranking",
    )
    soft_signals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Non-deterministic or qualitative modifiers to guide ranking",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall parsing confidence score")


class QueryUnderstandingResult(BaseModel):
    """Output from the query understanding subsystem (backward compatible wrapper)."""

    raw_query: str
    normalized_query: str
    intent: str = Field(default="product_search", description="Classified search intent")
    category: Optional[str] = Field(default=None, description="Detected product category")
    brand: Optional[str] = Field(default=None, description="Detected brand")
    price_min: Optional[float] = Field(default=None, description="Minimum price constraint")
    price_max: Optional[float] = Field(default=None, description="Maximum price constraint")
    currency: str = Field(default="USD", description="Currency identifier (default USD)")
    attributes: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted structured attributes")
    detected_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities (brands, categories, attributes, etc.)",
    )
    expanded_queries: List[str] = Field(
        default_factory=list,
        description="Expanded or synonym-augmented queries",
    )
    hard_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured filters suitable for index/metadata filtering",
    )
    soft_signals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Qualitative signals for neural ranking",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Parser confidence")


class RetrievalSignal(BaseModel):
    """First-stage retrieval diagnostic signal."""

    stage: str = "dense_retrieval"
    initial_score: float = Field(..., description="Cosine similarity / vector distance score")
    initial_rank: int = Field(..., description="1-indexed rank from vector index")


class RerankSignal(BaseModel):
    """Second-stage cross-encoder reranking diagnostic signal."""

    stage: str = "cross_encoder"
    rerank_score: float = Field(..., description="Neural cross-attention relevance score")
    rerank_rank: int = Field(..., description="Rank after cross-encoder scoring")


class PipelineStageTiming(BaseModel):
    """Latency breakdown per search pipeline stage in milliseconds."""

    query_understanding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    cross_encoder_rerank_ms: float = 0.0
    business_ranking_ms: float = 0.0
    explanation_generation_ms: float = 0.0
    total_latency_ms: float = 0.0


class SearchResultItem(BaseModel):
    """A ranked search result with attached multi-stage scoring signals and explanations."""

    product: Product
    final_score: float
    retrieval_signal: Optional[RetrievalSignal] = None
    rerank_signal: Optional[RerankSignal] = None
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable rationale or LLM explanation for why this item was retrieved",
    )
    grounded_explanation: Optional[GroundedExplanation] = Field(
        default=None,
        description="Full structured grounded explanation details (reasons, evidence tokens, warnings)",
    )


class SearchRequest(BaseModel):
    """Search endpoint input schema."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k_retrieval: Optional[int] = Field(default=100, ge=1, le=1000, description="Stage 1 retrieval candidates")
    top_k_reranking: Optional[int] = Field(default=20, ge=1, le=100, description="Stage 2 reranked results")
    filters: Optional[ProductFilter] = Field(default=None, description="Optional attribute/price filters")
    enable_reranking: bool = Field(default=True, description="Enable 2-stage Cross-Encoder reranking")
    enable_explanation: bool = Field(default=False, description="Generate LLM rationale for top results")
    ranking_strategy: str = Field(default="cross_encoder", description="Ranking strategy: cross_encoder | hybrid | dense_only")


class SearchResponse(BaseModel):
    """Search endpoint output schema."""

    query: str
    query_understanding: QueryUnderstandingResult
    total_retrieved: int
    total_returned: int
    results: List[SearchResultItem]
    timings: PipelineStageTiming
