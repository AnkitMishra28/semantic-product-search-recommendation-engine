"""Abstract interfaces for second-stage rerankers and hybrid ranking algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.models.product import Product


class RankedCandidate(BaseModel):
    """Product candidate enriched with ranking scores."""

    doc_id: str = Field(default="", description="Document / Product unique identifier (ASIN)")
    product_id: Optional[str] = Field(default=None, description="Alias for doc_id (ASIN)")
    score: float = Field(..., description="Computed relevance score")
    cross_encoder_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score")
    rank: int = Field(..., description="1-indexed rank after scoring")
    final_rank: Optional[int] = Field(default=None, description="1-indexed final rank")
    product: Optional[Product] = Field(default=None, description="Product model if available")
    first_stage_score: Optional[float] = Field(default=None, description="Score from first-stage retrieval")
    original_retrieval_score: Optional[float] = Field(default=None, description="Alias for first_stage_score")
    first_stage_rank: Optional[int] = Field(default=None, description="Rank from first-stage retrieval")
    original_rank: Optional[int] = Field(default=None, description="Alias for first_stage_rank")
    features: Dict[str, Any] = Field(default_factory=dict, description="Feature signals used in scoring")

    def model_post_init(self, __context: Any) -> None:
        if not self.doc_id and self.product_id:
            self.doc_id = self.product_id
        elif not self.doc_id and self.product is not None:
            self.doc_id = getattr(self.product, "parent_asin", "") or getattr(self.product, "asin", "")
        if self.product_id is None and self.doc_id:
            self.product_id = self.doc_id

        if self.cross_encoder_score is None:
            self.cross_encoder_score = self.score
        if self.final_rank is None:
            self.final_rank = self.rank

        if self.original_retrieval_score is None and self.first_stage_score is not None:
            self.original_retrieval_score = self.first_stage_score
        elif self.first_stage_score is None and self.original_retrieval_score is not None:
            self.first_stage_score = self.original_retrieval_score

        if self.original_rank is None and self.first_stage_rank is not None:
            self.original_rank = self.first_stage_rank
        elif self.first_stage_rank is None and self.original_rank is not None:
            self.first_stage_rank = self.original_rank



class BaseReranker(ABC):
    """Abstract interface for deep neural cross-encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        products: List[Product],
        top_k: int = 20,
        candidate_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Score (query, product) pairs using cross-attention model.

        Args:
            query: The search query string.
            products: List of candidate products from first-stage retrieval.
            top_k: Number of top-ranked products to return.
            candidate_k: Optional candidate budget cap to score.

        Returns:
            List of RankedCandidate sorted descending by neural relevance score.
        """
        pass


class BaseRanker(ABC):
    """Abstract interface for multi-signal / hybrid ranking strategies."""

    @abstractmethod
    def rank(
        self,
        query: str,
        candidates: List[RankedCandidate],
        top_k: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RankedCandidate]:
        """Apply business rules, diversity, and feature combinations to ranked candidates."""
        pass
