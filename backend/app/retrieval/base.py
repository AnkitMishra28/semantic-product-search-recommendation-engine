"""Abstract base interface for first-stage vector and lexical retrievers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import numpy as np


class CandidateResult(BaseModel):
    """Output candidate from a retrieval index."""

    doc_id: str = Field(..., description="Document / Product unique identifier (ASIN)")
    score: float = Field(..., description="Retrieval similarity score / distance")
    rank: int = Field(..., description="1-indexed rank from first stage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata stored in the index")


class FusedCandidateResult(CandidateResult):
    """Output candidate from hybrid score fusion (e.g. Reciprocal Rank Fusion)."""

    rrf_score: float = Field(..., description="Combined Reciprocal Rank Fusion score")
    bm25_rank: Optional[int] = Field(default=None, description="1-indexed rank in BM25 retrieval, or None if absent")
    dense_rank: Optional[int] = Field(default=None, description="1-indexed rank in Dense FAISS retrieval, or None if absent")
    bm25_score: Optional[float] = Field(default=None, description="Raw BM25 lexical score if retrieved")
    dense_score: Optional[float] = Field(default=None, description="Raw Dense cosine/inner-product score if retrieved")
    retrieved_by: List[str] = Field(default_factory=list, description="Source retrievers that retrieved this document (e.g. ['bm25', 'dense'])")

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Return standardized provenance dictionary for research analysis and diagnostic APIs."""
        return {
            "product_id": self.doc_id,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "rrf_score": self.rrf_score,
            "retrieved_by": self.retrieved_by,
        }


class HybridRetrievalResult(BaseModel):
    """Result of hybrid multi-retriever candidate generation and fusion."""

    candidates: List[FusedCandidateResult] = Field(default_factory=list, description="Rank-fused candidates")
    candidate_count_before_fusion: int = Field(default=0, description="Total unique candidates retrieved across all source retrievers before top-k truncation")
    candidate_count_after_fusion: int = Field(default=0, description="Number of candidates retained after top-k selection")
    bm25_count: int = Field(default=0, description="Count of candidates retrieved by BM25")
    dense_count: int = Field(default=0, description="Count of candidates retrieved by Dense retriever")
    overlap_count: int = Field(default=0, description="Count of candidates retrieved by both BM25 and Dense")
    timings: Dict[str, float] = Field(default_factory=dict, description="Latency breakdown per retrieval and fusion stage in milliseconds")


class BaseRetriever(ABC):
    """Abstract interface defining first-stage candidate retrieval.

    Allows interchangeable implementations (FAISS, Qdrant, Milvus, BM25, etc.)
    without modifying downstream ranking or API layers.
    """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateResult]:
        """Search vector index for nearest neighbors.

        Args:
            query_vector: 1D or 2D numpy array representing the query embedding.
            top_k: Maximum number of candidate items to retrieve.
            filters: Optional metadata filtering dictionary.

        Returns:
            List of CandidateResult sorted in descending order of relevance score.
        """
        pass

    @abstractmethod
    def index(
        self,
        vectors: np.ndarray,
        doc_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add dense vectors and corresponding document IDs to the index.

        Args:
            vectors: 2D numpy array (num_docs, embedding_dim).
            doc_ids: List of document identifier strings matching vectors row-wise.
            metadata: Optional list of metadata dictionaries.
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """Serialize the vector index and ID mappings to disk."""
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Load a serialized vector index and ID mappings from disk."""
        pass

    @property
    @abstractmethod
    def total_documents(self) -> int:
        """Return the count of documents currently indexed."""
        pass
