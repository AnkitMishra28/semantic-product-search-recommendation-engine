"""Retrieval module for vector and candidate search."""

from backend.app.retrieval.base import (
    BaseRetriever,
    CandidateResult,
    FusedCandidateResult,
    HybridRetrievalResult,
)
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.embeddings import (
    EmbeddingService,
    ExactDenseRetriever,
    ExactRetriever,
)
from backend.app.retrieval.faiss_retriever import FaissRetriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.rrf import (
    DEFAULT_RRF_K,
    calculate_candidate_overlap,
    compute_rrf_score,
    reciprocal_rank_fusion,
)
from backend.app.retrieval.tokenizer import tokenize_lexical

__all__ = [
    "BaseRetriever",
    "CandidateResult",
    "FusedCandidateResult",
    "HybridRetrievalResult",
    "FaissRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "tokenize_lexical",
    "EmbeddingService",
    "ExactDenseRetriever",
    "ExactRetriever",
    "reciprocal_rank_fusion",
    "compute_rrf_score",
    "calculate_candidate_overlap",
    "DEFAULT_RRF_K",
]
