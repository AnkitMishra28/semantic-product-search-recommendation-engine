"""Hybrid candidate retriever combining Lexical (BM25) and Dense (FAISS) retrieval via Reciprocal Rank Fusion."""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.app.query_understanding.base import BaseQueryProcessor
from backend.app.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.retrieval.base import (
    BaseRetriever,
    CandidateResult,
    FusedCandidateResult,
    HybridRetrievalResult,
)
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.faiss_retriever import FaissRetriever
from backend.app.retrieval.rrf import (
    DEFAULT_RRF_K,
    calculate_candidate_overlap,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates first-stage candidate generation by querying both BM25 and FAISS HNSW
    indexes, then fusing their ranked lists using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: FaissRetriever,
        rrf_k: int = DEFAULT_RRF_K,
        default_top_k: int = 100,
        query_processor: Optional[BaseQueryProcessor] = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.default_top_k = default_top_k
        self.query_processor = query_processor or QueryUnderstandingPipeline()

    def search_hybrid(
        self,
        query_text: str,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        candidate_k: int = 100,
        rrf_k: Optional[int] = None,
    ) -> HybridRetrievalResult:
        """Execute parallel lexical and dense candidate retrieval followed by RRF fusion.

        Args:
            query_text: Raw or normalized search query string.
            top_k: Number of fused candidates to return for downstream reranking.
            filters: Optional structured metadata filters (brand, category, price_min, price_max).
            candidate_k: Number of candidates to retrieve from EACH first-stage retriever (default: 100).
            rrf_k: Optional override for the RRF smoothing constant (default: 60).

        Returns:
            HybridRetrievalResult with fused candidate list and full provenance metrics.
        """
        k_const = rrf_k if rrf_k is not None else self.rrf_k

        # 1. Lexical BM25 Retrieval
        t_bm0 = time.perf_counter()
        bm25_candidates = self.bm25_retriever.search_text(
            query=query_text,
            top_k=candidate_k,
            filters=filters,
        )
        t_bm1 = time.perf_counter()
        bm25_ms = (t_bm1 - t_bm0) * 1000.0

        # 2. Dense FAISS Retrieval
        t_dn0 = time.perf_counter()
        if self.dense_retriever.embedding_service is not None:
            dense_candidates, enc_ms, search_ms = self.dense_retriever.search_query(
                query_text=query_text,
                top_k=candidate_k,
                filters=filters,
            )
            dense_ms = enc_ms + search_ms
        else:
            dense_candidates = []
            dense_ms = 0.0
        t_dn1 = time.perf_counter()

        # 3. Reciprocal Rank Fusion
        candidate_rankings = {
            "bm25": bm25_candidates,
            "dense": dense_candidates,
        }

        t_rf0 = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(
            candidate_rankings=candidate_rankings,
            k=k_const,
            top_k=top_k,
        )
        t_rf1 = time.perf_counter()
        fusion_ms = (t_rf1 - t_rf0) * 1000.0

        # Overlap analysis
        overlap_stats = calculate_candidate_overlap(candidate_rankings)
        union_count = overlap_stats["union_count"]
        intersection_count = overlap_stats["intersection_count"]

        total_first_stage_ms = bm25_ms + dense_ms + fusion_ms

        timings = {
            "bm25_latency_ms": float(round(bm25_ms, 3)),
            "dense_latency_ms": float(round(dense_ms, 3)),
            "fusion_latency_ms": float(round(fusion_ms, 3)),
            "total_latency_ms": float(round(total_first_stage_ms, 3)),
        }

        return HybridRetrievalResult(
            candidates=fused_candidates,
            candidate_count_before_fusion=union_count,
            candidate_count_after_fusion=len(fused_candidates),
            bm25_count=len(bm25_candidates),
            dense_count=len(dense_candidates),
            overlap_count=intersection_count,
            timings=timings,
        )

    def search_with_query_understanding(
        self,
        query_text: str,
        top_k: int = 100,
        candidate_k: int = 100,
        rrf_k: Optional[int] = None,
        override_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[HybridRetrievalResult, Any]:
        """Process query through Query Understanding pipeline, extract deterministic hard filters,
        and execute hybrid candidate retrieval.

        Args:
            query_text: Raw user search query.
            top_k: Final top-K candidates to return.
            candidate_k: Candidates per individual retriever.
            rrf_k: RRF smoothing constant.
            override_filters: Optional external filters to merge with extracted hard filters.

        Returns:
            Tuple of (HybridRetrievalResult, QueryUnderstandingResult).
        """
        # Run Query Understanding
        qu_result = self.query_processor.process(query_text)

        # Merge extracted hard filters with any manual override filters
        effective_filters = dict(qu_result.hard_filters) if qu_result.hard_filters else {}
        if override_filters:
            effective_filters.update(override_filters)

        # Execute hybrid search with effective hard filters
        hybrid_res = self.search_hybrid(
            query_text=qu_result.normalized_query or query_text,
            top_k=top_k,
            filters=effective_filters if effective_filters else None,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )

        return hybrid_res, qu_result
