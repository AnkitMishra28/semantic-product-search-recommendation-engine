"""Information Retrieval (IR) Evaluation Metrics and Latency Profiling.

Implements standard ranking and retrieval evaluation formulas:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)
- Latency percentile tracking
"""

import math
import time
from typing import Dict, List, Optional, Sequence
import numpy as np


class LatencyTracker:
    """Utility to track and compute percentile latency statistics for queries."""

    def __init__(self) -> None:
        self.latencies_ms: List[float] = []

    def record(self, latency_ms: float) -> None:
        """Record a single query latency in milliseconds."""
        self.latencies_ms.append(latency_ms)

    def summary(self) -> Dict[str, float]:
        """Compute latency summary statistics."""
        if not self.latencies_ms:
            return {
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "total_queries": 0,
            }

        arr = np.array(self.latencies_ms)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "total_queries": len(arr),
        }


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Compute Recall@K for a single query.

    Recall@K = |{retrieved in top K} ∩ {relevant}| / |{relevant}|
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Compute Precision@K for a single query.

    Precision@K = |{retrieved in top K} ∩ {relevant}| / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def reciprocal_rank_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Compute Reciprocal Rank (RR@K) for a single query.

    RR@K = 1 / rank of the first relevant document within top K, or 0.0 if not found.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at K (DCG@K).

    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log_2(i + 1)
    """
    scores = relevance_scores[:k]
    dcg = 0.0
    for idx, rel in enumerate(scores, start=1):
        if rel > 0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 1.0)
    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    graded_relevance: Dict[str, float],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K).

    NDCG@K = DCG@K / IDCG@K
    """
    if not graded_relevance or k <= 0:
        return 0.0

    # Actual ranking relevance
    actual_rels = [graded_relevance.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(actual_rels, k)

    # Ideal ranking relevance
    ideal_rels = sorted(graded_relevance.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_rels, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Compute Average Precision (AP) for a single query."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = 0
    sum_precisions = 0.0
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / idx
    return sum_precisions / len(relevant_set)
