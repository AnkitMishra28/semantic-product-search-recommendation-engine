r"""Reciprocal Rank Fusion (RRF) algorithms for hybrid candidate retrieval.

Implements standard rank-based score fusion:
    RRF(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}

Where:
- R is the set of retrieval systems (e.g. ['bm25', 'dense'])
- \text{rank}_r(d) is the 1-indexed rank of document d in retriever r
- k is a smoothing constant (default: 60)

Provenance tracking preserves individual retriever ranks, raw scores,
source attribution, and original metadata.
"""

from typing import Any, Dict, List, Optional, Sequence
from backend.app.retrieval.base import CandidateResult, FusedCandidateResult

DEFAULT_RRF_K = 60


def compute_rrf_score(ranks: Sequence[int], k: int = DEFAULT_RRF_K) -> float:
    """Compute Reciprocal Rank Fusion score for a list of 1-indexed ranks.
    
    Args:
        ranks: Collection of 1-indexed rank positions.
        k: Smoothing constant (default 60).
        
    Returns:
        Sum of 1 / (k + rank) across all provided ranks.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}")
    return sum(1.0 / (k + float(r)) for r in ranks)


def reciprocal_rank_fusion(
    candidate_rankings: Dict[str, Sequence[CandidateResult]],
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[FusedCandidateResult]:
    """Fuse multiple candidate result sets using standard Reciprocal Rank Fusion.
    
    Args:
        candidate_rankings: Mapping of retriever name -> list of CandidateResult items.
                            e.g. {"bm25": bm25_candidates, "dense": dense_candidates}
        k: Configurable RRF smoothing constant (default: 60).
        top_k: Optional maximum number of fused candidate results to return.
        
    Returns:
        List of FusedCandidateResult sorted in descending order of RRF score.
    """
    if k <= 0:
        raise ValueError(f"RRF smoothing constant k must be positive, got {k}")

    if not candidate_rankings:
        return []

    # Map doc_id -> intermediate fusion accumulator
    # doc_id -> {
    #   "ranks": {retriever_name: rank},
    #   "scores": {retriever_name: score},
    #   "metadata": metadata_dict,
    #   "first_seen_order": int
    # }
    doc_map: Dict[str, Dict[str, Any]] = {}
    insertion_counter = 0

    for retriever_name, candidates in candidate_rankings.items():
        for idx, item in enumerate(candidates, start=1):
            doc_id = str(item.doc_id)
            if not doc_id:
                continue

            item_rank = getattr(item, "rank", None)
            if item_rank is None or item_rank <= 0:
                item_rank = idx

            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "ranks": {},
                    "scores": {},
                    "metadata": dict(item.metadata) if item.metadata else {},
                    "first_seen_order": insertion_counter,
                }
                insertion_counter += 1
            else:
                # Merge metadata if richer
                if item.metadata:
                    for m_k, m_v in item.metadata.items():
                        if m_k not in doc_map[doc_id]["metadata"] or doc_map[doc_id]["metadata"][m_k] is None:
                            doc_map[doc_id]["metadata"][m_k] = m_v

            # Record rank and score for this retriever
            # If doc appears multiple times in same retriever list, record best (lowest) rank
            existing_rank = doc_map[doc_id]["ranks"].get(retriever_name)
            if existing_rank is None or item_rank < existing_rank:
                doc_map[doc_id]["ranks"][retriever_name] = item_rank
                doc_map[doc_id]["scores"][retriever_name] = float(item.score)

    if not doc_map:
        return []

    # Compute RRF score and build FusedCandidateResult objects
    fused_candidates: List[FusedCandidateResult] = []

    for doc_id, doc_info in doc_map.items():
        ranks_dict = doc_info["ranks"]
        scores_dict = doc_info["scores"]
        
        # Calculate sum of 1 / (k + rank)
        rrf_score = sum(1.0 / (k + float(r)) for r in ranks_dict.values())

        bm25_rank = ranks_dict.get("bm25")
        dense_rank = ranks_dict.get("dense")
        bm25_score = scores_dict.get("bm25")
        dense_score = scores_dict.get("dense")

        retrieved_by = sorted(list(ranks_dict.keys()))

        fused_candidates.append(
            FusedCandidateResult(
                doc_id=doc_id,
                score=rrf_score,
                rank=0,  # assigned after sorting
                rrf_score=rrf_score,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                bm25_score=bm25_score,
                dense_score=dense_score,
                retrieved_by=retrieved_by,
                metadata=doc_info["metadata"],
            )
        )

    # Sort descending by RRF score
    # Secondary tie-breaker: sum of ranks ascending (lower total rank sum is better)
    # Tertiary tie-breaker: doc_id stable alphabetical
    def sort_key(c: FusedCandidateResult) -> Any:
        ranks_sum = (c.bm25_rank or 999999) + (c.dense_rank or 999999)
        return (-c.rrf_score, ranks_sum, c.doc_id)

    fused_candidates.sort(key=sort_key)

    # Assign 1-indexed rank
    for idx, candidate in enumerate(fused_candidates, start=1):
        candidate.rank = idx

    if top_k is not None and top_k > 0:
        return fused_candidates[:top_k]

    return fused_candidates


def calculate_candidate_overlap(
    candidate_rankings: Dict[str, Sequence[CandidateResult]],
) -> Dict[str, Any]:
    """Calculate overlap statistics between multiple candidate result sets.
    
    Args:
        candidate_rankings: Mapping of retriever name -> list of CandidateResult items.
        
    Returns:
        Dictionary with union count, intersection count, per-retriever counts, and Jaccard overlap.
    """
    id_sets: Dict[str, set] = {
        name: {item.doc_id for item in candidates}
        for name, candidates in candidate_rankings.items()
    }

    all_ids = set()
    for s in id_sets.values():
        all_ids.update(s)

    union_count = len(all_ids)

    # Compute pairwise and total intersection
    if len(id_sets) >= 2:
        keys = list(id_sets.keys())
        intersection_set = id_sets[keys[0]].intersection(*[id_sets[k] for k in keys[1:]])
        intersection_count = len(intersection_set)
        jaccard = intersection_count / union_count if union_count > 0 else 0.0
    else:
        intersection_count = union_count
        jaccard = 1.0

    per_retriever_counts = {name: len(s) for name, s in id_sets.items()}

    return {
        "union_count": union_count,
        "intersection_count": intersection_count,
        "jaccard_similarity": jaccard,
        "per_retriever_counts": per_retriever_counts,
    }
