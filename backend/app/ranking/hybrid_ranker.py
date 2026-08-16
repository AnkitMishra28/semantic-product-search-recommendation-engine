"""Hybrid ranker combining neural semantic relevance with product business signals."""

import math
from typing import Any, Dict, List, Optional
from backend.app.ranking.base import BaseRanker, RankedCandidate


class HybridRanker(BaseRanker):
    """Blends semantic cross-encoder relevance with commercial quality signals (ratings, review volume)."""

    def __init__(
        self,
        relevance_weight: float = 0.7,
        rating_weight: float = 0.2,
        popularity_weight: float = 0.1,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.rating_weight = rating_weight
        self.popularity_weight = popularity_weight

    def rank(
        self,
        query: str,
        candidates: List[RankedCandidate],
        top_k: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RankedCandidate]:
        """Compute hybrid composite score for each candidate."""
        if not candidates:
            return []

        scored_candidates: List[RankedCandidate] = []

        for cand in candidates:
            p = cand.product
            # Normalized neural score
            rel_score = cand.score

            # Normalized rating score (0 to 1)
            rating_score = (p.average_rating / 5.0) if p.average_rating else 0.5

            # Log-scaled popularity score (capped at 1.0)
            pop_score = math.log1p(p.rating_count) / 10.0 if p.rating_count > 0 else 0.0
            pop_score = min(pop_score, 1.0)

            # Composite linear combination
            final_score = (
                self.relevance_weight * rel_score
                + self.rating_weight * rating_score
                + self.popularity_weight * pop_score
            )

            scored_candidates.append(
                RankedCandidate(
                    product=p,
                    score=float(final_score),
                    rank=0,
                    features={
                        "raw_semantic_score": rel_score,
                        "rating_signal": rating_score,
                        "popularity_signal": pop_score,
                        "composite_score": final_score,
                    },
                )
            )

        # Sort descending by hybrid composite score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        for idx, item in enumerate(scored_candidates, start=1):
            item.rank = idx

        return scored_candidates[:top_k]
