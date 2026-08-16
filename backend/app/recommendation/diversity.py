"""Diversity-aware recommendation reranking via Maximal Marginal Relevance (MMR)."""

import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np

from backend.app.recommendation.base import RecommendationCandidate

logger = logging.getLogger(__name__)


class MMRReranker:
    """Maximal Marginal Relevance (MMR) diversity reranker.
    
    Objective:
        MMR(d) = lambda * Relevance(d) - (1 - lambda) * max_{s in S} Sim(d, s)
        
    Where:
        Relevance(d) = normalized candidate score in [0, 1]
        S = set of already selected items in the top-K list
        Sim(d, s) = cosine similarity between product embedding vectors e_d and e_s
        lambda in [0.0, 1.0] controls the relevance vs diversity trade-off:
            lambda = 1.0 -> pure relevance (no diversity penalty)
            lambda = 0.0 -> maximum diversity
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
        default_lambda: float = 0.7,
    ) -> None:
        self.default_lambda = default_lambda
        self.embeddings: Optional[np.ndarray] = None
        self.doc_to_idx: Dict[str, int] = {}

        if embeddings is not None and doc_ids is not None:
            self.set_embeddings(embeddings, doc_ids)

    def set_embeddings(self, embeddings: np.ndarray, doc_ids: List[str]) -> None:
        """Register product embeddings for fast vector cosine similarity computation."""
        if len(embeddings) != len(doc_ids):
            raise ValueError("Embeddings and doc_ids length mismatch.")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.doc_to_idx = {str(d): idx for idx, d in enumerate(doc_ids)}

    def rerank(
        self,
        candidates: List[RecommendationCandidate],
        top_k: int = 10,
        lambda_param: Optional[float] = None,
    ) -> List[RecommendationCandidate]:
        """Rerank candidates using greedy MMR optimization.
        
        Args:
            candidates: Initial pool of ranked candidate items.
            top_k: Number of diverse items to select.
            lambda_param: Diversity weight (defaults to self.default_lambda).
            
        Returns:
            Diversified top_k list of RecommendationCandidate.
        """
        if not candidates:
            return []

        if top_k <= 0:
            return []

        lmbda = self.default_lambda if lambda_param is None else float(lambda_param)
        lmbda = max(0.0, min(1.0, lmbda))

        # If lambda == 1.0 or only 1 item requested, no diversity reranking needed
        if (lmbda == 1.0 or len(candidates) <= 1) and len(candidates) <= top_k:
            return candidates[:top_k]

        # Filter candidate vectors
        cand_indices: List[Optional[int]] = []
        cand_scores: List[float] = []

        # Min-Max normalize input candidate relevance scores if needed
        raw_scores = [c.score for c in candidates]
        max_s = max(raw_scores)
        min_s = min(raw_scores)
        s_range = max(max_s - min_s, 1e-8)

        for c in candidates:
            idx = self.doc_to_idx.get(c.product_id)
            cand_indices.append(idx)
            norm_rel = (c.score - min_s) / s_range if max_s > min_s else 1.0
            cand_scores.append(norm_rel)

        selected_candidates: List[RecommendationCandidate] = []
        selected_set_indices: List[int] = []  # Indices into self.embeddings
        remaining_indices = list(range(len(candidates)))

        # 1. First item: Item with highest relevance score
        first_idx = int(np.argmax(cand_scores))
        first_cand = candidates[first_idx]
        first_emb_idx = cand_indices[first_idx]

        first_cand.signals["mmr_score"] = float(cand_scores[first_idx])
        first_cand.signals["max_intra_similarity"] = 0.0
        first_cand.rank = 1
        first_cand.recommendation_type = "hybrid_mmr"

        selected_candidates.append(first_cand)
        if first_emb_idx is not None:
            selected_set_indices.append(first_emb_idx)
        remaining_indices.remove(first_idx)

        # 2. Greedy iterative selection for steps 2..top_k
        while len(selected_candidates) < top_k and remaining_indices:
            best_mmr = -float("inf")
            best_idx = remaining_indices[0]
            best_sim_to_s = 0.0

            for r_idx in remaining_indices:
                rel = cand_scores[r_idx]
                emb_idx = cand_indices[r_idx]

                max_sim = 0.0
                if emb_idx is not None and selected_set_indices and self.embeddings is not None:
                    # Cosine similarities to already selected vectors
                    cand_vec = self.embeddings[emb_idx]
                    sel_vecs = self.embeddings[selected_set_indices]
                    sims = np.dot(sel_vecs, cand_vec)
                    max_sim = float(np.max(sims))
                elif selected_set_indices:
                    # Fallback if embedding not found: check category overlap as proxy
                    c_meta = candidates[r_idx].metadata.get("categories", [])
                    sims = [
                        1.0 if set(c_meta).intersection(set(s.metadata.get("categories", []))) else 0.0
                        for s in selected_candidates
                    ]
                    max_sim = max(sims) if sims else 0.0

                mmr_val = (lmbda * rel) - ((1.0 - lmbda) * max_sim)

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = r_idx
                    best_sim_to_s = max_sim

            selected_cand = candidates[best_idx]
            selected_emb_idx = cand_indices[best_idx]

            selected_cand.signals["mmr_score"] = float(best_mmr)
            selected_cand.signals["max_intra_similarity"] = float(best_sim_to_s)
            selected_cand.rank = len(selected_candidates) + 1
            selected_cand.recommendation_type = "hybrid_mmr"

            selected_candidates.append(selected_cand)
            if selected_emb_idx is not None:
                selected_set_indices.append(selected_emb_idx)
            remaining_indices.remove(best_idx)

        return selected_candidates
