"""Item-Item Collaborative Filtering recommendation baseline using sparse co-occurrence graphs."""

from collections import defaultdict
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import pandas as pd

from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate

logger = logging.getLogger(__name__)


class CollaborativeRecommender(BaseRecommender):
    """Sparse Item-Item Collaborative Filtering recommender.
    
    Computes normalized cosine co-occurrence similarity:
        Sim(i, j) = C_{i, j} / sqrt(C_{i, i} * C_{j, j})
        
    User recommendation scoring:
        Score(u, j) = sum_{i in H_u} w(i) * Sim(i, j)
        
    Where:
        C_{i, j} = count of distinct historical users who interacted with both items i and j
        w(i) = user interaction weight (rating / 5.0)
    """

    def __init__(
        self,
        min_support: int = 1,
        max_neighbors_per_item: int = 100,
        product_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.min_support = min_support
        self.max_neighbors_per_item = max_neighbors_per_item
        self.catalog = product_catalog or {}

        # item_id -> {neighbor_item_id: similarity_score}
        self.item_similarities: Dict[str, Dict[str, float]] = {}
        # item_id -> interaction count (support)
        self.item_support: Dict[str, int] = {}
        # user_id -> set of interacted items (for historical lookup)
        self.user_history_map: Dict[str, List[Tuple[str, float, int]]] = {}

    def fit(
        self,
        interactions_df: pd.DataFrame,
        product_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "CollaborativeRecommender":
        """Build item-item co-occurrence graph strictly from historical interactions.
        
        Args:
            interactions_df: DataFrame with ['user_id', 'parent_asin', 'rating', 'timestamp'].
            product_catalog: Optional catalog metadata mapping parent_asin -> metadata dict.
        """
        if product_catalog is not None:
            self.catalog = product_catalog

        if interactions_df.empty:
            logger.warning("Empty interactions DataFrame passed to CollaborativeRecommender.fit()")
            return self

        # 1. Group interactions by user to build user baskets
        user_baskets: Dict[str, Set[str]] = defaultdict(set)
        self.user_history_map = defaultdict(list)
        item_user_counts: Dict[str, int] = defaultdict(int)

        for _, row in interactions_df.iterrows():
            u = str(row["user_id"])
            item = str(row["parent_asin"])
            rating = float(row.get("rating", 5.0) or 5.0)
            ts = int(row.get("timestamp", 0) or 0)

            user_baskets[u].add(item)
            self.user_history_map[u].append((item, rating, ts))

        for items in user_baskets.values():
            for item in items:
                item_user_counts[item] += 1

        self.item_support = dict(item_user_counts)

        # 2. Count pairwise co-occurrences: C_{i, j}
        co_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for items in user_baskets.values():
            item_list = list(items)
            n_items = len(item_list)
            if n_items < 2:
                continue

            for i in range(n_items):
                item_a = item_list[i]
                for j in range(i + 1, n_items):
                    item_b = item_list[j]
                    co_counts[item_a][item_b] += 1
                    co_counts[item_b][item_a] += 1

        # 3. Calculate cosine similarity and prune to top neighbors
        self.item_similarities = {}
        total_edges = 0

        for item_a, neighbors in co_counts.items():
            count_a = item_user_counts.get(item_a, 0)
            if count_a == 0:
                continue

            sim_list: List[Tuple[str, float]] = []
            for item_b, co_val in neighbors.items():
                if co_val < self.min_support:
                    continue

                count_b = item_user_counts.get(item_b, 0)
                if count_b == 0:
                    continue

                # Cosine similarity: co_val / sqrt(count_a * count_b)
                sim = float(co_val) / math.sqrt(float(count_a * count_b))
                sim_list.append((item_b, sim))

            # Prune to top-K neighbors
            sim_list.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = sim_list[: self.max_neighbors_per_item]

            if top_neighbors:
                self.item_similarities[item_a] = dict(top_neighbors)
                total_edges += len(top_neighbors)

        sparsity = 1.0 - (total_edges / max(1, len(item_user_counts) ** 2))
        logger.info(
            f"CollaborativeRecommender fitted on {len(interactions_df)} interactions. "
            f"Co-occurrence graph: {len(self.item_similarities)} items with neighbors, "
            f"{total_edges} sparse edges (sparsity: {sparsity:.6f})."
        )
        return self

    def _matches_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if an item's metadata satisfies hard constraints."""
        if not filters:
            return True

        if "category" in filters and filters["category"]:
            target_cat = str(filters["category"]).strip().lower()
            cats = [str(c).strip().lower() for c in metadata.get("categories", [])]
            if not any(target_cat in c or c in target_cat for c in cats):
                return False

        if "brand" in filters and filters["brand"]:
            target_brand = str(filters["brand"]).strip().lower()
            item_brand = str(metadata.get("brand", "")).strip().lower()
            if target_brand != item_brand and target_brand not in item_brand:
                return False

        price = metadata.get("price")
        if "price_min" in filters and filters["price_min"] is not None:
            if price is None or float(price) < float(filters["price_min"]):
                return False

        if "price_max" in filters and filters["price_max"] is not None:
            if price is None or float(price) > float(filters["price_max"]):
                return False

        return True

    def recommend(
        self,
        user_id: Optional[str] = None,
        history_asins: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_consumed: bool = True,
    ) -> List[RecommendationCandidate]:
        """Aggregate collaborative co-occurrence scores across user's past items."""
        # 1. Resolve user history
        user_items: List[Tuple[str, float]] = []
        seen_asins: Set[str] = set()

        if history_asins:
            for asin in history_asins:
                user_items.append((asin, 1.0))
                seen_asins.add(asin)
        elif user_id and user_id in self.user_history_map:
            for item, rating, _ in self.user_history_map[user_id]:
                weight = max(0.1, float(rating) / 5.0)
                user_items.append((item, weight))
                seen_asins.add(item)

        if not user_items:
            # Cold-start / no collaborative history
            return []

        # 2. Accumulate collaborative affinity scores for candidate items
        candidate_scores: Dict[str, float] = defaultdict(float)
        candidate_anchor_sources: Dict[str, str] = {}

        for anchor_asin, weight in user_items:
            neighbors = self.item_similarities.get(anchor_asin, {})
            for neighbor_asin, sim in neighbors.items():
                if exclude_consumed and neighbor_asin in seen_asins:
                    continue

                added_score = weight * sim
                candidate_scores[neighbor_asin] += added_score
                if neighbor_asin not in candidate_anchor_sources or sim > candidate_scores.get(neighbor_asin, 0.0):
                    candidate_anchor_sources[neighbor_asin] = anchor_asin

        if not candidate_scores:
            return []

        # 3. Min-Max Normalize scores to [0.0, 1.0]
        max_score = max(candidate_scores.values())
        min_score = min(candidate_scores.values())
        score_range = max(max_score - min_score, 1e-8)

        # Sort candidate items
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[RecommendationCandidate] = []
        for asin, raw_score in sorted_candidates:
            meta = self.catalog.get(asin, {})
            if filters and not self._matches_filters(meta, filters):
                continue

            norm_score = float((raw_score - min_score) / score_range) if max_score > min_score else 1.0
            anchor_src = candidate_anchor_sources.get(asin, "interacted items")
            anchor_title = self.catalog.get(anchor_src, {}).get("title", anchor_src)

            cand = RecommendationCandidate(
                product_id=asin,
                score=norm_score,
                rank=len(results) + 1,
                recommendation_type="collaborative",
                signals={
                    "collaborative": norm_score,
                    "raw_collab_score": raw_score,
                    "rating": float(meta.get("rating", 4.0) or 4.0) / 5.0,
                },
                reasons=[
                    f"Frequently co-interacted with '{anchor_title[:35]}...' by other customers."
                ],
                metadata=meta,
            )
            results.append(cand)
            if len(results) >= top_k:
                break

        return results

    def recommend_for_item(
        self,
        anchor_asin: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RecommendationCandidate]:
        """Produce collaborative neighbor recommendations for an anchor item."""
        return self.recommend(
            history_asins=[anchor_asin],
            top_k=top_k,
            filters=filters,
            exclude_consumed=True,
        )
