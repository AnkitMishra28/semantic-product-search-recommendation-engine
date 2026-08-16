"""Popularity-based recommendation baseline using Bayesian shrinkage rating priors."""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np
import pandas as pd

from backend.app.models.product import Product
from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate

logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """Popularity recommender applying Bayesian mean rating shrinkage and volume scaling.
    
    Formula:
        RawScore(i) = [ (v_i * r_i + m * C) / (v_i + m) ] * log1p(v_i)
        
    Where:
        v_i = historical interaction/review count for item i
        r_i = historical mean rating for item i
        C   = global average rating across all items
        m   = Bayesian shrinkage parameter (minimum confidence pseudo-counts, default 5)
    """

    def __init__(
        self,
        m_prior: float = 5.0,
        product_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.m_prior = m_prior
        self.catalog = product_catalog or {}
        self.item_scores: Dict[str, float] = {}
        self.item_volumes: Dict[str, int] = {}
        self.item_ratings: Dict[str, float] = {}
        self.ranked_item_ids: List[str] = []
        self.global_mean_rating: float = 4.0

    def fit(
        self,
        interactions_df: pd.DataFrame,
        product_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "PopularityRecommender":
        """Compute Bayesian smoothed popularity statistics strictly from historical interactions.
        
        Args:
            interactions_df: Historical interactions with columns ['parent_asin', 'rating'].
            product_catalog: Optional catalog metadata mapping parent_asin -> metadata dict.
        """
        if product_catalog is not None:
            self.catalog = product_catalog

        if interactions_df.empty:
            logger.warning("Empty interactions DataFrame passed to PopularityRecommender.fit()")
            return self

        # Calculate empirical item stats strictly from historical window
        stats = (
            interactions_df.groupby("parent_asin")
            .agg(
                volume=("rating", "count"),
                mean_rating=("rating", "mean"),
            )
            .reset_index()
        )

        self.global_mean_rating = float(interactions_df["rating"].mean()) if not interactions_df.empty else 4.0
        C = self.global_mean_rating
        m = self.m_prior

        raw_scores: Dict[str, float] = {}
        self.item_volumes = {}
        self.item_ratings = {}

        for _, row in stats.iterrows():
            asin = str(row["parent_asin"])
            v = int(row["volume"])
            r = float(row["mean_rating"])
            self.item_volumes[asin] = v
            self.item_ratings[asin] = r

            # Bayesian smoothed rating
            bayesian_rating = (v * r + m * C) / (v + m)
            # Volume scaling using log1p
            raw_score = bayesian_rating * math.log1p(v)
            raw_scores[asin] = raw_score

        # Also incorporate items in catalog with 0 interactions as baseline prior
        for asin in self.catalog.keys():
            if asin not in raw_scores:
                # Default baseline score
                raw_scores[asin] = (m * C / m) * math.log1p(0)  # = 0.0
                self.item_volumes[asin] = 0
                self.item_ratings[asin] = C

        # Min-Max Normalization to [0.0, 1.0]
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        min_score = min(raw_scores.values()) if raw_scores else 0.0
        range_score = max(max_score - min_score, 1e-8)

        self.item_scores = {
            asin: float((score - min_score) / range_score)
            for asin, score in raw_scores.items()
        }

        # Sort ranked list descending by score
        self.ranked_item_ids = sorted(
            self.item_scores.keys(),
            key=lambda x: (self.item_scores[x], self.item_volumes.get(x, 0)),
            reverse=True,
        )

        logger.info(
            f"PopularityRecommender fitted on {len(interactions_df)} interactions. "
            f"Ranked {len(self.ranked_item_ids)} items (Global mean rating: {C:.2f})."
        )
        return self

    def _matches_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if an item's metadata satisfies hard constraints."""
        if not filters:
            return True

        # Category filter
        if "category" in filters and filters["category"]:
            target_cat = str(filters["category"]).strip().lower()
            cats = [str(c).strip().lower() for c in metadata.get("categories", [])]
            if not any(target_cat in c or c in target_cat for c in cats):
                return False

        # Brand filter
        if "brand" in filters and filters["brand"]:
            target_brand = str(filters["brand"]).strip().lower()
            item_brand = str(metadata.get("brand", "")).strip().lower()
            if target_brand != item_brand and target_brand not in item_brand:
                return False

        # Price min filter
        price = metadata.get("price")
        if "price_min" in filters and filters["price_min"] is not None:
            if price is None or float(price) < float(filters["price_min"]):
                return False

        # Price max filter
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
        """Return top popular items satisfying filters and excluding consumed history."""
        seen: Set[str] = set(history_asins or []) if exclude_consumed else set()
        results: List[RecommendationCandidate] = []

        for asin in self.ranked_item_ids:
            if asin in seen:
                continue

            meta = self.catalog.get(asin, {})
            if filters and not self._matches_filters(meta, filters):
                continue

            score = self.item_scores.get(asin, 0.0)
            rating = self.item_ratings.get(asin, self.global_mean_rating)
            vol = self.item_volumes.get(asin, 0)

            cand = RecommendationCandidate(
                product_id=asin,
                score=score,
                rank=len(results) + 1,
                recommendation_type="popularity",
                signals={
                    "popularity": score,
                    "rating": float(rating / 5.0),
                    "volume": float(vol),
                },
                reasons=[
                    f"Popular and highly-rated product ({rating:.1f}/5 from {vol} reviews) in the catalog."
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
        """Recommend popular products in the same category as the anchor item."""
        anchor_meta = self.catalog.get(anchor_asin, {})
        anchor_cats = anchor_meta.get("categories", [])

        merged_filters = dict(filters or {})
        if anchor_cats and "category" not in merged_filters:
            merged_filters["category"] = anchor_cats[0]

        return self.recommend(
            history_asins=[anchor_asin],
            top_k=top_k,
            filters=merged_filters,
            exclude_consumed=True,
        )
