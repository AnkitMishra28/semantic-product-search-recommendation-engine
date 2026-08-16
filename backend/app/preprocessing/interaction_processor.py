"""Interaction data ingestion, cleaning, referential integrity filtering, and temporal splitting."""

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


def clean_interaction_record(
    raw: Dict[str, Any],
    valid_parent_asins: Set[str],
) -> Optional[Dict[str, Any]]:
    """Clean and validate a single raw Amazon 2023 review/interaction record.
    
    Returns None if:
      - parent_asin is missing or NOT present in the selected product catalog.
      - user_id is missing or empty.
      - rating is missing or outside valid range [1.0, 5.0].
      - timestamp is missing, zero, or non-numeric.
    """
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not parent_asin or not isinstance(parent_asin, str):
        return None
    parent_asin = parent_asin.strip()
    if parent_asin not in valid_parent_asins:
        return None
        
    user_id = raw.get("user_id")
    if not user_id or not isinstance(user_id, str):
        return None
    user_id = user_id.strip()
    if not user_id:
        return None
        
    # Rating validation
    raw_rating = raw.get("rating")
    try:
        rating = float(raw_rating)
        if not (1.0 <= rating <= 5.0):
            return None
    except (ValueError, TypeError):
        return None
        
    # Timestamp validation
    raw_ts = raw.get("timestamp")
    try:
        timestamp = int(raw_ts)
        if timestamp <= 0:
            return None
    except (ValueError, TypeError):
        return None
        
    verified = bool(raw.get("verified_purchase", False))
    helpful_vote = int(raw.get("helpful_vote", 0) or 0)
    
    return {
        "user_id": user_id,
        "parent_asin": parent_asin,
        "rating": rating,
        "timestamp": timestamp,
        "verified_purchase": verified,
        "helpful_vote": helpful_vote,
    }


def process_interactions(
    reviews_iter: Iterator[Dict[str, Any]],
    valid_parent_asins: Set[str],
    max_interactions: Optional[int] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw review records, ensure catalog referential integrity, and apply a temporal split.
    
    Temporal Splitting Methodology:
      Interactions are strictly ordered by chronological timestamp. Quantile cutoffs
      (train_cutoff, val_cutoff) partition the timeline into:
        - Train / History (e.g. earliest 70% of timeline)
        - Validation (e.g. middle 15% of timeline)
        - Test / Future Evaluation (e.g. most recent 15% of timeline)
        
      This temporal cutoff guarantees that historical recommendation training
      never observes future user ratings, preventing temporal leakage.
      
    Args:
        reviews_iter: Iterator yielding raw review dictionaries.
        valid_parent_asins: Set of canonical parent_asins present in products.parquet.
        max_interactions: Optional limit on total cleaned interactions to retain.
        train_ratio: Fraction of temporal history for training (default 0.70).
        val_ratio: Fraction of temporal history for validation (default 0.15).
        test_ratio: Fraction of temporal history for future test evaluation (default 0.15).
        
    Returns:
        Tuple of (interactions_df, split_metadata_dict).
    """
    cleaned_records: List[Dict[str, Any]] = []
    
    for raw in reviews_iter:
        rec = clean_interaction_record(raw, valid_parent_asins)
        if rec is not None:
            cleaned_records.append(rec)
            if max_interactions and len(cleaned_records) >= max_interactions:
                break
                
    if not cleaned_records:
        empty_df = pd.DataFrame(columns=[
            "user_id", "parent_asin", "rating", "timestamp", "verified_purchase", "helpful_vote", "split"
        ])
        return empty_df, {"total_interactions": 0}

    df = pd.DataFrame(cleaned_records)
    
    # Deduplicate (user_id, parent_asin) keeping the most recent interaction
    df = df.sort_values(by=["timestamp"], ascending=True)
    df = df.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")
    
    # Re-sort chronologically
    df = df.sort_values(by=["timestamp"], ascending=True).reset_index(drop=True)
    
    # Compute deterministic temporal quantile cutoffs
    t_train_cutoff = df["timestamp"].quantile(train_ratio)
    t_val_cutoff = df["timestamp"].quantile(train_ratio + val_ratio)
    
    # Assign split labels deterministically
    conditions = [
        df["timestamp"] <= t_train_cutoff,
        (df["timestamp"] > t_train_cutoff) & (df["timestamp"] <= t_val_cutoff),
        df["timestamp"] > t_val_cutoff,
    ]
    choices = ["train", "val", "test"]
    df["split"] = np.select(conditions, choices, default="test")
    
    split_counts = df["split"].value_counts().to_dict()
    
    split_meta = {
        "total_interactions": len(df),
        "unique_users": int(df["user_id"].nunique()),
        "unique_products": int(df["parent_asin"].nunique()),
        "min_timestamp": int(df["timestamp"].min()),
        "max_timestamp": int(df["timestamp"].max()),
        "train_cutoff_timestamp": int(t_train_cutoff),
        "val_cutoff_timestamp": int(t_val_cutoff),
        "split_counts": split_counts,
    }
    
    return df, split_meta
