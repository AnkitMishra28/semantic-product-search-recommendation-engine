"""Automated data validation and integrity suite."""

import json
from typing import Any, Dict, List, Set, Tuple
import pandas as pd


class ValidationError(Exception):
    """Raised when dataset integrity or schema constraints are violated."""
    pass


def validate_products_catalog(products_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate processed products DataFrame against structural and quality constraints."""
    if products_df.empty:
        raise ValidationError("Products DataFrame is empty.")
        
    required_cols = {"parent_asin", "title", "categories", "features", "embedding_text"}
    missing_cols = required_cols - set(products_df.columns)
    if missing_cols:
        raise ValidationError(f"Products catalog is missing mandatory columns: {missing_cols}")

    # Check 1: Unique product IDs
    total_rows = len(products_df)
    unique_asins = products_df["parent_asin"].nunique()
    if total_rows != unique_asins:
        raise ValidationError(f"Duplicate parent_asins detected! Total rows: {total_rows}, Unique ASINs: {unique_asins}")

    # Check 2: Non-empty titles
    empty_titles = products_df["title"].isna() | (products_df["title"].astype(str).str.strip() == "")
    if empty_titles.any():
        bad_count = empty_titles.sum()
        raise ValidationError(f"Found {bad_count} products with empty or null titles.")

    # Check 3: Valid prices
    if "price" in products_df.columns:
        invalid_prices = products_df["price"].notna() & (
            (products_df["price"] <= 0) | (products_df["price"] > 100000)
        )
        if invalid_prices.any():
            bad_count = invalid_prices.sum()
            raise ValidationError(f"Found {bad_count} products with invalid prices (< 0 or > $100k).")

    # Check 4: Valid ratings
    if "average_rating" in products_df.columns:
        invalid_ratings = products_df["average_rating"].notna() & (
            (products_df["average_rating"] < 1.0) | (products_df["average_rating"] > 5.0)
        )
        if invalid_ratings.any():
            bad_count = invalid_ratings.sum()
            raise ValidationError(f"Found {bad_count} products with ratings outside [1.0, 5.0].")

    # Check 5: Non-empty embedding text
    empty_emb = products_df["embedding_text"].isna() | (products_df["embedding_text"].astype(str).str.strip() == "")
    if empty_emb.any():
        bad_count = empty_emb.sum()
        raise ValidationError(f"Found {bad_count} products with empty embedding_text.")

    return {
        "status": "PASSED",
        "total_products": total_rows,
        "unique_parent_asins": unique_asins,
    }


def validate_interactions(
    interactions_df: pd.DataFrame,
    catalog_asins: Set[str],
) -> Dict[str, Any]:
    """Validate processed interactions DataFrame and check referential integrity with products catalog."""
    if interactions_df.empty:
        raise ValidationError("Interactions DataFrame is empty.")

    required_cols = {"user_id", "parent_asin", "rating", "timestamp", "split"}
    missing_cols = required_cols - set(interactions_df.columns)
    if missing_cols:
        raise ValidationError(f"Interactions DataFrame is missing mandatory columns: {missing_cols}")

    # Check 1: Valid user IDs
    empty_users = interactions_df["user_id"].isna() | (interactions_df["user_id"].astype(str).str.strip() == "")
    if empty_users.any():
        raise ValidationError(f"Found {empty_users.sum()} interactions with empty user_id.")

    # Check 2: Valid rating values (1.0 to 5.0)
    invalid_ratings = (interactions_df["rating"] < 1.0) | (interactions_df["rating"] > 5.0)
    if invalid_ratings.any():
        raise ValidationError(f"Found {invalid_ratings.sum()} interactions with invalid rating values.")

    # Check 3: Valid positive timestamps
    invalid_ts = interactions_df["timestamp"] <= 0
    if invalid_ts.any():
        raise ValidationError(f"Found {invalid_ts.sum()} interactions with invalid non-positive timestamp.")

    # Check 4: Referential integrity (no orphaned product references)
    interaction_asins = set(interactions_df["parent_asin"].unique())
    orphaned_asins = interaction_asins - catalog_asins
    if orphaned_asins:
        raise ValidationError(f"Referential integrity failure: {len(orphaned_asins)} interacted ASINs not in product catalog.")

    # Check 5: Valid split values
    valid_splits = {"train", "val", "test"}
    invalid_splits = set(interactions_df["split"].unique()) - valid_splits
    if invalid_splits:
        raise ValidationError(f"Invalid split partition labels found: {invalid_splits}")

    return {
        "status": "PASSED",
        "total_interactions": len(interactions_df),
        "unique_users": int(interactions_df["user_id"].nunique()),
        "unique_products": int(interactions_df["parent_asin"].nunique()),
        "splits": interactions_df["split"].value_counts().to_dict(),
    }


def validate_evaluation_queries(
    eval_queries_path: str,
    catalog_asins: Set[str],
) -> Dict[str, Any]:
    """Validate ground truth evaluation queries schema and referential integrity against product catalog."""
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not queries or not isinstance(queries, list):
        raise ValidationError("Evaluation queries file is empty or not a list.")

    seen_qids = set()
    for q in queries:
        qid = q.get("query_id")
        if not qid or qid in seen_qids:
            raise ValidationError(f"Duplicate or empty query_id: {qid}")
        seen_qids.add(qid)

        qtext = q.get("query")
        if not qtext or not isinstance(qtext, str) or len(qtext.strip()) == 0:
            raise ValidationError(f"Empty query text in query {qid}")

        rel_ids = q.get("relevant_product_ids")
        if not rel_ids or not isinstance(rel_ids, list) or len(rel_ids) == 0:
            raise ValidationError(f"Query {qid} has no relevant_product_ids.")

        # Check uniqueness of relevant_product_ids within query
        if len(rel_ids) != len(set(rel_ids)):
            raise ValidationError(f"Query {qid} contains duplicate relevant product IDs.")

        # Check every relevant ID exists in catalog
        for pid in rel_ids:
            if pid not in catalog_asins:
                raise ValidationError(f"Query {qid} references unknown product ID '{pid}' not in catalog.")

    return {
        "status": "PASSED",
        "total_queries": len(queries),
        "query_ids": list(seen_qids),
    }
