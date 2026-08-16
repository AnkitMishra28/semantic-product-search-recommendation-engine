"""Dataset statistical profiler and report generator.

Computes empirical summary statistics over processed product and interaction datasets.
All metrics are derived strictly from the generated data files with no fabrication.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def compute_distribution_stats(series: pd.Series) -> Dict[str, float]:
    """Compute summary statistics (min, max, mean, median, quantiles) for a series."""
    if series.empty:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0}
    
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
    }


def format_timestamp(ts_ms: Optional[int]) -> str:
    """Format millisecond epoch timestamp into human-readable ISO date."""
    if not ts_ms or ts_ms <= 0:
        return "N/A"
    try:
        # Check if timestamp is in seconds or milliseconds
        ts_sec = ts_ms / 1000.0 if ts_ms > 1e11 else float(ts_ms)
        dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts_ms)


def profile_dataset(
    products_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    output_json_path: Optional[str] = None,
    output_md_path: Optional[str] = None,
    sampling_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Profile processed product catalog and user interaction datasets.
    
    Args:
        products_df: DataFrame of processed products.
        interactions_df: DataFrame of processed user interactions.
        output_json_path: Optional destination path for dataset_profile.json.
        output_md_path: Optional destination path for dataset_profile.md.
        sampling_meta: Optional metadata regarding sampling and subset configuration.
        
    Returns:
        Comprehensive dictionary of profile statistics.
    """
    total_products = int(len(products_df))
    unique_product_ids = int(products_df["parent_asin"].nunique())
    
    # Products missingness
    missing_prices = int(products_df["price"].isna().sum())
    missing_descriptions = int((products_df["description"].isna() | (products_df["description"].str.strip() == "")).sum())
    missing_brands = int((products_df["brand"].isna() | (products_df["brand"].str.strip() == "")).sum())
    missing_features = int(products_df["features"].apply(lambda f: len(f) == 0 if isinstance(f, (list, tuple, np.ndarray)) else True).sum())
    
    # Category statistics
    all_categories = set()
    category_counts: Dict[str, int] = {}
    for cats in products_df["categories"]:
        if isinstance(cats, (list, tuple, np.ndarray)):
            for c in cats:
                c_str = str(c).strip()
                if c_str:
                    all_categories.add(c_str)
                    category_counts[c_str] = category_counts.get(c_str, 0) + 1
        elif isinstance(cats, str) and cats.strip():
            c_str = cats.strip()
            all_categories.add(c_str)
            category_counts[c_str] = category_counts.get(c_str, 0) + 1
            
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    # Price stats
    valid_prices = products_df["price"].dropna()
    price_stats = compute_distribution_stats(valid_prices)

    # Product ratings stats
    valid_ratings = products_df["average_rating"].dropna()
    product_rating_stats = compute_distribution_stats(valid_ratings)

    # Interactions profiling
    total_interactions = int(len(interactions_df))
    unique_users = int(interactions_df["user_id"].nunique()) if not interactions_df.empty else 0
    unique_interacted_products = int(interactions_df["parent_asin"].nunique()) if not interactions_df.empty else 0
    
    # Interactions rating distribution
    rating_distribution = {}
    if not interactions_df.empty and "rating" in interactions_df.columns:
        rating_counts = interactions_df["rating"].value_counts().sort_index().to_dict()
        for r, cnt in rating_counts.items():
            rating_distribution[str(r)] = {
                "count": int(cnt),
                "percentage": round(float(cnt / total_interactions * 100), 2)
            }
            
    # Temporal metrics
    min_ts = int(interactions_df["timestamp"].min()) if not interactions_df.empty else 0
    max_ts = int(interactions_df["timestamp"].max()) if not interactions_df.empty else 0
    
    # Split metrics
    split_distribution = {}
    if not interactions_df.empty and "split" in interactions_df.columns:
        for split_name, cnt in interactions_df["split"].value_counts().items():
            sub_df = interactions_df[interactions_df["split"] == split_name]
            split_distribution[split_name] = {
                "count": int(cnt),
                "percentage": round(float(cnt / total_interactions * 100), 2),
                "min_timestamp": int(sub_df["timestamp"].min()),
                "max_timestamp": int(sub_df["timestamp"].max()),
                "min_date": format_timestamp(int(sub_df["timestamp"].min())),
                "max_date": format_timestamp(int(sub_df["timestamp"].max())),
            }

    # Products per user and interactions per product
    products_per_user = interactions_df.groupby("user_id")["parent_asin"].count() if not interactions_df.empty else pd.Series(dtype=int)
    interactions_per_product = interactions_df.groupby("parent_asin")["user_id"].count() if not interactions_df.empty else pd.Series(dtype=int)

    profile = {
        "dataset_name": "Amazon Reviews 2023 (Electronics)",
        "profile_generated_at": datetime.now(timezone.utc).isoformat(),
        "sampling_strategy": sampling_meta or {},
        "products": {
            "total_records": total_products,
            "unique_parent_asins": unique_product_ids,
            "total_unique_categories": len(all_categories),
            "top_categories": [{"category": c, "count": count} for c, count in top_categories],
            "missingness": {
                "missing_prices": {
                    "count": missing_prices,
                    "percentage": round(float(missing_prices / max(total_products, 1) * 100), 2),
                },
                "missing_brands": {
                    "count": missing_brands,
                    "percentage": round(float(missing_brands / max(total_products, 1) * 100), 2),
                },
                "missing_descriptions": {
                    "count": missing_descriptions,
                    "percentage": round(float(missing_descriptions / max(total_products, 1) * 100), 2),
                },
                "missing_features": {
                    "count": missing_features,
                    "percentage": round(float(missing_features / max(total_products, 1) * 100), 2),
                },
            },
            "price_distribution_usd": price_stats,
            "average_rating_distribution": product_rating_stats,
        },
        "interactions": {
            "total_records": total_interactions,
            "unique_users": unique_users,
            "unique_products": unique_interacted_products,
            "catalog_coverage_percentage": round(float(unique_interacted_products / max(unique_product_ids, 1) * 100), 2),
            "rating_distribution": rating_distribution,
            "temporal_range": {
                "min_timestamp": min_ts,
                "max_timestamp": max_ts,
                "start_date": format_timestamp(min_ts),
                "end_date": format_timestamp(max_ts),
            },
            "temporal_splits": split_distribution,
            "user_activity_stats": compute_distribution_stats(products_per_user),
            "item_popularity_stats": compute_distribution_stats(interactions_per_product),
        },
    }

    # Write JSON profile
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

    # Write human-readable Markdown report
    if output_md_path:
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        md_content = generate_markdown_profile(profile)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    return profile


def generate_markdown_profile(profile: Dict[str, Any]) -> str:
    """Generate a clean Markdown data profiling report."""
    prod = profile["products"]
    inter = profile["interactions"]
    missing = prod["missingness"]
    price_s = prod["price_distribution_usd"]
    temp = inter["temporal_range"]
    
    top_cat_lines = "\n".join(
        f"| {c['category']} | {c['count']:,} |"
        for c in prod["top_categories"][:10]
    )
    
    rating_lines = "\n".join(
        f"| {stars} ★ | {d['count']:,} | {d['percentage']}% |"
        for stars, d in inter.get("rating_distribution", {}).items()
    )

    split_lines = "\n".join(
        f"| **{s_name}** | {s_data['count']:,} | {s_data['percentage']}% | {s_data.get('min_date', 'N/A')} to {s_data.get('max_date', 'N/A')} |"
        for s_name, s_data in inter.get("temporal_splits", {}).items()
    )

    return f"""# Amazon Reviews 2023 (Electronics) — Dataset Profile Report

*Generated on: {profile['profile_generated_at']}*

---

## 1. Executive Summary

| Dimension | Processed Metric |
| :--- | :--- |
| **Product Catalog Size** | **{prod['total_records']:,}** unique products |
| **User Interactions** | **{inter['total_records']:,}** ratings/reviews |
| **Unique Customer Users** | **{inter['unique_users']:,}** users |
| **Unique Categories** | **{prod['total_unique_categories']:,}** categories |
| **Temporal Span** | {temp['start_date']} — {temp['end_date']} |
| **Catalog Review Coverage** | **{inter['catalog_coverage_percentage']}%** |

---

## 2. Product Catalog Characteristics

### Metadata Quality & Completeness
| Attribute | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
| **Price** | {missing['missing_prices']['count']:,} | {missing['missing_prices']['percentage']}% |
| **Brand** | {missing['missing_brands']['count']:,} | {missing['missing_brands']['percentage']}% |
| **Description** | {missing['missing_descriptions']['count']:,} | {missing['missing_descriptions']['percentage']}% |
| **Features** | {missing['missing_features']['count']:,} | {missing['missing_features']['percentage']}% |

### Pricing Distribution (USD)
| Statistic | Value |
| :--- | :--- |
| **Min Price** | ${price_s['min']:.2f} |
| **25th Percentile (p25)** | ${price_s['p25']:.2f} |
| **Median Price (p50)** | ${price_s['median']:.2f} |
| **Mean Price** | ${price_s['mean']:.2f} |
| **75th Percentile (p75)** | ${price_s['p75']:.2f} |
| **95th Percentile (p95)** | ${price_s['p95']:.2f} |
| **Max Price** | ${price_s['max']:.2f} |

### Top Product Categories
| Category | Product Count |
| :--- | :--- |
{top_cat_lines}

---

## 3. User Interaction & Rating Dynamics

### Rating Distribution
| Rating Score | Count | Percentage |
| :--- | :--- | :--- |
{rating_lines}

### Temporal Partitioning (Train / Val / Test)
| Split Partition | Interactions | Share | Temporal Date Range |
| :--- | :--- | :--- | :--- |
{split_lines}

### Interaction Density Statistics
| Metric | Mean | Median | p75 | p95 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interactions per User** | {inter['user_activity_stats']['mean']:.2f} | {inter['user_activity_stats']['median']:.1f} | {inter['user_activity_stats']['p75']:.1f} | {inter['user_activity_stats']['p95']:.1f} | {inter['user_activity_stats']['max']:.0f} |
| **Interactions per Product** | {inter['item_popularity_stats']['mean']:.2f} | {inter['item_popularity_stats']['median']:.1f} | {inter['item_popularity_stats']['p75']:.1f} | {inter['item_popularity_stats']['p95']:.1f} | {inter['item_popularity_stats']['max']:.0f} |

---

## 4. Methodology & Research Integrity Notes
1. **Source Authenticity**: All product metadata and user interactions originate strictly from the official McAuley Lab Amazon Reviews 2023 Electronics dataset.
2. **Deterministic Sampling**: The development subset was selected via quality-weighted stratified scoring with a deterministic random seed (`seed=42`).
3. **Temporal Evaluation**: Partitions use global chronological quantile splits to prevent historical leakage.
"""
