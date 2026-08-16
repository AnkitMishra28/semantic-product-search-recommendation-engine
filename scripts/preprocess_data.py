#!/usr/bin/env python3
"""Preprocessing and dataset preparation pipeline for Amazon Reviews 2023 (Electronics).

Loads raw JSONL streams, cleans and normalizes metadata, generates deterministic
embedding document representations, applies temporal splits, extracts ground-truth
evaluation queries, and exports Parquet datasets.

Usage:
    python scripts/preprocess_data.py --target-products 60000 --seed 42
    python scripts/preprocess_data.py --help
"""

import argparse
import json
import os
import sys
import time
from typing import Iterator, Dict, Any, Set
import pandas as pd

# Add root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.preprocessing.sampler import sample_and_deduplicate_products
from backend.app.preprocessing.interaction_processor import process_interactions
from backend.app.preprocessing.eval_queries import build_evaluation_queries
from backend.app.preprocessing.profiler import profile_dataset
from backend.app.preprocessing.validator import (
    validate_products_catalog,
    validate_interactions,
    validate_evaluation_queries,
)


def read_jsonl_stream(file_path: str) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON records line-by-line from a JSONL file without loading entire file into memory."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}. Run scripts/download_data.py first.")
        
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end preprocessing pipeline for Amazon Reviews 2023 Electronics dataset."
    )
    parser.add_argument(
        "--raw-products",
        type=str,
        default="data/raw/meta_Electronics.jsonl",
        help="Path to raw product metadata JSONL file (default: data/raw/meta_Electronics.jsonl)",
    )
    parser.add_argument(
        "--raw-reviews",
        type=str,
        default="data/raw/Electronics_reviews.jsonl",
        help="Path to raw reviews JSONL file (default: data/raw/Electronics_reviews.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--target-products",
        type=int,
        default=60000,
        help="Target number of unique high-quality products to sample (50,000 - 100,000, default: 60000)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=2.5,
        help="Minimum quality richness score for product inclusion (default: 2.5)",
    )
    parser.add_argument(
        "--text-variant",
        type=str,
        default="title_brand_category_features_description",
        choices=[
            "title_brand_category",
            "title_brand_category_features",
            "title_brand_category_features_description",
        ],
        help="Default text representation variant for product embedding (default: title_brand_category_features_description)",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=None,
        help="Optional maximum number of cleaned interactions to retain (default: all matching)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of temporal history for train split (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of temporal history for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling reproducibility (default: 42)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    products_parquet_path = os.path.join(args.output_dir, "products.parquet")
    interactions_parquet_path = os.path.join(args.output_dir, "interactions.parquet")
    eval_queries_path = os.path.join(args.output_dir, "evaluation_queries.json")
    profile_json_path = os.path.join(args.output_dir, "dataset_profile.json")
    profile_md_path = os.path.join(args.output_dir, "dataset_profile.md")

    print("======================================================================")
    print(" Amazon Reviews 2023 (Electronics) — Preprocessing & Ingestion")
    print("======================================================================")
    print(f"[*] Target Product Catalog Size: {args.target_products:,}")
    print(f"[*] Text Representation Variant: {args.text_variant}")
    print(f"[*] Random Seed: {args.seed}")
    print(f"[*] Temporal Splits: {int(args.train_ratio*100)}% Train / {int(args.val_ratio*100)}% Val / {int((1.0-args.train_ratio-args.val_ratio)*100)}% Test")
    print("======================================================================\n")

    # Step 1: Process and sample product catalog
    start_time = time.time()
    print("[1/6] Ingesting, cleaning, deduplicating, and sampling product catalog...")
    products_stream = read_jsonl_stream(args.raw_products)
    selected_products = sample_and_deduplicate_products(
        products_iter=products_stream,
        target_size=args.target_products,
        min_quality_score=args.min_quality,
        text_variant=args.text_variant,
        seed=args.seed,
    )
    
    products_df = pd.DataFrame(selected_products)
    # Save products to Parquet
    products_df.to_parquet(products_parquet_path, index=False, engine="pyarrow")
    prod_time = time.time() - start_time
    print(f"[+] Processed & saved {len(products_df):,} products to {products_parquet_path} ({prod_time:.1f}s)")

    catalog_asins: Set[str] = set(products_df["parent_asin"].unique())

    # Step 2: Ingest, clean, filter, and temporally split user interactions
    print("\n[2/6] Ingesting and temporally partitioning user interactions...")
    inter_start = time.time()
    reviews_stream = read_jsonl_stream(args.raw_reviews)
    test_ratio = max(0.0, round(1.0 - args.train_ratio - args.val_ratio, 4))
    
    interactions_df, split_meta = process_interactions(
        reviews_iter=reviews_stream,
        valid_parent_asins=catalog_asins,
        max_interactions=args.max_interactions,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )
    
    # Save interactions to Parquet
    interactions_df.to_parquet(interactions_parquet_path, index=False, engine="pyarrow")
    inter_time = time.time() - inter_start
    print(f"[+] Processed & saved {len(interactions_df):,} interactions to {interactions_parquet_path} ({inter_time:.1f}s)")
    print(f"    -> Split breakdown: {split_meta.get('split_counts', {})}")

    # Step 3: Curate and match ground-truth evaluation queries
    print("\n[3/6] Curating ground-truth evaluation queries against processed catalog...")
    eval_queries = build_evaluation_queries(products_df, eval_queries_path)
    print(f"[+] Created {len(eval_queries)} evaluation queries with catalog-grounded relevance labels in {eval_queries_path}")

    # Step 4: Run comprehensive data validation
    print("\n[4/6] Running automated data validation checks...")
    val_prod = validate_products_catalog(products_df)
    print(f"    [+] Products Validation: {val_prod['status']} ({val_prod['unique_parent_asins']:,} unique ASINs)")
    
    val_inter = validate_interactions(interactions_df, catalog_asins)
    print(f"    [+] Interactions Validation: {val_inter['status']} ({val_inter['total_interactions']:,} valid interactions, {val_inter['unique_users']:,} users)")
    
    val_eval = validate_evaluation_queries(eval_queries_path, catalog_asins)
    print(f"    [+] Evaluation Queries Validation: {val_eval['status']} ({val_eval['total_queries']} queries verified)")

    # Step 5: Statistical profiling and Markdown report generation
    print("\n[5/6] Generating dataset statistical profile...")
    sampling_info = {
        "target_size": args.target_products,
        "actual_size": len(products_df),
        "min_quality_score": args.min_quality,
        "seed": args.seed,
        "text_variant": args.text_variant,
    }
    profile_dataset(
        products_df=products_df,
        interactions_df=interactions_df,
        output_json_path=profile_json_path,
        output_md_path=profile_md_path,
        sampling_meta=sampling_info,
    )
    print(f"[+] JSON profile saved to {profile_json_path}")
    print(f"[+] Markdown report saved to {profile_md_path}")

    total_elapsed = time.time() - start_time
    print(f"\n[6/6] Preprocessing pipeline completed successfully in {total_elapsed:.1f}s.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
