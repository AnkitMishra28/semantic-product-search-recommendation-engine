#!/usr/bin/env python3
"""Dataset profiling CLI tool.

Computes comprehensive statistical profiling over processed products and interactions
and writes both structured JSON and human-readable Markdown reports.

Usage:
    python scripts/profile_dataset.py
    python scripts/profile_dataset.py --help
"""

import argparse
import os
import sys
import pandas as pd

# Add root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.preprocessing.profiler import profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute statistical profile of processed Amazon Reviews 2023 datasets."
    )
    parser.add_argument(
        "--products-path",
        type=str,
        default="data/processed/products.parquet",
        help="Path to processed products.parquet (default: data/processed/products.parquet)",
    )
    parser.add_argument(
        "--interactions-path",
        type=str,
        default="data/processed/interactions.parquet",
        help="Path to processed interactions.parquet (default: data/processed/interactions.parquet)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/processed/dataset_profile.json",
        help="Path to output JSON profile (default: data/processed/dataset_profile.json)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="data/processed/dataset_profile.md",
        help="Path to output Markdown report (default: data/processed/dataset_profile.md)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.products_path):
        print(f"[!] Products file not found: {args.products_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.interactions_path):
        print(f"[!] Interactions file not found: {args.interactions_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading products from {args.products_path}...")
    products_df = pd.read_parquet(args.products_path)
    print(f"    Loaded {len(products_df):,} product records.")

    print(f"[*] Loading interactions from {args.interactions_path}...")
    interactions_df = pd.read_parquet(args.interactions_path)
    print(f"    Loaded {len(interactions_df):,} interaction records.")

    print("[*] Computing empirical dataset profiling metrics...")
    profile = profile_dataset(
        products_df=products_df,
        interactions_df=interactions_df,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
    )

    print(f"[+] Successfully wrote profile JSON: {args.output_json}")
    print(f"[+] Successfully wrote Markdown report: {args.output_md}")
    print("\nSummary:")
    print(f"  - Unique Products: {profile['products']['total_records']:,}")
    print(f"  - Total Interactions: {profile['interactions']['total_records']:,}")
    print(f"  - Unique Users: {profile['interactions']['unique_users']:,}")
    print(f"  - Unique Categories: {profile['products']['total_unique_categories']:,}")
    print(f"  - Date Range: {profile['interactions']['temporal_range']['start_date']} -> {profile['interactions']['temporal_range']['end_date']}")


if __name__ == "__main__":
    main()
