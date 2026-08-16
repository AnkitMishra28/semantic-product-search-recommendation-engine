#!/usr/bin/env python3
"""Dataset validation CLI suite.

Executes comprehensive integrity, schema, and referential consistency checks
on the processed products catalog, interactions, and ground-truth evaluation queries.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --help
"""

import argparse
import os
import sys
import pandas as pd

# Add root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.preprocessing.validator import (
    ValidationError,
    validate_products_catalog,
    validate_interactions,
    validate_evaluation_queries,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate schema, constraints, and referential integrity of processed datasets."
    )
    parser.add_argument(
        "--products-path",
        type=str,
        default="data/processed/products.parquet",
        help="Path to products.parquet (default: data/processed/products.parquet)",
    )
    parser.add_argument(
        "--interactions-path",
        type=str,
        default="data/processed/interactions.parquet",
        help="Path to interactions.parquet (default: data/processed/interactions.parquet)",
    )
    parser.add_argument(
        "--eval-queries-path",
        type=str,
        default="data/processed/evaluation_queries.json",
        help="Path to evaluation_queries.json (default: data/processed/evaluation_queries.json)",
    )

    args = parser.parse_args()

    print("======================================================================")
    print(" Amazon Reviews 2023 — Dataset Validation Suite")
    print("======================================================================\n")

    errors = 0

    # 1. Validate Products Catalog
    print(f"[*] Validating products catalog: {args.products_path}")
    if not os.path.exists(args.products_path):
        print(f"    [!] Error: Products file missing at {args.products_path}")
        sys.exit(1)

    try:
        products_df = pd.read_parquet(args.products_path)
        val_res = validate_products_catalog(products_df)
        print(f"    [+] PASSED: {val_res['total_products']:,} valid products, all parent_asins unique.")
        catalog_asins = set(products_df["parent_asin"].unique())
    except ValidationError as e:
        print(f"    [!] FAILED: {e}")
        errors += 1
        catalog_asins = set()

    # 2. Validate Interactions
    print(f"\n[*] Validating interactions: {args.interactions_path}")
    if not os.path.exists(args.interactions_path):
        print(f"    [!] Error: Interactions file missing at {args.interactions_path}")
        sys.exit(1)

    try:
        interactions_df = pd.read_parquet(args.interactions_path)
        val_res = validate_interactions(interactions_df, catalog_asins)
        print(f"    [+] PASSED: {val_res['total_interactions']:,} valid interactions across {val_res['unique_users']:,} users.")
        print(f"        Splits: {val_res['splits']}")
    except ValidationError as e:
        print(f"    [!] FAILED: {e}")
        errors += 1

    # 3. Validate Evaluation Queries
    print(f"\n[*] Validating evaluation queries: {args.eval_queries_path}")
    if not os.path.exists(args.eval_queries_path):
        print(f"    [!] Error: Evaluation queries file missing at {args.eval_queries_path}")
        sys.exit(1)

    try:
        val_res = validate_evaluation_queries(args.eval_queries_path, catalog_asins)
        print(f"    [+] PASSED: {val_res['total_queries']} queries validated. All ground-truth IDs verified in catalog.")
    except ValidationError as e:
        print(f"    [!] FAILED: {e}")
        errors += 1

    print("\n======================================================================")
    if errors == 0:
        print(" [+] ALL VALIDATION CHECKS PASSED.")
        print("======================================================================")
        sys.exit(0)
    else:
        print(f" [!] VALIDATION FAILED WITH {errors} ERROR(S).")
        print("======================================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
