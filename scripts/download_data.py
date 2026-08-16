#!/usr/bin/env python3
"""Dataset acquisition CLI for Amazon Reviews 2023 (Electronics).

Downloads raw JSONL metadata and user interaction streams from the official
McAuley Lab Hugging Face repository (https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

Usage:
    python scripts/download_data.py --max-products 75000 --max-reviews 250000
    python scripts/download_data.py --full
"""

import argparse
import os
import sys
import time
from typing import Optional
import requests


DEFAULT_META_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"
DEFAULT_REVIEWS_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"


def stream_download_jsonl(
    url: str,
    output_path: str,
    max_records: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
    force: bool = False,
) -> int:
    """Stream download and write JSONL records from Hugging Face endpoint."""
    if os.path.exists(output_path) and not force:
        print(f"[*] Target file already exists: {output_path} (use --force to re-download)")
        with open(output_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        print(f"[*] Found {count:,} existing records in {output_path}")
        return count

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".tmp"
    
    limit_str = f"{max_records:,} records" if max_records else "FULL dataset"
    print(f"[*] Streaming from {url}")
    print(f"[*] Target output: {output_path} ({limit_str})")
    
    headers = {"User-Agent": "Amazon-Scale-Semantic-Search-Research/1.0"}
    start_time = time.time()
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] Network error connecting to {url}: {e}", file=sys.stderr)
        raise

    count = 0
    bytes_downloaded = 0
    
    with open(temp_path, "wb") as f:
        for line in response.iter_lines():
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            
            f.write(line + b"\n")
            count += 1
            bytes_downloaded += len(line) + 1
            
            if count % 10000 == 0:
                elapsed = max(time.time() - start_time, 0.001)
                speed_mb = (bytes_downloaded / (1024 * 1024)) / elapsed
                print(f"    -> Streamed {count:,} records ({(bytes_downloaded / (1024*1024)):.1f} MB, {speed_mb:.2f} MB/s)")
                
            if max_records and count >= max_records:
                break

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)
    
    total_time = max(time.time() - start_time, 0.001)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Download complete: {count:,} records ({file_size_mb:.2f} MB) in {total_time:.1f}s ({(count / total_time):.1f} rec/s)\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw Amazon Reviews 2023 dataset from official Hugging Face repository."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Electronics",
        help="Amazon product category (default: Electronics)",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=75000,
        help="Maximum raw product metadata records to download (default: 75000)",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=250000,
        help="Maximum raw review/interaction records to download (default: 250000)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download complete raw files without truncating",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw downloaded files (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing downloaded raw files",
    )
    
    args = parser.parse_args()
    
    meta_url = DEFAULT_META_URL.format(category=args.category)
    reviews_url = DEFAULT_REVIEWS_URL.format(category=args.category)
    
    meta_out = os.path.join(args.output_dir, f"meta_{args.category}.jsonl")
    reviews_out = os.path.join(args.output_dir, f"{args.category}_reviews.jsonl")
    
    prod_limit = None if args.full else args.max_products
    rev_limit = None if args.full else args.max_reviews
    
    print("======================================================================")
    print(" Amazon Reviews 2023 — Data Acquisition Pipeline")
    print("======================================================================")
    print(f"[*] Category: {args.category}")
    print(f"[*] Output Directory: {args.output_dir}")
    print(f"[*] Max Products: {'ALL' if prod_limit is None else f'{prod_limit:,}'}")
    print(f"[*] Max Reviews: {'ALL' if rev_limit is None else f'{rev_limit:,}'}")
    print("======================================================================\n")
    
    # 1. Download product metadata
    stream_download_jsonl(
        url=meta_url,
        output_path=meta_out,
        max_records=prod_limit,
        force=args.force,
    )
    
    # 2. Download user reviews/interactions
    stream_download_jsonl(
        url=reviews_url,
        output_path=reviews_out,
        max_records=rev_limit,
        force=args.force,
    )
    
    print("[+] All requested raw data successfully acquired under data/raw/.")


if __name__ == "__main__":
    main()
