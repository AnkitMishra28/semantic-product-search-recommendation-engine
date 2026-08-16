#!/usr/bin/env python3
"""Batch Dense Embedding Generation CLI.

Encodes the 60,000 processed products from Amazon Reviews 2023 (Electronics)
using sentence-transformers/all-MiniLM-L6-v2 across text representation ablation variants.

Outputs:
    - data/embeddings/products_<variant>.npy (float32, L2-normalized)
    - data/embeddings/products_<variant>_metadata.json (ID mappings and build provenance)

Usage:
    python scripts/build_embeddings.py --variant all --batch-size 256
    python scripts/build_embeddings.py --variant title_brand_category
"""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Add repo root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.preprocessing.product_document import (
    TextRepresentationVariant,
    build_product_text,
)
from backend.app.retrieval.embeddings import (
    DEFAULT_MODEL_NAME,
    EXPECTED_EMBEDDING_DIM,
    EmbeddingService,
)

VARIANTS = [
    "title_brand_category",
    "title_brand_category_features",
    "title_brand_category_features_description",
]


def get_git_commit() -> str:
    """Retrieve Git commit hash or return 'untracked_repo'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "untracked_repo"


def generate_embeddings_for_variant(
    products_df: pd.DataFrame,
    variant: str,
    embedder: EmbeddingService,
    batch_size: int = 256,
    output_dir: str = "data/embeddings",
) -> Dict[str, Any]:
    """Generate and persist embeddings and metadata for a specific text variant."""
    print(f"\n[*] Processing Representation Variant: '{variant}'")
    total_docs = len(products_df)
    
    # 1. Format document texts
    print(f"  [1/4] Constructing structured document representations for {total_docs:,} items...")
    t0 = time.perf_counter()
    doc_ids = products_df["parent_asin"].astype(str).tolist()
    texts: List[str] = []
    
    for _, row in products_df.iterrows():
        p_dict = row.to_dict()
        doc_text = build_product_text(p_dict, variant=variant)
        texts.append(doc_text)
    format_time = time.perf_counter() - t0
    print(f"  [+] Formatted {total_docs:,} texts in {format_time:.2f}s")

    # Sample text preview
    print(f"  [*] Sample Document Text ({variant}):\n" + "-" * 50)
    sample_preview = texts[0][:300] + ("..." if len(texts[0]) > 300 else "")
    for line in sample_preview.split("\n"):
        print(f"      {line}")
    print("-" * 50)

    # 2. Batch encode documents
    print(f"  [2/4] Encoding {total_docs:,} documents with {embedder.model_name} (batch_size={batch_size})...")
    t1 = time.perf_counter()
    embeddings = embedder.encode_documents(
        texts=texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    encode_time = time.perf_counter() - t1
    throughput = total_docs / encode_time if encode_time > 0 else 0.0
    print(f"  [+] Encoded {total_docs:,} vectors in {encode_time:.2f}s ({throughput:.1f} docs/sec)")

    # 3. Numerical verification
    print("  [3/4] Verifying vector matrix dimensions and L2 normalization...")
    assert embeddings.shape == (total_docs, EXPECTED_EMBEDDING_DIM), (
        f"Shape mismatch: {embeddings.shape} vs expected ({total_docs}, {EXPECTED_EMBEDDING_DIM})"
    )
    assert embeddings.dtype == np.float32, f"Expected float32 dtype, got {embeddings.dtype}"
    
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = float(np.mean(norms))
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    print(f"  [+] Norm statistics: Mean={mean_norm:.6f}, Min={min_norm:.6f}, Max={max_norm:.6f}")
    assert np.allclose(norms, 1.0, atol=1e-3), "Vectors are not unit-normalized!"

    # Compute storage footprint
    storage_bytes = embeddings.nbytes
    storage_mb = storage_bytes / (1024 * 1024)
    print(f"  [+] In-memory array size: {storage_mb:.2f} MB ({storage_bytes:,} bytes)")

    # 4. Persist binary array and metadata
    print("  [4/4] Persisting vector array and metadata mappings...")
    os.makedirs(output_dir, exist_ok=True)
    npy_filename = f"products_{variant}.npy"
    meta_filename = f"products_{variant}_metadata.json"
    npy_path = os.path.join(output_dir, npy_filename)
    meta_path = os.path.join(output_dir, meta_filename)

    np.save(npy_path, embeddings)
    print(f"  [+] Saved binary embeddings to {npy_path}")

    # Build metadata lookup
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    metadata_payload = {
        "variant": variant,
        "model_name": embedder.model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_products": total_docs,
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "device": embedder.device,
        "batch_size": batch_size,
        "generation_time_sec": float(round(encode_time, 2)),
        "throughput_docs_per_sec": float(round(throughput, 1)),
        "storage_mb": float(round(storage_mb, 2)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "doc_ids": doc_ids,
        "id_to_index": id_to_idx,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    print(f"  [+] Saved metadata and ID index mappings to {meta_path}")

    return {
        "variant": variant,
        "npy_path": npy_path,
        "meta_path": meta_path,
        "encode_time_sec": encode_time,
        "throughput": throughput,
        "storage_mb": storage_mb,
        "shape": list(embeddings.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate SentenceTransformer embeddings for products catalog."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=[
            "all",
            "title_brand_category",
            "title_brand_category_features",
            "title_brand_category_features_description",
        ],
        help="Text representation variant to generate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for SentenceTransformer inference (default: 256)",
    )
    parser.add_argument(
        "--products-path",
        type=str,
        default="data/processed/products.parquet",
        help="Path to processed products catalog (default: data/processed/products.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Destination directory for embeddings (default: data/embeddings)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME})",
    )

    args = parser.parse_args()

    print("======================================================================")
    print(" Amazon-Scale Semantic Search — Batch Embedding Generation")
    print("======================================================================")
    print(f"[*] Products Path:     {args.products_path}")
    print(f"[*] Output Directory:  {args.output_dir}")
    print(f"[*] Model:             {args.model_name}")
    print(f"[*] Variant Target:    {args.variant}")
    print(f"[*] Batch Size:        {args.batch_size}")
    print("======================================================================\n")

    if not os.path.exists(args.products_path):
        raise FileNotFoundError(f"Products file not found at {args.products_path}")

    # Load products catalog
    t_load = time.perf_counter()
    products_df = pd.read_parquet(args.products_path)
    print(f"[+] Loaded {len(products_df):,} products in {time.perf_counter() - t_load:.2f}s")

    # Initialize embedder singleton
    embedder = EmbeddingService(
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=True,
    )
    print(f"[+] Embedder initialized: dim={embedder.embedding_dimension}, device={embedder.device}")

    variants_to_run = VARIANTS if args.variant == "all" else [args.variant]
    
    summary_records = []
    total_start = time.perf_counter()
    for v in variants_to_run:
        rec = generate_embeddings_for_variant(
            products_df=products_df,
            variant=v,
            embedder=embedder,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
        summary_records.append(rec)

    total_elapsed = time.perf_counter() - total_start
    print("\n======================================================================")
    print(" Embedding Generation Complete")
    print("======================================================================")
    print(f"[*] Total Time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    for s in summary_records:
        print(f"  - {s['variant']}: {s['shape']} | {s['storage_mb']:.2f} MB | {s['encode_time_sec']:.2f}s ({s['throughput']:.1f} docs/sec)")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
