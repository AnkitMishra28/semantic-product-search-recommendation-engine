#!/usr/bin/env python3
"""BM25 Lexical Retrieval Baseline Experiment Runner.

Evaluates BM25 Okapi retrieval across catalog-grounded evaluation queries on the
Amazon Reviews 2023 (Electronics) dataset, computes standard IR metrics (Recall@K,
MRR@K, NDCG@K), measures latency percentiles, and generates a structured failure analysis.

Usage:
    python scripts/run_bm25_baseline.py
    python scripts/run_bm25_baseline.py --k1 1.5 --b 0.75 --top-k 100
"""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd
import yaml

# Add repository root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.tokenizer import tokenize_lexical
from evaluation.metrics import (
    LatencyTracker,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def get_git_commit() -> str:
    """Retrieve Git commit hash or return 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "untracked_repo"


def run_bm25_experiment(
    products_path: str = "data/processed/products.parquet",
    eval_queries_path: str = "data/processed/evaluation_queries.json",
    top_k: int = 100,
    k1: float = 1.5,
    b: float = 0.75,
    output_results_path: str = "experiments/results/bm25_baseline.json",
    output_analysis_path: str = "experiments/baseline/bm25_failure_analysis.md",
    config_path: str = "experiments/baseline/config.yaml",
    save_index_path: str = None,
) -> Dict[str, Any]:
    """Execute end-to-end BM25 lexical baseline evaluation experiment."""
    print("======================================================================")
    print(" Amazon-Scale Semantic Search — BM25 Lexical Retrieval Baseline")
    print("======================================================================")
    print(f"[*] Products Path:        {products_path}")
    print(f"[*] Evaluation Queries:   {eval_queries_path}")
    print(f"[*] BM25 Parameters:      k1={k1}, b={b}, top_k={top_k}")
    print("======================================================================\n")

    # 1. Load products catalog
    if not os.path.exists(products_path):
        raise FileNotFoundError(f"Products file missing at {products_path}")
    if not os.path.exists(eval_queries_path):
        raise FileNotFoundError(f"Evaluation queries file missing at {eval_queries_path}")

    print("[1/5] Loading products catalog...")
    t0 = time.perf_counter()
    products_df = pd.read_parquet(products_path)
    total_products = len(products_df)
    load_time = time.perf_counter() - t0
    print(f"[+] Loaded {total_products:,} products in {load_time:.2f}s")

    # 2. Build BM25 index
    print(f"\n[2/5] Building BM25Okapi index over {total_products:,} documents...")
    retriever = BM25Retriever(k1=k1, b=b, tokenizer=tokenize_lexical)
    index_build_sec = retriever.index_corpus(products_df, id_column="parent_asin")
    print(f"[+] BM25 index built in {index_build_sec:.2f}s ({total_products/index_build_sec:.1f} docs/sec)")

    if save_index_path:
        print(f"[*] Saving index to {save_index_path}...")
        retriever.save(save_index_path)

    # 3. Load evaluation queries
    print(f"\n[3/5] Loading ground-truth evaluation queries...")
    with open(eval_queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)
    print(f"[+] Loaded {len(eval_queries)} evaluation queries.")

    # Create quick product lookup map for failure analysis
    product_title_map = {}
    for _, row in products_df.iterrows():
        product_title_map[str(row["parent_asin"])] = str(row.get("title") or "Unknown Title")

    # 4. Evaluate queries
    print("\n[4/5] Running evaluation queries against BM25 index...")
    per_query_results: List[Dict[str, Any]] = []
    latency_tracker = LatencyTracker()
    
    recalls_10: List[float] = []
    recalls_50: List[float] = []
    recalls_100: List[float] = []
    mrrs_10: List[float] = []
    ndcgs_10: List[float] = []

    for q in eval_queries:
        qid = q["query_id"]
        qtext = q["query"]
        relevant_asins: List[str] = q.get("relevant_product_ids", [])
        graded_rel = {asin: 1.0 for asin in relevant_asins}

        # Measure query latency
        t_start = time.perf_counter()
        candidates = retriever.search_text(qtext, top_k=top_k)
        t_end = time.perf_counter()
        q_latency_ms = (t_end - t_start) * 1000.0
        latency_tracker.record(q_latency_ms)

        retrieved_ids = [c.doc_id for c in candidates]
        
        # Calculate standard IR metrics
        r10 = recall_at_k(retrieved_ids, relevant_asins, 10)
        r50 = recall_at_k(retrieved_ids, relevant_asins, 50)
        r100 = recall_at_k(retrieved_ids, relevant_asins, 100)
        mrr10 = reciprocal_rank_at_k(retrieved_ids, relevant_asins, 10)
        ndcg10 = ndcg_at_k(retrieved_ids, graded_rel, 10)

        recalls_10.append(r10)
        recalls_50.append(r50)
        recalls_100.append(r100)
        mrrs_10.append(mrr10)
        ndcgs_10.append(ndcg10)

        # Record rank positions of relevant items found
        retrieved_relevant_ranks = {}
        for rank_idx, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_asins:
                retrieved_relevant_ranks[doc_id] = rank_idx

        per_query_results.append({
            "query_id": qid,
            "query": qtext,
            "category": q.get("category"),
            "intent_type": q.get("intent_type"),
            "num_relevant": len(relevant_asins),
            "relevant_product_ids": relevant_asins,
            "retrieved_count": len(retrieved_ids),
            "retrieved_product_ids": retrieved_ids[:10],
            "relevant_ranks_found": retrieved_relevant_ranks,
            "recall_at_10": float(r10),
            "recall_at_50": float(r50),
            "recall_at_100": float(r100),
            "mrr_at_10": float(mrr10),
            "ndcg_at_10": float(ndcg10),
            "latency_ms": float(round(q_latency_ms, 2)),
        })

    # Summary metrics
    lat_summary = latency_tracker.summary()
    aggregate_metrics = {
        "recall_at_10": float(np.mean(recalls_10)),
        "recall_at_50": float(np.mean(recalls_50)),
        "recall_at_100": float(np.mean(recalls_100)),
        "mrr_at_10": float(np.mean(mrrs_10)),
        "ndcg_at_10": float(np.mean(ndcgs_10)),
    }

    latency_metrics = {
        "p50": float(round(lat_summary["p50_ms"], 2)),
        "p90": float(round(lat_summary["p90_ms"], 2)),
        "p95": float(round(lat_summary["p95_ms"], 2)),
        "p99": float(round(lat_summary["p99_ms"], 2)),
        "mean": float(round(lat_summary["mean_ms"], 2)),
        "min": float(round(lat_summary["min_ms"], 2)),
        "max": float(round(lat_summary["max_ms"], 2)),
    }

    print("\n======================================================================")
    print(" BM25 Baseline Experimental Results Summary")
    print("======================================================================")
    print(f"[*] Total Queries Evaluated: {len(eval_queries)}")
    print(f"[*] Recall@10:               {aggregate_metrics['recall_at_10']:.4f}")
    print(f"[*] Recall@50:               {aggregate_metrics['recall_at_50']:.4f}")
    print(f"[*] Recall@100:              {aggregate_metrics['recall_at_100']:.4f}")
    print(f"[*] MRR@10:                  {aggregate_metrics['mrr_at_10']:.4f}")
    print(f"[*] NDCG@10:                 {aggregate_metrics['ndcg_at_10']:.4f}")
    print(f"[*] Query Latency p50:       {latency_metrics['p50']:.2f} ms")
    print(f"[*] Query Latency p95:       {latency_metrics['p95']:.2f} ms")
    print(f"[*] Query Latency p99:       {latency_metrics['p99']:.2f} ms")
    print("======================================================================\n")

    # 5. Persist results
    print("[5/5] Persisting experiment results and failure analysis...")
    experiment_payload = {
        "experiment_id": "bm25_baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "BM25Okapi",
        "dataset": {
            "name": "Amazon Reviews 2023 (Electronics)",
            "num_products": total_products,
            "num_queries": len(eval_queries),
            "products_file": products_path,
            "queries_file": eval_queries_path,
        },
        "parameters": {
            "k1": k1,
            "b": b,
            "top_k": top_k,
            "tokenizer": "tokenize_lexical (NFKC + alphanumeric/tech preserve)",
        },
        "index_build_time_sec": float(round(index_build_sec, 2)),
        "metrics": aggregate_metrics,
        "latency_ms": latency_metrics,
        "system_provenance": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "git_commit": get_git_commit(),
        },
        "per_query_results": per_query_results,
    }

    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(experiment_payload, f, indent=2)
    print(f"[+] Saved aggregate and per-query results to {output_results_path}")

    # Generate failure analysis
    generate_failure_analysis(
        per_query_results=per_query_results,
        product_title_map=product_title_map,
        output_path=output_analysis_path,
        aggregate_metrics=aggregate_metrics,
        latency_metrics=latency_metrics,
    )
    print(f"[+] Saved failure analysis report to {output_analysis_path}")

    # Update experiments/baseline/config.yaml
    update_baseline_config(
        config_path=config_path,
        products_path=products_path,
        eval_queries_path=eval_queries_path,
        k1=k1,
        b=b,
        top_k=top_k,
    )
    print(f"[+] Updated configuration in {config_path}")

    return experiment_payload


def generate_failure_analysis(
    per_query_results: List[Dict[str, Any]],
    product_title_map: Dict[str, str],
    output_path: str,
    aggregate_metrics: Dict[str, float],
    latency_metrics: Dict[str, float],
) -> None:
    """Generate Markdown failure analysis highlighting zero-hit queries and representative failure patterns."""
    zero_hit_top10 = [q for q in per_query_results if q["recall_at_10"] == 0.0]
    zero_hit_top50 = [q for q in per_query_results if q["recall_at_50"] == 0.0]
    zero_hit_top100 = [q for q in per_query_results if q["recall_at_100"] == 0.0]

    failure_sections = []
    # Analyze representative failure cases
    for q in zero_hit_top10[:6]:
        qid = q["query_id"]
        qtext = q["query"]
        intent_type = q.get("intent_type", "general")
        relevant_ids = q["relevant_product_ids"]
        retrieved_ids = q["retrieved_product_ids"][:5]

        rel_details = []
        for rid in relevant_ids[:3]:
            title = product_title_map.get(rid, "Unknown Product")
            rel_details.append(f"  - `{rid}`: {title}")

        ret_details = []
        for rank, rid in enumerate(retrieved_ids, start=1):
            title = product_title_map.get(rid, "Unknown Product")
            ret_details.append(f"  {rank}. `{rid}`: {title}")

        # Diagnosis logic based on intent
        if "travel" in qtext or "running" in qtext or "programming" in qtext or "wrist pain" in qtext:
            failure_cat = "Vocabulary Mismatch / Contextual Intent Blindness"
            explanation = (
                f"The user query contains colloquial use-case terms (e.g. *'{qtext}'*). "
                f"Relevant items emphasize technical specs without necessarily repeating the exact use-case descriptor, "
                f"leading BM25 to score irrelevant accessory items with partial keyword overlap higher."
            )
        elif "under" in qtext:
            failure_cat = "Attribute & Constraint Blindness"
            explanation = (
                f"The query specifies a numeric or budget filter (e.g. *'{qtext}'*). "
                f"BM25 treats numerical constraints as literal text tokens rather than structured range predicates, "
                f"retrieving items with numbers in descriptions regardless of actual product suitability."
            )
        elif "for" in qtext:
            failure_cat = "Compatibility / Paraphrase Semantic Gap"
            explanation = (
                f"The query specifies cross-device compatibility (e.g. *'{qtext}'*). "
                f"BM25 fails to perform relational reasoning between the host device and peripheral product."
            )
        else:
            failure_cat = "Lexical Synonymy / Term Sparsity"
            explanation = (
                f"Exact keyword overlap is insufficient to bridge lexical variance between query terminology and product metadata."
            )

        section = f"""### Query `{qid}`: *"{qtext}"*
- **Intent Type**: `{intent_type}`
- **Identified Failure Category**: **{failure_cat}**
- **Ground-Truth Relevant Products (Expected)**:
{chr(10).join(rel_details)}
- **Top Retreived by BM25 (Actual)**:
{chr(10).join(ret_details)}
- **Root Cause Analysis**:
  {explanation}
"""
        failure_sections.append(section)

    md_content = f"""# BM25 Lexical Baseline — Failure Analysis & Diagnostic Report

*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*

---

## 1. Baseline Performance Summary

| Metric | Measured Value |
| :--- | :--- |
| **Total Evaluation Queries** | **{len(per_query_results)}** |
| **Recall@10** | **{aggregate_metrics['recall_at_10']:.4f}** ({aggregate_metrics['recall_at_10']*100:.2f}%) |
| **Recall@50** | **{aggregate_metrics['recall_at_50']:.4f}** ({aggregate_metrics['recall_at_50']*100:.2f}%) |
| **Recall@100** | **{aggregate_metrics['recall_at_100']:.4f}** ({aggregate_metrics['recall_at_100']*100:.2f}%) |
| **MRR@10** | **{aggregate_metrics['mrr_at_10']:.4f}** |
| **NDCG@10** | **{aggregate_metrics['ndcg_at_10']:.4f}** |
| **Query Latency (p50)** | **{latency_metrics['p50']:.2f} ms** |
| **Query Latency (p95)** | **{latency_metrics['p95']:.2f} ms** |
| **Query Latency (p99)** | **{latency_metrics['p99']:.2f} ms** |

---

## 2. Zero-Hit Query Distribution

| Threshold | Zero-Hit Query Count | Failure Rate |
| :--- | :--- | :--- |
| **Zero Relevant in Top 10** | **{len(zero_hit_top10)}** / {len(per_query_results)} | **{(len(zero_hit_top10)/len(per_query_results))*100:.1f}%** |
| **Zero Relevant in Top 50** | **{len(zero_hit_top50)}** / {len(per_query_results)} | **{(len(zero_hit_top50)/len(per_query_results))*100:.1f}%** |
| **Zero Relevant in Top 100** | **{len(zero_hit_top100)}** / {len(per_query_results)} | **{(len(zero_hit_top100)/len(per_query_results))*100:.1f}%** |

---

## 3. Detailed Failure Case Studies

{chr(10).join(failure_sections)}

---

## 4. Key Takeaways & Motivation for Dense Semantic Retrieval (Phase 3)
1. **Vocabulary Mismatch**: BM25 requires exact token overlap and fails when customers search with colloquial intent rather than exact catalog keywords.
2. **Context Blindness**: Modifiers such as *"for travel"*, *"for running"*, or *"for programming"* dilutes the lexical score across irrelevant accessories containing those words.
3. **Control Condition Established**: This empirical BM25 benchmark serves as the rigorous control baseline for dense bi-encoder retrieval comparisons.
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def update_baseline_config(
    config_path: str,
    products_path: str,
    eval_queries_path: str,
    k1: float,
    b: float,
    top_k: int,
) -> None:
    """Update experiments/baseline/config.yaml with exact execution parameters."""
    cfg = {
        "experiment_id": "bm25_baseline",
        "description": "BM25 Okapi lexical retrieval baseline on Amazon Reviews 2023 Electronics dataset",
        "track": "baseline",
        "random_seed": 42,
        "dataset": {
            "name": "Amazon Reviews 2023 (Electronics)",
            "products_path": products_path,
            "eval_queries_path": eval_queries_path,
        },
        "retrieval": {
            "method": "bm25_okapi",
            "k_values": [10, 50, 100],
            "bm25_params": {
                "k1": k1,
                "b": b,
                "top_k": top_k,
            },
            "tokenizer": "tokenize_lexical (NFKC + lowercase + technical token preservation)",
        },
        "metrics": [
            "recall@10",
            "recall@50",
            "recall@100",
            "mrr@10",
            "ndcg@10",
            "latency_ms",
        ],
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BM25 lexical retrieval baseline on Amazon Reviews 2023 Electronics."
    )
    parser.add_argument(
        "--products-path",
        type=str,
        default="data/processed/products.parquet",
        help="Path to processed products catalog (default: data/processed/products.parquet)",
    )
    parser.add_argument(
        "--eval-queries-path",
        type=str,
        default="data/processed/evaluation_queries.json",
        help="Path to evaluation queries JSON (default: data/processed/evaluation_queries.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum candidate results to retrieve per query (default: 100)",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 term frequency saturation parameter k1 (default: 1.5)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 document length normalization parameter b (default: 0.75)",
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default="experiments/results/bm25_baseline.json",
        help="Path to save experiment JSON output (default: experiments/results/bm25_baseline.json)",
    )
    parser.add_argument(
        "--output-analysis",
        type=str,
        default="experiments/baseline/bm25_failure_analysis.md",
        help="Path to save failure analysis Markdown report (default: experiments/baseline/bm25_failure_analysis.md)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="experiments/baseline/config.yaml",
        help="Path to experiment configuration YAML (default: experiments/baseline/config.yaml)",
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Optional path to save serialized BM25 index pkl file",
    )

    args = parser.parse_args()

    run_bm25_experiment(
        products_path=args.products_path,
        eval_queries_path=args.eval_queries_path,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        output_results_path=args.output_results,
        output_analysis_path=args.output_analysis,
        config_path=args.config_path,
        save_index_path=args.save_index,
    )


if __name__ == "__main__":
    main()
