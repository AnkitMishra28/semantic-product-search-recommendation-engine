#!/usr/bin/env python3
"""Query Understanding & Structured Search Intent Benchmark Runner.

Evaluates deterministic and catalog-aware Query Understanding:
- Category extraction & synonym resolution
- Brand extraction & multi-word handling
- Price ceiling/floor extraction & shorthand normalization
- Fine-grained technical attribute parsing
- Intent classification
- Hard vs. soft filter assignment
- Micro-benchmarking processing latency (p50, p95, p99)

Outputs:
- experiments/results/query_understanding_benchmark.json
- experiments/query_understanding/report.md

Usage:
    python scripts/run_query_understanding_benchmark.py
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.query_understanding.pipeline import QueryUnderstandingPipeline
from evaluation.metrics import LatencyTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Retrieve current Git commit hash or return 'untracked_repo'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "untracked_repo"


def compute_field_metrics(
    predictions: List[Any],
    ground_truths: List[Any],
) -> Dict[str, float]:
    """Compute Precision, Recall, and F1 for categorical/scalar field extraction."""
    tp = 0
    fp = 0
    fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_has_val = pred is not None and pred != ""
        gt_has_val = gt is not None and gt != ""

        if pred_has_val and gt_has_val:
            if str(pred).strip().lower() == str(gt).strip().lower():
                tp += 1
            else:
                fp += 1
                fn += 1
        elif pred_has_val and not gt_has_val:
            fp += 1
        elif not pred_has_val and gt_has_val:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_attributes_metrics(
    predicted_attrs: List[Dict[str, List[str]]],
    expected_attrs: List[Dict[str, List[str]]],
) -> Dict[str, float]:
    """Compute token-level Precision, Recall, F1 for multi-valued attribute dictionaries."""
    tp = 0
    fp = 0
    fn = 0

    for pred_dict, gt_dict in zip(predicted_attrs, expected_attrs):
        # Flatten (category, val) pairs
        pred_set = {(k.lower(), v.lower()) for k, vals in pred_dict.items() for v in vals}
        gt_set = {(k.lower(), v.lower()) for k, vals in gt_dict.items() for v in vals}

        tp += len(pred_set & gt_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_query_understanding_benchmark(
    eval_dataset_path: str = "evaluation/query_understanding_dataset.json",
    output_json_path: str = "experiments/results/query_understanding_benchmark.json",
    output_report_path: str = "experiments/query_understanding/report.md",
    repetitions: int = 100,
) -> Dict[str, Any]:
    """Execute evaluation and latency micro-benchmarking on Query Understanding."""
    print("=" * 80)
    print(" Phase 6: Query Understanding & Structured Intent Benchmark")
    print("=" * 80)

    # 1. Load evaluation dataset
    print(f"\n[1/4] Loading evaluation queries from '{eval_dataset_path}'...")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"      Loaded {len(eval_data)} annotated evaluation queries.")

    # 2. Initialize pipeline
    print("\n[2/4] Initializing QueryUnderstandingPipeline...")
    pipeline = QueryUnderstandingPipeline(default_currency="INR")

    # 3. Evaluate structured field extraction
    print(f"\n[3/4] Evaluating extraction accuracy and classifying failures...")
    preds_category = []
    gts_category = []
    preds_brand = []
    gts_brand = []
    preds_price_max = []
    gts_price_max = []
    preds_price_min = []
    gts_price_min = []
    preds_intent = []
    gts_intent = []
    preds_attributes = []
    gts_attributes = []

    per_query_results = []
    exact_matches = 0
    failures_by_category: Dict[str, List[Dict[str, Any]]] = {
        "missing_entity": [],
        "incorrect_entity": [],
        "incorrect_price": [],
        "incorrect_category": [],
        "ambiguous_intent": [],
        "unsupported_attribute": [],
    }

    for item in eval_data:
        qid = item["id"]
        raw_query = item["query"]
        expected_cat = item.get("expected_category")
        expected_brd = item.get("expected_brand")
        expected_pmin = item.get("expected_price_min")
        expected_pmax = item.get("expected_price_max")
        expected_cur = item.get("expected_currency", "INR")
        expected_int = item.get("expected_intent", "product_search")
        expected_att = item.get("expected_attributes", {})

        # Process query
        res = pipeline.process_to_intent(raw_query)

        preds_category.append(res.category)
        gts_category.append(expected_cat)
        preds_brand.append(res.brand)
        gts_brand.append(expected_brd)
        preds_price_max.append(res.price_max)
        gts_price_max.append(expected_pmax)
        preds_price_min.append(res.price_min)
        gts_price_min.append(expected_pmin)
        preds_intent.append(res.intent)
        gts_intent.append(expected_int)
        preds_attributes.append(res.attributes)
        gts_attributes.append(expected_att)

        # Exact match check
        cat_match = (res.category or "").lower() == (expected_cat or "").lower()
        brd_match = (res.brand or "").lower() == (expected_brd or "").lower()
        pmax_match = res.price_max == expected_pmax
        pmin_match = res.price_min == expected_pmin
        int_match = res.intent == expected_int

        # Attributes match
        pred_attr_set = {(k.lower(), v.lower()) for k, vals in res.attributes.items() for v in vals}
        gt_attr_set = {(k.lower(), v.lower()) for k, vals in expected_att.items() for v in vals}
        attr_match = pred_attr_set == gt_attr_set

        is_exact_match = all([cat_match, brd_match, pmax_match, pmin_match, int_match, attr_match])
        if is_exact_match:
            exact_matches += 1

        # Failure categorization
        detected_failures = []
        if not cat_match:
            if expected_cat and not res.category:
                failures_by_category["missing_entity"].append({"query": raw_query, "field": "category", "expected": expected_cat, "got": res.category})
            else:
                failures_by_category["incorrect_category"].append({"query": raw_query, "expected": expected_cat, "got": res.category})
            detected_failures.append("category_mismatch")

        if not brd_match:
            if expected_brd and not res.brand:
                failures_by_category["missing_entity"].append({"query": raw_query, "field": "brand", "expected": expected_brd, "got": res.brand})
            else:
                failures_by_category["incorrect_entity"].append({"query": raw_query, "field": "brand", "expected": expected_brd, "got": res.brand})
            detected_failures.append("brand_mismatch")

        if not (pmax_match and pmin_match):
            failures_by_category["incorrect_price"].append({
                "query": raw_query,
                "expected": {"min": expected_pmin, "max": expected_pmax},
                "got": {"min": res.price_min, "max": res.price_max},
            })
            detected_failures.append("price_mismatch")

        if not int_match:
            failures_by_category["ambiguous_intent"].append({"query": raw_query, "expected": expected_int, "got": res.intent})
            detected_failures.append("intent_mismatch")

        if not attr_match:
            failures_by_category["unsupported_attribute"].append({
                "query": raw_query,
                "expected": expected_att,
                "got": res.attributes,
            })
            detected_failures.append("attribute_mismatch")

        per_query_results.append({
            "id": qid,
            "raw_query": raw_query,
            "normalized_query": res.normalized_query,
            "extracted": {
                "category": res.category,
                "brand": res.brand,
                "price_min": res.price_min,
                "price_max": res.price_max,
                "currency": res.currency,
                "intent": res.intent,
                "attributes": res.attributes,
                "hard_filters": res.hard_filters,
                "soft_signals": res.soft_signals,
                "confidence": res.confidence,
            },
            "expected": {
                "category": expected_cat,
                "brand": expected_brd,
                "price_min": expected_pmin,
                "price_max": expected_pmax,
                "currency": expected_cur,
                "intent": expected_int,
                "attributes": expected_att,
            },
            "is_exact_match": is_exact_match,
            "detected_failures": detected_failures,
        })

    # 4. Latency Profiling across repetitions
    print(f"\n[4/4] Profiling Query Understanding latency ({repetitions} repetitions)...")
    latency_tracker = LatencyTracker()
    for _ in range(repetitions):
        for item in eval_data:
            t0 = time.perf_counter()
            _ = pipeline.process_to_intent(item["query"])
            t1 = time.perf_counter()
            latency_tracker.record((t1 - t0) * 1000.0)

    lat_summary = latency_tracker.summary()

    # Field-level metrics calculation
    field_metrics = {
        "category": compute_field_metrics(preds_category, gts_category),
        "brand": compute_field_metrics(preds_brand, gts_brand),
        "price_max": compute_field_metrics(preds_price_max, gts_price_max),
        "price_min": compute_field_metrics(preds_price_min, gts_price_min),
        "intent": compute_field_metrics(preds_intent, gts_intent),
        "attributes": compute_attributes_metrics(preds_attributes, gts_attributes),
    }

    # Macro & Micro F1
    f1_scores = [m["f1"] for m in field_metrics.values()]
    macro_f1 = float(np.mean(f1_scores))
    total_tp = sum(m["tp"] for m in field_metrics.values())
    total_fp = sum(m["fp"] for m in field_metrics.values())
    total_fn = sum(m["fn"] for m in field_metrics.values())
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    exact_match_accuracy = exact_matches / len(eval_data)

    print("\n" + "=" * 80)
    print(" SUMMARY: QUERY UNDERSTANDING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Exact Match Accuracy: {exact_match_accuracy * 100:.1f}% ({exact_matches}/{len(eval_data)})")
    print(f"  Macro F1 Score:       {macro_f1:.4f}")
    print(f"  Micro F1 Score:       {micro_f1:.4f}")
    print(f"  Category F1:          {field_metrics['category']['f1']:.4f} (P: {field_metrics['category']['precision']:.4f}, R: {field_metrics['category']['recall']:.4f})")
    print(f"  Brand F1:             {field_metrics['brand']['f1']:.4f} (P: {field_metrics['brand']['precision']:.4f}, R: {field_metrics['brand']['recall']:.4f})")
    print(f"  Price Max F1:         {field_metrics['price_max']['f1']:.4f} (P: {field_metrics['price_max']['precision']:.4f}, R: {field_metrics['price_max']['recall']:.4f})")
    print(f"  Price Min F1:         {field_metrics['price_min']['f1']:.4f} (P: {field_metrics['price_min']['precision']:.4f}, R: {field_metrics['price_min']['recall']:.4f})")
    print(f"  Intent F1:            {field_metrics['intent']['f1']:.4f} (P: {field_metrics['intent']['precision']:.4f}, R: {field_metrics['intent']['recall']:.4f})")
    print(f"  Attributes F1:        {field_metrics['attributes']['f1']:.4f} (P: {field_metrics['attributes']['precision']:.4f}, R: {field_metrics['attributes']['recall']:.4f})")
    print(f"  Latency (p50):        {lat_summary['p50_ms']:.3f} ms | (p95): {lat_summary['p95_ms']:.3f} ms | (p99): {lat_summary['p99_ms']:.3f} ms")

    benchmark_payload = {
        "experiment_id": "query_understanding_benchmark_phase6",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_dataset": {
            "path": eval_dataset_path,
            "total_queries": len(eval_data),
        },
        "metrics": {
            "exact_match_accuracy": float(round(exact_match_accuracy, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "micro_f1": float(round(micro_f1, 4)),
            "field_metrics": field_metrics,
        },
        "latency_ms": lat_summary,
        "system_provenance": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "git_commit": get_git_commit(),
        },
        "failure_counts": {k: len(v) for k, v in failures_by_category.items()},
        "failures_by_category": failures_by_category,
        "per_query_results": per_query_results,
    }

    # Save JSON result
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\n[+] Saved full JSON artifact to: {output_json_path}")

    # Generate Markdown Report
    generate_query_understanding_report(
        payload=benchmark_payload,
        output_report_path=output_report_path,
    )
    print(f"[+] Saved markdown report to: {output_report_path}")

    return benchmark_payload


def generate_query_understanding_report(
    payload: Dict[str, Any],
    output_report_path: str,
) -> None:
    """Generate professional scientific report for query understanding."""
    m = payload["metrics"]
    fm = m["field_metrics"]
    lat = payload["latency_ms"]
    prov = payload["system_provenance"]
    fc = payload["failure_counts"]
    queries = payload["per_query_results"]

    correct_samples = [q for q in queries if q["is_exact_match"]][:4]
    failure_samples = [q for q in queries if not q["is_exact_match"]][:4]

    lines = [
        "# Query Understanding & Structured Search Intent Report",
        "",
        "## 1. Executive Summary & Objective",
        "",
        "This experiment benchmarks the deterministic, catalog-aware **Query Understanding subsystem** on a dedicated evaluation dataset of **35 multi-faceted queries** spanning categories, brands, price boundaries, technical attributes, and search intents.",
        "",
        "- **Exact Match Accuracy**: **" + f"{m['exact_match_accuracy']*100:.1f}%" + "**",
        "- **Macro F1 Score**: **" + f"{m['macro_f1']:.4f}" + "**",
        "- **Micro F1 Score**: **" + f"{m['micro_f1']:.4f}" + "**",
        "- **Processing Latency (p50)**: **" + f"{lat['p50_ms']:.3f} ms**",
        "- **Processing Latency (p95)**: **" + f"{lat['p95_ms']:.3f} ms**",
        "",
        "---",
        "",
        "## 2. Field-Level Precision, Recall, and F1 Metrics",
        "",
        "| Extracted Field | Precision | Recall | F1 Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Category** | {fm['category']['precision']:.4f} | {fm['category']['recall']:.4f} | **{fm['category']['f1']:.4f}** | {fm['category']['tp']} | {fm['category']['fp']} | {fm['category']['fn']} |",
        f"| **Brand** | {fm['brand']['precision']:.4f} | {fm['brand']['recall']:.4f} | **{fm['brand']['f1']:.4f}** | {fm['brand']['tp']} | {fm['brand']['fp']} | {fm['brand']['fn']} |",
        f"| **Price Maximum ($price\\_max$)** | {fm['price_max']['precision']:.4f} | {fm['price_max']['recall']:.4f} | **{fm['price_max']['f1']:.4f}** | {fm['price_max']['tp']} | {fm['price_max']['fp']} | {fm['price_max']['fn']} |",
        f"| **Price Minimum ($price\\_min$)** | {fm['price_min']['precision']:.4f} | {fm['price_min']['recall']:.4f} | **{fm['price_min']['f1']:.4f}** | {fm['price_min']['tp']} | {fm['price_min']['fp']} | {fm['price_min']['fn']} |",
        f"| **Intent Classification** | {fm['intent']['precision']:.4f} | {fm['intent']['recall']:.4f} | **{fm['intent']['f1']:.4f}** | {fm['intent']['tp']} | {fm['intent']['fp']} | {fm['intent']['fn']} |",
        f"| **Product Attributes** | {fm['attributes']['precision']:.4f} | {fm['attributes']['recall']:.4f} | **{fm['attributes']['f1']:.4f}** | {fm['attributes']['tp']} | {fm['attributes']['fp']} | {fm['attributes']['fn']} |",
        "",
        "---",
        "",
        "## 3. Systems Latency Breakdown",
        "",
        "Deterministic CPU-bound regex and vocabulary extraction achieves sub-millisecond execution, adding negligible overhead to the retrieval pipeline.",
        "",
        "| Metric | Measured Latency |",
        "| :--- | :--- |",
        f"| **p50 Latency** | {lat['p50_ms']:.3f} ms |",
        f"| **p90 Latency** | {lat['p90_ms']:.3f} ms |",
        f"| **p95 Latency** | {lat['p95_ms']:.3f} ms |",
        f"| **p99 Latency** | {lat['p99_ms']:.3f} ms |",
        f"| **Mean Latency** | {lat['mean_ms']:.3f} ms |",
        "",
        "---",
        "",
        "## 4. Hard Filter vs. Soft Signal Policy",
        "",
        "> [!IMPORTANT]",
        "> **Safe Filtering Architecture**",
        "> - **Hard Filters**: Deterministic quantitative constraints applied *prior* to expensive vector search / Cross-Encoder reranking.",
        ">   - `price_max` / `price_min`: Drops out-of-budget products from the candidate pool.",
        ">   - `brand` / `category`: Applied when high confidence exact matches exist.",
        "> - **Soft Signals**: Subjective, qualitative, or use-case modifiers retained for neural ranking and Cross-Encoder attention scoring.",
        ">   - `gaming`, `travel`, `office`, `ergonomic`, `compact`, `fast charging`, `best`.",
        "",
        "---",
        "",
        "## 5. Representative Extraction Examples",
        "",
        "### Exact Matches (Successful Extractions)",
    ]

    for cs in correct_samples:
        ex = cs["extracted"]
        lines.extend([
            f"#### Query: *\"{cs['raw_query']}\"*",
            f"- **Normalized**: `{cs['normalized_query']}`",
            f"- **Category**: `{ex['category']}` | **Brand**: `{ex['brand']}`",
            f"- **Price Limits**: `min={ex['price_min']}`, `max={ex['price_max']} {ex['currency']}`",
            f"- **Intent**: `{ex['intent']}`",
            f"- **Attributes**: `{ex['attributes']}`",
            f"- **Hard Filters**: `{ex['hard_filters']}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 6. Failure Mode Analysis",
        "",
        f"- **Missing Entities**: {fc['missing_entity']}",
        f"- **Incorrect Entities**: {fc['incorrect_entity']}",
        f"- **Incorrect Price Limits**: {fc['incorrect_price']}",
        f"- **Incorrect Category**: {fc['incorrect_category']}",
        f"- **Ambiguous Intent**: {fc['ambiguous_intent']}",
        f"- **Unsupported Attributes**: {fc['unsupported_attribute']}",
        "",
    ])

    if failure_samples:
        lines.append("### Diagnostic Failure Cases")
        for fs in failure_samples:
            lines.extend([
                f"#### Query: *\"{fs['raw_query']}\"*",
                f"- **Detected Failures**: `{fs['detected_failures']}`",
                f"- **Got**: `{fs['extracted']}`",
                f"- **Expected**: `{fs['expected']}`",
                "",
            ])
    else:
        lines.append("*Zero failure cases detected on benchmark evaluation dataset.*")

    lines.extend([
        "---",
        "",
        "## 7. System Provenance",
        "",
        f"- **Platform**: {prov['platform']}",
        f"- **Python Version**: {prov['python_version']}",
        f"- **Git Commit**: `{prov['git_commit']}`",
        f"- **Timestamp**: {payload['timestamp']}",
    ])

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Query Understanding Benchmark")
    parser.add_argument("--eval-dataset", default="evaluation/query_understanding_dataset.json", help="Path to eval dataset")
    parser.add_argument("--output-json", default="experiments/results/query_understanding_benchmark.json", help="Output JSON path")
    parser.add_argument("--output-report", default="experiments/query_understanding/report.md", help="Output report path")
    parser.add_argument("--repetitions", type=int, default=100, help="Latency timing repetitions")
    args = parser.parse_args()

    run_query_understanding_benchmark(
        eval_dataset_path=args.eval_dataset,
        output_json_path=args.output_json,
        output_report_path=args.output_report,
        repetitions=args.repetitions,
    )
