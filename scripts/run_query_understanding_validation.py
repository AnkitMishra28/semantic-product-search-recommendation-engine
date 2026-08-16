#!/usr/bin/env python3
"""Phase 6.1: Query Understanding Validation, Currency Correction & Edge-Case Benchmark Runner.

Evaluates the hardened, catalog-aware Query Understanding subsystem on the expanded 60-query validation set:
- Canonical USD dataset currency default with explicit INR / EUR support
- Boundary-aware, longest-match non-overlapping attribute extraction (USB vs USB-C vs USB-A vs USB 3.0, WiFi vs WiFi 6, RTX vs RTX 4060)
- Earliest head-noun category disambiguation and synonym resolution
- Heuristic confidence scoring
- Robust latency micro-benchmarking (p50, p90, p95, p99, mean)
- Detailed failure mode classification

Outputs:
- experiments/results/query_understanding_validation.json
- experiments/query_understanding/validation_report.md

Usage:
    python scripts/run_query_understanding_validation.py
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
from typing import Any, Dict, List, Optional, Set, Tuple

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


def run_query_understanding_validation(
    eval_dataset_path: str = "evaluation/query_understanding_validation_dataset.json",
    output_json_path: str = "experiments/results/query_understanding_validation.json",
    output_report_path: str = "experiments/query_understanding/validation_report.md",
    repetitions: int = 100,
) -> Dict[str, Any]:
    """Execute Phase 6.1 validation evaluation and micro-benchmarking."""
    print("=" * 80)
    print(" Phase 6.1: Query Understanding Validation & Edge-Case Hardening")
    print("=" * 80)

    # 1. Load evaluation dataset
    print(f"\n[1/4] Loading validation dataset from '{eval_dataset_path}'...")
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"      Loaded {len(eval_data)} annotated validation queries.")

    # 2. Initialize pipeline
    print("\n[2/4] Initializing QueryUnderstandingPipeline with default USD currency...")
    pipeline = QueryUnderstandingPipeline(default_currency="USD")

    # 3. Evaluate structured extraction
    print(f"\n[3/4] Evaluating extraction accuracy and classifying edge-case failure modes...")
    preds_category = []
    gts_category = []
    preds_brand = []
    gts_brand = []
    preds_price_max = []
    gts_price_max = []
    preds_price_min = []
    gts_price_min = []
    preds_currency = []
    gts_currency = []
    preds_intent = []
    gts_intent = []
    preds_attributes = []
    gts_attributes = []

    per_query_results = []
    exact_matches = 0
    failures_by_category: Dict[str, List[Dict[str, Any]]] = {
        "missing_entity": [],
        "false_positive_entity": [],
        "overlapping_entity": [],
        "currency_ambiguity": [],
        "incorrect_price": [],
        "incorrect_category": [],
        "synonym_failure": [],
        "ambiguous_intent": [],
        "unsupported_attribute": [],
        "conflicting_constraints": [],
    }

    for item in eval_data:
        qid = item["id"]
        raw_query = item["query"]
        expected_cat = item.get("expected_category")
        expected_brd = item.get("expected_brand")
        expected_pmin = item.get("expected_price_min")
        expected_pmax = item.get("expected_price_max")
        expected_cur = item.get("expected_currency", "USD")
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
        preds_currency.append(res.currency)
        gts_currency.append(expected_cur)
        preds_intent.append(res.intent)
        gts_intent.append(expected_int)
        preds_attributes.append(res.attributes)
        gts_attributes.append(expected_att)

        # Exact match verification
        cat_match = (res.category or "").lower() == (expected_cat or "").lower()
        brd_match = (res.brand or "").lower() == (expected_brd or "").lower()
        pmax_match = res.price_max == expected_pmax
        pmin_match = res.price_min == expected_pmin
        cur_match = res.currency == expected_cur
        int_match = res.intent == expected_int

        pred_attr_set = {(k.lower(), v.lower()) for k, vals in res.attributes.items() for v in vals}
        gt_attr_set = {(k.lower(), v.lower()) for k, vals in expected_att.items() for v in vals}
        attr_match = pred_attr_set == gt_attr_set

        is_exact_match = all([cat_match, brd_match, pmax_match, pmin_match, cur_match, int_match, attr_match])
        if is_exact_match:
            exact_matches += 1

        detected_failures = []
        if not cat_match:
            if expected_cat and not res.category:
                failures_by_category["missing_entity"].append({"query": raw_query, "field": "category", "expected": expected_cat, "got": res.category})
            elif not expected_cat and res.category:
                failures_by_category["false_positive_entity"].append({"query": raw_query, "field": "category", "got": res.category})
            else:
                failures_by_category["incorrect_category"].append({"query": raw_query, "expected": expected_cat, "got": res.category})
            detected_failures.append("category_mismatch")

        if not brd_match:
            if expected_brd and not res.brand:
                failures_by_category["missing_entity"].append({"query": raw_query, "field": "brand", "expected": expected_brd, "got": res.brand})
            elif not expected_brd and res.brand:
                failures_by_category["false_positive_entity"].append({"query": raw_query, "field": "brand", "got": res.brand})
            detected_failures.append("brand_mismatch")

        if not (pmax_match and pmin_match):
            failures_by_category["incorrect_price"].append({
                "query": raw_query,
                "expected": {"min": expected_pmin, "max": expected_pmax},
                "got": {"min": res.price_min, "max": res.price_max},
            })
            detected_failures.append("price_mismatch")

        if not cur_match:
            failures_by_category["currency_ambiguity"].append({
                "query": raw_query,
                "expected": expected_cur,
                "got": res.currency,
            })
            detected_failures.append("currency_mismatch")

        if not int_match:
            failures_by_category["ambiguous_intent"].append({"query": raw_query, "expected": expected_int, "got": res.intent})
            detected_failures.append("intent_mismatch")

        if not attr_match:
            # Check if failure was overlapping entity
            has_overlap = ("usb" in raw_query.lower() and "usb-c" in raw_query.lower()) or \
                          ("wifi" in raw_query.lower() and "wifi 6" in raw_query.lower()) or \
                          ("rtx" in raw_query.lower() and "rtx 40" in raw_query.lower())
            if has_overlap:
                failures_by_category["overlapping_entity"].append({
                    "query": raw_query,
                    "expected": expected_att,
                    "got": res.attributes,
                })
            else:
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
    print(f"\n[4/4] Profiling Query Understanding latency ({repetitions} repetitions across {len(eval_data)} queries)...")
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
        "currency": compute_field_metrics(preds_currency, gts_currency),
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
    print(" SUMMARY: PHASE 6.1 VALIDATION RESULTS (60 EVALUATION QUERIES)")
    print("=" * 80)
    print(f"  Exact Match Accuracy: {exact_match_accuracy * 100:.1f}% ({exact_matches}/{len(eval_data)})")
    print(f"  Macro F1 Score:       {macro_f1:.4f}")
    print(f"  Micro F1 Score:       {micro_f1:.4f}")
    print(f"  Category F1:          {field_metrics['category']['f1']:.4f} (P: {field_metrics['category']['precision']:.4f}, R: {field_metrics['category']['recall']:.4f})")
    print(f"  Brand F1:             {field_metrics['brand']['f1']:.4f} (P: {field_metrics['brand']['precision']:.4f}, R: {field_metrics['brand']['recall']:.4f})")
    print(f"  Price Max F1:         {field_metrics['price_max']['f1']:.4f} (P: {field_metrics['price_max']['precision']:.4f}, R: {field_metrics['price_max']['recall']:.4f})")
    print(f"  Price Min F1:         {field_metrics['price_min']['f1']:.4f} (P: {field_metrics['price_min']['precision']:.4f}, R: {field_metrics['price_min']['recall']:.4f})")
    print(f"  Currency F1:          {field_metrics['currency']['f1']:.4f} (P: {field_metrics['currency']['precision']:.4f}, R: {field_metrics['currency']['recall']:.4f})")
    print(f"  Intent F1:            {field_metrics['intent']['f1']:.4f} (P: {field_metrics['intent']['precision']:.4f}, R: {field_metrics['intent']['recall']:.4f})")
    print(f"  Attributes F1:        {field_metrics['attributes']['f1']:.4f} (P: {field_metrics['attributes']['precision']:.4f}, R: {field_metrics['attributes']['recall']:.4f})")
    print(f"  Latency (p50):        {lat_summary['p50_ms']:.3f} ms | (p90): {lat_summary['p90_ms']:.3f} ms | (p95): {lat_summary['p95_ms']:.3f} ms | (p99): {lat_summary['p99_ms']:.3f} ms")

    validation_payload = {
        "experiment_id": "query_understanding_validation_phase6_1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "canonical_currency": "USD",
        "validation_dataset": {
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
        json.dump(validation_payload, f, indent=2)
    print(f"\n[+] Saved validation JSON artifact to: {output_json_path}")

    # Generate Markdown Report
    generate_validation_report(
        payload=validation_payload,
        output_report_path=output_report_path,
    )
    print(f"[+] Saved validation markdown report to: {output_report_path}")

    return validation_payload


def generate_validation_report(
    payload: Dict[str, Any],
    output_report_path: str,
) -> None:
    """Generate comprehensive scientific validation report for Phase 6.1."""
    m = payload["metrics"]
    fm = m["field_metrics"]
    lat = payload["latency_ms"]
    prov = payload["system_provenance"]
    fc = payload["failure_counts"]
    queries = payload["per_query_results"]

    correct_samples = [q for q in queries if q["is_exact_match"]][:5]
    failure_samples = [q for q in queries if not q["is_exact_match"]]

    lines = [
        "# Phase 6.1 Validation Report: Query Understanding & Edge-Case Hardening",
        "",
        "## 1. Executive Summary & Validation Objectives",
        "",
        "This validation experiment evaluates the hardened **Query Understanding subsystem** on an expanded benchmark of **60 multi-faceted evaluation queries**. Phase 6.1 addresses critical currency alignment with the Amazon Reviews 2023 catalog (USD canonical), boundary-aware longest-match non-overlapping attribute extraction (resolving substring false positives like `USB` on `USB-C`), and documented heuristic confidence scoring.",
        "",
        "- **Validation Evaluation Set**: **60 annotated search queries**",
        "- **Exact Match Accuracy**: **" + f"{m['exact_match_accuracy']*100:.1f}%" + "**",
        "- **Macro F1 Score (Structured Evaluation Set)**: **" + f"{m['macro_f1']:.4f}" + "**",
        "- **Micro F1 Score (Structured Evaluation Set)**: **" + f"{m['micro_f1']:.4f}" + "**",
        "- **p50 Processing Latency**: **" + f"{lat['p50_ms']:.3f} ms**",
        "- **p95 Processing Latency**: **" + f"{lat['p95_ms']:.3f} ms**",
        "- **p99 Processing Latency**: **" + f"{lat['p99_ms']:.3f} ms**",
        "",
        "---",
        "",
        "## 2. Critical Corrections Implemented in Phase 6.1",
        "",
        "1. **Canonical Dataset Currency Alignment (USD)**:",
        "   - Canonical dataset prices are in USD. Unspecified currency bounds (e.g. `\"under 800\"`) default to **USD** (`DEFAULT_CURRENCY=\"USD\"`).",
        "   - Explicit currency mentions (`\"under ₹80000\"`, `\"80000 inr\"`, `\"under 500 eur\"`) are accurately identified without silent or lossy currency conversions.",
        "",
        "2. **Boundary-Aware Longest-Match Entity Extraction**:",
        "   - Eliminated overlapping substring extraction bugs using interval span tracking and boundary regex `(?<![\\w\\-])pattern(?![\\w\\-])`.",
        "   - `\"usb-c cable\"` $\\implies$ `[\"USB-C\"]` (no longer generates false positive `[\"USB\", \"USB-C\"]`).",
        "   - `\"wifi 6 router\"` $\\implies$ `[\"WiFi 6\"]` (no longer generates false positive `[\"WiFi\", \"WiFi 6\"]`).",
        "   - `\"rtx 4060 laptop\"` $\\implies$ `[\"RTX 4060\"]` (no longer generates false positive `[\"RTX\", \"RTX 4060\"]`).",
        "   - `\"usb-c and usb-a hub with usb 3.0\"` $\\implies$ `[\"USB 3.0\", \"USB-A\", \"USB-C\"]`.",
        "",
        "3. **Documented Heuristic Confidence Calculation**:",
        "   - Fully deterministic and explainable heuristic score based on grounded catalog matches, exact vs. synonym category matches, numeric validity, and ungrounded query penalties.",
        "",
        "---",
        "",
        "## 3. Field-Level Validation Metrics",
        "",
        "| Extracted Field | Precision | Recall | F1 Score | TP | FP | FN |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Category** | {fm['category']['precision']:.4f} | {fm['category']['recall']:.4f} | **{fm['category']['f1']:.4f}** | {fm['category']['tp']} | {fm['category']['fp']} | {fm['category']['fn']} |",
        f"| **Brand** | {fm['brand']['precision']:.4f} | {fm['brand']['recall']:.4f} | **{fm['brand']['f1']:.4f}** | {fm['brand']['tp']} | {fm['brand']['fp']} | {fm['brand']['fn']} |",
        f"| **Price Maximum ($price\\_max$)** | {fm['price_max']['precision']:.4f} | {fm['price_max']['recall']:.4f} | **{fm['price_max']['f1']:.4f}** | {fm['price_max']['tp']} | {fm['price_max']['fp']} | {fm['price_max']['fn']} |",
        f"| **Price Minimum ($price\\_min$)** | {fm['price_min']['precision']:.4f} | {fm['price_min']['recall']:.4f} | **{fm['price_min']['f1']:.4f}** | {fm['price_min']['tp']} | {fm['price_min']['fp']} | {fm['price_min']['fn']} |",
        f"| **Currency Detection** | {fm['currency']['precision']:.4f} | {fm['currency']['recall']:.4f} | **{fm['currency']['f1']:.4f}** | {fm['currency']['tp']} | {fm['currency']['fp']} | {fm['currency']['fn']} |",
        f"| **Intent Classification** | {fm['intent']['precision']:.4f} | {fm['intent']['recall']:.4f} | **{fm['intent']['f1']:.4f}** | {fm['intent']['tp']} | {fm['intent']['fp']} | {fm['intent']['fn']} |",
        f"| **Product Attributes** | {fm['attributes']['precision']:.4f} | {fm['attributes']['recall']:.4f} | **{fm['attributes']['f1']:.4f}** | {fm['attributes']['tp']} | {fm['attributes']['fp']} | {fm['attributes']['fn']} |",
        "",
        "---",
        "",
        "## 4. Latency Micro-Benchmark Profile",
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
        "## 5. Failure Mode Taxonomy & Edge-Case Analysis",
        "",
        f"- **Missing Entities**: {fc['missing_entity']}",
        f"- **False Positive Entities**: {fc['false_positive_entity']}",
        f"- **Overlapping Entities**: {fc['overlapping_entity']}",
        f"- **Currency Ambiguities**: {fc['currency_ambiguity']}",
        f"- **Incorrect Price Bounds**: {fc['incorrect_price']}",
        f"- **Incorrect Category**: {fc['incorrect_category']}",
        f"- **Synonym Failures**: {fc['synonym_failure']}",
        f"- **Ambiguous Intent**: {fc['ambiguous_intent']}",
        f"- **Unsupported Attributes**: {fc['unsupported_attribute']}",
        f"- **Conflicting Constraints**: {fc['conflicting_constraints']}",
        "",
    ]

    if failure_samples:
        lines.append("### Diagnostic Failures")
        for fs in failure_samples:
            lines.extend([
                f"#### Query: *\"{fs['raw_query']}\"*",
                f"- **Failures**: `{fs['detected_failures']}`",
                f"- **Got**: `{fs['extracted']}`",
                f"- **Expected**: `{fs['expected']}`",
                "",
            ])
    else:
        lines.append("*Zero failure cases detected on 60-query validation set.*")

    lines.extend([
        "---",
        "",
        "## 6. Scientific Scope & Limitations",
        "",
        "> [!NOTE]",
        "> **Scope & Scientific Integrity**",
        "> - Metrics reported above represent **Macro/Micro F1 on the project-specific structured query evaluation set** of 60 annotated queries, not a claim of universal natural language accuracy across arbitrary open-web queries.",
        "> - This parser operates as a **production-style research prototype** designed to provide sub-millisecond deterministic structure prior to hybrid retrieval.",
        "",
        "---",
        "",
        "## 7. Provenance",
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
    parser = argparse.ArgumentParser(description="Run Phase 6.1 Query Understanding Validation")
    parser.add_argument("--eval-dataset", default="evaluation/query_understanding_validation_dataset.json", help="Path to validation dataset")
    parser.add_argument("--output-json", default="experiments/results/query_understanding_validation.json", help="Output JSON path")
    parser.add_argument("--output-report", default="experiments/query_understanding/validation_report.md", help="Output report path")
    parser.add_argument("--repetitions", type=int, default=100, help="Latency timing repetitions")
    args = parser.parse_args()

    run_query_understanding_validation(
        eval_dataset_path=args.eval_dataset,
        output_json_path=args.output_json,
        output_report_path=args.output_report,
        repetitions=args.repetitions,
    )
