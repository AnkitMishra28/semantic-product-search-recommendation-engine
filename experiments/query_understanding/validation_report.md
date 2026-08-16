# Phase 6.1 Validation Report: Query Understanding & Edge-Case Hardening

## 1. Executive Summary & Validation Objectives

This validation experiment evaluates the hardened **Query Understanding subsystem** on an expanded benchmark of **60 multi-faceted evaluation queries**. Phase 6.1 addresses critical currency alignment with the Amazon Reviews 2023 catalog (USD canonical), boundary-aware longest-match non-overlapping attribute extraction (resolving substring false positives like `USB` on `USB-C`), and documented heuristic confidence scoring.

- **Validation Evaluation Set**: **60 annotated search queries**
- **Exact Match Accuracy**: **100.0%**
- **Macro F1 Score (Structured Evaluation Set)**: **1.0000**
- **Micro F1 Score (Structured Evaluation Set)**: **1.0000**
- **p50 Processing Latency**: **0.839 ms**
- **p95 Processing Latency**: **1.358 ms**
- **p99 Processing Latency**: **1.599 ms**

---

## 2. Critical Corrections Implemented in Phase 6.1

1. **Canonical Dataset Currency Alignment (USD)**:
   - Canonical dataset prices are in USD. Unspecified currency bounds (e.g. `"under 800"`) default to **USD** (`DEFAULT_CURRENCY="USD"`).
   - Explicit currency mentions (`"under ₹80000"`, `"80000 inr"`, `"under 500 eur"`) are accurately identified without silent or lossy currency conversions.

2. **Boundary-Aware Longest-Match Entity Extraction**:
   - Eliminated overlapping substring extraction bugs using interval span tracking and boundary regex `(?<![\w\-])pattern(?![\w\-])`.
   - `"usb-c cable"` $\implies$ `["USB-C"]` (no longer generates false positive `["USB", "USB-C"]`).
   - `"wifi 6 router"` $\implies$ `["WiFi 6"]` (no longer generates false positive `["WiFi", "WiFi 6"]`).
   - `"rtx 4060 laptop"` $\implies$ `["RTX 4060"]` (no longer generates false positive `["RTX", "RTX 4060"]`).
   - `"usb-c and usb-a hub with usb 3.0"` $\implies$ `["USB 3.0", "USB-A", "USB-C"]`.

3. **Documented Heuristic Confidence Calculation**:
   - Fully deterministic and explainable heuristic score based on grounded catalog matches, exact vs. synonym category matches, numeric validity, and ungrounded query penalties.

---

## 3. Field-Level Validation Metrics

| Extracted Field | Precision | Recall | F1 Score | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Category** | 1.0000 | 1.0000 | **1.0000** | 53 | 0 | 0 |
| **Brand** | 1.0000 | 1.0000 | **1.0000** | 23 | 0 | 0 |
| **Price Maximum ($price\_max$)** | 1.0000 | 1.0000 | **1.0000** | 21 | 0 | 0 |
| **Price Minimum ($price\_min$)** | 1.0000 | 1.0000 | **1.0000** | 4 | 0 | 0 |
| **Currency Detection** | 1.0000 | 1.0000 | **1.0000** | 60 | 0 | 0 |
| **Intent Classification** | 1.0000 | 1.0000 | **1.0000** | 60 | 0 | 0 |
| **Product Attributes** | 1.0000 | 1.0000 | **1.0000** | 87 | 0 | 0 |

---

## 4. Latency Micro-Benchmark Profile

| Metric | Measured Latency |
| :--- | :--- |
| **p50 Latency** | 0.839 ms |
| **p90 Latency** | 1.150 ms |
| **p95 Latency** | 1.358 ms |
| **p99 Latency** | 1.599 ms |
| **Mean Latency** | 0.893 ms |

---

## 5. Failure Mode Taxonomy & Edge-Case Analysis

- **Missing Entities**: 0
- **False Positive Entities**: 0
- **Overlapping Entities**: 0
- **Currency Ambiguities**: 0
- **Incorrect Price Bounds**: 0
- **Incorrect Category**: 0
- **Synonym Failures**: 0
- **Ambiguous Intent**: 0
- **Unsupported Attributes**: 0
- **Conflicting Constraints**: 0

*Zero failure cases detected on 60-query validation set.*
---

## 6. Scientific Scope & Limitations

> [!NOTE]
> **Scope & Scientific Integrity**
> - Metrics reported above represent **Macro/Micro F1 on the project-specific structured query evaluation set** of 60 annotated queries, not a claim of universal natural language accuracy across arbitrary open-web queries.
> - This parser operates as a **production-style research prototype** designed to provide sub-millisecond deterministic structure prior to hybrid retrieval.

---

## 7. Provenance

- **Platform**: Windows-11-10.0.26200-SP0
- **Python Version**: 3.14.2
- **Git Commit**: `untracked_repo`
- **Timestamp**: 2026-08-14T17:40:43.023594+00:00
