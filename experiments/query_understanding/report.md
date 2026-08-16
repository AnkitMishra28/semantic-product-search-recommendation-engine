# Query Understanding & Structured Search Intent Report

## 1. Executive Summary & Objective

This experiment benchmarks the deterministic, catalog-aware **Query Understanding subsystem** on a dedicated evaluation dataset of **35 multi-faceted queries** spanning categories, brands, price boundaries, technical attributes, and search intents.

- **Exact Match Accuracy**: **94.3%**
- **Macro F1 Score**: **0.9968**
- **Micro F1 Score**: **0.9933**
- **Processing Latency (p50)**: **0.714 ms**
- **Processing Latency (p95)**: **0.996 ms**

---

## 2. Field-Level Precision, Recall, and F1 Metrics

| Extracted Field | Precision | Recall | F1 Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Category** | 1.0000 | 1.0000 | **1.0000** | 32 | 0 | 0 |
| **Brand** | 1.0000 | 1.0000 | **1.0000** | 18 | 0 | 0 |
| **Price Maximum ($price\_max$)** | 1.0000 | 1.0000 | **1.0000** | 9 | 0 | 0 |
| **Price Minimum ($price\_min$)** | 1.0000 | 1.0000 | **1.0000** | 3 | 0 | 0 |
| **Intent Classification** | 1.0000 | 1.0000 | **1.0000** | 35 | 0 | 0 |
| **Product Attributes** | 0.9623 | 1.0000 | **0.9808** | 51 | 2 | 0 |

---

## 3. Systems Latency Breakdown

Deterministic CPU-bound regex and vocabulary extraction achieves sub-millisecond execution, adding negligible overhead to the retrieval pipeline.

| Metric | Measured Latency |
| :--- | :--- |
| **p50 Latency** | 0.714 ms |
| **p90 Latency** | 0.821 ms |
| **p95 Latency** | 0.996 ms |
| **p99 Latency** | 1.289 ms |
| **Mean Latency** | 0.741 ms |

---

## 4. Hard Filter vs. Soft Signal Policy

> [!IMPORTANT]
> **Safe Filtering Architecture**
> - **Hard Filters**: Deterministic quantitative constraints applied *prior* to expensive vector search / Cross-Encoder reranking.
>   - `price_max` / `price_min`: Drops out-of-budget products from the candidate pool.
>   - `brand` / `category`: Applied when high confidence exact matches exist.
> - **Soft Signals**: Subjective, qualitative, or use-case modifiers retained for neural ranking and Cross-Encoder attention scoring.
>   - `gaming`, `travel`, `office`, `ergonomic`, `compact`, `fast charging`, `best`.

---

## 5. Representative Extraction Examples

### Exact Matches (Successful Extractions)
#### Query: *"best gaming laptop under 80k with rtx"*
- **Normalized**: `best gaming laptop under 80000 with rtx`
- **Category**: `laptop` | **Brand**: `None`
- **Price Limits**: `min=None`, `max=80000.0 INR`
- **Intent**: `price_constrained_search`
- **Attributes**: `{'gpu': ['RTX'], 'use_case': ['gaming']}`
- **Hard Filters**: `{'price_max': 80000.0, 'category': 'laptop'}`

#### Query: *"sony noise cancelling bluetooth headphones for travel"*
- **Normalized**: `sony noise cancelling bluetooth headphones for travel`
- **Category**: `headphones` | **Brand**: `Sony`
- **Price Limits**: `min=None`, `max=None INR`
- **Intent**: `product_search`
- **Attributes**: `{'connectivity': ['Bluetooth'], 'use_case': ['travel'], 'features': ['noise_cancelling']}`
- **Hard Filters**: `{'brand': 'Sony', 'category': 'headphones'}`

#### Query: *"wireless earbuds below 5000 inr"*
- **Normalized**: `wireless earbuds below 5000 inr`
- **Category**: `earbuds` | **Brand**: `None`
- **Price Limits**: `min=None`, `max=5000.0 INR`
- **Intent**: `price_constrained_search`
- **Attributes**: `{'connectivity': ['Wireless']}`
- **Hard Filters**: `{'price_max': 5000.0, 'category': 'earbuds'}`

#### Query: *"mechanical gaming keyboard with rgb"*
- **Normalized**: `mechanical gaming keyboard with rgb`
- **Category**: `keyboard` | **Brand**: `None`
- **Price Limits**: `min=None`, `max=None INR`
- **Intent**: `product_search`
- **Attributes**: `{'use_case': ['gaming'], 'features': ['mechanical', 'rgb']}`
- **Hard Filters**: `{'category': 'keyboard'}`

---

## 6. Failure Mode Analysis

- **Missing Entities**: 0
- **Incorrect Entities**: 0
- **Incorrect Price Limits**: 0
- **Incorrect Category**: 0
- **Ambiguous Intent**: 0
- **Unsupported Attributes**: 2

### Diagnostic Failure Cases
#### Query: *"usb-c fast charger for macbook and ipad"*
- **Detected Failures**: `['attribute_mismatch']`
- **Got**: `{'category': 'charger', 'brand': None, 'price_min': None, 'price_max': None, 'currency': 'INR', 'intent': 'product_search', 'attributes': {'connectivity': ['USB', 'USB-C'], 'features': ['fast_charging']}, 'hard_filters': {'category': 'charger'}, 'soft_signals': {'connectivity': ['USB', 'USB-C'], 'features': ['fast_charging']}, 'confidence': 1.0}`
- **Expected**: `{'category': 'charger', 'brand': None, 'price_min': None, 'price_max': None, 'currency': 'INR', 'intent': 'product_search', 'attributes': {'connectivity': ['USB-C'], 'features': ['fast_charging']}}`

#### Query: *"amazon basics usb-c cable pack"*
- **Detected Failures**: `['attribute_mismatch']`
- **Got**: `{'category': 'cable', 'brand': 'Amazon Basics', 'price_min': None, 'price_max': None, 'currency': 'INR', 'intent': 'product_search', 'attributes': {'connectivity': ['USB', 'USB-C']}, 'hard_filters': {'brand': 'Amazon Basics', 'category': 'cable'}, 'soft_signals': {'connectivity': ['USB', 'USB-C']}, 'confidence': 1.0}`
- **Expected**: `{'category': 'cable', 'brand': 'Amazon Basics', 'price_min': None, 'price_max': None, 'currency': 'INR', 'intent': 'product_search', 'attributes': {'connectivity': ['USB-C']}}`

---

## 7. System Provenance

- **Platform**: Windows-11-10.0.26200-SP0
- **Python Version**: 3.14.2
- **Git Commit**: `untracked_repo`
- **Timestamp**: 2026-08-14T17:26:47.978230+00:00
