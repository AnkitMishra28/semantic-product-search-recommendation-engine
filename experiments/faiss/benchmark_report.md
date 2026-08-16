# FAISS Approximate Nearest Neighbor (ANN) Retrieval Benchmark Report

## 1. Executive Summary & Research Objective

> **Research Question**: *How much retrieval latency can we reduce using FAISS approximate nearest-neighbor indexes while preserving acceptable retrieval recall?*

This experiment benchmarks **Exact FlatIP**, **Hierarchical Navigable Small World (HNSW)**, and **Inverted File Flat (IVFFlat)** indexes on **60,000 normalized product vectors** (dimension = 384, dtype = float32) generated from the `title_brand_category_features` representation of the Amazon Reviews 2023 (Electronics) dataset.

All queries were evaluated against both:
1. **Exact Reference Nearest Neighbors (`exact_flat_ip`)**: Computing true **ANN Recall@K** ($|\text{Approx}_K \cap \text{Exact}_K| / K$).
2. **Task Ground-Truth Relevance Labels**: Computing end-to-end task search quality (**Recall@10**, **Recall@50**, **Recall@100**, **MRR@10**, **NDCG@10**).

---

## 2. Quantitative Benchmark Comparison

| Index | Parameters | ANN Recall@10 | Relevance Recall@10 | MRR@10 | NDCG@10 | Retrieval Latency (p50) | Retrieval Latency (p95) | End-to-End Latency (p95) | Index Memory | Build/Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FlatIP** | `—` | **1.0000** | 0.0333 | 0.0972 | 0.0400 | 4.295 ms | 5.693 ms | 19.31 ms | 87.9 MB | 0.03s |
| **HNSW** | `M=32, efConstruction=200, efSearch=32` | **0.9900** | 0.0333 | 0.0972 | 0.0400 | 0.495 ms | 0.577 ms | 7.92 ms | 103.5 MB | 3.79s |
| **HNSW** | `M=32, efConstruction=200, efSearch=64` | **0.9967** | 0.0333 | 0.0972 | 0.0400 | 0.556 ms | 0.671 ms | 8.17 ms | 103.5 MB | 3.73s |
| **HNSW** | `M=32, efConstruction=200, efSearch=128` | **0.9967** | 0.0333 | 0.0972 | 0.0400 | 0.697 ms | 0.856 ms | 8.18 ms | 103.5 MB | 3.76s |
| **IVFFlat** | `nlist=256, nprobe=4` | **0.9600** | 0.0292 | 0.0889 | 0.0364 | 0.506 ms | 0.599 ms | 7.52 ms | 88.7 MB | 0.73s (0.65s train) |
| **IVFFlat** | `nlist=256, nprobe=16` | **0.9867** | 0.0292 | 0.0889 | 0.0364 | 0.805 ms | 0.952 ms | 8.24 ms | 88.7 MB | 0.72s (0.64s train) |
| **IVFFlat** | `nlist=256, nprobe=32` | **0.9900** | 0.0292 | 0.0889 | 0.0364 | 1.174 ms | 1.409 ms | 8.63 ms | 88.7 MB | 0.72s (0.64s train) |

---

## 3. ANN Recall vs. Search Relevance Recall Distinction

> [!IMPORTANT]
> **Methodological Distinction**:
> - **ANN Recall@K**: Measures vector-space approximation fidelity — how closely the ANN index reproduces the mathematical ground-truth $k$-nearest neighbors produced by exact exhaustive dot product search (`IndexFlatIP`).
> - **Search Relevance Recall@K**: Measures task fulfillment — whether the retrieved $k$ items contain human-annotated relevant products from catalog queries.

### Key Findings:
1. **Near-Perfect ANN Fidelity**: HNSW configurations with $efSearch \ge 64$ achieve **99.67% ANN Recall@10** and **100% preservation of task Relevance Recall@10 (0.0333)**, MRR@10 (0.0972), and NDCG@10 (0.0400).
2. **Massive Vector Search Speedup**: Vector search latency dropped from **4.295 ms** (exact FlatIP p50) to **0.556 ms** (HNSW efSearch=64 p50) and **0.495 ms** (HNSW efSearch=32 p50) — an empirical **~8x to 10x retrieval latency speedup**.
3. **Query Encoding Dominance**: End-to-end latency is predominantly governed by Sentence Transformer inference (~5.40 ms p50), while FAISS HNSW vector search consumes **< 0.8 ms (p95)**, making the vector retrieval stage negligible in the overall online serving budget.
4. **IVFFlat Trade-off**: IVFFlat ($nlist=256$) demonstrated a compact memory footprint (88.7 MB) and rapid indexing (0.65s train, 0.07s add), but achieved lower ANN Recall@10 at $nprobe=4$ (96.0%) and required $nprobe=32$ to reach 99.0% at higher search latency (1.409 ms p95).

---

## 4. Evidence-Based Recommendation

**HNSW (configuration: `{'M': 32, 'efConstruction': 200, 'efSearch': 64, 'metric': 'inner_product'}`)** provides the optimal recall-latency-memory trade-off on this dataset:
- **ANN Recall@10**: 99.67%
- **Search Relevance Recall@10**: 0.0333 (identical to exact reference search)
- **Vector Retrieval Latency (p95)**: 0.671 ms
- **Persisted Index Location**: `data/indexes\hnsw_m32_efc200_efs64.index`

---

## 5. System & Reproducibility Provenance

- **Platform**: Windows-11-10.0.26200-SP0
- **Python Version**: 3.14.2
- **FAISS Version**: 1.15.0
- **Git Commit**: `untracked_repo`
- **Benchmark Timestamp**: 2026-08-14T16:32:03.486617+00:00
