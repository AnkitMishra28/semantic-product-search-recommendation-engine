# Hybrid Lexical + Dense Candidate Retrieval & Reciprocal Rank Fusion

## 1. Motivation & Problem Formulation

First-stage candidate retrieval in e-commerce search must satisfy two conflicting requirements under strict latency budgets:
1. **High Precision on Exact Identifiers**: Recognizing exact technical specifications, model identifiers (e.g. `RTX 4060`, `Cat8`, `HDMI 2.1`), and brand names.
2. **High Recall on Semantic & Colloquial Intent**: Capturing conceptual use cases (e.g. *"for programming"*, *"for travel"*, *"for running"*) where relevant items may not repeat the exact query phrase.

### The First-Stage Retrieval Bottleneck
In a multi-stage funnel architecture:
- **BM25 alone** suffers from vocabulary mismatch and contextual intent blindness (Stage-1 Recall@100 = 18.75%).
- **Dense Bi-Encoder alone** struggles with fine-grained technical identifiers and exact numerical boundaries (Stage-1 Recall@100 = 19.58%).
- **Stage 2 Cross-Encoder rerankers cannot score items that were not retrieved in Stage 1**.

Therefore, Phase 7 introduces a **hybrid candidate generation layer** combining BM25 and FAISS HNSW via **Reciprocal Rank Fusion (RRF)** before neural reranking.

---

## 2. Target Architecture

```
                        USER QUERY
                            │
                            ▼
                   QUERY UNDERSTANDING
             (Deterministic Hard Filters)
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
                BM25                FAISS
             Top-K=100           Top-K=100
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    RRF SCORE FUSION
                  (RRF Constant k=60)
                            │
                     Candidate Pool
                         Top-100
                            │
                            ▼
                      Cross-Encoder
                         Top-20
                            │
                            ▼
                        Final Top-K
```

---

## 3. Reciprocal Rank Fusion (RRF) Mathematics

Given a set of retrievers $R = \{\text{bm25}, \text{dense}\}$ and an arbitrary document $d$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where:
- $\text{rank}_r(d) \in \{1, 2, \dots, K\}$ is the 1-indexed rank of document $d$ within retriever $r$'s candidate list.
- If document $d$ is missing from retriever $r$, its term is omitted (or rank $\to \infty$).
- $k$ is a configurable smoothing parameter (default $k=60$, based on Cormack et al., SIGIR 2009).

### Why Rank-Based RRF over Linear Score Normalization?
1. **Incommensurate Score Distributions**: BM25 produces unbounded positive scores $[0, \infty)$, while dense bi-encoders produce cosine/inner-product scores $[-1, 1]$.
2. **Distribution Instability**: Min-max and z-score normalization are vulnerable to score outliers per query.
3. **Zero Parameter Tuning**: RRF requires no manual alpha score weights and is invariant to score calibration differences across query types.

---

## 4. Empirical Evaluation Results

### Master Benchmark Comparison Table

Evaluated on **60,000 products** across **30 catalog-grounded evaluation queries**:

| Architecture Pipeline | Stage-1 Recall@100 | Stage-2 Recall@20 | MRR@10 | NDCG@5 | NDCG@10 | Latency (p50) | Latency (p95) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. BM25 Only** | 0.1875 | N/A (0.0625 @20) | 0.1140 | 0.0574 | 0.0512 | 287.92 ms | 445.40 ms |
| **B. Dense FAISS Only** | 0.1958 | N/A (0.0458 @20) | 0.0972 | 0.0537 | 0.0400 | 42.81 ms | 62.46 ms |
| **C. Hybrid RRF (BM25 + FAISS)** | **0.1958** | N/A (0.0667 @20) | **0.1159** | **0.0526** | **0.0448** | **336.30 ms** | **493.75 ms** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D. Dense $\to$ Cross-Encoder** | 0.1958 | 0.0500 | 0.1528 | 0.0707 | 0.0584 | 8890.09 ms | 9892.61 ms |
| **E. Hybrid RRF $\to$ Cross-Encoder** | **0.1958** | **0.0542** | **0.1528** | **0.0707** | **0.0584** | **9188.81 ms** | **10199.44 ms** |

> [!IMPORTANT]
> **Core Research Finding**: Hybrid RRF does not improve Stage-1 Recall@100 over Dense FAISS in this evaluation (both achieve 0.1958). However, Hybrid improves first-stage MRR@10 from 0.0972 to 0.1159 and improves downstream Stage-2 Recall@20 from 0.0500 to 0.0542 after Cross-Encoder reranking. The results therefore indicate improved candidate ranking and complementary retrieval rather than an increase in the Top-100 recall ceiling.

---

## 5. Complementary Recovery Analysis

Across all **240 ground truth relevant product annotations**:

- **Recovered by Both retrievers**: 31 items (12.92%)
- **Recovered by BM25 ONLY**: 14 items (5.83%)
- **Recovered by Dense FAISS ONLY**: 16 items (6.67%)
- **Missed by Both**: 179 items (74.58%)
- **Total in Untruncated Candidate Union**: **61 items** (**25.42%**)

> [!NOTE]
> **Complementary Coverage vs. Truncated Funnel**: The untruncated BM25 ∪ Dense candidate union captures 25.42% of relevant instances, compared with 19.58% for Dense alone. After RRF ranking and truncation to the Top-100 candidate pool, Hybrid achieves Recall@100 = 19.58%, tying Dense on this metric.

---

## 6. RRF Ablation Analysis ($k \in \{10, 30, 60, 100\}$)

| RRF $k$ Parameter | Recall@10 | Recall@20 | Recall@50 | Recall@100 | MRR@10 | NDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **k = 10** | 0.0417 | 0.0583 | 0.1000 | 0.1958 | 0.1194 | 0.0480 |
| **k = 30** | 0.0417 | 0.0667 | 0.1083 | 0.1958 | 0.1159 | 0.0474 |
| **k = 60** | 0.0375 | 0.0667 | 0.1125 | 0.1958 | 0.1159 | 0.0448 |
| **k = 100** | 0.0375 | 0.0625 | 0.1125 | 0.1958 | 0.1159 | 0.0448 |

> [!TIP]
> **Ablation Insight**: Among the evaluated configurations, no single RRF constant dominates every metric. k=10 achieves the highest MRR@10, while k=30 and k=60 provide stronger Recall@20/50 performance. k=60 is retained as the default conventional RRF setting for the remainder of the experiment.

---

## 7. Representative Failure Mode Analysis

### 1. BM25 succeeds, dense fails
- **Query**: *"magnetic wireless car charger mount for iPhone"* (`q_018`)
- **Product**: `B08N6PZR6Y` (*"JETech Wireless FM Transmitter Radio Car Kit for Smart Phones Bundle with 3.5mm Audio Plug and Car Charger (Black)"*)
- **Ranks**: BM25: `9` | Dense: `None` | Hybrid: `39` | Cross-Encoder: `56`
- **Explanation**: BM25 captures exact lexical keywords and technical terms directly present in product title/features that the dense embedding space placed outside the top vector neighborhood.

### 2. Dense succeeds, BM25 fails
- **Query**: *"external DVD drive USB 3.0 portable optical drive"* (`q_025`)
- **Product**: `B00E6GUJ4G` (*"External USB DVD/CD"*)
- **Ranks**: BM25: `None` | Dense: `28` | Hybrid: `59` | Cross-Encoder: `58`
- **Explanation**: Dense embeddings understand semantic synonyms and contextual use-case intent where the product description uses alternative terminology rather than the exact query keywords.

### 3. Both succeed
- **Query**: *"high capacity power bank fast charging 20000mAh"* (`q_007`)
- **Product**: `B0BHY8TMT7` (*"JBL Pulse 4 - Waterproof Portable Bluetooth Speaker with Light Show and InfinityLab InstantGo 10000mAh Wireless Power Bank (White)"*)
- **Ranks**: BM25: `2` | Dense: `4` | Hybrid: `2` | Cross-Encoder: `4`
- **Explanation**: Strong dual agreement: item has high lexical term density and strong embedding geometric proximity, receiving reciprocal rank boosts from both systems into top ranks.

### 4. Both fail
- **Query**: *"noise cancelling bluetooth headphones for travel"* (`q_001`)
- **Product**: `B0BW4PFM58` (*"OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)"*)
- **Ranks**: BM25: `None` | Dense: `None` | Hybrid: `None` | Cross-Encoder: `None`
- **Explanation**: Extreme vocabulary gap combined with sparse product metadata where neither lexical terms nor bi-encoder vector representations captured the association within top-100 candidates.

### 5. Hybrid candidate inclusion where one individual retriever fails
- **Query**: *"magnetic wireless car charger mount for iPhone"* (`q_018`)
- **Product**: `B08N6PZR6Y` (*"JETech Wireless FM Transmitter Radio Car Kit for Smart Phones Bundle with 3.5mm Audio Plug and Car Charger (Black)"*)
- **Ranks**: BM25: `9` | Dense: `None` | Hybrid: `39` | Cross-Encoder: `56`
- **Explanation**: The product was retrieved exclusively by BM25 and therefore entered the hybrid candidate pool despite being absent from Dense Top-100. However, it ranked 39th under RRF and 56th after Cross-Encoder reranking, so this case demonstrates candidate-pool inclusion rather than successful final Top-20 ranking.

### 6. Hybrid fails despite both retrievers retrieving candidates
- **Query**: *"portable bluetooth speaker waterproof with deep bass"* (`q_006`)
- **Product**: `B099V8GPR4` (*"JBL Flip 4, Black - Waterproof, Portable & Durable Bluetooth Speaker - Up to 12 Hours of Wireless Streaming - Includes Noise-Cancelling Speakerphone, Voice Assistant & JBL Connect+"*)
- **Ranks**: BM25: `68` | Dense: `65` | Hybrid: `47` | Cross-Encoder: `95`
- **Explanation**: When a document appears at the very tail of both retriever rankings (e.g. rank 60+ in both), the combined RRF score is lower than high single-retriever candidates (e.g. rank 2 in one retriever yields 1/62 = 0.016 vs 0.013).

---

## 8. Limitations & Scope

1. **Dataset Scope**: Evaluated on 60,000 products from the Amazon Reviews 2023 Electronics domain across 30 catalog-grounded queries.
2. **Latency Considerations**: The target production latency budget is ≤50 ms; the current research prototype does not meet this target because BM25 is implemented as an in-memory Python retrieval layer. Production deployment would require an optimized inverted-index implementation such as Lucene/OpenSearch/Elasticsearch.
3. **Incommensurate Candidate Depths**: Equal Top-100 allocation from both retrievers was used; future work can explore adaptive allocation based on Query Understanding intent classification.
