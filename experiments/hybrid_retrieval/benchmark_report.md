# Track E: Hybrid First-Stage Retrieval (BM25 + FAISS + Reciprocal Rank Fusion) Benchmark Report

## 1. Executive Summary & Research Objective

> **Research Question**: *Can hybrid lexical + dense retrieval improve candidate recall over either BM25 or dense retrieval alone, particularly for exact product attributes, brands, model numbers, and semantic intent?*

In modern e-commerce search architectures (inspired by Amazon multi-stage search pipelines), candidate generation is the critical first stage. **Stage 2 Cross-Encoder reranking can only score candidates that survive Stage 1**. If a relevant product is missing from first-stage retrieval, it is impossible for downstream models to recover it.

This experiment evaluates a hybrid candidate generation layer combining:
1. **Lexical Retrieval (BM25 Okapi)**: High precision on exact keywords, model numbers, brand identifiers, and technical specifications.
2. **Dense Vector Retrieval (FAISS HNSW)**: High recall on semantic intent, colloquial synonyms, and conceptual descriptions.
3. **Reciprocal Rank Fusion (RRF)**: Parameterized rank-based score fusion ($k=60$) producing a balanced top-100 candidate pool for Stage 2 Cross-Encoder reranking.

---

## 2. Master Comparative Benchmark Results Table

Evaluated on **60,000 products** from the Amazon Reviews 2023 Electronics dataset across **30 catalog-grounded evaluation queries**:

| Architecture Pipeline | Stage-1 Recall@100 | Stage-2 Recall@20 | MRR@10 | NDCG@5 | NDCG@10 | Latency (p50) | Latency (p95) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. BM25 Only** | 0.1875 | N/A (0.0625 @20) | 0.1140 | 0.0574 | 0.0512 | 287.92 ms | 445.40 ms |
| **B. Dense FAISS Only** | 0.1958 | N/A (0.0458 @20) | 0.0972 | 0.0537 | 0.0400 | 42.81 ms | 62.46 ms |
| **C. Hybrid RRF (BM25 + FAISS)** | **0.1958** | N/A (0.0667 @20) | **0.1159** | **0.0526** | **0.0448** | **336.30 ms** | **493.75 ms** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D. Dense $\to$ Cross-Encoder** | 0.1958 | 0.0500 | 0.1528 | 0.0707 | 0.0584 | 8890.09 ms | 9892.61 ms |
| **E. Hybrid RRF $\to$ Cross-Encoder** | **0.1958** | **0.0542** | **0.1528** | **0.0707** | **0.0584** | **9188.81 ms** | **10199.44 ms** |

---

## 3. Core Research Findings: Dense+CE vs. Hybrid+CE

Comparing **Dense $\to$ Cross-Encoder** against **Hybrid RRF $\to$ Cross-Encoder**:

- **Stage-1 Recall@100 (Candidate Funnel)**: `0.1958` $\to$ `0.1958` (**+0.0000** absolute | **+0.00%** relative)
- **Stage-2 Recall@20 (Final Top-20 List)**: `0.0500` $\to$ `0.0542` (**+0.0042** absolute | **+8.40%** relative)
- **MRR@10 (First Relevant Rank)**: `0.1528` $\to$ `0.1528` (**+0.0000** absolute | **+0.00%** relative)
- **NDCG@10 (Overall Ranking Quality)**: `0.0584` $\to$ `0.0584` (**+0.0000** absolute | **+0.00%** relative)

> [!IMPORTANT]
> **Scientific Finding**: Hybrid RRF does not improve Stage-1 Recall@100 over Dense FAISS in this evaluation (both achieve 0.1958). However, Hybrid improves first-stage MRR@10 from 0.0972 to 0.1159 and improves downstream Stage-2 Recall@20 from 0.0500 to 0.0542 after Cross-Encoder reranking. The results therefore indicate improved candidate ranking and complementary retrieval rather than an increase in the Top-100 recall ceiling.

---

## 4. Complementary Retrieval & Overlap Analysis

- **Mean Candidate Pool Size Before Fusion (Union)**: **155.43** products / query
- **Mean Candidate Overlap Size (Intersection)**: **44.57** products / query
- **Mean Jaccard Candidate Similarity**: **0.3083**

### Ground Truth Relevant Items Recovery Distribution

Across all **240 ground truth relevant product instances**:

| Recovery Category | Relevant Count | Percentage of Total Relevant |
| :--- | :--- | :--- |
| **Recovered by BOTH BM25 and Dense** | 31 | 12.92% |
| **Recovered by BM25 ONLY** | 14 | 5.83% |
| **Recovered by Dense FAISS ONLY** | 16 | 6.67% |
| **Missed by BOTH Retrievers** | 179 | 74.58% |
| **Total Captured in Untruncated Union Pool** | **61** | **25.42%** |

> [!NOTE]
> **Complementary Coverage vs. Truncated Funnel**: The untruncated BM25 ∪ Dense candidate union captures 25.42% of relevant instances, compared with 19.58% for Dense alone. After RRF ranking and truncation to the Top-100 candidate pool, Hybrid achieves Recall@100 = 19.58%, tying Dense on this metric.

---

## 5. RRF Constant $k$ Ablation Study

Reciprocal Rank Fusion uses smoothing constant $k$ to balance shallow vs deep rank contributions:
$$RRF(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

| RRF $k$ Parameter | Recall@10 | Recall@20 | Recall@50 | Recall@100 | MRR@10 | NDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **k = 10** | 0.0417 | 0.0583 | 0.1000 | 0.1958 | 0.1194 | 0.0480 |
| **k = 30** | 0.0417 | 0.0667 | 0.1083 | 0.1958 | 0.1159 | 0.0474 |
| **k = 60** | 0.0375 | 0.0667 | 0.1125 | 0.1958 | 0.1159 | 0.0448 |
| **k = 100** | 0.0375 | 0.0625 | 0.1125 | 0.1958 | 0.1159 | 0.0448 |

> [!TIP]
> **Ablation Insight**: Among the evaluated configurations, no single RRF constant dominates every metric. k=10 achieves the highest MRR@10, while k=30 and k=60 provide stronger Recall@20/50 performance. k=60 is retained as the default conventional RRF setting for the remainder of the experiment.

---

## 6. Latency Benchmark Breakdown (Systems Performance)

| Pipeline Component | Latency (p50) | Latency (p95) | Latency (p99) | Latency (Mean) |
| :--- | :--- | :--- | :--- | :--- |
| **BM25 Inverted Search** | 287.92 ms | 445.40 ms | 479.36 ms | 288.51 ms |
| **Dense Query Encoding** | 40.96 ms | 60.37 ms | 70.35 ms | 40.54 ms |
| **FAISS HNSW Search** | 1.98 ms | 2.41 ms | 2.78 ms | 1.95 ms |
| **Total Dense First Stage** | 42.81 ms | 62.46 ms | 72.32 ms | 42.49 ms |
| **RRF Score Fusion** | **3.048 ms** | **3.893 ms** | **5.081 ms** | **3.016 ms** |
| **Total Hybrid First Stage** | **336.30 ms** | **493.75 ms** | **530.21 ms** | **334.02 ms** |
| **Cross-Encoder Scoring ($N=100$)** | 8847.54 ms | 9840.21 ms | 10091.30 ms | 8292.42 ms |
| **Dense $\to$ CE End-to-End** | 8890.09 ms | 9892.61 ms | 10143.92 ms | 8334.91 ms |
| **Hybrid $\to$ CE End-to-End** | **9188.81 ms** | **10199.44 ms** | **10485.28 ms** | **8626.44 ms** |

> [!NOTE]
> **Latency Budget & Hardware Profile**: The target production latency budget is ≤50 ms; the current research prototype does not meet this target because BM25 is implemented as an in-memory Python retrieval layer. Production deployment would require an optimized inverted-index implementation such as Lucene/OpenSearch/Elasticsearch.

---

## 7. Representative Failure & Success Case Studies

### 1. BM25 succeeds, dense fails
- **Query**: *"magnetic wireless car charger mount for iPhone"* (`q_018`)
- **Product**: `B08N6PZR6Y` — *"JETech Wireless FM Transmitter Radio Car Kit for Smart Phones Bundle with 3.5mm Audio Plug and Car Charger (Black)"*
- **Retrieval Provenance**: BM25 Rank: `9` | Dense Rank: `None` | Hybrid RRF Rank: `39` | Cross-Encoder Rank: `56`
- **Technical Rationale**: BM25 captures exact lexical keywords and technical terms directly present in product title/features that the dense embedding space placed outside the top vector neighborhood.

### 2. Dense succeeds, BM25 fails
- **Query**: *"external DVD drive USB 3.0 portable optical drive"* (`q_025`)
- **Product**: `B00E6GUJ4G` — *"External USB DVD/CD"*
- **Retrieval Provenance**: BM25 Rank: `None` | Dense Rank: `28` | Hybrid RRF Rank: `59` | Cross-Encoder Rank: `58`
- **Technical Rationale**: Dense embeddings understand semantic synonyms and contextual use-case intent where the product description uses alternative terminology rather than the exact query keywords.

### 3. Both succeed
- **Query**: *"high capacity power bank fast charging 20000mAh"* (`q_007`)
- **Product**: `B0BHY8TMT7` — *"JBL Pulse 4 - Waterproof Portable Bluetooth Speaker with Light Show and InfinityLab InstantGo 10000mAh Wireless Power Bank (White)"*
- **Retrieval Provenance**: BM25 Rank: `2` | Dense Rank: `4` | Hybrid RRF Rank: `2` | Cross-Encoder Rank: `4`
- **Technical Rationale**: Strong dual agreement: item has high lexical term density and strong embedding geometric proximity, receiving reciprocal rank boosts from both systems into top ranks.

### 4. Both fail
- **Query**: *"noise cancelling bluetooth headphones for travel"* (`q_001`)
- **Product**: `B0BW4PFM58` — *"OontZ Angle 3 Bluetooth Speaker, Portable Wireless Bluetooth 5.0 Speaker, 10 Watts, Crystal Clear Stereo Sound, Rich Bass, IPX5 Water Resistant, Loud Portable Bluetooth Speaker (Black)"*
- **Retrieval Provenance**: BM25 Rank: `None` | Dense Rank: `None` | Hybrid RRF Rank: `None` | Cross-Encoder Rank: `None`
- **Technical Rationale**: Extreme vocabulary gap combined with sparse product metadata where neither lexical terms nor bi-encoder vector representations captured the association within top-100 candidates.

### 5. Hybrid candidate inclusion where one individual retriever fails
- **Query**: *"magnetic wireless car charger mount for iPhone"* (`q_018`)
- **Product**: `B08N6PZR6Y` — *"JETech Wireless FM Transmitter Radio Car Kit for Smart Phones Bundle with 3.5mm Audio Plug and Car Charger (Black)"*
- **Retrieval Provenance**: BM25 Rank: `9` | Dense Rank: `None` | Hybrid RRF Rank: `39` | Cross-Encoder Rank: `56`
- **Technical Rationale**: The product was retrieved exclusively by BM25 and therefore entered the hybrid candidate pool despite being absent from Dense Top-100. However, it ranked 39th under RRF and 56th after Cross-Encoder reranking, so this case demonstrates candidate-pool inclusion rather than successful final Top-20 ranking.

### 6. Hybrid fails despite both retrievers retrieving candidates
- **Query**: *"portable bluetooth speaker waterproof with deep bass"* (`q_006`)
- **Product**: `B099V8GPR4` — *"JBL Flip 4, Black - Waterproof, Portable & Durable Bluetooth Speaker - Up to 12 Hours of Wireless Streaming - Includes Noise-Cancelling Speakerphone, Voice Assistant & JBL Connect+"*
- **Retrieval Provenance**: BM25 Rank: `68` | Dense Rank: `65` | Hybrid RRF Rank: `47` | Cross-Encoder Rank: `95`
- **Technical Rationale**: When a document appears at the very tail of both retriever rankings (e.g. rank 60+ in both), the combined RRF score is lower than high single-retriever candidates (e.g. rank 2 in one retriever yields 1/62 = 0.016 vs 0.013).

---

## 8. System Provenance & Scientific Reproducibility

- **Platform**: Windows-11-10.0.26200-SP0
- **Python Version**: 3.14.2
- **PyTorch Version**: 2.10.0+cpu
- **Git Commit**: `untracked_repo`
- **Timestamp**: 2026-08-15T07:14:32.991821+00:00
