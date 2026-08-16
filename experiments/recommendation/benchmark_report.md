# Track D & Phase 8: Hybrid Personalized Recommendation Engine Benchmark Report

## 1. Executive Summary & Research Objective

> **Research Question**: *Can a hybrid recommendation model combining user preference embeddings, item-item collaborative signals, popularity/rating priors, and diversity-aware reranking outperform simple popularity and content-based baselines while maintaining recommendation diversity?*

In large-scale e-commerce platforms (such as Amazon-inspired product recommendation architectures), personalizing product discovery requires balancing multiple distinct signals:
1. **Semantic Preference Matching (Content-Based)**: Capturing long-term user affinity across technical attributes and categories using dense Sentence Transformer vector profiles.
2. **Co-Occurrence Behavioral Consensus (Collaborative Filtering)**: Discovering complementary and substitute items frequently co-viewed or co-purchased in customer interaction graphs.
3. **Bayesian Popularity Priors**: Scaling recommendation confidence by historical review volume and Bayesian-smoothed rating distributions.
4. **Diversity-Aware Reranking (MMR)**: Balancing relevance with catalog diversity using Maximal Marginal Relevance to prevent homogeneous recommendation lists.

> [!IMPORTANT]
> **Core Finding on Accuracy vs. Coverage Trade-off**: The Multi-Signal Hybrid Recommender does not outperform the popularity baseline on raw held-out accuracy metrics in this evaluation. However, it substantially increases catalog coverage (0.0713 vs. 0.0002) while incorporating personalized semantic and collaborative signals. The results demonstrate a relevance–coverage trade-off rather than universal accuracy superiority. MMR further improves recommendation diversity at additional computational cost.

---

## 2. Master Comparative Benchmark Results Table (Held-Out Test Cohort)

Evaluated on **60,000 catalog products** across **1,621 known evaluation users** under strict chronological zero-leakage evaluation protocol:

| Recommendation Strategy | HitRate@5 | HitRate@10 | HitRate@20 | Recall@10 | Precision@10 | MRR@10 | NDCG@10 | Catalog Coverage@10 | Intra-List Similarity@10 | Category Diversity@10 | Latency (p50) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Popularity Baseline** | 0.0105 | **0.0253** | 0.0302 | **0.0187** | 0.0025 | **0.0050** | **0.0077** | 0.0002 | 0.3057 | 1.9125 | 0.08 ms |
| **B. Content-Based Baseline** | 0.0043 | **0.0062** | 0.0099 | **0.0059** | 0.0006 | **0.0028** | **0.0034** | 0.1310 | 0.7069 | 0.7846 | 3.27 ms |
| **C. Collaborative Filtering** | 0.0006 | **0.0012** | 0.0043 | **0.0009** | 0.0001 | **0.0007** | **0.0007** | 0.0593 | 0.2193 | 2.6074 | 0.18 ms |
| **D. Multi-Signal Hybrid Recommender** | 0.0093 | **0.0136** | 0.0204 | **0.0115** | 0.0014 | **0.0042** | **0.0058** | 0.0713 | 0.6496 | 0.8456 | 9.02 ms |
| **E. Hybrid + MMR Reranking** | 0.0074 | **0.0130** | 0.0204 | **0.0108** | 0.0013 | **0.0039** | **0.0054** | 0.0728 | 0.6225 | 0.9507 | 34.72 ms |
| **F. Validation-Optimal (Popularity Only)** | 0.0105 | **0.0253** | 0.0302 | **0.0187** | 0.0025 | **0.0050** | **0.0077** | 0.0002 | 0.3057 | 1.9125 | 7.06 ms |

---

## 3. Validation Set Ablation Studies

### 3.1 Component Weight Ablation (Validation Cohort)

Hyperparameter selection performed strictly on the validation partition (never tuned on test):

| Configuration Label | Content Weight | Collab Weight | Pop Weight | Rating Weight | Recall@10 | MRR@10 | NDCG@10 | Intra-List Sim@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Content Only | 1.00 | 0.00 | 0.00 | 0.00 | 0.0126 | 0.0073 | **0.0079** | 0.7071 |
| Collaborative Only | 0.00 | 1.00 | 0.00 | 0.00 | 0.0051 | 0.0033 | **0.0033** | 0.3168 |
| Popularity Only | 0.00 | 0.00 | 1.00 | 0.00 | 0.0351 | 0.0082 | **0.0134** | 0.3065 |
| Content + Collaborative | 0.50 | 0.50 | 0.00 | 0.00 | 0.0116 | 0.0052 | **0.0061** | 0.4958 |
| Content + Popularity | 0.50 | 0.00 | 0.50 | 0.00 | 0.0163 | 0.0092 | **0.0102** | 0.5624 |
| Collaborative + Popularity | 0.00 | 0.50 | 0.50 | 0.00 | 0.0158 | 0.0044 | **0.0066** | 0.2425 |
| Full Hybrid (Balanced) | 0.40 | 0.30 | 0.15 | 0.15 | 0.0175 | 0.0099 | **0.0108** | 0.6466 |
| Full Hybrid (Collab-Heavy) | 0.35 | 0.45 | 0.10 | 0.10 | 0.0142 | 0.0064 | **0.0076** | 0.4195 |
| Full Hybrid (Content-Heavy) | 0.45 | 0.25 | 0.15 | 0.15 | 0.0168 | 0.0099 | **0.0106** | 0.6532 |

> **Selected Validation Configuration**: `Popularity Only` with weights: $w_{content}=0.00, w_{collab}=0.00, w_{pop}=1.00, w_{rating}=0.00$.

### 3.2 MMR Diversity Lambda Parameter Sweep

Evaluating trade-off between recommendation relevance (NDCG) and list diversity (ILS):

| MMR $\lambda$ | NDCG@10 | Recall@10 | HitRate@10 | Intra-List Similarity@10 | Category Diversity@10 | Catalog Coverage@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\lambda = 0.00$ | 0.0022 | 0.0034 | 0.0041 | **0.1189** | **2.5212** | 0.0006 |
| $\lambda = 0.25$ | 0.0093 | 0.0216 | 0.0278 | **0.1427** | **2.7142** | 0.0005 |
| $\lambda = 0.50$ | 0.0112 | 0.0255 | 0.0332 | **0.2132** | **2.4132** | 0.0003 |
| $\lambda = 0.70$ | 0.0127 | 0.0286 | 0.0373 | **0.2399** | **2.3172** | 0.0003 |
| $\lambda = 0.75$ | 0.0117 | 0.0278 | 0.0366 | **0.2383** | **2.2946** | 0.0003 |
| $\lambda = 0.85$ | 0.0114 | 0.0277 | 0.0366 | **0.2700** | **2.1146** | 0.0003 |
| $\lambda = 1.00$ | 0.0134 | 0.0351 | 0.0461 | **0.3065** | **1.9917** | 0.0002 |

> **Selected MMR Parameter**: $\lambda = 0.70$ provides a practical relevance–diversity trade-off, reducing intra-list similarity relative to the non-MMR hybrid configuration while maintaining competitive recommendation quality.

---

## 4. Latency Breakdown & Computational Efficiency

Measured on single-thread CPU execution across 100 evaluation users:

| Subsystem Stage | Latency (p50) | Latency (p90) | Latency (p95) | Latency (p99) | Latency (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Generation** | 0.26 ms | 0.32 ms | 0.34 ms | 0.36 ms | 0.27 ms |
| **Content Embedding Search (FAISS)** | 2.29 ms | 2.56 ms | 2.72 ms | 3.07 ms | 2.32 ms |
| **Collaborative Graph Lookup (Sparse)** | 0.33 ms | 0.63 ms | 0.72 ms | 1.20 ms | 0.34 ms |
| **MMR Diversity Reranking** | 2.11 ms | 2.34 ms | 2.42 ms | 2.51 ms | 2.14 ms |
| **Total End-to-End Hybrid (No MMR)** | 6.81 ms | 7.96 ms | 8.16 ms | 8.54 ms | 6.84 ms |

---

## 5. Case Studies & Qualitative Findings

### 1. Collaborative Filtering succeeds where Content-Based fails
- **User**: `AHEZFYPU77XDQZRAWLK7VEVEA5EQ`
- **Explanation**: Collaborative filtering leveraged co-interaction baskets across customers to recommend a complementary product from a different category that semantic vector proximity alone did not connect.

### 2. Content-Based succeeds where Collaborative Filtering fails
- **User**: `AE5UB4BPNWVM6XJCU55PUEZK5PPQ`
- **Explanation**: Semantic vector embeddings matched product attributes and technical features despite sparse historical co-purchase interaction graph links.

### 3. Cold-start routing policy for user with zero interaction history
- **User**: `anonymous_cold_user`
- **Explanation**: Cold-start users with no browsing history seamlessly receive top Bayesian popularity choices scaled by confidence rating priors and diverse top-level categories.

### 4. MMR Diversity Reranking de-duplication effect
- **User**: `AE22Z3RLVIRU6RT5PNRK5CFFNEFQ`
- **Explanation**: MMR penalized redundant near-identical products in the candidate list, introducing greater variety in product categories and brands.
