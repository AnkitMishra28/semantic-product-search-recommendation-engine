# Personalized Recommendation System Architecture & Engineering Guide

## 1. System Overview

The Amazon-inspired personalized recommendation architecture is built as a multi-strategy hybrid architecture combining:
- **Bayesian Popularity Prior** (`PopularityRecommender`)
- **Semantic User Preference Vectors** (`ContentBasedRecommender`)
- **Sparse Item-Item Co-occurrence Graph** (`CollaborativeRecommender`)
- **Multi-Signal Hybrid Combination** (`HybridRecommender`)
- **Maximal Marginal Relevance Diversity Reranker** (`MMRReranker`)
- **Unified Service Interface** (`RecommendationService`)

```
                            USER REQUEST
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              User Profile               Item Anchor
            (Past Asins, Timestamps)    (Anchor Asin)
                    │                         │
          ┌─────────┼─────────┐               │
          ▼         ▼         ▼               ▼
       Content    Collab     Pop          Filtered
       Vector     Graph    Bayesian       Candidate
       Search    Lookup     Prior           Pool
       (FAISS)   (Sparse)                     │
          │         │         │               │
          └─────────┼─────────┘               │
                    ▼                         │
            HYBRID SCORE FUSION ◄─────────────┘
          (Weighted Signal Union)
                    │
                    ▼
             MMR RERANKER
          (Diversity Optimization)
                    │
                    ▼
          FINAL RECOMMENDATIONS
        (With Structured Reasons)
```

---

## 2. Mathematical Formulation

### 2.1 Bayesian Popularity Prior
$$\text{Score}_{\text{pop}}(i) = \frac{v_i \cdot \bar{r}_i + m \cdot C}{v_i + m} \cdot \log(1 + v_i)$$

### 2.2 Semantic User Profile Embedding
$$\mathbf{u} = \frac{\sum_{i \in H_u} w_i \mathbf{e}_i}{\|\sum_{i \in H_u} w_i \mathbf{e}_i\|_2} \quad \text{where } w_i = 2^{-\frac{\Delta t_i}{t_{\text{half}}}} \cdot \frac{r_i}{5.0}$$

### 2.3 Sparse Item-Item Cosine Similarity
$$\text{Sim}(i, j) = \frac{C_{i, j}}{\sqrt{C_{i, i} \cdot C_{j, j}}}$$

### 2.4 Hybrid Score Fusion
$$S_{\text{hybrid}}(u, d) = w_{\text{content}} \cdot \hat{S}_{\text{content}}(u, d) + w_{\text{collab}} \cdot \hat{S}_{\text{collab}}(u, d) + w_{\text{pop}} \cdot \hat{S}_{\text{pop}}(d) + w_{\text{rating}} \cdot \hat{S}_{\text{rating}}(d)$$

### 2.5 Maximal Marginal Relevance (MMR)
$$\text{MMR}(d) = \lambda \cdot S_{\text{hybrid}}(u, d) - (1 - \lambda) \cdot \max_{s \in S} \text{CosineSim}(\mathbf{e}_d, \mathbf{e}_s)$$

---

## 3. Empirical Accuracy & Diversity Characteristics

The Multi-Signal Hybrid Recommender does not outperform the popularity baseline on raw held-out accuracy metrics in this evaluation (Popularity Recall@10 = 0.0187 vs. Hybrid Recall@10 = 0.0115). However, it substantially increases catalog coverage (0.0713 vs. 0.0002) while incorporating personalized semantic and collaborative signals. The results demonstrate a relevance–coverage trade-off rather than universal accuracy superiority. MMR diversity reranking ($\lambda = 0.70$) further reduces intra-list similarity (ILS = 0.2399 vs. 0.3065) at additional computational cost.

---

## 4. Reproducibility

Run the benchmark from CLI:
```bash
python scripts/run_recommendation_benchmark.py
```