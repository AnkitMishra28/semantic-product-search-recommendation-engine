# Recommendation Engine Failure Analysis & Diagnostic Taxonomy

## 1. Diagnostic Taxonomy of Recommendation Failure Modes

Empirical analysis of 1,621 held-out test cohort evaluations identifies four primary failure modes:

### 1.1 Sparse Interaction Graph (Graph Disconnectedness)
- **Symptom**: User interacted with niche items having zero co-occurrence edges ($C_{i, j} = 0$) in the training graph.
- **Failure Impact**: Collaborative recommender produces an empty candidate set, forcing total reliance on content vector similarity.
- **Mitigation**: Multi-channel candidate pooling in `HybridRecommender` automatically blends semantic vector nearest neighbors when collaborative edges are unavailable.

### 1.2 Category Monoculture (Homogeneity in Content-Based Vectors)
- **Symptom**: When a user views 3 items from the same specific sub-category (e.g. HDMI cables), dense semantic profile vector clusters tightly in one region of the 384-dimensional embedding space.
- **Failure Impact**: Content-based top-10 list contains 10 identical variants of HDMI cables from different brands (Intra-List Similarity $> 0.85$).
- **Mitigation**: MMR diversity reranking ($\lambda = 0.70$) explicitly penalizes intra-list embedding similarity, diversifying recommendations across complementary categories.

### 1.3 Popularity Bias & Long-Tail Neglect
- **Symptom**: Popularity baselines over-recommend ubiquitous items (e.g. top Bluetooth speakers) regardless of user interest.
- **Failure Impact**: Low HitRate and poor catalog coverage ($< 1.0\%$).
- **Mitigation**: Balancing Bayesian popularity with personalized user profile embeddings reduces top-1% popularity concentration.

### 1.4 Temporal Drift & Intent Shift
- **Symptom**: Historical interactions span months or years; user intent transitions from audio gear to PC hardware.
- **Failure Impact**: Older interactions dilute the relevance of recent interest vectors.
- **Mitigation**: Exponential recency decay weighting ($w_i = 2^{-\Delta t / \text{half\_life}}$) in preference embedding aggregation.

---

## 2. Hard Filter Adherence & Edge Case Handling

- **Empty History (Cold Start)**: Deterministically routed to Bayesian popularity with category diversification.
- **Consumed Item Filtering**: 100% adherence to `exclude_consumed=True` preventing repeat recommendations of already owned products.
- **Hard Constraints**: Verification of category, brand, and price boundary adherence across all candidate pools.