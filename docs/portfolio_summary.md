# Portfolio & Recruiter Executive Summary

**Project**: Amazon-Scale Semantic Product Search & Recommendation Engine  
**Focus**: Information Retrieval, Neural Ranking, Graph Recommendations, Production ML Engineering  
**Dataset**: Amazon Reviews 2023 (60,000 Electronics Products, 31,286 Interactions, 1,621 Users)  
**Status**: 🟢 Complete, Reproducible & Release-Ready

---

## A. Two-Line Project Summary

An end-to-end, multi-stage e-commerce discovery engine combining lexical BM25, dense FAISS HNSW vector search, neural Cross-Encoder reranking, and multi-signal graph recommendations with evidence-grounded explainability. Evaluated across 10 reproducible benchmark tracks on 60,000 real Amazon catalog items with an interactive Next.js research dashboard and hardened FastAPI backend.

---

## B. 5 High-Impact Resume Bullets

- **Architected Multi-Stage Hybrid Search**: Engineered a production-grade dual-track retrieval engine combining Lexical BM25 and Dense FAISS HNSW vector search (384-dim) via Reciprocal Rank Fusion ($k=10/60$), expanding untruncated candidate pool recall by **+29.8%** (19.58% to **25.42%**) over 60,000 Amazon Electronics products.
- **Boosted Ranking Relevance with Cross-Encoders**: Implemented Stage-2 neural cross-attention reranking (`ms-marco-MiniLM-L-6-v2`), translating first-stage candidate diversity gains into a **+19.2% MRR@10 improvement** (0.0972 to **0.1159**) and an **+8.33% downstream Recall@20 gain** (5.00% to **5.42%**).
- **Expanded Recommendation Catalog Coverage by 356x**: Designed a Multi-Signal Hybrid Recommender blending user-item co-purchases, item-item co-occurrence graphs, and dense semantic embeddings with Maximal Marginal Relevance (MMR $\lambda=0.70$), expanding active catalog discovery from **0.02% to 7.13%** (4,280 unique items) while reducing intra-list redundancy to **0.240**.
- **Built Hallucination-Free Explainability**: Developed a deterministic, evidence-grounded explanation service that verifies query intent tokens, spec constraints, and category hierarchies against true catalog metadata, achieving 100% attribute alignment without external LLM latency or hallucination risk.
- **Engineered Full-Stack Production System**: Delivered a typed FastAPI microservice and Next.js 14 research dashboard featuring interactive parameter sweep charts, latency percentiles, and an experiment registry; validated with **125 backend tests**, 100% Playwright browser automation, and a one-command validation suite (`scripts/validate_all.py`).

---

## C. 30-Second Recruiter Pitch

> "I built an Amazon-scale semantic search and recommendation platform that solves the classic e-commerce vocabulary mismatch problem. It uses a two-stage hybrid retrieval architecture—combining BM25 keyword search and FAISS vector search, followed by deep neural cross-encoder reranking. On 60,000 real Amazon products, it achieved a 19.2% boost in MRR and expanded recommendation catalog discovery by 356 times. The entire system is live with a FastAPI backend, a Next.js 14 dashboard, and 125 automated tests."

---

## D. 60-Second Technical Overview

> "In e-commerce search, bi-encoder vector retrieval struggles on exact part numbers and brand names, while BM25 keyword matching fails on colloquial natural-language intent. I designed a dual-track Stage-1 candidate generator that queries BM25 and a 384-dimensional FAISS HNSW index in parallel, fusing the top 100 candidates with Reciprocal Rank Fusion. This expands the candidate pool coverage to 25.42%.  
> Next, a Stage-2 Cross-Encoder reranks the top 50 candidates using joint cross-attention, boosting downstream Recall@20 from 5.00% to 5.42%. For recommendations, I combined co-purchase interaction graphs with semantic embeddings and MMR diversity reranking, reaching an optimal balance at $\lambda=0.70$. Everything is instrumented with millisecond stage profiling, evidence-grounded explainability, and 10 tracked offline benchmark JSON artifacts."

---

## E. 2-Minute Applied Scientist Deep Dive

> "The core scientific problem I tackled is optimizing the trade-off between first-stage candidate recall depth and second-stage neural ranking precision under computational constraints.  
> Evaluating against 60,000 Amazon Electronics products and 30 curated benchmark queries, I demonstrated that BM25 and Dense bi-encoders retrieve largely complementary relevance sets with only a 30.8% Jaccard overlap: BM25 uniquely recovers 5.83% of relevant items and Dense uniquely recovers 6.67%. Fusing them via RRF expands the untruncated candidate pool from 19.58% to 25.42%.  
> When the Stage-2 Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) rescores this fused pool, the extra candidate diversity directly translates into an 8.33% downstream Recall@20 gain and lifts MRR@10 from 0.0972 to 0.1159. Through an RRF $k$ ablation across $\{10, 30, 60, 100\}$, I found $k=10$ provides the steepest top-rank discount on this catalog, while $k=60$ serves as a conservative default.  
> On the recommendation side across 1,621 evaluation users, while popularity baselines show high raw precision on dense head items (0.0025), they only touch 0.02% of the catalog. My Multi-Signal Hybrid recommender expands coverage to 7.13% (4,280 items), and an empirical MMR $\lambda$ sweep identified $\lambda=0.70$ as the Pareto-optimal operating point for relevance vs intra-list category diversity."

---

## F. Key Machine Learning Concepts Demonstrated

1. **Two-Stage Information Retrieval**: Candidate generation (bi-encoder + lexical) vs high-capacity neural rescoring (cross-encoder).
2. **Dense Vector Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` 384-dim normalized representations.
3. **Approximate Nearest Neighbor (ANN)**: FAISS HNSW ($M=32$, $\text{efConstruction}=200$, $\text{efSearch}=64$) achieving 99.8% recall at 680 QPS.
4. **Rank Fusion**: Reciprocal Rank Fusion (RRF) with parameter smoothing ablation.
5. **Re-ranking with Cross-Encoders**: Joint query-document cross-attention (`ms-marco-MiniLM-L-6-v2`).
6. **Multi-Signal Recommendations**: Blending collaborative filtering co-occurrence graphs with dense content embeddings.
7. **Diversity Optimization**: Maximal Marginal Relevance (MMR) for anti-redundancy and long-tail exploration.
8. **Intent & Slot Understanding**: Real-time query normalization, intent classification (94.2% accuracy), and price/rating constraint extraction.
9. **Grounded Explainability**: Rule-based feature-to-query constraint verification preventing LLM hallucinations.
10. **Offline Evaluation Metrics**: NDCG@K, MRR@K, Recall@K, HitRate@K, Intra-List Cosine Distance, Catalog Coverage.

---

## G. Key Software Engineering Concepts Demonstrated

1. **Microservice Architecture**: Asynchronous FastAPI ASGI backend with lifespan singleton resource management.
2. **Modern Frontend**: Next.js 14 App Router, TypeScript 5, TailwindCSS, and custom responsive SVG chart visualizers.
3. **Container Orchestration**: Multi-stage `Dockerfile` and `docker-compose.yml` configurations.
4. **Resilient API Contracts**: Pydantic v2 domain schemas, normalized `ApiError` handling, and timeout safeguards.
5. **Deterministic Testing**: 125 Pytest backend unit/integration tests and automated Playwright browser test suites.
6. **Unified Reproducibility**: One-command validation runner (`scripts/validate_all.py`) checking environment, artifacts, tests, builds, and live endpoints.
7. **Security & Zero Credentials**: Zero committed secrets, environment variable configuration template (`.env.example`), and protected `.gitignore`.

---

## H. Most Important Empirical Findings

1. **Dual-Track Complementarity**: BM25 and Dense FAISS have a 30.8% Jaccard candidate overlap; fusing them unlocks **25.42% candidate coverage** (+29.8% expansion).
2. **Downstream Cross-Encoder Gain**: Neural reranking over fused candidates delivers **+8.33% Recall@20** and **+19.2% MRR@10**.
3. **Catalog Discovery Expansion**: Multi-Signal Hybrid recommendations expand catalog coverage from **0.02% (Popularity) to 7.13% (Hybrid)**—a 356x expansion.
4. **Optimal Diversity Operating Point**: MMR with **$\lambda=0.70$** balances high precision (0.0038) with rich category diversity (2.32) and reduced intra-list similarity (0.240).

---

## I. Honest Scientific Limitations & Trade-offs

1. **CPU Cross-Encoder Latency**: Sequential CPU cross-attention over 50 candidate pairs requires ~1.1 seconds. In a GPU cloud environment with TensorRT or ONNX Runtime batching, latency is expected to drop below 15 ms.
2. **Binary Review Relevancy**: Benchmark relevance labels are derived from user review interaction graphs rather than multi-graded human expert editorial judgments.
3. **Cold-Start ASINs**: Items with zero review interactions rely solely on dense semantic text embedding similarity.

---

## J. Likely Applied Scientist / ML Interview Questions

1. *Why did you choose Reciprocal Rank Fusion over a linear weighted combination of BM25 and Dense scores?*
2. *How does candidate depth at Stage 1 affect Cross-Encoder precision and end-to-end latency at Stage 2?*
3. *Why does the Popularity baseline achieve higher raw precision than collaborative or hybrid recommender models on this dataset?*
4. *How do you prevent hallucinations in e-commerce search explanations?*
5. *Why did you choose FAISS HNSW over IVFFlat or FlatIP for 60,000 vectors?*

---

## K. Strong Answers & Talking Points

1. **RRF vs Linear Blend**:  
   *"BM25 scores and cosine similarities operate on completely different, non-calibrated numerical scales that shift across query lengths. Reciprocal Rank Fusion is scale-invariant because it operates purely on ordinal ranks ($\frac{1}{k + r}$), making it robust across queries without requiring per-query score normalization."*
2. **Candidate Depth Trade-offs**:  
   *"Stage-1 Recall@100 defines the theoretical upper bound for all downstream stages. Scoring 100 candidates with a Cross-Encoder yields the highest recall but increases latency linearly. Truncating to the top 50 candidates for Stage 2 preserves ~92% of relevant candidates while cutting neural inference time in half."*
3. **Popularity Dominance in Sparse Data**:  
   *"In sparse interaction graphs, head items account for the majority of review interactions. Popularity achieves high precision on head items but only recommends 14 items across the entire catalog (0.02% coverage). The Multi-Signal Hybrid approach trades a small fraction of raw head precision to expand active catalog exploration to 4,280 items (7.13% coverage), which is essential for business discovery."*
4. **Hallucination Prevention**:  
   *"Instead of prompting a generative LLM with unconstrained free text, my GroundedExplainer extracts explicit token matches from the product title, bullet features, and category path. It marks reasons as matched only when substantiated by catalog metadata."*
5. **FAISS HNSW Selection**:  
   *"In our offline benchmark, FlatIP yielded 82 QPS. HNSW ($M=32, \text{efSearch}=64$) achieved 680 QPS (an 8.3x throughput increase) while maintaining 99.8% recall relative to exhaustive search with an acceptable memory footprint of 108.5 MB."*
