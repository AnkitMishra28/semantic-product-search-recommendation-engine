# Phase 14 — Comprehensive System Architecture & Dependency Audit

**Project**: Amazon-Scale Semantic Product Search & Recommendation Engine  
**Author**: Applied Scientist Pair-Programming Agent  
**Date**: August 2026 (Phase 14 Production & Research Finalization)  
**Status**: 🟢 VERIFIED

---

## 1. Executive Summary

This system audit provides a verified architectural inspection of the repository across backend microservices, vector search infrastructure, neural rerankers, multi-signal recommendation pipelines, grounded explanation services, frontend UI layers, testing matrices, and container configurations.

All data, indexes, models, and evaluation artifacts exist as physical files in the repository. No simulated data layers, fake latency values, or synthetic evaluation metrics are utilized.

---

## 2. System Architecture Map

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               NEXT.JS 14 FRONTEND                                      │
│  - App Router (/search, /recommendations, /evaluation, /about, /dashboard -> /)        │
│  - Glassmorphic UI Design System (Obsidian slate & cyan accents, zero Tailwind pills)  │
│  - Interactive Scientific Evaluation Dashboard (6 Tabs, Parameter Sweeps, Inspector)  │
│  - Typed API Client (lib/api.ts) with timeout, retry & error normalization             │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ HTTP/JSON REST API (/api/v1/*)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND ENGINE                                   │
│  - Endpoints: /search, /recommend, /explain, /products/{id}, /metrics, /experiments    │
│  - FastAPI Lifespan Singleton: ModelRegistry & SearchEngine Orchestrator               │
│  - In-Memory Catalog: 60,000 Verified Amazon Reviews 2023 Electronics Products         │
└───────────────────┬──────────────────────────────┬─────────────────────────────┬───────┘
                    │                              │                             │
                    ▼                              ▼                             ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐ ┌────────────────────────┐
│    RETRIEVAL & RANKING       │ │   RECOMMENDATION ENGINE     │ │  GROUNDED EXPLAINER    │
│ • Query Understanding:       │ │ • Multi-Signal Service:     │ │ • GroundedExplainer    │
│   Normalized + Intent/Slots  │ │   - Popularity Recommender  │ │ • Feature Matcher      │
│ • Dense Vector Retrieval:    │ │   - Content-Based (Cosine)  │ │ • Category Alignment   │
│   FAISS HNSW (384-dim)       │ │   - Collaborative (Co-buy)  │ │ • Hallucination Guard  │
│ • Lexical Retrieval:         │ │   - Multi-Signal Hybrid     │ │ • Deterministic        │
│   BM25Okapi baseline         │ │ • MMR Diversity Reranker:   │ │   Fallback Provider    │
│ • Hybrid RRF Fusion:         │ │   (λ=0.70 trade-off)        │ └────────────────────────┘
│   k=10/60 rank smoothing     │ └─────────────────────────────┘
│ • Neural Cross-Encoder:      │
│   ms-marco-MiniLM-L-6-v2     │
│ • Business & Rating Reranker │
└──────────────────────────────┘
```

---

## 3. Detailed Component Audit

### 3.1 Backend Architecture (`backend/app/`)
- **Framework**: FastAPI with Pydantic v2 schemas and Uvicorn ASGI server.
- **Model Registry (`backend/app/services/model_registry.py`)**: Singleton managing thread-safe model weights and catalog structures in RAM:
  - `catalog`: 60,000 `Product` objects loaded from `data/processed/products.parquet` (110.6 MB).
  - `retriever`: `FAISSRetriever` loaded from `data/indexes/hnsw_m32_efc200_efs64.index` (108.5 MB, 60,000 indexed vectors).
  - `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
  - `reranker`: `cross-encoder/ms-marco-MiniLM-L-6-v2` (full cross-attention scoring).
  - `recommendation_service`: Unified `RecommendationService` combining `PopularityRecommender`, `ContentBasedRecommender`, `CollaborativeRecommender`, `HybridRecommender`, and `MMRDiversityReranker`.
- **Search Engine (`backend/app/services/engine.py`)**: Synchronously orchestrates the 5-stage search pipeline and instruments millisecond timings (`PipelineStageTiming`).
- **Explanation Service (`backend/app/explanations/`)**: Evidence-grounded rule-based and fallback explainer validating product features and query intent tokens against true catalog metadata.

### 3.2 Frontend Architecture (`frontend/src/`)
- **Framework**: Next.js 14.2.35 (React 18, TypeScript 5, TailwindCSS 3.4.1).
- **Design System**: Obsidian Glass aesthetic with CSS variables, high contrast typography, accessible ARIA labels, and zero unauthorized third-party styling gimmicks.
- **Pages**:
  - `/`: Search Landing Page with instant query launcher and architecture summary.
  - `/search`: Live Multi-Stage Semantic Search Engine with pipeline inspection drawer and query understanding pills.
  - `/recommendations`: Multi-Strategy Recommendation Laboratory with anchor product selector and MMR diversity slider.
  - `/evaluation`: Scientific Evaluation & Experiment Analysis Dashboard featuring 6 interactive tabs, parameter sweep charts, latency distributions, and raw JSON artifact inspector.
  - `/about`: System Architecture, Offline Evaluation Methodology, and Provenance reference.
  - `/dashboard`: Clean redirect route to `/` ensuring no 404 broken link exists.
- **API Client (`frontend/src/lib/api.ts`)**: Strongly typed asynchronous client supporting error normalization (`ApiError`), timeout safeguards (30s), and automatic JSON parsing.

### 3.3 Data Layer & Indexes (`data/`)
- `data/processed/products.parquet`: 60,000 products with ASIN, title, description, features, price, brand, categories, rating, rating count, and bought-together graphs.
- `data/processed/interactions.parquet`: 31,286 interactions across 1,621 users.
- `data/indexes/hnsw_m32_efc200_efs64.index`: Pre-built HNSW FAISS index (M=32, efConstruction=200, efSearch=64, index size 108.5 MB).
- `data/embeddings/products_title_brand_category_features_description.npy`: 60,000 x 384 float32 embeddings array (92.1 MB).
- `data/processed/evaluation_queries.json`: 30 curated electronic domain evaluation queries with ground truth ASIN relevance sets.

### 3.4 Experiment Artifacts Registry (`experiments/results/`)
All 10 offline evaluation runs are persisted as immutable JSON files:
1. `hybrid_retrieval.json` (Track E: Hybrid BM25 + FAISS + RRF)
2. `cross_encoder_reranking.json` (Phase 9: Neural Reranking & Budget Ablations)
3. `recommendation.json` (Phase 8: Multi-Signal Hybrid & MMR λ Sweep)
4. `faiss_benchmark.json` (Flat vs IVF100 vs HNSW32 vs IVFPQ)
5. `query_understanding_benchmark.json` (Phase 6 Intent Classification & Slot Extraction)
6. `query_understanding_validation.json` (Phase 6.1 Validation Query Set)
7. `bm25_baseline.json` (Okapi BM25 Baseline)
8. `semantic_title_brand_category.json` (Embedding Ablation 1)
9. `semantic_title_brand_category_features.json` (Embedding Ablation 2)
10. `semantic_title_brand_category_features_description.json` (Embedding Ablation 3)

---

## 4. Test Suite Audit

- **Backend Pytest Suite (`backend/tests/`)**:
  - `test_api_serving.py`: 13 tests verifying FastAPI endpoints, search, recommend, explain, error handling, and CORS.
  - `test_bm25.py`: 10 tests verifying BM25 tokenization, scoring, and edge cases.
  - `test_cross_encoder_reranking.py`: 11 tests verifying cross-encoder model ranking, score monotonicity, and candidate truncation.
  - `test_embeddings.py`: 8 tests verifying sentence transformer embeddings and vector normalization.
  - `test_evaluation_metrics.py`: 5 tests verifying Precision@K, Recall@K, MRR@K, NDCG@K, and Catalog Coverage.
  - `test_faiss_ann.py`: 8 tests verifying HNSW/Flat index construction, querying, and serialization.
  - `test_grounded_explanations.py`: 8 tests verifying evidence extraction, guardrails, and hallucination rejection.
  - `test_health.py`: 2 tests verifying health and readiness status endpoints.
  - `test_hybrid_retrieval.py`: 3 tests verifying hybrid retrieval pipeline execution.
  - `test_preprocessing.py`: 20 tests verifying text normalization, price parsing, and category extraction.
  - `test_query_understanding.py`: 17 tests verifying intent classification, slot extraction, price ranges, and entity tagging.
  - `test_ranking_interface.py`: 2 tests verifying ranking protocol contracts.
  - `test_recommendation.py`: 6 tests verifying Popularity, Content, Collaborative, Hybrid, and MMR diversity recommenders.
  - `test_retrieval_interface.py`: 2 tests verifying retrieval protocol contracts.
  - `test_rrf.py`: 10 tests verifying Reciprocal Rank Fusion formula and tie-breaking.
  - **Total Backend Tests**: **125 passed** (0 failures).

- **Frontend Quality Assurance**:
  - `npm run typecheck`: **0 errors**.
  - `npm run lint`: **0 warnings, 0 errors**.
  - `npm run build`: **9/9 routes compiled**.
  - `Playwright Verification`: **100% automated browser checks passed**.

---

## 5. Audit Conclusions & Hardening Roadmap

1. **Environment Configuration**: `.env.example` should align with the active HNSW index artifact path (`./data/indexes/hnsw_m32_efc200_efs64.index`).
2. **Container Build**: `frontend/public/` directory must contain a `.gitkeep` placeholder so Next.js multi-stage Docker builds succeed without errors.
3. **Reproducibility**: Provide a unified `scripts/validate_all.py` script that executes environment checks, test suites, static analysis, builds, and live API verifications in a single command.
