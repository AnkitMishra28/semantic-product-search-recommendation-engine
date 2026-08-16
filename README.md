# Semantic Product Search & Recommendation Engine

> **Applied Research & Production Prototype**: An end-to-end multi-stage semantic product search, neural reranking, graph recommendation, and evidence-grounded explanation platform built over **60,000 Amazon Reviews 2023 Electronics products**. Features physical FAISS HNSW indexes, dual-track candidate fusion, cross-encoder neural reranking, multi-signal recommendation pipelines, and an interactive Next.js research console.

---

## 1. Executive Summary & Problem Statement

Standard keyword search in e-commerce frequently suffers from vocabulary mismatch, inability to parse natural language specifications (e.g., *"fast charging usb-c power bank for travel"*), and an inability to rank semantically relevant items when exact keyword matches are absent. Conversely, pure dense retrieval can miss exact brand or SKU identifiers and introduce high computational latency when scaled to large catalogs. Furthermore, recommendation engines often suffer from severe popularity bias, trapping users in narrow feedback loops.

This project implements a decoupled, research-grade search and recommendation architecture inspired by modern large-scale e-commerce retrieval systems:
1. **Query Understanding**: Intent classification, regex-based slot extraction (brand, category, price bounds, technical attributes), and query normalization.
2. **Stage 1 (Dual-Track Candidate Retrieval)**: Parallel candidate generation using Dense Semantic Vector Search (`FAISS HNSW` + `sentence-transformers/all-MiniLM-L6-v2`) and Lexical Search (`BM25Okapi`), merged via Reciprocal Rank Fusion (RRF).
3. **Stage 2 (Neural Cross-Encoder Reranking)**: Full cross-attention reranking over retrieved candidate pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
4. **Stage 3 (Business Scoring & Hybrid Blending)**: Dynamic linear interpolation between cross-encoder relevance scores and item quality signals (average star rating and review log volume).
5. **Multi-Signal Recommendation Service**: Blending co-purchase graphs (`bought_together`), dense semantic cosine neighborhoods, and popularity baselines with Maximal Marginal Relevance (MMR) diversity reranking.
6. **Deterministic Grounded Explanations**: Evidence-grounded feature verification directly cross-referencing catalog attributes against user queries to prevent hallucinated assertions.
7. **Scientific Evaluation Dashboard**: Interactive Next.js console tracking 10 physical, immutable offline benchmark experiment runs.

---

## 2. System Architecture

```
                                ┌─────────────────────────────┐
                                │         User Query          │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │  Query Understanding (QU)   │
                                │  • Intent Classification    │
                                │  • Entity & Slot Extraction │
                                │  • Price Floor / Ceiling    │
                                └──────────────┬──────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │   Dense Vector Retrieval    │                 │   Lexical BM25 Retrieval    │
        │  (FAISS HNSW - 384 dims)    │                 │     (Inverted Index)        │
        └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                       │ (Top-100 Candidates)                          │ (Top-100 Candidates)
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │ Reciprocal Rank Fusion (RRF)│
                                │   Candidate Pool: 25.42%    │
                                └──────────────┬──────────────┘
                                               │ (Top-50 Fusion Candidates)
                                               ▼
                                ┌─────────────────────────────┐
                                │ Neural Cross-Encoder Rerank │
                                │  (ms-marco-MiniLM-L-6-v2)   │
                                └──────────────┬──────────────┘
                                               │ (Top-20 Neural Ranks)
                                               ▼
                                ┌─────────────────────────────┐
                                │  Business & Rating Reranker │
                                │    (Score Linear Blend)     │
                                └──────────────┬──────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │     Final Search Results    │                 │ Grounded Explanation Engine │
        │    (Top-10 Product Cards)   │                 │ ("Why Matched" Evidence)    │
        └─────────────────────────────┘                 └─────────────────────────────┘
```

---

## 3. End-to-End Search Pipeline

Every search request (`POST /api/v1/search`) executes through a modular multi-stage pipeline:

1. **Query Understanding (QU)**:
   - Normalizes unicode, accents, spelling variants, and price shorthands (e.g., `$50`, `50usd`, `under 50 dollars`).
   - Identifies user intent: `SPECIFIC_PRODUCT`, `CATEGORY_EXPLORATION`, `FEATURE_FOCUSED`, or `PRICE_CONSTRAINED`.
   - Extracts structured entity slots (brand, category, price constraints, hardware specifications like `16gb ram`, `rtx 4060`, `usb-c`).
2. **Dense Retrieval (Bi-Encoder + FAISS HNSW)**:
   - Generates 384-dimensional query embedding via `sentence-transformers/all-MiniLM-L6-v2`.
   - Performs approximate nearest neighbor (ANN) search on a graph-based FAISS HNSW index ($M=32, \text{efConstruction}=200, \text{efSearch}=64$).
   - Returns Top-100 semantic candidates with cosine similarity scores.
3. **Lexical Retrieval (BM25Okapi)**:
   - Queries an in-memory inverted index tokenized over product titles, brands, categories, and technical bullet features.
   - Returns Top-100 keyword candidates with exact lexical scores.
4. **Reciprocal Rank Fusion (RRF)**:
   - Combines dense and lexical rankings via standard RRF formula:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{k + r_m(d)}$$
   - Selects Top-50 fused candidates for downstream neural scoring.
5. **Stage 2 Neural Cross-Encoder Reranking**:
   - Evaluates full cross-attention for all $(q, d_i)$ pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Captures rich semantic interaction, query-item token alignment, and negative constraint signals.
6. **Business Score Blending & Hard Filtering**:
   - Applies extracted hard filters (price ceiling, brand constraint).
   - Combines cross-encoder logit $\sigma(s_{\text{ce}})$ with product rating multiplier and log review volume:
     $$\text{FinalScore} = 0.75 \cdot \sigma(s_{\text{ce}}) + 0.15 \cdot \left(\frac{\text{Rating}}{5.0}\right) + 0.10 \cdot \min\left(1.0, \frac{\ln(1 + \text{Reviews})}{\ln(10000)}\right)$$
7. **Grounded Explanations**:
   - Checks retrieved product metadata against query constraints to produce verifiable explanation bullets and highlight matched features.

---

## 4. Multi-Strategy Recommendation Pipeline

The recommendation engine (`POST /api/v1/recommend`) supports five complementary strategies:

1. **Popularity Recommender**:
   - Fallback baseline ranked by Wilson-score confidence interval over review count and average star rating.
   - Supports category-level and brand-level filtering.
2. **Content-Based Recommender**:
   - Computes cosine similarity across pre-computed 384-dimensional dense product embeddings.
   - Retrieves items with matching functional, technical, and taxonomic properties.
3. **Collaborative Item-to-Item Recommender**:
   - Mines graph co-occurrences from the `bought_together` co-purchase interaction graph.
   - Surfaces items frequently co-purchased or co-reviewed by customers.
4. **Multi-Signal Hybrid Recommender**:
   - Linearly blends collaborative co-purchase weights (0.50), content semantic similarity (0.35), and global item quality (0.15).
   - Seamlessly handles cold-start items by dynamically falling back to content embeddings when co-purchase edges are absent.
5. **Maximal Marginal Relevance (MMR) Diversity Reranker**:
   - Applies greedy submodular selection to balance relevance against intra-list redundancy:
     $$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
   - Evaluated at $\lambda = 0.70$, reducing intra-list redundancy from 0.412 to 0.240.

---

## 5. Technical Stack

### Backend
- **Framework**: FastAPI (Asynchronous ASGI microservice)
- **Runtime**: Python 3.10 / 3.11 / 3.14 (Uvicorn ASGI server)
- **Data Schemas & Settings**: Pydantic v2 & Pydantic Settings
- **Testing**: Pytest (125 unit and integration tests)

### Machine Learning & Retrieval
- **Dense Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector Search Engine**: `faiss-cpu` (HNSW graph topology: $M=32$, $\text{efConstruction}=200$, $\text{efSearch}=64$)
- **Lexical Search**: `rank-bm25` (BM25Okapi inverted index)
- **Neural Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (PyTorch / Hugging Face Transformers)
- **Data Processing**: NumPy, Pandas, PyArrow (Apache Parquet), Scikit-Learn

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS (Obsidian glass visual design system)
- **Animation & Icons**: Framer Motion, Lucide React
- **Browser Testing**: Playwright regression test suite

---

## 6. Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── api/v1/                  # FastAPI endpoints (/search, /recommend, /explain, /metrics, /evaluate)
│   │   ├── core/                    # Settings, CORS, logging, and singleton lifecycle
│   │   ├── models/                  # Pydantic schemas (Product, SearchResponse, RecommendResponse)
│   │   ├── preprocessing/           # Ingestion, cleaning, quality scoring, deduplication
│   │   ├── query_understanding/     # Pipeline, Normalizer, IntentClassifier, SlotExtractor
│   │   ├── retrieval/               # FAISSRetriever, BM25Retriever, HybridRetriever, RRF
│   │   ├── ranking/                 # CrossEncoderReranker, HybridRanker
│   │   ├── recommendation/          # Popularity, ContentBased, Collaborative, Hybrid, MMR
│   │   ├── explanations/            # GroundedExplainer, EvidenceBuilder, Hallucination Guardrails
│   │   ├── services/                # ModelRegistry (singleton) & SearchEngine orchestrator
│   │   └── main.py                  # ASGI application entrypoint
│   ├── tests/                       # 125 Pytest unit and integration tests (100% pass rate)
│   └── requirements.txt             # Backend dependencies
├── data/
│   ├── processed/                   # interactions.parquet, evaluation_queries.json, dataset_profile.json
│   ├── indexes/                     # hnsw_m32_efc200_efs64.meta.json
│   ├── embeddings/                  # products_title_brand_category_features_description_metadata.json
│   └── raw/                         # Raw acquisition documentation & .gitkeep
├── docs/                            # Deep-dive architecture specs, system audits, and reproducibility guides
├── experiments/
│   └── results/                     # 10 immutable offline benchmark JSON artifacts
├── frontend/
│   ├── src/
│   │   ├── app/                     # Next.js 14 App Router routes (/, /search, /recommendations, /evaluation, /about)
│   │   ├── components/              # UI components, pipeline visualizer, interactive metric charts
│   │   ├── lib/                     # Typed ApiClient, error normalization, formatting utilities
│   │   └── types/                   # TypeScript interfaces matching backend schemas
│   ├── package.json                 # Frontend dependencies & scripts
│   └── run-phase13-verification.mjs # Automated Playwright browser verification
├── scripts/
│   ├── validate_all.py              # One-command unified system test suite
│   ├── download_data.py             # Stream downloader for Amazon Reviews 2023
│   ├── preprocess_data.py           # Ingestion, cleaning, and Parquet creation
│   └── build_embeddings.py          # Bi-encoder vector encoding & FAISS HNSW indexing
├── docker-compose.yml               # Multi-container orchestration descriptor
├── Dockerfile                       # Multi-stage backend container image
└── .env.example                     # Root environment configuration template
```

---

## 7. Key API Endpoints

The FastAPI backend exposes both `/api/v1/*` (versioned) and `/api/*` (direct) routes:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Liveness check returning uptime and service status |
| `GET` | `/api/v1/ready` | Readiness check verifying models, FAISS index, and product catalog are loaded |
| `POST` | `/api/v1/search` | Full multi-stage semantic search with query understanding, RRF fusion, and neural reranking |
| `POST` | `/api/v1/recommend` | Multi-strategy recommendations by anchor product ASIN or user interaction history |
| `POST` | `/api/v1/explain` | Evidence-grounded explanation generation with attribute verification |
| `GET` | `/api/v1/products/{id}` | Verified catalog lookup by ASIN identifier |
| `GET` | `/api/v1/metrics` | Serves authoritative offline benchmark results |
| `GET` | `/api/v1/evaluate/experiments` | Enumerates all 10 tracked experiment runs |
| `GET` | `/api/v1/evaluate/experiments/{id}` | Full raw JSON artifact inspector payload |

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs` when the backend is running.

---

## 8. Local Setup & Execution Guide

### Prerequisites
- **Python**: 3.10, 3.11, or 3.14 (64-bit)
- **Node.js**: 18.x or 20.x LTS
- **Hardware**: 8 GB RAM minimum (16 GB recommended for in-memory 60K FAISS index & PyTorch models)

### Step 1: Clone Repository
```powershell
git clone git@github-personal:AnkitMishra28/semantic-product-search-recommendation-engine.git
cd "semantic-product-search-recommendation-engine"
```

### Step 2: Backend Setup (Windows PowerShell)
```powershell
# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Step 3: Dataset Ingestion & Index Generation (If starting fresh)
```powershell
# Download raw Amazon Reviews 2023 dataset
python scripts/download_data.py --max-products 75000 --max-reviews 250000

# Clean and create processed Parquet catalog
python scripts/preprocess_data.py --target-products 60000 --seed 42

# Generate dense embeddings and build FAISS HNSW index
python scripts/build_embeddings.py
```

### Step 4: Start Backend Server
```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/v1/health`

### Step 5: Start Frontend Server
In a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:3000`

---

## 9. Verification & Testing

### A. Run Backend Unit & Integration Tests (125 tests)
```powershell
python -m pytest backend/tests/ -v
```
*Result: 125 passed, 0 failures, 100% pass rate.*

### B. Frontend Static Analysis & Production Build
```powershell
cd frontend
npm run typecheck
npm run lint
npm run build
```

### C. One-Command Unified System Validation Suite
```powershell
python scripts/validate_all.py
```

---

## 10. Empirical Scientific Benchmarks (Source of Truth)

All metrics below reflect physical, immutable offline benchmark results stored under `experiments/results/`:

| Metric / Experiment | Benchmark Result | Scientific Finding |
| :--- | :-: | :--- |
| **Stage-1 Candidate Depth (Recall@100)** | **19.58%** | Dense FAISS matches Hybrid RRF on candidate depth (100) |
| **First-Stage Ranking Gain (MRR@10)** | **0.0972 $\to$ 0.1159** | **+19.2%** ranking accuracy gain via Reciprocal Rank Fusion over BM25 |
| **Stage-2 Neural Gain (Recall@20)** | **5.00% $\to$ 5.42%** | **+8.33%** downstream relevance gain post Cross-Encoder neural reranking |
| **Untruncated Candidate Pool Coverage** | **25.42%** | 61 / 240 relevant items recovered across BM25 $\cup$ Dense pool |
| **Candidate Overlap Jaccard Similarity** | **30.8%** | Lexical and semantic retrievers retrieve complementary item spaces |
| **Optimal RRF Smoothing ($k$)** | **$k = 10$** (MRR 0.1194) | Lower $k$ sharpens top-1 discrimination; $k=60$ provides balanced smoothing |
| **Recommender Catalog Coverage** | **7.13%** (4,280 unique items) | **356x** catalog discovery expansion over Popularity baseline (0.02%) |
| **MMR Diversity Balance Point** | **$\lambda = 0.70$** | Reduces intra-list similarity from 0.412 to 0.240 while preserving top-10 relevance |

---

## 11. Deployment Architecture

### Frontend Deployment (Vercel)
The Next.js 14 application in `frontend/` is fully optimized for deployment on **Vercel**:
- **Root Directory**: `frontend`
- **Framework Preset**: Next.js
- **Build Command**: `npm run build`
- **Install Command**: `npm install`
- **Environment Variable**: `NEXT_PUBLIC_API_URL` pointing to the deployed backend endpoint (e.g. `https://api.yourdomain.com`).

### Backend Deployment (Containerized / Cloud Host)
> [!IMPORTANT]
> The Python ML backend loads a **60,000-vector FAISS HNSW index**, PyTorch models (`all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`), and in-memory catalog data (~1.5–2 GB RAM). It is **not suitable for standard serverless execution (such as Vercel Serverless Functions or AWS Lambda with 250MB limits)**.

**Recommended Production Target**:
- **Container**: `backend/Dockerfile` (Multi-stage `python:3.11-slim` image)
- **Host**: AWS ECS Fargate, AWS EC2 (t3.medium or g4dn.xlarge for GPU), GCP Cloud Run (configured with 2–4 GB memory), Render, or Railway.
- **Orchestration**: `docker-compose.yml` for single-node containerized environments:
  ```bash
  docker-compose up --build
  ```

---

## 12. Engineering Highlights & Trade-offs

1. **Approximate Nearest Neighbors with HNSW**:  
   Hierarchical Navigable Small World graphs ($M=32$, $\text{efConstruction}=200$) achieve **99.8% recall at ~680 QPS** on CPU, avoiding the quadratic complexity of exhaustive vector scans.
2. **Complementary Rank Fusion (RRF)**:  
   Because BM25 and Dense FAISS share only a 30.8% Jaccard candidate overlap, fusing both candidate pools yields a **25.42% untruncated candidate coverage** (+29.8% expansion over either retriever alone).
3. **Decoupled Two-Stage Inference Budgeting**:  
   Heavy cross-attention neural reranking is restricted to the top 50 fused candidates, keeping latency predictable while capturing deep semantic interactions.
4. **Submodular Diversity Optimization (MMR)**:  
   MMR reranking at $\lambda=0.70$ systematically prevents category homogenization in recommendations.
5. **Deterministic Hallucination Guardrails**:  
   Grounded explanation generation relies on exact metadata constraint checking rather than unconstrained generative models, ensuring zero hallucinated product specifications.
6. **Singleton Model Registry with Async Lifespan**:  
   Models, indexes, and catalog DataFrames are loaded once during FastAPI lifespan startup, preventing duplicate memory allocations across incoming requests.

---

## 13. Future Extensions & Production Roadmap

The following areas represent natural extensions for production scaling:
- **GPU Inference & Batching**: Deploying the cross-encoder behind Triton Inference Server with dynamic batching and TensorRT optimization to reduce 50-pair reranking latency below 15 ms.
- **Quantization**: INT8 ONNX runtime quantization for sub-50ms CPU cross-encoder scoring.
- **Learned Sparse Retrieval**: Investigating SPLADE v2 to unify lexical inverted indexes with learned term expansion.
- **Graph Neural Networks**: GraphSAGE embeddings over session-based user-item co-purchase subgraphs.
- **Multi-Graded Human Annotations**: Supplementing implicit review interaction graphs with multi-graded editorial relevance judgments.

---

## 14. Applied Scientist & Engineering Competencies

This project demonstrates practical applied science and engineering proficiency in:
- **Information Retrieval (IR)**: Lexical indexing (BM25), dense representation learning, HNSW vector indexing, Reciprocal Rank Fusion, and ranking metrics (NDCG@K, Recall@K, MRR@K).
- **Recommender Systems**: Collaborative filtering, content-based embedding spaces, multi-signal hybrid scoring, catalog coverage metrics, and MMR diversity optimization.
- **Natural Language Processing (NLP)**: Bi-encoder sentence representations, cross-encoder neural cross-attention, intent classification, and slot extraction.
- **System Architecture**: High-throughput FastAPI asynchronous services, singleton model caching, typed Next.js UI, Docker containerization, and automated end-to-end testing suites.
- **Empirical Rigor**: Hypothesis-driven parameter sweeps, failure mode analysis, and reproducible offline benchmark reporting.

---

## 15. License & Citations

- **Dataset**: Amazon Reviews 2023 (Electronics Category) curated by McAuley Lab, UC San Diego (*Hou et al., 2024*).
- **Pretrained Models**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **License**: [MIT License](LICENSE)
