# System Architecture & Technical Design

## 1. Overview & High-Level Architecture

The **Amazon-Scale Semantic Product Search & Recommendation Engine** is a research-oriented, production-style prototype designed to simulate the multi-stage discovery pipelines employed in modern e-commerce systems.

Rather than relying on single-stage keyword matching or heavy monolithic models, the system employs an asynchronous, staged ranking and recommendation topology:

```
                  ┌──────────────────────────────────────────────┐
                  │                 User Query                   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Query Understanding Pipeline        │
                  │  • Normalization & Tokenization              │
                  │  • Intent Classification                     │
                  │  • Entity & Attribute Extraction             │
                  │  • Query Expansion / Rewriting               │
                  └──────────────────────┬───────────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
             ┌─────────────────────┐           ┌─────────────────────┐
             │ Dense Vector Search │           │ Lexical / Filter    │
             │ (FAISS / MinILM-L6) │           │ (Category/Price/BM25│
             └──────────┬──────────┘           └──────────┬──────────┘
                        │ (Top-K ~ 100)                   │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         First-Stage Score Fusion             │
                  │  • Reciprocal Rank Fusion / Alpha blending   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Cross-Encoder Reranking (Stage 2)    │
                  │  • ms-marco-MiniLM-L-6-v2                    │
                  │  • Query-Product deep interaction            │
                  │  • (Top-K ~ 20)                              │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Hybrid Business & Context Ranking    │
                  │  • Rerank Score + Rating / Popularity / Margin
                  │  • Diversity / MMR (Maximal Marginal Rel.)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
             ┌─────────────────────┐           ┌─────────────────────┐
             │  Final Search Rank  │           │ LLM Explanation Gen │
             │  (Top-K Results)    │           │ ("Why recommended") │
             └─────────────────────┘           └─────────────────────┘
```

---

## 2. Core Subsystems & Interfaces

### 2.1 Backend Decoupling & Interface Design

All machine learning, retrieval, and ranking operations are abstracted behind clean abstract base classes (`ABC` / Protocols). The HTTP API layer (`FastAPI`) only interacts with domain models and orchestrator services, ensuring that:
1. Retrieval engines can be swapped (e.g. from local `FAISS` to distributed `Qdrant` or `Milvus`) with zero changes to business logic or API endpoints.
2. Ranking strategies (Cross-Encoder, LambdaMART, GBDT, Hybrid rule-based) can be benchmarked side-by-side.
3. Search and recommendation modules are completely testable in isolation using unit test doubles and mocks.

### 2.2 Model Lifecycle & Singleton Management

Large ML models (`SentenceTransformer`, `CrossEncoder`) and vector index structures are memory-intensive. Reloading them per HTTP request is anti-pattern.
- The `ModelRegistry` service handles singleton model instances.
- FastAPI's `lifespan` context manager initializes models once upon application startup and gracefully releases GPU/CPU memory upon shutdown.

### 2.3 Vector Store Abstraction (`BaseRetriever`)

```python
class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalResult]:
        """Execute vector similarity search."""
        pass

    @abstractmethod
    def index(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        """Add vectors and document IDs to index."""
        pass
```

### 2.4 Ranking Abstraction (`BaseRanker` & `BaseReranker`)

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[ProductCandidate], top_k: int) -> list[RankedResult]:
        """Score (query, candidate) pairs using a deep cross-encoder or neural ranker."""
        pass
```

### 2.5 Query Understanding Pipeline (`BaseQueryProcessor`)

Processes raw query strings into structured search intent:
- Canonical spelling / normalization
- Category / Brand / Price intent extraction
- Multi-representation query vector generation

### 2.6 Explanation Service (`BaseExplainer`)

Generates human-interpretable rationale for why products were retrieved or recommended:
- Highlighting matched feature tokens and semantic proximity
- Contextual personalized justifications

### 2.7 Data Ingestion, Preprocessing & Temporal Partitioning Pipeline

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     Official McAuley Lab Amazon Reviews 2023                      │
│            (meta_Electronics.jsonl ~5.0GB, Electronics_reviews.jsonl ~21.5GB)     │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼ [Streaming Acquisition: scripts/download_data.py]
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      Raw Storage (Git-Ignored: data/raw/)                         │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼ [Ingestion & Cleaning Engine: scripts/preprocess_data.py]
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    Product Processing & Quality Scoring Engine                    │
│  • HTML decoding & Unicode NFKC normalization                                     │
│  • Non-destructive token preservation (no stemming for SentenceTransformers)       │
│  • Currency parsing & validation (USD floats)                                     │
│  • Brand disambiguation from store/details                                        │
│  • Canonical parent_asin deduplication & quality-weighted stratified sampling     │
│  • Deterministic representation builder (Variants A, B, C)                        │
└──────────────────┬────────────────────────────────────────────────┬───────────────┘
                   │                                                │
                   ▼                                                ▼
┌──────────────────────────────────────┐         ┌──────────────────────────────────────┐
│  data/processed/products.parquet     │         │ Interaction Cleaning & Partitioning  │
│  • 60,000 unique products            │         │ • Catalog referential filtering      │
│  • Embedding-ready document text     │         │ • Chronological sorting              │
│  • Isolated business ranking signals │         │ • Global quantile temporal splitting │
└──────────────────┬───────────────────┘         └──────────────────┬───────────────────┘
                   │                                                │
                   │                                                ▼
                   │                             ┌──────────────────────────────────────┐
                   │                             │  data/processed/interactions.parquet │
                   │                             │  • 31,286 interactions (70/15/15%)   │
                   │                             │  • Train: 2002-2020 | Val: 2020-2022 │
                   │                             │  • Test: 2022-2023                   │
                   │                             └──────────────────┬───────────────────┘
                   │                                                │
                   └───────────────────────┬────────────────────────┘
                                           │
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                 Evaluation Queries & Automated Validation Suite                   │
│  • data/processed/evaluation_queries.json (30 catalog-grounded search intents)    │
│  • scripts/validate_dataset.py (zero orphaned IDs, unique ASINs, valid schemas)   │
│  • scripts/profile_dataset.py (data/processed/dataset_profile.json & .md)         │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Evaluation & Research Reproducibility Framework

Reproducibility is a primary design pillar. The `evaluation/` module is isolated from the live serving path and provides:
1. **Deterministic Metrics**: Standard IR metrics implemented without third-party black-box dependencies:
   - `Recall@K`
   - `MRR@K` (Mean Reciprocal Rank)
   - `NDCG@K` (Normalized Discounted Cumulative Gain)
   - `Precision@K`
   - `Latency (p50, p95, p99)`
2. **Experiment Manifest**: Structured YAML configuration per experiment tracking:
   - Dataset subset / version
   - Model checkpoints / hashes
   - Vector index parameters (e.g. `nlist`, `nprobe`, metric type)
   - Preprocessing flags and random seeds
3. **Structured Outputs**: Machine-readable JSON results logged to `experiments/results/` tagged with timestamp and commit ID.

---

## 4. Multi-Stage Retrieval & Ranking Funnel Architecture

The system implements a classic industrial multi-stage candidate retrieval and neural ranking funnel:

```
[User Query]
     │
     ▼
[Stage 0: Query Understanding Pipeline]
  ├── Category Extraction & Canonical Taxonomy Mapping
  ├── Brand Normalization & Disambiguation
  ├── Price Bound Extraction & Currency Tagging (USD)
  └── Attribute Matching & Search Intent Classification
     │
     ▼
[Stage 1: High-Recall Candidate Generation]
  ├── BM25 Okapi Inverted Index (Top 100 lexical candidates)
  └── FAISS HNSW Vector Index (Top 100 dense cosine candidates via all-MiniLM-L6-v2)
     │
     ▼
[Stage 1.5: Reciprocal Rank Fusion (RRF)]
  └── Score Fusion: RRF(d) = Σ 1 / (60 + rank_i(d)) -> Fused Candidate Pool (Top 30-100)
     │
     ▼
[Stage 2: Precision Neural Reranking (Phase 9)]
  └── CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
      • Full all-to-all cross-attention: [CLS] query [SEP] title + brand + specs [SEP]
      • Candidate Budget: candidate_k = 30-50 optimal Pareto frontier
      • Single-load memory lifecycle with torch.no_grad() eval mode
     │
     ▼
[Stage 3: Business Signal Blending & Diversity (Phase 8)]
  ├── Bayesian Popularity Priors & Temporal Decay
  └── Maximal Marginal Relevance (MMR) Diversity Reranking (λ=0.70)
     │
     ▼
[Final Top-20 Ranked Results Delivered to User]
```

