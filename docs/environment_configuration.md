# Environment & Configuration Guide

**Project**: Amazon-Scale Semantic Product Search & Recommendation Engine  
**Release**: Phase 15 Production Hardening & Research Finalization  
**Status**: 🟢 VERIFIED

---

## 1. Overview

This document provides a definitive reference for all configuration variables supported by the backend and frontend services.

### Core Configuration Principle
**Zero Mandatory Environment Variables for Local Execution.**  
The application is architected with robust repository defaults. When launched without any `.env` file, the backend automatically binds to `http://0.0.0.0:8000`, loads the physical 60,000 product catalog from `data/processed/products.parquet`, connects to the pre-built FAISS HNSW index at `data/indexes/hnsw_m32_efc200_efs64.index`, and activates the local deterministic evidence-grounded explainer.

---

## 2. Complete Environment Variable Reference

| Variable Name | Classification | Default Value | Purpose & Description | Safe to Commit? |
| :--- | :---: | :--- | :--- | :---: |
| `ENVIRONMENT` | Non-Secret | `development` | Runtime mode (`development`, `staging`, `production`). Controls debug flags and logging detail. | YES |
| `LOG_LEVEL` | Non-Secret | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | YES |
| `APP_HOST` | Non-Secret | `0.0.0.0` | IP interface for FastAPI Uvicorn listener. | YES |
| `APP_PORT` | Non-Secret | `8000` | Port for FastAPI Uvicorn server. | YES |
| `CORS_ORIGINS` | Non-Secret | `["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"]` | JSON array or comma-separated string of allowed browser origins for CORS middleware. | YES |
| `EMBEDDING_MODEL_NAME` | Non-Secret | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model identifier for 384-dim dense query and document embeddings. | YES |
| `RERANKER_MODEL_NAME` | Non-Secret | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model identifier for Stage-2 neural cross-attention reranking. | YES |
| `DEVICE` | Non-Secret | `cpu` | PyTorch execution device (`cpu` or `cuda`). | YES |
| `DATA_RAW_DIR` | Non-Secret | `./data/raw` | Path to raw dataset files. | YES |
| `DATA_PROCESSED_DIR` | Non-Secret | `./data/processed` | Path to processed parquet catalog and interaction graphs. | YES |
| `INDEX_DIR` | Non-Secret | `./data/indexes` | Directory containing serialized FAISS indexes. | YES |
| `MODELS_CACHE_DIR` | Non-Secret | `./models/weights` | Local cache directory for downloaded transformer weights. | YES |
| `EXPERIMENTS_DIR` | Non-Secret | `./experiments` | Directory containing offline benchmark configs and immutable result JSONs. | YES |
| `PRODUCTS_CATALOG_PATH` | Non-Secret | `./data/processed/products.parquet` | Verified 60,000 product parquet catalog. | YES |
| `INTERACTIONS_PATH` | Non-Secret | `./data/processed/interactions.parquet` | 31,286 user interaction records for collaborative filtering and evaluation. | YES |
| `CONTENT_EMBEDDINGS_PATH` | Non-Secret | `./data/embeddings/products_title_brand_category_features_description.npy` | Pre-computed 60,000 x 384 numpy array for content-based recommendations and MMR diversity reranking. | YES |
| `CONTENT_EMBEDDINGS_METADATA_PATH` | Non-Secret | `./data/embeddings/products_title_brand_category_features_description_metadata.json` | Row index to ASIN mapping dictionary. | YES |
| `VECTOR_STORE_BACKEND` | Non-Secret | `faiss` | Vector search provider (`faiss` or `qdrant`). | YES |
| `FAISS_INDEX_TYPE` | Non-Secret | `HNSW` | Active FAISS index structure (`HNSW`, `FlatIP`, `IVFFlat`). | YES |
| `FAISS_INDEX_PATH` | Non-Secret | `./data/indexes/hnsw_m32_efc200_efs64.index` | Path to physical HNSW index file (M=32, efConstruction=200, efSearch=64). | YES |
| `DEFAULT_RETRIEVAL_TOP_K` | Non-Secret | `100` | Default candidate pool size retrieved during Stage-1 vector/lexical search. | YES |
| `DEFAULT_RERANKING_TOP_K` | Non-Secret | `20` | Default number of candidates scored by the Stage-2 Cross-Encoder. | YES |
| `DEFAULT_RECOMMENDATION_TOP_K` | Non-Secret | `10` | Default number of recommendations returned per item or user. | YES |
| `HYBRID_ALPHA` | Non-Secret | `0.7` | Weight assigned to dense retrieval vs business/rating signals in hybrid score blending. | YES |
| `ENABLE_LLM_EXPLANATIONS` | Non-Secret | `false` | When `false`, uses the deterministic rule-based grounded explainer locally. When `true`, enables optional OpenAI GPT API calls. | YES |
| `LLM_MODEL_NAME` | Non-Secret | `gpt-4o-mini` | OpenAI model name used only when `ENABLE_LLM_EXPLANATIONS=true`. | YES |
| `OPENAI_API_KEY` | **Secret** | `""` | OpenAI API key. **Required ONLY if** `ENABLE_LLM_EXPLANATIONS=true`. Leave blank for default local execution. | **NO (Keep Private)** |
| `OPENAI_BASE_URL` | Non-Secret | `None` | Optional custom base URL for Azure OpenAI or local LLM proxy endpoints (e.g. vLLM, Ollama). | YES |
| `NEXT_PUBLIC_API_URL` | Non-Secret | `http://localhost:8000` | Base URL used by the Next.js frontend to communicate with the FastAPI backend. | YES |
| `BACKEND_URL` | Test-Only | `http://localhost:8000` | Target URL used by Playwright and verification scripts. | YES |
| `FRONTEND_URL` | Test-Only | `http://localhost:3000` | Target URL used by Playwright browser test runners. | YES |

---

## 3. Secret Credentials & Obtaining Optional API Keys

### `OPENAI_API_KEY`
- **Is it required to run the project?**  
  **NO.** The system operates completely without an OpenAI API key. The primary grounded explanation engine (`GroundedExplainer` in `backend/app/explanations/`) evaluates feature presence and category matches against actual catalog metadata deterministically with zero remote API dependencies and zero hallucination risk.
- **When is it needed?**  
  Only when an operator explicitly sets `ENABLE_LLM_EXPLANATIONS=true` to test natural-language narrative synthesis.
- **Where to obtain it?**  
  1. Navigate to the OpenAI Developer Platform: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  2. Create a new secret key.
  3. Create a local `.env` file in the repository root and set `OPENAI_API_KEY=sk-...` (this file is excluded by `.gitignore`).
- **Free Tier / Pricing**:  
  OpenAI provides trial credits upon initial account creation; standard pay-per-token pricing applies for `gpt-4o-mini` (~$0.15 per million input tokens).

---

## 4. Local Development vs Containerized Execution

### Local Development (Default)
No `.env` file required. All path resolutions in `backend/app/core/config.py` compute repository-relative absolute paths dynamically using `Path(__file__).resolve().parents[3]`, ensuring that running from the repo root or subdirectories resolves correctly.

### Docker / Production Deployment
When deploying via `docker-compose.yml`, environment variables are injected into container runtimes:
```yaml
services:
  backend:
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
      - RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
      - DEVICE=cpu
      - VECTOR_STORE_BACKEND=faiss
      - CORS_ORIGINS=["http://localhost:3000", "http://frontend:3000"]
  frontend:
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
```
