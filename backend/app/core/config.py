"""Application settings and environment configuration using Pydantic Settings."""

from functools import lru_cache
import json
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, resolved from this file's location (backend/app/core/config.py),
# so data/index/model paths are correct regardless of the process's working
# directory at launch time — not just when uvicorn happens to be started from
# the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Central configuration class loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    app_name: str = "Amazon-Scale Semantic Product Search & Recommendation Engine"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://frontend:3000",
        "http://frontend:3001",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        """Allow CORS_ORIGINS to be provided as JSON list or comma-separated string."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("cors_origins JSON must be a list")
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    # ML Models
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"  # 'cuda' or 'cpu'
    embedding_dim: int = 384
    max_seq_length: int = 256

    # Data & Indexes
    data_raw_dir: str = str(_PROJECT_ROOT / "data" / "raw")
    data_processed_dir: str = str(_PROJECT_ROOT / "data" / "processed")
    index_dir: str = str(_PROJECT_ROOT / "data" / "indexes")
    models_cache_dir: str = str(_PROJECT_ROOT / "models" / "weights")
    experiments_dir: str = str(_PROJECT_ROOT / "experiments")

    # Real product catalog artifact (60,000 Amazon Reviews 2023 Electronics
    # products) produced by scripts/preprocess_data.py — see docs/architecture.md.
    products_catalog_path: str = str(_PROJECT_ROOT / "data" / "processed" / "products.parquet")

    # Real user interaction history (data/processed/interactions.parquet, built by
    # scripts/preprocess_data.py) — powers PopularityRecommender/CollaborativeRecommender
    # and RecommendationService.get_user_history().
    interactions_path: str = str(_PROJECT_ROOT / "data" / "processed" / "interactions.parquet")

    # Dense product embeddings used for content-based recommendations and MMR diversity
    # reranking. Must be the same representation variant used to build faiss_index_path
    # (see data/embeddings/*_metadata.json "id_to_index" for the row<->ASIN mapping).
    content_embeddings_path: str = str(
        _PROJECT_ROOT / "data" / "embeddings" / "products_title_brand_category_features_description.npy"
    )
    content_embeddings_metadata_path: str = str(
        _PROJECT_ROOT
        / "data"
        / "embeddings"
        / "products_title_brand_category_features_description_metadata.json"
    )

    # Search & Retrieval Defaults
    default_retrieval_top_k: int = 100
    default_reranking_top_k: int = 20
    default_recommendation_top_k: int = 10
    hybrid_alpha: float = 0.7

    # Vector Storage Backend
    # NOTE: must match the persisted artifact built by scripts/build_embeddings.py
    # (data/indexes/hnsw_m32_efc200_efs64.index + .meta.json), which is HNSW —
    # the previous "FlatIP" / "electronics_all_minilm_l6_v2.index" defaults pointed
    # at a path that was never produced, silently leaving the retriever empty.
    vector_store_backend: str = "faiss"
    faiss_index_type: str = "HNSW"
    faiss_index_path: str = str(_PROJECT_ROOT / "data" / "indexes" / "hnsw_m32_efc200_efs64.index")

    # Optional LLM Explanation Service (Phase 10/11)
    openai_api_key: str = ""
    openai_base_url: Optional[str] = Field(
        default=None,
        description="Optional OpenAI-compatible base URL (e.g. Azure OpenAI endpoint or local proxy). Unset uses the default OpenAI API endpoint.",
    )
    llm_model_name: str = "gpt-4o-mini"
    enable_llm_explanations: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
