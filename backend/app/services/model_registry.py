"""ModelRegistry Singleton: Manages startup model loading and memory handles."""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.models.product import Product
from backend.app.ranking.cross_encoder import CrossEncoderReranker
from backend.app.recommendation.collaborative import CollaborativeRecommender
from backend.app.recommendation.content_based import ContentBasedRecommender
from backend.app.recommendation.diversity import MMRReranker
from backend.app.recommendation.hybrid import HybridRecommender
from backend.app.recommendation.popularity import PopularityRecommender
from backend.app.recommendation.service import RecommendationService
from backend.app.retrieval.base import BaseRetriever
from backend.app.retrieval.faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)


def _to_str_list(value: Any) -> List[str]:
    """Coerce a parquet cell (numpy object array, list, or None) into List[str]."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


def _clean_str(value: Any) -> Optional[str]:
    """Normalize a parquet cell to a trimmed string, or None for missing/NaN values."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_float(value: Any) -> Optional[float]:
    """Normalize a parquet cell to a float, or None for missing/NaN/unparseable values."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


class ModelRegistry:
    """Singleton service to load models and vector indexes once at server startup."""

    _instance: Optional["ModelRegistry"] = None

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model: Any = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.retriever: Optional[BaseRetriever] = None
        self.catalog: Dict[str, Product] = {}
        self.recommendation_service: Optional[RecommendationService] = None
        self._catalog_loaded_from_disk = False
        self._is_initialized = False

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "ModelRegistry":
        """Get or initialize the singleton ModelRegistry instance."""
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def initialize(self) -> None:
        """Load embedding model, reranker, and FAISS index into memory."""
        if self._is_initialized:
            logger.info("ModelRegistry is already initialized.")
            return

        logger.info(f"Initializing ModelRegistry on device: {self.settings.device}")

        # 1. Initialize Vector Retriever and load the persisted FAISS index built by
        #    scripts/build_embeddings.py. Without this, the retriever stays empty and
        #    every search silently returns zero candidates.
        self.retriever = FaissRetriever(
            dimension=self.settings.embedding_dim,
            index_type=self.settings.faiss_index_type,
        )
        self._load_faiss_index(self.settings.faiss_index_path)

        # 1b. Load the real product catalog. Guarded on an empty catalog so tests
        #     that pre-seed ModelRegistry.catalog with fixture products (see
        #     backend/tests/conftest.py) are never overwritten by the real 60k
        #     catalog — initialize() is re-entered fresh for every test.
        if not self.catalog:
            self.catalog = self._load_catalog(self.settings.products_catalog_path)
            self._catalog_loaded_from_disk = bool(self.catalog)

        # 1c. Build the unified RecommendationService (popularity / content-based /
        #     collaborative / hybrid / MMR diversity) from real interaction and
        #     embedding artifacts. Only attempted when we just loaded the real
        #     catalog ourselves — under tests the catalog is pre-seeded with 3
        #     fixture products with no matching real interactions/embeddings, so
        #     SearchEngine keeps its lightweight ItemToItemRecommender fallback.
        if self._catalog_loaded_from_disk:
            self.recommendation_service = self._build_recommendation_service()

        # 2. Initialize Reranker
        self.reranker = CrossEncoderReranker(
            model_name=self.settings.reranker_model_name,
            device=self.settings.device,
            max_seq_length=self.settings.max_seq_length,
        )

        # 3. Optional warm-up / load neural weights if available
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.settings.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.settings.embedding_model_name,
                device=self.settings.device,
            )
        except ImportError:
            logger.warning("sentence_transformers not installed. Embedding generation will use mock vectors.")

        # Load reranker model weights
        self.reranker.load_model()

        self._is_initialized = True
        logger.info("ModelRegistry successfully initialized.")

    def _load_faiss_index(self, index_path: str) -> None:
        """Load the persisted FAISS index (data/indexes/*.index + .meta.json) into
        self.retriever. Missing/unreadable index degrades to an empty retriever
        (search returns zero candidates) rather than crashing server startup —
        the caller can inspect GET /api/v1/ready to see the real vector_index status.
        """
        path = Path(index_path)
        if not path.exists():
            logger.warning(
                f"FAISS index file not found at '{path}'. Retriever will remain empty "
                "until a real index is built (see scripts/build_embeddings.py)."
            )
            return
        try:
            t0 = time.perf_counter()
            self.retriever.load(str(path))
            logger.info(
                f"Loaded FAISS index ({self.retriever.total_documents:,} vectors) "
                f"from '{path}' in {time.perf_counter() - t0:.2f}s."
            )
        except Exception:
            logger.exception(f"Failed to load FAISS index from '{path}'. Retriever will remain empty.")

    def _load_catalog(self, products_path: str) -> Dict[str, Product]:
        """Load the real product catalog from the processed parquet artifact
        (data/processed/products.parquet, built by scripts/preprocess_data.py) into
        Product domain objects, keyed by parent_asin — matching the doc_id convention
        used by the persisted FAISS index's id_to_doc_map.
        """
        path = Path(products_path)
        if not path.exists():
            logger.warning(
                f"Product catalog parquet not found at '{path}'. Catalog will remain empty "
                "until the dataset is preprocessed (see scripts/preprocess_data.py)."
            )
            return {}

        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        catalog: Dict[str, Product] = {}
        for row in df.itertuples(index=False):
            fields = row._asdict()
            asin = str(fields.get("parent_asin"))
            quality_score = fields.get("quality_score")
            metadata = {"quality_score": float(quality_score)} if quality_score is not None and not (
                isinstance(quality_score, float) and math.isnan(quality_score)
            ) else {}

            catalog[asin] = Product(
                parent_asin=asin,
                title=_clean_str(fields.get("title")) or "Unknown Product",
                description=_clean_str(fields.get("description")) or "",
                features=_to_str_list(fields.get("features")),
                price=_clean_float(fields.get("price")),
                brand=_clean_str(fields.get("brand")),
                categories=_to_str_list(fields.get("categories")),
                average_rating=_clean_float(fields.get("average_rating")),
                rating_number=int(fields.get("rating_number") or 0),
                image_url=_clean_str(fields.get("image_url")),
                images=_to_str_list(fields.get("images")),
                bought_together=_to_str_list(fields.get("bought_together")),
                embedding_text=_clean_str(fields.get("embedding_text")),
                metadata=metadata,
            )

        logger.info(f"Loaded {len(catalog):,} products from '{path}' in {time.perf_counter() - t0:.2f}s.")
        return catalog

    def _load_interactions(self, interactions_path: str) -> pd.DataFrame:
        """Load real user-item interaction history (data/processed/interactions.parquet).
        Powers PopularityRecommender/CollaborativeRecommender fitting and
        RecommendationService.get_user_history(). Missing file degrades to an empty
        frame — recommenders handle that gracefully (cold-start / popularity-only).
        """
        path = Path(interactions_path)
        if not path.exists():
            logger.warning(f"Interactions parquet not found at '{path}'. Collaborative signals will be unavailable.")
            return pd.DataFrame(columns=["user_id", "parent_asin", "rating", "timestamp"])
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} interactions from '{path}'.")
        return df

    def _load_content_embeddings(
        self, embeddings_path: str, metadata_path: str
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Load the dense product embeddings used for content-based recommendations and
        MMR diversity reranking (data/embeddings/*.npy + matching *_metadata.json
        'id_to_index' mapping). Missing files degrade to (None, []) — HybridRecommender
        and RecommendationService both tolerate a missing content_recommender.
        """
        emb_path = Path(embeddings_path)
        meta_path = Path(metadata_path)
        if not emb_path.exists() or not meta_path.exists():
            logger.warning(
                f"Content embeddings not found at '{emb_path}' / '{meta_path}'. "
                "Content-based recommendations and MMR diversity will be unavailable."
            )
            return None, []

        vectors = np.load(emb_path).astype(np.float32)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        id_to_index: Dict[str, int] = meta.get("id_to_index", {})
        doc_ids: List[str] = [""] * len(id_to_index)
        for asin, idx in id_to_index.items():
            if 0 <= idx < len(doc_ids):
                doc_ids[idx] = str(asin)

        if len(doc_ids) != len(vectors):
            logger.warning(
                f"Embedding count ({len(vectors)}) does not match id_to_index size "
                f"({len(doc_ids)}) in '{meta_path}'. Content-based recommendations will be unavailable."
            )
            return None, []

        logger.info(f"Loaded {len(vectors):,} content embeddings ({vectors.shape[1]}-dim) from '{emb_path}'.")
        return vectors, doc_ids

    def _build_recommendation_service(self) -> Optional[RecommendationService]:
        """Construct the unified RecommendationService — popularity, content-based,
        collaborative, hybrid, and MMR diversity reranking — wired from the real
        catalog, interaction history, and content embeddings. Mirrors the construction
        pattern validated by backend/tests/test_recommendation.py::test_recommendation_service_end_to_end.
        """
        if not self.catalog:
            return None

        # Recommenders below filter/explain using plain metadata dicts (not Pydantic
        # models) — this mirrors the exact shape used in the validated test fixtures.
        catalog_dict: Dict[str, Dict[str, Any]] = {
            asin: {
                "asin": p.asin,
                "parent_asin": p.parent_asin or p.asin,
                "title": p.title,
                "brand": p.brand,
                "price": p.price,
                "rating": p.average_rating or 4.0,
                "average_rating": p.average_rating or 4.0,
                "rating_number": p.rating_number,
                "categories": p.categories,
                "features": p.features,
            }
            for asin, p in self.catalog.items()
        }

        interactions_df = self._load_interactions(self.settings.interactions_path)
        embeddings, doc_ids = self._load_content_embeddings(
            self.settings.content_embeddings_path, self.settings.content_embeddings_metadata_path
        )

        t0 = time.perf_counter()

        popularity_rec = PopularityRecommender(product_catalog=catalog_dict)
        if not interactions_df.empty:
            popularity_rec.fit(interactions_df)

        content_rec: Optional[ContentBasedRecommender] = None
        mmr_reranker: Optional[MMRReranker] = None
        if embeddings is not None and doc_ids:
            content_rec = ContentBasedRecommender(
                embeddings=embeddings, doc_ids=doc_ids, product_catalog=catalog_dict
            )
            mmr_reranker = MMRReranker(embeddings=embeddings, doc_ids=doc_ids, default_lambda=0.7)

        collaborative_rec: Optional[CollaborativeRecommender] = None
        if not interactions_df.empty:
            collaborative_rec = CollaborativeRecommender(product_catalog=catalog_dict).fit(interactions_df)

        hybrid_rec = HybridRecommender(
            popularity_recommender=popularity_rec,
            content_recommender=content_rec,
            collaborative_recommender=collaborative_rec,
            diversity_reranker=mmr_reranker,
            product_catalog=catalog_dict,
        )

        service = RecommendationService(
            popularity_recommender=popularity_rec,
            content_recommender=content_rec,
            collaborative_recommender=collaborative_rec,
            hybrid_recommender=hybrid_rec,
            diversity_reranker=mmr_reranker,
            product_catalog=self.catalog,
            interactions_df=interactions_df,
        )

        logger.info(f"Built RecommendationService in {time.perf_counter() - t0:.2f}s.")
        return service

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text string into dense vector embedding."""
        if self.embedding_model is not None:
            vector = self.embedding_model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(vector, dtype=np.float32)

        # Mock deterministic embedding fallback for development without GPU / models loaded
        dim = self.settings.embedding_dim
        seed = sum(ord(c) for c in text) % 10000
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def shutdown(self) -> None:
        """Release models and GPU/CPU memory on shutdown."""
        logger.info("Shutting down ModelRegistry and clearing memory handles.")
        self.embedding_model = None
        self.reranker = None
        self.retriever = None
        self.catalog.clear()
        self.recommendation_service = None
        self._catalog_loaded_from_disk = False
        self._is_initialized = False
