"""ModelRegistry Singleton: Manages startup model loading and memory handles."""

import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
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
    """Coerce a cell (json string, numpy object array, list, or None) into List[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
        except Exception:
            return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


def _clean_str(value: Any) -> Optional[str]:
    """Normalize a cell to a trimmed string, or None for missing/NaN values."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_float(value: Any) -> Optional[float]:
    """Normalize a cell to a float, or None for missing/NaN/unparseable values."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


class SqliteProductCatalog(dict):
    """Memory-efficient catalog mapping backed by an indexed SQLite database on disk.
    
    Acts as a standard Dict[str, Product] for transparent drop-in compatibility across
    all API routers, recommenders, and search pipelines while using near-zero RAM (~1 MB).
    """

    def __init__(self, db_path: str, max_cache_size: int = 1000) -> None:
        super().__init__()
        self.db_path = str(db_path)
        self.max_cache_size = max_cache_size
        self._conn: Optional[sqlite3.Connection] = None
        self._total_count: Optional[int] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _row_to_product(self, row: sqlite3.Row) -> Product:
        asin = str(row["parent_asin"])
        quality_score = row["quality_score"] if "quality_score" in row.keys() else None
        meta = {}
        if quality_score is not None:
            try:
                meta["quality_score"] = float(quality_score)
            except (TypeError, ValueError):
                pass

        return Product(
            parent_asin=asin,
            asin=asin,
            title=_clean_str(row["title"]) or "Unknown Product",
            description=_clean_str(row["description"]) or "",
            features=_to_str_list(row["features"]),
            price=_clean_float(row["price"]),
            brand=_clean_str(row["brand"]),
            categories=_to_str_list(row["categories"]),
            average_rating=_clean_float(row["average_rating"]),
            rating_number=int(row["rating_number"] or 0),
            image_url=_clean_str(row["image_url"]),
            images=_to_str_list(row["images"]),
            bought_together=_to_str_list(row["bought_together"]),
            embedding_text=_clean_str(row["embedding_text"]) if "embedding_text" in row.keys() else None,
            metadata=meta,
        )

    def __getitem__(self, key: str) -> Product:
        if super().__contains__(key):
            return super().__getitem__(key)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products WHERE parent_asin = ? LIMIT 1", (str(key),))
        row = cursor.fetchone()
        if not row:
            raise KeyError(key)
        prod = self._row_to_product(row)
        if len(self) >= self.max_cache_size:
            super().clear()
        super().__setitem__(key, prod)
        return prod

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM products WHERE parent_asin = ? LIMIT 1", (str(key),))
        return cursor.fetchone() is not None

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        if self._total_count is None:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products")
            self._total_count = int(cursor.fetchone()[0])
        return self._total_count

    def keys(self) -> List[str]:  # type: ignore
        cursor = self.conn.cursor()
        cursor.execute("SELECT parent_asin FROM products")
        return [str(r[0]) for r in cursor.fetchall()]

    def items(self) -> Iterator[Tuple[str, Product]]:  # type: ignore
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products")
        for row in cursor.fetchall():
            prod = self._row_to_product(row)
            yield str(prod.parent_asin or prod.asin), prod

    def values(self) -> Iterator[Product]:  # type: ignore
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products")
        for row in cursor.fetchall():
            yield self._row_to_product(row)


class ModelRegistry:
    """Singleton service to manage model handles, vector indexes, and catalog access."""

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
        """Initialize lightweight data handles on startup; ML weights are loaded efficiently."""
        if self._is_initialized:
            logger.info("ModelRegistry is already initialized.")
            return

        logger.info(f"Initializing ModelRegistry on device: {self.settings.device}")

        # Constrain thread allocations for single-core / low-memory containers
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        # 1. Initialize Vector Retriever and load the persisted FAISS index
        self.retriever = FaissRetriever(
            dimension=self.settings.embedding_dim,
            index_type=self.settings.faiss_index_type,
        )
        self._load_faiss_index(self.settings.faiss_index_path)

        # 1b. Load the product catalog
        if not self.catalog:
            self.catalog = self._load_catalog(self.settings.products_catalog_path)
            self._catalog_loaded_from_disk = bool(self.catalog)

        # 2. Initialize Reranker handle (weights are loaded on first use or in background)
        self.reranker = CrossEncoderReranker(
            model_name=self.settings.reranker_model_name,
            device=self.settings.device,
            max_seq_length=self.settings.max_seq_length,
        )

        self._is_initialized = True
        logger.info("ModelRegistry successfully initialized.")

    def _load_faiss_index(self, index_path: str) -> None:
        """Load the persisted FAISS index into self.retriever."""
        path = Path(index_path)
        if not path.exists():
            logger.warning(f"FAISS index file not found at '{path}'. Retriever will remain empty.")
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
        """Load catalog from indexed SQLite database (0 MB RAM) or parquet fallback."""
        db_path = getattr(self.settings, "products_db_path", None) or str(Path(products_path).with_suffix(".db"))
        if Path(db_path).exists():
            logger.info(f"Connecting to memory-optimized SQLite catalog at '{db_path}'")
            return SqliteProductCatalog(db_path)

        path = Path(products_path)
        if not path.exists():
            logger.warning(f"Product catalog parquet not found at '{path}'. Catalog will remain empty.")
            return {}

        # If database does not exist, build indexed SQLite database for near-zero memory footprint
        try:
            logger.info(f"Generating optimized SQLite catalog '{db_path}' from parquet...")
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pandas()
            conn = sqlite3.connect(db_path)
            for col in ["categories", "features", "images", "bought_together"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: json.dumps(list(x)) if x is not None and hasattr(x, "__iter__") and not isinstance(x, str) else json.dumps([]) if x is not None else None
                    )
            df.to_sql("products", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_asin ON products (parent_asin);")
            conn.close()
            del df, table
            import gc; gc.collect()
            logger.info(f"Generated SQLite catalog '{db_path}' successfully.")
            return SqliteProductCatalog(db_path)
        except Exception as e:
            logger.warning(f"Could not build SQLite catalog ({e}); falling back to memory parquet read.")

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
                asin=asin,
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

    def get_recommendation_service(self) -> Optional[RecommendationService]:
        """Lazy load unified RecommendationService on demand."""
        if self.recommendation_service is None and self._catalog_loaded_from_disk:
            self.recommendation_service = self._build_recommendation_service()
        return self.recommendation_service

    def _build_recommendation_service(self) -> Optional[RecommendationService]:
        """Construct the unified RecommendationService — popularity, content-based,
        collaborative, hybrid, and MMR diversity reranking — wired directly from the
        catalog, interaction history, and content embeddings.
        """
        if not self.catalog:
            return None

        interactions_df = self._load_interactions(self.settings.interactions_path)
        embeddings, doc_ids = self._load_content_embeddings(
            self.settings.content_embeddings_path, self.settings.content_embeddings_metadata_path
        )

        t0 = time.perf_counter()

        popularity_rec = PopularityRecommender(product_catalog=self.catalog)  # type: ignore
        if not interactions_df.empty:
            popularity_rec.fit(interactions_df)

        content_rec: Optional[ContentBasedRecommender] = None
        mmr_reranker: Optional[MMRReranker] = None
        if embeddings is not None and doc_ids:
            content_rec = ContentBasedRecommender(
                embeddings=embeddings, doc_ids=doc_ids, product_catalog=self.catalog  # type: ignore
            )
            mmr_reranker = MMRReranker(embeddings=embeddings, doc_ids=doc_ids, default_lambda=0.7)

        collaborative_rec: Optional[CollaborativeRecommender] = None
        if not interactions_df.empty:
            collaborative_rec = CollaborativeRecommender(product_catalog=self.catalog).fit(interactions_df)  # type: ignore

        hybrid_rec = HybridRecommender(
            popularity_recommender=popularity_rec,
            content_recommender=content_rec,
            collaborative_recommender=collaborative_rec,
            diversity_reranker=mmr_reranker,
            product_catalog=self.catalog,  # type: ignore
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
        """Encode text string into dense vector embedding with lazy loading."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {self.settings.embedding_model_name}")
                self.embedding_model = SentenceTransformer(
                    self.settings.embedding_model_name,
                    device=self.settings.device,
                )
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using fallback vectors.")

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
