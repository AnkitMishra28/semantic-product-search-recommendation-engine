"""FAISS-backed vector retrieval implementation supporting Flat, HNSW, and IVF indexes."""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.app.retrieval.base import BaseRetriever, CandidateResult

logger = logging.getLogger(__name__)

# Optional import handling for faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class FaissRetriever(BaseRetriever):
    """Vector similarity retriever powered by Facebook AI Similarity Search (FAISS).

    Supports:
    - Exact inner product search: IndexFlatIP (cosine on normalized vectors)
    - Hierarchical Navigable Small World: IndexHNSWFlat (configurable M, efConstruction, efSearch)
    - Inverted File Index: IndexIVFFlat (configurable nlist, nprobe)
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "FlatIP",
        metric: str = "inner_product",
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        nlist: int = 256,
        nprobe: int = 16,
        embedding_service: Optional[Any] = None,
    ) -> None:
        self.dimension = dimension
        self.index_type = index_type.strip()
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.nlist = nlist
        self.nprobe = nprobe
        self.embedding_service = embedding_service

        self.index_handle: Any = None
        self.quantizer: Any = None
        self.id_to_doc_map: Dict[int, str] = {}
        self.doc_to_id_map: Dict[str, int] = {}
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        self._is_trained = False

        if FAISS_AVAILABLE:
            self._init_index()
        else:
            logger.warning("FAISS library not installed. FaissRetriever running in uninitialized mode.")

    def _get_metric_type(self) -> int:
        """Resolve FAISS metric constant."""
        if self.metric in ("inner_product", "cosine", "ip", "dot"):
            return faiss.METRIC_INNER_PRODUCT
        elif self.metric in ("l2", "euclidean"):
            return faiss.METRIC_L2
        return faiss.METRIC_INNER_PRODUCT

    def _init_index(self) -> None:
        """Instantiate underlying FAISS index structure."""
        if not FAISS_AVAILABLE:
            return

        metric_type = self._get_metric_type()
        itype = self.index_type.lower()

        if itype in ("flatip", "exact_flat_ip", "flat_ip", "exact"):
            self.index_handle = faiss.IndexFlatIP(self.dimension)
            self._is_trained = True
        elif itype in ("flatl2", "flat_l2"):
            self.index_handle = faiss.IndexFlatL2(self.dimension)
            self._is_trained = True
        elif itype in ("hnsw", "hnswflat", "indexhnswflat"):
            self.index_handle = faiss.IndexHNSWFlat(self.dimension, self.m, metric_type)
            self.index_handle.hnsw.efConstruction = self.ef_construction
            self.index_handle.hnsw.efSearch = self.ef_search
            self._is_trained = True
        elif itype in ("ivfflat", "ivf", "indexivfflat"):
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                self.quantizer = faiss.IndexFlatIP(self.dimension)
            else:
                self.quantizer = faiss.IndexFlatL2(self.dimension)
            self.index_handle = faiss.IndexIVFFlat(
                self.quantizer,
                self.dimension,
                self.nlist,
                metric_type,
            )
            self.index_handle.nprobe = self.nprobe
            self._is_trained = False
        else:
            logger.warning(f"Unknown index_type '{self.index_type}'. Defaulting to IndexFlatIP.")
            self.index_handle = faiss.IndexFlatIP(self.dimension)
            self._is_trained = True

    @property
    def is_trained(self) -> bool:
        """Return training status of underlying FAISS index."""
        if self.index_handle is None:
            return False
        return bool(self.index_handle.is_trained)

    def train(self, vectors: np.ndarray) -> float:
        """Train index clustering (e.g. for IVFFlat). Returns elapsed training time in seconds."""
        if not FAISS_AVAILABLE or self.index_handle is None:
            raise RuntimeError("Cannot train uninitialized FAISS index.")

        if self.index_handle.is_trained:
            logger.debug("Index is already trained; skipping train step.")
            return 0.0

        vecs = vectors.astype(np.float32)
        if self.metric in ("inner_product", "cosine", "ip"):
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            vecs = vecs / norms

        t0 = time.perf_counter()
        self.index_handle.train(vecs)
        t1 = time.perf_counter()
        self._is_trained = self.index_handle.is_trained
        elapsed_sec = t1 - t0
        logger.info(f"FAISS index trained in {elapsed_sec:.4f}s.")
        return elapsed_sec

    def index(
        self,
        vectors: np.ndarray,
        doc_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Add dense vectors and document mappings into the index.

        Returns:
            Elapsed time for vector addition in seconds.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Cannot add vectors to index.")

        if self.index_handle is None:
            self._init_index()

        if len(vectors) != len(doc_ids):
            raise ValueError(
                f"Vectors count ({len(vectors)}) does not match doc_ids count ({len(doc_ids)})"
            )

        vecs = vectors.astype(np.float32)
        if self.metric in ("inner_product", "cosine", "ip"):
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            vecs = vecs / norms

        # Automatically train if necessary
        if not self.index_handle.is_trained:
            self.train(vecs)

        start_idx = len(self.id_to_doc_map)

        t0 = time.perf_counter()
        self.index_handle.add(vecs)
        t1 = time.perf_counter()
        elapsed_sec = t1 - t0

        for offset, doc_id in enumerate(doc_ids):
            internal_id = start_idx + offset
            self.id_to_doc_map[internal_id] = doc_id
            self.doc_to_id_map[doc_id] = internal_id
            if metadata and offset < len(metadata):
                self.doc_metadata[doc_id] = metadata[offset]

        return elapsed_sec

    def set_ef_search(self, ef_search: int) -> None:
        """Dynamically update HNSW efSearch parameter."""
        self.ef_search = ef_search
        if self.index_handle is not None and hasattr(self.index_handle, "hnsw"):
            self.index_handle.hnsw.efSearch = ef_search

    def set_nprobe(self, nprobe: int) -> None:
        """Dynamically update IVFFlat nprobe parameter."""
        self.nprobe = nprobe
        if self.index_handle is not None and hasattr(self.index_handle, "nprobe"):
            self.index_handle.nprobe = nprobe

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateResult]:
        """Search vector index for nearest neighbors.

        Args:
            query_vector: 1D or 2D numpy array representing query embedding.
            top_k: Number of candidates to retrieve.
            filters: Optional metadata filter dict (e.g. {'brand': 'Sony', 'max_price': 100.0}).

        Returns:
            List of CandidateResult ordered descending by similarity score.
        """
        if self.index_handle is None or self.total_documents == 0:
            logger.debug("FAISS index is empty or uninitialized.")
            return []

        # Ensure 2D float32 array
        if query_vector.ndim == 1:
            q_vec = query_vector.reshape(1, -1).astype(np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        # Normalize query vector for cosine equivalence if using inner product
        if self.metric in ("inner_product", "cosine", "ip"):
            norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
            norm[norm == 0] = 1e-10
            q_vec = q_vec / norm

        # If filtering is required, over-fetch candidates then filter down to top_k
        fetch_k = top_k
        if filters:
            fetch_k = min(top_k * 5, self.total_documents)

        actual_k = min(fetch_k, self.total_documents)
        distances, indices = self.index_handle.search(q_vec, actual_k)

        results: List[CandidateResult] = []
        rank = 1
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self.id_to_doc_map.get(int(idx), str(idx))
            meta = self.doc_metadata.get(doc_id, {})

            # Filter validation
            if filters:
                if "brand" in filters and filters["brand"]:
                    req_brand = str(filters["brand"]).lower()
                    item_brand = str(meta.get("brand") or "").lower()
                    if req_brand != item_brand:
                        continue
                if "category" in filters and filters["category"]:
                    req_cat = str(filters["category"]).lower()
                    item_cats = meta.get("categories")
                    if isinstance(item_cats, (list, tuple, np.ndarray)):
                        cat_str = " ".join(str(c).lower() for c in item_cats)
                    else:
                        cat_str = str(item_cats or "").lower()
                    if req_cat not in cat_str:
                        continue
                max_p = filters.get("max_price") if "max_price" in filters else filters.get("price_max")
                if max_p is not None:
                    p = meta.get("price")
                    if p is None or p > float(max_p):
                        continue
                min_p = filters.get("min_price") if "min_price" in filters else filters.get("price_min")
                if min_p is not None:
                    p = meta.get("price")
                    if p is None or p < float(min_p):
                        continue

            results.append(
                CandidateResult(
                    doc_id=doc_id,
                    score=float(score),
                    rank=rank,
                    metadata=meta,
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def search_query(
        self,
        query_text: str,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[CandidateResult], float, float]:
        """Encode query string and execute FAISS search.

        Returns:
            Tuple of (results, encoding_latency_ms, retrieval_latency_ms).
        """
        if self.embedding_service is None:
            raise ValueError("EmbeddingService is required for raw query string search.")

        t0 = time.perf_counter()
        q_vec = self.embedding_service.encode_queries(query_text)
        t1 = time.perf_counter()
        encoding_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        results = self.search(query_vector=q_vec, top_k=top_k, filters=filters)
        t3 = time.perf_counter()
        search_ms = (t3 - t2) * 1000.0

        return results, encoding_ms, search_ms

    def save(self, file_path: str) -> None:
        """Serialize FAISS index binary and JSON metadata mapping to disk."""
        if not FAISS_AVAILABLE or self.index_handle is None:
            raise RuntimeError("Cannot save uninitialized FAISS index.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index_file = path if path.suffix == ".index" else path.with_suffix(".index")
        faiss.write_index(self.index_handle, str(index_file))

        meta_file = index_file.with_suffix(".meta.json")
        model_name = getattr(self.embedding_service, "model_name", "sentence-transformers/all-MiniLM-L6-v2")

        meta_payload = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "total_documents": self.total_documents,
            "embedding_model": model_name,
            "id_to_doc_map": {str(k): v for k, v in self.id_to_doc_map.items()},
            "doc_metadata": self.doc_metadata,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(f"Saved FAISS index ({self.index_type}, {self.total_documents} items) to {index_file}")

    def load(self, file_path: str) -> None:
        """Load FAISS index binary and JSON metadata mapping from disk."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed.")

        path = Path(file_path)
        index_file = path if path.suffix == ".index" else path.with_suffix(".index")

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")

        self.index_handle = faiss.read_index(str(index_file))
        self.dimension = self.index_handle.d
        self._is_trained = bool(self.index_handle.is_trained)

        meta_file = index_file.with_suffix(".meta.json")
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.dimension = meta.get("dimension", self.dimension)
                self.index_type = meta.get("index_type", self.index_type)
                self.metric = meta.get("metric", self.metric)
                self.m = meta.get("m", self.m)
                self.ef_construction = meta.get("ef_construction", self.ef_construction)
                self.ef_search = meta.get("ef_search", self.ef_search)
                self.nlist = meta.get("nlist", self.nlist)
                self.nprobe = meta.get("nprobe", self.nprobe)
                self.id_to_doc_map = {int(k): v for k, v in meta.get("id_to_doc_map", {}).items()}
                self.doc_to_id_map = {v: k for k, v in self.id_to_doc_map.items()}
                self.doc_metadata = meta.get("doc_metadata", {})

        # Re-apply runtime search parameters
        if hasattr(self.index_handle, "hnsw"):
            self.index_handle.hnsw.efSearch = self.ef_search
        if hasattr(self.index_handle, "nprobe"):
            self.index_handle.nprobe = self.nprobe

        logger.info(f"Loaded FAISS index ({self.index_type}, {self.total_documents} items) from {index_file}")

    def get_memory_usage_bytes(self) -> int:
        """Estimate memory footprint of the FAISS index."""
        if self.index_handle is None:
            return 0
        try:
            # Serialized size is an exact representation of C++ index data structure
            buf = faiss.serialize_index(self.index_handle)
            return len(buf)
        except Exception:
            # Approximate calculation: N * D * 4 bytes
            return int(self.total_documents * self.dimension * 4)

    @property
    def total_documents(self) -> int:
        """Return total count of indexed items."""
        if self.index_handle is None:
            return 0
        return int(self.index_handle.ntotal)

    search_vector = search

