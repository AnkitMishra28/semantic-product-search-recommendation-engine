"""Sentence Transformer Embedding Service and Exact Dense Vector Retriever.

Provides high-throughput batch document encoding, normalized query embedding,
and vectorized exact cosine similarity retrieval (control baseline before ANN).
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from backend.app.preprocessing.product_document import (
    TextRepresentationVariant,
    build_product_text,
)
from backend.app.retrieval.base import BaseRetriever, CandidateResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384


class EmbeddingService:
    """Singleton-style or reusable service for Sentence Transformer inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing SentenceTransformer '{model_name}' on device '{self.device}'")
        self.model = SentenceTransformer(model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._embedding_dim = self.model.get_embedding_dimension()
        else:
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        
        if self._embedding_dim != EXPECTED_EMBEDDING_DIM:
            logger.warning(
                f"Model dimension {self._embedding_dim} differs from expected {EXPECTED_EMBEDDING_DIM}"
            )

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimension of the underlying transformer."""
        return self._embedding_dim

    def encode_queries(
        self,
        queries: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """Encode one or multiple search queries into normalized dense vectors.
        
        Args:
            queries: Single query string or sequence of query strings.
            
        Returns:
            1D numpy array of shape (dim,) for single query, or 2D array of shape (N, dim).
        """
        if isinstance(queries, str):
            single = True
            query_list = [queries]
        else:
            single = False
            query_list = list(queries)

        if not query_list:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        vectors = self.model.encode(
            query_list,
            batch_size=len(query_list),
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        return vectors[0] if single else vectors

    encode_query = encode_queries

    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Batch encode product document representations into normalized dense vectors.
        
        Args:
            texts: List of product document text representations.
            batch_size: Mini-batch size for transformer forward pass.
            show_progress_bar: Whether to display a tqdm progress bar during encoding.
            
        Returns:
            2D numpy array of shape (len(texts), dim) with float32 dtype and unit L2 norm.
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ).astype(np.float32)

        return vectors


class ExactDenseRetriever(BaseRetriever):
    """Exact nearest-neighbor candidate retriever using vectorized inner products.
    
    Because product vectors and query vectors are L2-normalized:
        cosine_similarity(a, b) = dot(a, b)
    This provides mathematically exact cosine nearest-neighbor search.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        embeddings: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.embeddings = embeddings.astype(np.float32) if embeddings is not None else None
        self.doc_ids = doc_ids or []
        self.metadata = metadata or []
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def set_corpus(
        self,
        embeddings: np.ndarray,
        doc_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Assign precomputed dense embeddings and identifier mappings to the retriever."""
        if len(embeddings) != len(doc_ids):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) must match doc_ids count ({len(doc_ids)})"
            )
        self.embeddings = embeddings.astype(np.float32)
        self.doc_ids = list(doc_ids)
        self.metadata = metadata or [{} for _ in range(len(doc_ids))]
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateResult]:
        """Perform exact cosine similarity search against indexed product embeddings.
        
        Args:
            query_vector: 1D numpy array of shape (dim,) representing normalized query.
            top_k: Maximum candidate count to return.
            filters: Optional metadata filtering criteria.
            
        Returns:
            List of CandidateResult sorted in descending order of cosine similarity score.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Ensure 1D query vector
        q_vec = np.asarray(query_vector, dtype=np.float32).ravel()
        
        # Ensure query vector is unit normalized for cosine equivalence
        norm = np.linalg.norm(q_vec)
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
            q_vec = q_vec / norm

        # Vectorized dot product against all N document vectors
        # X is (N, D), q is (D,) -> scores is (N,)
        scores = np.dot(self.embeddings, q_vec)

        num_docs = len(scores)
        k = min(top_k, num_docs)

        if num_docs > k:
            # Partial partition for O(N + K log K) performance
            partitioned_indices = np.argpartition(scores, -k)[-k:]
            sorted_order = partitioned_indices[np.argsort(-scores[partitioned_indices])]
        else:
            sorted_order = np.argsort(-scores)

        results: List[CandidateResult] = []
        rank = 1
        for idx in sorted_order:
            score_val = float(scores[idx])
            doc_id = self.doc_ids[idx]
            meta = self.metadata[idx] if idx < len(self.metadata) else {}

            # Metadata filtering
            if filters:
                if "brand" in filters and filters["brand"]:
                    req_brand = str(filters["brand"]).lower()
                    item_brand = str(meta.get("brand") or "").lower()
                    if req_brand != item_brand:
                        continue
                if "max_price" in filters and filters["max_price"] is not None:
                    p = meta.get("price")
                    if p is None or p > float(filters["max_price"]):
                        continue

            results.append(
                CandidateResult(
                    doc_id=doc_id,
                    score=score_val,
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
    ) -> Tuple[List[CandidateResult], float]:
        """Encode query text and perform exact semantic search.
        
        Returns:
            Tuple of (List[CandidateResult], query_encoding_time_ms).
        """
        if self.embedding_service is None:
            raise ValueError("EmbeddingService is required to search with raw text queries.")

        t0 = time.perf_counter()
        query_vector = self.embedding_service.encode_queries(query_text)
        t1 = time.perf_counter()
        encoding_ms = (t1 - t0) * 1000.0

        candidates = self.search(query_vector=query_vector, top_k=top_k, filters=filters)
        return candidates, encoding_ms

    def index(
        self,
        vectors: np.ndarray,
        doc_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """BaseRetriever interface compliance."""
        self.set_corpus(embeddings=vectors, doc_ids=doc_ids, metadata=metadata)

    def save(self, file_path: str) -> None:
        """Serialize embeddings and metadata to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        meta_path = file_path.replace(".npy", ".json")
        if not meta_path.endswith(".json"):
            meta_path += ".json"
        npy_path = file_path if file_path.endswith(".npy") else file_path + ".npy"

        np.save(npy_path, self.embeddings)
        meta_payload = {
            "num_documents": len(self.doc_ids),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
            "doc_ids": self.doc_ids,
            "metadata": self.metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f)

    def load(self, file_path: str) -> None:
        """Load serialized embeddings and metadata from disk."""
        npy_path = file_path if file_path.endswith(".npy") else file_path + ".npy"
        meta_path = file_path.replace(".npy", ".json")
        if not meta_path.endswith(".json"):
            meta_path += ".json"

        self.embeddings = np.load(npy_path).astype(np.float32)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_payload = json.load(f)

        self.doc_ids = meta_payload["doc_ids"]
        self.metadata = meta_payload.get("metadata", [])
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    @property
    def total_documents(self) -> int:
        """Return total count of indexed product documents."""
        return len(self.doc_ids)


# Architectural class alias
ExactRetriever = ExactDenseRetriever
