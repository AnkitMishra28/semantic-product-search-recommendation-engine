"""Cross-Encoder neural reranking implementation."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from backend.app.models.product import Product
from backend.app.preprocessing.product_document import build_product_text
from backend.app.ranking.base import BaseReranker, RankedCandidate

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Stage 2 neural reranker using full cross-attention over (query, document) pairs."""

    _instance: Optional["CrossEncoderReranker"] = None

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Any = None
        self._is_loaded = False

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        batch_size: int = 32,
    ) -> "CrossEncoderReranker":
        """Get or initialize singleton instance of CrossEncoderReranker."""
        if cls._instance is None:
            cls._instance = cls(
                model_name=model_name,
                device=device,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
            )
            cls._instance.load_model()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Return model loading status."""
        return self._is_loaded and self.model is not None

    def load_model(self) -> None:
        """Load pretrained cross-encoder weights into memory once."""
        if self._is_loaded and self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(
                f"Loading CrossEncoder model '{self.model_name}' on device '{self.device}' "
                f"(max_length={self.max_seq_length}, batch_size={self.batch_size})"
            )
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_seq_length,
            )
            # Ensure model is strictly in eval mode
            if hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
                self.model.model.eval()
            self._is_loaded = True
            logger.info(f"CrossEncoder '{self.model_name}' successfully loaded and set to eval mode.")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder '{self.model_name}': {e}. Running in fallback mode.")
            self._is_loaded = False

    def predict_pairs(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """Score a collection of (query, document_text) pairs in batches with gradients disabled."""
        if not pairs:
            return np.array([], dtype=np.float32)

        if not self._is_loaded or self.model is None:
            self.load_model()

        if self._is_loaded and self.model is not None:
            pair_list = [[q, d] for q, d in pairs]
            with torch.no_grad():
                raw_scores = self.model.predict(
                    pair_list,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            return np.asarray(raw_scores, dtype=np.float32)

        # Fallback dummy scores if model is unavailable
        return np.array([1.0 / (i + 1.0) for i in range(len(pairs))], dtype=np.float32)

    def rerank(
        self,
        query: str,
        products: List[Product],
        top_k: int = 20,
        candidate_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Score candidate Product objects against the query using cross-attention."""
        if not products:
            return []

        if candidate_k is not None and candidate_k > 0:
            candidate_products = products[:candidate_k]
        else:
            candidate_products = products

        # Construct pairs with canonical title_brand_category_features representation
        pairs: List[Tuple[str, str]] = []
        for p in candidate_products:
            prod_dict = p.model_dump()
            doc_text = build_product_text(prod_dict, variant="title_brand_category_features")
            pairs.append((query, doc_text))

        scores = self.predict_pairs(pairs)

        ranked: List[RankedCandidate] = []
        for orig_idx, (p, score) in enumerate(zip(candidate_products, scores), start=1):
            doc_id = p.parent_asin or p.asin
            ranked.append(
                RankedCandidate(
                    doc_id=doc_id,
                    product_id=doc_id,
                    product=p,
                    score=float(score),
                    cross_encoder_score=float(score),
                    rank=0,
                    final_rank=0,
                    first_stage_score=None,
                    original_retrieval_score=None,
                    first_stage_rank=orig_idx,
                    original_rank=orig_idx,
                    features={"cross_encoder_score": float(score)},
                )
            )

        # Sort descending by cross-encoder score
        ranked.sort(key=lambda x: x.score, reverse=True)
        for idx, item in enumerate(ranked, start=1):
            item.rank = idx
            item.final_rank = idx

        return ranked[:top_k]

    def rerank_candidates(
        self,
        query: str,
        candidate_ids: List[str],
        doc_text_map: Dict[str, str],
        first_stage_scores: Optional[Dict[str, float]] = None,
        first_stage_ranks: Optional[Dict[str, int]] = None,
        top_k: int = 20,
        candidate_k: Optional[int] = None,
    ) -> List[RankedCandidate]:
        """Score candidate document IDs using precomputed product texts."""
        if not candidate_ids:
            return []

        if candidate_k is not None and candidate_k > 0:
            target_ids = candidate_ids[:candidate_k]
        else:
            target_ids = candidate_ids

        pairs: List[Tuple[str, str]] = []
        for doc_id in target_ids:
            text = doc_text_map.get(doc_id, f"Product {doc_id}")
            pairs.append((query, text))

        scores = self.predict_pairs(pairs)

        first_stage_scores = first_stage_scores or {}
        first_stage_ranks = first_stage_ranks or {}

        ranked: List[RankedCandidate] = []
        for orig_idx, (doc_id, score) in enumerate(zip(target_ids, scores), start=1):
            f_score = first_stage_scores.get(doc_id)
            f_rank = first_stage_ranks.get(doc_id, orig_idx)
            ranked.append(
                RankedCandidate(
                    doc_id=doc_id,
                    product_id=doc_id,
                    score=float(score),
                    cross_encoder_score=float(score),
                    rank=0,
                    final_rank=0,
                    first_stage_score=f_score,
                    original_retrieval_score=f_score,
                    first_stage_rank=f_rank,
                    original_rank=f_rank,
                    features={
                        "cross_encoder_score": float(score),
                        "first_stage_score": f_score,
                        "first_stage_rank": f_rank,
                    },
                )
            )

        # Sort descending by cross-encoder score
        ranked.sort(key=lambda x: x.score, reverse=True)
        for idx, item in enumerate(ranked, start=1):
            item.rank = idx
            item.final_rank = idx

        return ranked[:top_k]
