"""BM25 Okapi lexical retrieval engine for catalog search."""

import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from backend.app.retrieval.base import BaseRetriever, CandidateResult
from backend.app.retrieval.tokenizer import tokenize_lexical


class BM25Retriever(BaseRetriever):
    """Lexical candidate retriever based on BM25 Okapi algorithm."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or tokenize_lexical
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self._doc_id_to_idx: Dict[str, int] = {}

    def index_corpus(
        self,
        products_df: pd.DataFrame,
        id_column: str = "parent_asin",
        text_column: Optional[str] = None,
    ) -> float:
        """Build the BM25 index from a pandas DataFrame of products.
        
        Args:
            products_df: Processed products DataFrame.
            id_column: Column name for canonical product ID.
            text_column: Optional column name for pre-computed search text. If None,
                         builds text dynamically from title, brand, categories, features, description.
                         
        Returns:
            Index construction elapsed time in seconds.
        """
        start_time = time.perf_counter()
        
        self.doc_ids = products_df[id_column].astype(str).tolist()
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        
        # Build metadata cache for candidate results
        self.doc_metadata = []
        tokenized_corpus: List[List[str]] = []
        
        for _, row in products_df.iterrows():
            if text_column and text_column in row and pd.notna(row[text_column]):
                doc_text = str(row[text_column])
            else:
                title = str(row.get("title") or "")
                brand = str(row.get("brand") or "")
                
                cats_raw = row.get("categories")
                if isinstance(cats_raw, (list, tuple, np.ndarray)):
                    cats = " ".join([str(c) for c in cats_raw if str(c).strip()])
                else:
                    cats = str(cats_raw or "")
                    
                feats_raw = row.get("features")
                if isinstance(feats_raw, (list, tuple, np.ndarray)):
                    feats = " ".join([str(f) for f in feats_raw if str(f).strip()])
                else:
                    feats = str(feats_raw or "")
                    
                desc = str(row.get("description") or "")
                
                # Compose complete lexical representation (excluding price/rating business signals)
                doc_text = f"{title} {brand} {cats} {feats} {desc}".strip()
                
            tokenized_corpus.append(self.tokenizer(doc_text))
            
            meta = {
                "title": row.get("title"),
                "brand": row.get("brand"),
                "categories": row.get("categories"),
                "price": row.get("price") if pd.notna(row.get("price")) else None,
                "average_rating": row.get("average_rating") if pd.notna(row.get("average_rating")) else None,
            }
            self.doc_metadata.append(meta)

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        # Enforce positive lower bound on zero/negative IDFs so valid terms retain positive score
        avg_idf = getattr(self.bm25, "average_idf", 1.0) or 1.0
        floor_val = max(avg_idf * getattr(self.bm25, "epsilon", 0.25), 1e-4)
        for term, val in self.bm25.idf.items():
            if val <= 0.0:
                self.bm25.idf[term] = floor_val

        elapsed_sec = time.perf_counter() - start_time
        return elapsed_sec

    def search_text(
        self,
        query: str,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateResult]:
        """Search the BM25 index with a raw text query.
        
        Args:
            query: Raw user query string.
            top_k: Maximum candidate count to return.
            filters: Optional structured filter constraints (e.g. brand, category, price_min, price_max).
            
        Returns:
            List of CandidateResult items sorted in descending order of BM25 score.
        """
        if self.bm25 is None or not self.doc_ids:
            return []
            
        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Filter strictly positive scores
        positive_mask = scores > 0.0
        if not np.any(positive_mask):
            return []

        # Find top_k indices sorted descending
        positive_indices = np.where(positive_mask)[0]
        positive_scores = scores[positive_indices]
        
        if len(positive_indices) > top_k:
            # Partial sort for efficiency on large candidate sets
            top_partition = np.argpartition(positive_scores, -top_k)[-top_k:]
            sorted_order = top_partition[np.argsort(-positive_scores[top_partition])]
            selected_indices = positive_indices[sorted_order]
        else:
            sorted_order = np.argsort(-positive_scores)
            selected_indices = positive_indices[sorted_order]

        results: List[CandidateResult] = []
        rank = 1
        for idx in selected_indices:
            score_val = float(scores[idx])
            doc_id = self.doc_ids[idx]
            meta = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
            
            # Apply structured metadata filters if provided
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
                    score=score_val,
                    rank=rank,
                    metadata=meta,
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateResult]:
        """BaseRetriever interface compliance (raises NotImplementedError for dense vector inputs)."""
        raise NotImplementedError(
            "BM25Retriever requires text queries via search_text(query: str, top_k: int)."
        )

    def index(
        self,
        vectors: np.ndarray,
        doc_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """BaseRetriever interface compliance (raises NotImplementedError for raw dense vector inputs)."""
        raise NotImplementedError(
            "BM25Retriever operates on text tokens; use index_corpus(products_df) instead."
        )

    def save(self, file_path: str) -> None:
        """Serialize BM25 index and ID mappings to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        payload = {
            "bm25": self.bm25,
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
            "k1": self.k1,
            "b": self.b,
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path: str) -> None:
        """Load serialized BM25 index and ID mappings from disk."""
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
            
        self.bm25 = payload["bm25"]
        self.doc_ids = payload["doc_ids"]
        self.doc_metadata = payload.get("doc_metadata", [])
        self.k1 = payload.get("k1", 1.5)
        self.b = payload.get("b", 0.75)
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    @property
    def total_documents(self) -> int:
        """Return total document count in the index."""
        return len(self.doc_ids)

    search = search_text
    retrieve = search_text
