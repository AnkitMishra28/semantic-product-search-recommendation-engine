"""Integration tests for Hybrid First-Stage Retrieval (BM25 + FAISS + RRF) and Cross-Encoder pipeline."""

import numpy as np
import pandas as pd
import pytest

from backend.app.models.product import Product
from backend.app.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.ranking.cross_encoder import CrossEncoderReranker
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.embeddings import ExactRetriever
from backend.app.retrieval.faiss_retriever import FaissRetriever
from backend.app.retrieval.hybrid import HybridRetriever


class MockEmbeddingService:
    """Mock embedding service producing deterministic vectors for unit tests."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.model_name = "mock-embedding-model"

    def encode_queries(self, query: str) -> np.ndarray:
        seed = sum(ord(c) for c in query) % 10000
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 10000
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dimension).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-10)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def hybrid_fixture(sample_products):
    """Fixture to instantiate BM25, FAISS, and HybridRetriever over sample products."""
    products_data = [p.model_dump() for p in sample_products]
    df = pd.DataFrame(products_data)
    df["parent_asin"] = df["asin"]

    # 1. BM25
    bm25 = BM25Retriever()
    bm25.index_corpus(df)

    # 2. FAISS
    mock_embedder = MockEmbeddingService(dimension=384)
    texts = [f"{p.title} {p.brand} {' '.join(p.features)}" for p in sample_products]
    vecs = mock_embedder.encode_documents(texts)
    doc_ids = [p.asin for p in sample_products]
    metadata = [
        {"title": p.title, "brand": p.brand, "price": p.price, "categories": p.categories}
        for p in sample_products
    ]

    faiss_retriever = FaissRetriever(
        dimension=384,
        index_type="FlatIP",
        embedding_service=mock_embedder,
    )
    faiss_retriever.index(vecs, doc_ids, metadata=metadata)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=faiss_retriever,
        rrf_k=60,
        default_top_k=10,
    )

    return hybrid, bm25, faiss_retriever, df


def test_hybrid_search_basic(hybrid_fixture):
    """Verify hybrid search executes BM25, FAISS, and RRF correctly."""
    hybrid, bm25, faiss_retriever, _ = hybrid_fixture

    result = hybrid.search_hybrid("noise cancelling headphones", top_k=5, candidate_k=5)

    assert result.candidate_count_after_fusion > 0
    assert len(result.candidates) <= 5
    assert result.timings["total_latency_ms"] >= 0.0
    assert "bm25_latency_ms" in result.timings
    assert "dense_latency_ms" in result.timings
    assert "fusion_latency_ms" in result.timings

    # Top candidate should have provenance
    top_cand = result.candidates[0]
    assert top_cand.rank == 1
    assert top_cand.rrf_score > 0.0
    assert len(top_cand.retrieved_by) >= 1


def test_hybrid_search_with_query_understanding_filters(hybrid_fixture):
    """Verify Query Understanding hard filters are applied to both retrievers without hard-filtering soft signals."""
    hybrid, _, _, _ = hybrid_fixture

    # Query with brand and price constraints
    query = "Sony headphones under 400"
    result, qu_res = hybrid.search_with_query_understanding(query, top_k=5)

    assert qu_res.brand == "Sony"
    assert qu_res.price_max == 400.0
    assert "brand" in qu_res.hard_filters
    assert "price_max" in qu_res.hard_filters

    # Result should only contain Sony items under 400
    for cand in result.candidates:
        assert cand.metadata.get("brand") == "Sony"
        assert cand.metadata.get("price") <= 400.0


def test_end_to_end_hybrid_to_cross_encoder(hybrid_fixture):
    """Verify full Stage 1 Hybrid RRF -> Stage 2 Cross-Encoder reranking pipeline."""
    hybrid, _, _, df = hybrid_fixture

    query = "wireless noise cancelling headphones"

    # Stage 1: Hybrid First-Stage Candidate Generation
    hybrid_res = hybrid.search_hybrid(query, top_k=10, candidate_k=10)
    candidate_ids = [c.doc_id for c in hybrid_res.candidates]
    first_stage_scores = {c.doc_id: c.rrf_score for c in hybrid_res.candidates}
    first_stage_ranks = {c.doc_id: c.rank for c in hybrid_res.candidates}

    doc_text_map = {
        row["asin"]: f"{row['title']} {row['brand']} {' '.join(row['features'])}"
        for _, row in df.iterrows()
    }

    # Stage 2: Cross-Encoder Reranking
    reranker = CrossEncoderReranker(device="cpu")
    # Will use fallback scoring in mock mode if weights aren't loaded in test
    ranked_results = reranker.rerank_candidates(
        query=query,
        candidate_ids=candidate_ids,
        doc_text_map=doc_text_map,
        first_stage_scores=first_stage_scores,
        first_stage_ranks=first_stage_ranks,
        top_k=5,
    )

    assert len(ranked_results) == len(candidate_ids)
    assert ranked_results[0].rank == 1
    assert ranked_results[0].first_stage_score is not None
    assert ranked_results[0].first_stage_rank is not None
    assert "cross_encoder_score" in ranked_results[0].features
