"""Test ranking interfaces, cross-encoder wrapper, and hybrid ranking algorithms."""

from backend.app.models.product import Product
from backend.app.ranking.base import RankedCandidate
from backend.app.ranking.cross_encoder import CrossEncoderReranker
from backend.app.ranking.hybrid_ranker import HybridRanker


def test_cross_encoder_fallback(sample_products: list[Product]) -> None:
    """Test cross-encoder fallback scoring when neural weights are not yet loaded."""
    reranker = CrossEncoderReranker(model_name="dummy_model", device="cpu")
    ranked = reranker.rerank(query="noise cancelling headphones", products=sample_products, top_k=2)

    assert len(ranked) == 2
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_hybrid_ranker_combination(sample_products: list[Product]) -> None:
    """Test hybrid score combination of relevance, ratings, and review popularity."""
    ranker = HybridRanker(relevance_weight=0.6, rating_weight=0.3, popularity_weight=0.1)

    candidates = [
        RankedCandidate(product=sample_products[0], score=0.8, rank=1),
        RankedCandidate(product=sample_products[1], score=0.7, rank=2),
    ]

    ranked = ranker.rank(query="headphones", candidates=candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].score > 0.0
    assert "rating_signal" in ranked[0].features
    assert "popularity_signal" in ranked[0].features
