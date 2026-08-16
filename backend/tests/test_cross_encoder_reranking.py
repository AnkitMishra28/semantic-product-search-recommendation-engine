"""Unit and Integration Tests for Cross-Encoder Second-Stage Neural Reranking."""

import numpy as np
import pytest

from backend.app.models.product import Product
from backend.app.preprocessing.product_document import build_product_text
from backend.app.ranking.base import BaseRanker, BaseReranker, RankedCandidate
from backend.app.ranking.cross_encoder import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)


@pytest.fixture(scope="module")
def mock_products() -> list[Product]:
    """Provide structured sample products for testing reranking logic."""
    return [
        Product(
            parent_asin="B001",
            title="Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            brand="Sony",
            categories=["Electronics", "Headphones", "Over-Ear"],
            features=["Industry Leading Noise Cancellation", "30-hour battery life", "Multipoint connection"],
            price=398.0,
            average_rating=4.6,
            rating_number=12000,
        ),
        Product(
            parent_asin="B002",
            title="Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones",
            brand="Bose",
            categories=["Electronics", "Headphones", "Over-Ear"],
            features=["TriPort acoustic architecture", "Quiet and Aware Modes", "24-hour battery"],
            price=329.0,
            average_rating=4.5,
            rating_number=8500,
        ),
        Product(
            parent_asin="B003",
            title="Anker 737 Power Bank (PowerCore 24K) 3-Port Portable Charger",
            brand="Anker",
            categories=["Electronics", "Accessories", "Power Banks"],
            features=["140W Two-Way Fast Charging", "Smart Digital Display", "24,000mAh capacity"],
            price=149.99,
            average_rating=4.7,
            rating_number=5000,
        ),
        Product(
            parent_asin="B004",
            title="Apple AirPods Max Wireless Over-Ear Headphones",
            brand="Apple",
            categories=["Electronics", "Headphones", "Over-Ear"],
            features=["Apple-designed dynamic driver", "Active Noise Cancellation", "Transparency mode"],
            price=549.0,
            average_rating=4.4,
            rating_number=9200,
        ),
    ]


@pytest.fixture(scope="module")
def loaded_reranker() -> CrossEncoderReranker:
    """Module-scoped CrossEncoder instance to minimize load overhead."""
    reranker = CrossEncoderReranker(
        model_name=DEFAULT_RERANKER_MODEL,
        device="cpu",
        max_seq_length=128,
        batch_size=16,
    )
    reranker.load_model()
    return reranker


class TestCrossEncoderReranker:
    """Test suite for CrossEncoder neural reranker implementation."""

    def test_model_initialization_and_metadata(self, loaded_reranker: CrossEncoderReranker) -> None:
        assert loaded_reranker.model_name == DEFAULT_RERANKER_MODEL
        assert loaded_reranker.device == "cpu"
        assert loaded_reranker.max_seq_length == 128
        assert loaded_reranker.batch_size == 16
        assert loaded_reranker.is_loaded

    def test_pair_construction_and_batch_scoring(self, loaded_reranker: CrossEncoderReranker) -> None:
        pairs = [
            ("noise cancelling bluetooth headphones", "Title: Sony WH-1000XM5. Features: Active Noise Cancellation"),
            ("noise cancelling bluetooth headphones", "Title: Anker 737 Portable Charger Power Bank"),
        ]
        scores = loaded_reranker.predict_pairs(pairs)
        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32
        # Sony headphones should have higher score than Anker power bank for headphones query
        assert scores[0] > scores[1]

    def test_top_k_reranking_order_and_ranks(
        self,
        loaded_reranker: CrossEncoderReranker,
        mock_products: list[Product],
    ) -> None:
        query = "best noise cancelling headphones for travel"
        ranked = loaded_reranker.rerank(query=query, products=mock_products, top_k=3)

        assert len(ranked) == 3
        # Ranks must be 1-indexed sequential integers
        assert [r.rank for r in ranked] == [1, 2, 3]
        # Scores must be strictly descending
        assert ranked[0].score >= ranked[1].score >= ranked[2].score
        # Headphones should rank above power bank
        top_asins = [r.doc_id for r in ranked]
        assert "B003" not in top_asins or ranked[-1].doc_id == "B003"

    def test_singleton_behavior(self) -> None:
        inst1 = CrossEncoderReranker.get_instance(device="cpu", max_seq_length=128)
        inst2 = CrossEncoderReranker.get_instance(device="cpu", max_seq_length=128)
        assert inst1 is inst2
        assert inst1.is_loaded

    def test_batch_sizes_consistency(
        self,
        loaded_reranker: CrossEncoderReranker,
        mock_products: list[Product],
    ) -> None:
        query = "wireless noise cancelling headphones"
        pairs = [(query, build_product_text(p.model_dump())) for p in mock_products]

        scores_b1 = CrossEncoderReranker(device="cpu", batch_size=1, max_seq_length=128)
        scores_b1.load_model()
        res_1 = scores_b1.predict_pairs(pairs)

        scores_b16 = CrossEncoderReranker(device="cpu", batch_size=16, max_seq_length=128)
        scores_b16.load_model()
        res_16 = scores_b16.predict_pairs(pairs)

        assert len(res_1) == len(res_16) == 4
        np.testing.assert_allclose(res_1, res_16, rtol=1e-4, atol=1e-4)

    def test_candidate_budget_k_parameter(
        self,
        loaded_reranker: CrossEncoderReranker,
        mock_products: list[Product],
    ) -> None:
        query = "wireless headphones"
        # 4 products available; candidate_k=2 should only consider first 2
        ranked = loaded_reranker.rerank(query=query, products=mock_products, top_k=10, candidate_k=2)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        asins = [r.doc_id for r in ranked]
        assert "B003" not in asins and "B004" not in asins

    def test_candidate_preservation_and_first_stage_ranks(
        self,
        loaded_reranker: CrossEncoderReranker,
    ) -> None:
        candidate_ids = ["B003", "B001", "B002"]
        doc_text_map = {
            "B001": "Title: Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "B002": "Title: Bose QC45 Noise Cancelling Headphones",
            "B003": "Title: Anker Power Bank Portable Charger",
        }
        first_stage_scores = {"B003": 0.85, "B001": 0.80, "B002": 0.75}
        first_stage_ranks = {"B003": 1, "B001": 2, "B002": 3}

        ranked = loaded_reranker.rerank_candidates(
            query="noise cancelling headphones",
            candidate_ids=candidate_ids,
            doc_text_map=doc_text_map,
            first_stage_scores=first_stage_scores,
            first_stage_ranks=first_stage_ranks,
            top_k=3,
        )

        assert len(ranked) == 3
        # First-stage metadata and all required aliases must be preserved
        for r in ranked:
            assert r.product_id == r.doc_id
            assert r.cross_encoder_score == r.score
            assert r.final_rank == r.rank
            assert r.first_stage_score is not None
            assert r.original_retrieval_score == r.first_stage_score
            assert r.first_stage_rank in [1, 2, 3]
            assert r.original_rank == r.first_stage_rank
            assert r.first_stage_score == first_stage_scores[r.doc_id]
            assert r.first_stage_rank == first_stage_ranks[r.doc_id]

        # B001 or B002 should be promoted over B003
        assert ranked[0].doc_id in ["B001", "B002"]

    def test_empty_candidate_list(self, loaded_reranker: CrossEncoderReranker) -> None:
        ranked = loaded_reranker.rerank(query="test query", products=[], top_k=10)
        assert ranked == []

        ranked_cand = loaded_reranker.rerank_candidates(
            query="test query",
            candidate_ids=[],
            doc_text_map={},
            top_k=10,
        )
        assert ranked_cand == []

    def test_top_k_exceeding_candidate_count(
        self,
        loaded_reranker: CrossEncoderReranker,
        mock_products: list[Product],
    ) -> None:
        ranked = loaded_reranker.rerank(query="headphones", products=mock_products[:2], top_k=50)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_deterministic_scoring(
        self,
        loaded_reranker: CrossEncoderReranker,
        mock_products: list[Product],
    ) -> None:
        query = "portable power bank fast charge"
        run1 = loaded_reranker.rerank(query=query, products=mock_products, top_k=4)
        run2 = loaded_reranker.rerank(query=query, products=mock_products, top_k=4)

        assert [r.doc_id for r in run1] == [r.doc_id for r in run2]
        for r1, r2 in zip(run1, run2):
            assert np.isclose(r1.score, r2.score, atol=1e-5)

    def test_cpu_fallback_mode(self, mock_products: list[Product]) -> None:
        fallback_reranker = CrossEncoderReranker(model_name="nonexistent/dummy_model_name_for_fallback", device="cpu")
        # Explicitly don't crash when model cannot load
        fallback_reranker.load_model()
        assert not fallback_reranker.is_loaded

        ranked = fallback_reranker.rerank(query="test", products=mock_products, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

