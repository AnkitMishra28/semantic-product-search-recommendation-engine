"""Unit tests for Reciprocal Rank Fusion (RRF) algorithms and provenance attribution."""

import pytest
from backend.app.retrieval.base import CandidateResult, FusedCandidateResult
from backend.app.retrieval.rrf import (
    DEFAULT_RRF_K,
    calculate_candidate_overlap,
    compute_rrf_score,
    reciprocal_rank_fusion,
)


def test_compute_rrf_score_basic():
    """Verify manual mathematical calculation of RRF scores."""
    # Single rank 1 with k=60 -> 1/61
    score = compute_rrf_score([1], k=60)
    assert pytest.approx(score, rel=1e-6) == 1.0 / 61.0

    # Multiple ranks [1, 2] with k=60 -> 1/61 + 1/62
    score_multi = compute_rrf_score([1, 2], k=60)
    assert pytest.approx(score_multi, rel=1e-6) == (1.0 / 61.0 + 1.0 / 62.0)

    # Custom k=10 -> 1/11
    score_k10 = compute_rrf_score([1], k=10)
    assert pytest.approx(score_k10, rel=1e-6) == 1.0 / 11.0

    # Invalid k <= 0 raises ValueError
    with pytest.raises(ValueError):
        compute_rrf_score([1], k=0)
    with pytest.raises(ValueError):
        compute_rrf_score([1], k=-5)


def test_rrf_rank_ordering():
    """Verify descending score sort order and 1-indexed rank assignment."""
    bm25_cands = [
        CandidateResult(doc_id="doc_1", score=10.0, rank=1),
        CandidateResult(doc_id="doc_2", score=8.0, rank=2),
        CandidateResult(doc_id="doc_3", score=5.0, rank=3),
    ]
    dense_cands = [
        CandidateResult(doc_id="doc_1", score=0.95, rank=1),
        CandidateResult(doc_id="doc_2", score=0.85, rank=2),
        CandidateResult(doc_id="doc_3", score=0.75, rank=3),
    ]

    fused = reciprocal_rank_fusion({"bm25": bm25_cands, "dense": dense_cands}, k=60)

    assert len(fused) == 3
    assert fused[0].doc_id == "doc_1"
    assert fused[0].rank == 1
    assert fused[0].rrf_score == pytest.approx(1.0 / 61.0 + 1.0 / 61.0)

    assert fused[1].doc_id == "doc_2"
    assert fused[1].rank == 2
    assert fused[1].rrf_score == pytest.approx(1.0 / 62.0 + 1.0 / 62.0)

    assert fused[2].doc_id == "doc_3"
    assert fused[2].rank == 3
    assert fused[2].rrf_score == pytest.approx(1.0 / 63.0 + 1.0 / 63.0)


def test_rrf_missing_documents_and_complementary_boost():
    """Verify that items present in both retrievers receive additive boosts over single-retriever items."""
    # doc_both is rank 5 in both retrievers -> 1/65 + 1/65 = 2/65 = ~0.030769
    # doc_bm_only is rank 1 in BM25 only -> 1/61 = ~0.016393
    # doc_dense_only is rank 1 in Dense only -> 1/61 = ~0.016393
    bm25_cands = [
        CandidateResult(doc_id="doc_bm_only", score=15.0, rank=1),
        CandidateResult(doc_id="doc_both", score=6.0, rank=5),
    ]
    dense_cands = [
        CandidateResult(doc_id="doc_dense_only", score=0.92, rank=1),
        CandidateResult(doc_id="doc_both", score=0.78, rank=5),
    ]

    fused = reciprocal_rank_fusion({"bm25": bm25_cands, "dense": dense_cands}, k=60)

    assert len(fused) == 3
    # doc_both should rank 1st due to dual-retrieval boost
    assert fused[0].doc_id == "doc_both"
    assert fused[0].rank == 1
    assert fused[0].retrieved_by == ["bm25", "dense"]

    # doc_bm_only and doc_dense_only should follow
    single_ids = {fused[1].doc_id, fused[2].doc_id}
    assert single_ids == {"doc_bm_only", "doc_dense_only"}


def test_rrf_duplicate_documents_in_retriever():
    """Verify deduplication when an index returns the same document multiple times."""
    bm25_cands = [
        CandidateResult(doc_id="doc_dup", score=10.0, rank=1),
        CandidateResult(doc_id="doc_dup", score=8.0, rank=3),  # duplicate
        CandidateResult(doc_id="doc_other", score=5.0, rank=2),
    ]

    fused = reciprocal_rank_fusion({"bm25": bm25_cands}, k=60)
    assert len(fused) == 2
    # Best rank 1 should be recorded
    dup_cand = next(c for c in fused if c.doc_id == "doc_dup")
    assert dup_cand.bm25_rank == 1


def test_rrf_tied_ranks_and_deterministic_order():
    """Verify deterministic stable ordering when two documents have identical RRF scores."""
    bm25_cands = [
        CandidateResult(doc_id="doc_b", score=10.0, rank=1),
        CandidateResult(doc_id="doc_a", score=10.0, rank=1),
    ]

    fused = reciprocal_rank_fusion({"bm25": bm25_cands}, k=60)
    assert len(fused) == 2
    # Both receive valid distinct 1-indexed ranks 1 and 2
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[0].rrf_score == fused[1].rrf_score


def test_rrf_configurable_k():
    """Verify configurable k alters the relative rank scaling."""
    bm25_cands = [
        CandidateResult(doc_id="doc_top1", score=10.0, rank=1),
        CandidateResult(doc_id="doc_deep", score=2.0, rank=10),
    ]

    fused_k10 = reciprocal_rank_fusion({"bm25": bm25_cands}, k=10)
    fused_k100 = reciprocal_rank_fusion({"bm25": bm25_cands}, k=100)

    # At k=10: rank 1 is 1/11 (~0.0909), rank 10 is 1/20 (0.0500) -> ratio ~1.818
    # At k=100: rank 1 is 1/101 (~0.0099), rank 10 is 1/110 (~0.00909) -> ratio ~1.089
    top_k10 = next(c for c in fused_k10 if c.doc_id == "doc_top1").rrf_score
    deep_k10 = next(c for c in fused_k10 if c.doc_id == "doc_deep").rrf_score
    assert (top_k10 / deep_k10) > 1.8

    top_k100 = next(c for c in fused_k100 if c.doc_id == "doc_top1").rrf_score
    deep_k100 = next(c for c in fused_k100 if c.doc_id == "doc_deep").rrf_score
    assert (top_k100 / deep_k100) < 1.1


def test_rrf_candidate_union_and_source_attribution():
    """Verify candidate union and complete provenance attribution."""
    bm25_cands = [
        CandidateResult(doc_id="B001", score=12.5, rank=1, metadata={"brand": "Sony", "price": 348.0}),
        CandidateResult(doc_id="B002", score=8.2, rank=2, metadata={"brand": "AudioTech"}),
    ]
    dense_cands = [
        CandidateResult(doc_id="B002", score=0.88, rank=1, metadata={"brand": "AudioTech", "price": 19.99}),
        CandidateResult(doc_id="B003", score=0.79, rank=2, metadata={"brand": "Anker", "price": 29.99}),
    ]

    fused = reciprocal_rank_fusion({"bm25": bm25_cands, "dense": dense_cands}, k=60)

    # Union should contain B001, B002, B003
    assert len(fused) == 3

    b001 = next(c for c in fused if c.doc_id == "B001")
    assert b001.retrieved_by == ["bm25"]
    assert b001.bm25_rank == 1
    assert b001.dense_rank is None
    assert b001.bm25_score == 12.5
    assert b001.dense_score is None

    b002 = next(c for c in fused if c.doc_id == "B002")
    assert b002.retrieved_by == ["bm25", "dense"]
    assert b002.bm25_rank == 2
    assert b002.dense_rank == 1
    assert b002.bm25_score == 8.2
    assert b002.dense_score == 0.88

    b003 = next(c for c in fused if c.doc_id == "B003")
    assert b003.retrieved_by == ["dense"]
    assert b003.bm25_rank is None
    assert b003.dense_rank == 2

    # Verify provenance dictionary
    prov = b002.to_provenance_dict()
    assert prov["product_id"] == "B002"
    assert prov["bm25_rank"] == 2
    assert prov["dense_rank"] == 1
    assert prov["retrieved_by"] == ["bm25", "dense"]


def test_rrf_top_k_truncation():
    """Verify top_k parameter truncates candidate pool properly."""
    cands = [CandidateResult(doc_id=f"doc_{i}", score=float(20 - i), rank=i) for i in range(1, 21)]
    fused = reciprocal_rank_fusion({"bm25": cands}, k=60, top_k=5)
    assert len(fused) == 5
    assert [c.rank for c in fused] == [1, 2, 3, 4, 5]


def test_rrf_metadata_preservation():
    """Verify document metadata is preserved during fusion."""
    bm25_cands = [
        CandidateResult(
            doc_id="B001",
            score=10.0,
            rank=1,
            metadata={"title": "Sony Headphones", "price": 348.0, "categories": ["Electronics"]},
        )
    ]
    fused = reciprocal_rank_fusion({"bm25": bm25_cands}, k=60)
    assert fused[0].metadata["title"] == "Sony Headphones"
    assert fused[0].metadata["price"] == 348.0
    assert fused[0].metadata["categories"] == ["Electronics"]


def test_calculate_candidate_overlap():
    """Verify candidate overlap helper computes union, intersection, and Jaccard."""
    bm25_cands = [
        CandidateResult(doc_id="doc_1", score=1.0, rank=1),
        CandidateResult(doc_id="doc_2", score=1.0, rank=2),
    ]
    dense_cands = [
        CandidateResult(doc_id="doc_2", score=1.0, rank=1),
        CandidateResult(doc_id="doc_3", score=1.0, rank=2),
    ]

    overlap = calculate_candidate_overlap({"bm25": bm25_cands, "dense": dense_cands})
    assert overlap["union_count"] == 3
    assert overlap["intersection_count"] == 1
    assert pytest.approx(overlap["jaccard_similarity"]) == 1.0 / 3.0
    assert overlap["per_retriever_counts"]["bm25"] == 2
    assert overlap["per_retriever_counts"]["dense"] == 2
