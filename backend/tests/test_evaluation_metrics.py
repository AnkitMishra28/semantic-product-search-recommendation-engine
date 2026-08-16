"""Test deterministic IR evaluation metrics and latency tracker."""

import pytest
from evaluation.metrics import (
    LatencyTracker,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_at_k() -> None:
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc2", "doc4"]

    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=4) == 1.0


def test_precision_at_k() -> None:
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc1", "doc3"]

    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_reciprocal_rank_at_k() -> None:
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc2"]

    assert reciprocal_rank_at_k(retrieved, relevant, k=1) == 0.0
    assert reciprocal_rank_at_k(retrieved, relevant, k=2) == 0.5
    assert reciprocal_rank_at_k(retrieved, ["doc1"], k=3) == 1.0


def test_ndcg_at_k() -> None:
    retrieved = ["doc1", "doc2", "doc3"]
    graded = {"doc1": 3.0, "doc2": 2.0, "doc3": 0.0}

    # Perfect ranking order for top 2
    assert ndcg_at_k(retrieved, graded, k=2) == 1.0

    # Suboptimal ranking order
    suboptimal = ["doc2", "doc1", "doc3"]
    score = ndcg_at_k(suboptimal, graded, k=2)
    assert 0.0 < score < 1.0


def test_latency_tracker() -> None:
    tracker = LatencyTracker()
    for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
        tracker.record(latency)

    summary = tracker.summary()
    assert summary["total_queries"] == 5
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 50.0
    assert summary["p50_ms"] == 30.0
