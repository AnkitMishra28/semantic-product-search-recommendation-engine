"""Unit and integration tests for personalized recommendation algorithms, diversity rerankers, and service."""

import numpy as np
import pandas as pd
import pytest

from backend.app.models.product import Product
from backend.app.models.recommendation import RecommendRequest
from backend.app.recommendation.base import BaseRecommender, RecommendationCandidate
from backend.app.recommendation.collaborative import CollaborativeRecommender
from backend.app.recommendation.content_based import ContentBasedRecommender
from backend.app.recommendation.diversity import MMRReranker
from backend.app.recommendation.hybrid import HybridRecommender
from backend.app.recommendation.popularity import PopularityRecommender
from backend.app.recommendation.service import RecommendationService


@pytest.fixture
def sample_recommendation_data(sample_products):
    """Fixture providing sample products, synthetic interactions, and mock embeddings."""
    catalog = {
        p.asin: {
            "asin": p.asin,
            "parent_asin": p.parent_asin or p.asin,
            "title": p.title,
            "brand": p.brand,
            "price": p.price,
            "rating": p.average_rating or 4.0,
            "average_rating": p.average_rating or 4.0,
            "rating_number": p.rating_number,
            "categories": p.categories,
            "features": p.features,
        }
        for p in sample_products
    }
    asins = list(catalog.keys())

    # Generate synthetic interactions
    interactions = [
        # User 1 interacted with item 0 and 1
        {"user_id": "u1", "parent_asin": asins[0], "rating": 5.0, "timestamp": 1000, "split": "train"},
        {"user_id": "u1", "parent_asin": asins[1], "rating": 4.0, "timestamp": 2000, "split": "train"},
        # User 2 interacted with item 1 and 2
        {"user_id": "u2", "parent_asin": asins[1], "rating": 5.0, "timestamp": 3000, "split": "train"},
        {"user_id": "u2", "parent_asin": asins[2], "rating": 5.0, "timestamp": 4000, "split": "train"},
        # User 3 interacted with item 0 and 2
        {"user_id": "u3", "parent_asin": asins[0], "rating": 4.0, "timestamp": 5000, "split": "train"},
        {"user_id": "u3", "parent_asin": asins[2], "rating": 4.0, "timestamp": 6000, "split": "train"},
    ]
    interactions_df = pd.DataFrame(interactions)

    # Deterministic mock embeddings (384-dim)
    np.random.seed(42)
    embeddings = np.random.randn(len(asins), 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return catalog, interactions_df, embeddings, asins


def test_popularity_recommender_basic_and_filters(sample_recommendation_data):
    """Verify PopularityRecommender computes Bayesian scores, filters by brand/price, and bounds top_k."""
    catalog, interactions_df, _, asins = sample_recommendation_data

    pop_rec = PopularityRecommender(m_prior=2.0, product_catalog=catalog).fit(interactions_df)

    # 1. Basic recommendation
    recs = pop_rec.recommend(user_id="u_new", top_k=3, exclude_consumed=False)
    assert len(recs) == 3
    assert recs[0].score >= recs[1].score >= recs[2].score
    assert recs[0].rank == 1
    assert "popularity" in recs[0].signals

    # 2. Exclusion of consumed items
    consumed = [asins[0]]
    recs_ex = pop_rec.recommend(history_asins=consumed, top_k=3, exclude_consumed=True)
    rec_ids = [r.product_id for r in recs_ex]
    assert asins[0] not in rec_ids

    # 3. Hard filter: Brand
    sony_recs = pop_rec.recommend(top_k=5, filters={"brand": "Sony"})
    for r in sony_recs:
        assert r.metadata.get("brand") == "Sony"

    # 4. Hard filter: Price max
    cheap_recs = pop_rec.recommend(top_k=5, filters={"price_max": 200.0})
    for r in cheap_recs:
        assert r.metadata.get("price") <= 200.0


def test_content_based_recommender(sample_recommendation_data):
    """Verify ContentBasedRecommender builds user preference embeddings and excludes consumed items."""
    catalog, _, embeddings, asins = sample_recommendation_data

    content_rec = ContentBasedRecommender(
        embeddings=embeddings,
        doc_ids=asins,
        product_catalog=catalog,
    )

    # Known history
    history = [asins[0], asins[1]]
    recs = content_rec.recommend(history_asins=history, top_k=3, exclude_consumed=True)

    assert len(recs) <= 3
    for r in recs:
        assert r.product_id not in history
        assert 0.0 <= r.score <= 1.0
        assert "content" in r.signals

    # Empty history returns empty list (cold start signal)
    assert content_rec.recommend(history_asins=[], top_k=5) == []


def test_collaborative_recommender(sample_recommendation_data):
    """Verify CollaborativeRecommender creates sparse co-occurrence graph and ranks neighbor items."""
    catalog, interactions_df, _, asins = sample_recommendation_data

    collab_rec = CollaborativeRecommender(
        min_support=1,
        max_neighbors_per_item=50,
        product_catalog=catalog,
    ).fit(interactions_df)

    # User who interacted with item 0 -> should co-recommend item 1 and item 2
    recs = collab_rec.recommend(history_asins=[asins[0]], top_k=2, exclude_consumed=True)

    assert len(recs) > 0
    rec_ids = [r.product_id for r in recs]
    assert asins[0] not in rec_ids
    assert asins[1] in rec_ids or asins[2] in rec_ids
    assert recs[0].recommendation_type == "collaborative"


def test_hybrid_recommender_and_cold_start(sample_recommendation_data):
    """Verify HybridRecommender multi-signal fusion, cold start routing, and score normalization."""
    catalog, interactions_df, embeddings, asins = sample_recommendation_data

    pop_rec = PopularityRecommender(product_catalog=catalog).fit(interactions_df)
    content_rec = ContentBasedRecommender(embeddings=embeddings, doc_ids=asins, product_catalog=catalog)
    collab_rec = CollaborativeRecommender(product_catalog=catalog).fit(interactions_df)
    mmr_reranker = MMRReranker(embeddings=embeddings, doc_ids=asins, default_lambda=0.7)

    hybrid_rec = HybridRecommender(
        popularity_recommender=pop_rec,
        content_recommender=content_rec,
        collaborative_recommender=collab_rec,
        diversity_reranker=mmr_reranker,
        content_weight=0.4,
        collaborative_weight=0.3,
        popularity_weight=0.15,
        rating_weight=0.15,
        product_catalog=catalog,
    )

    # 1. Warm user recommendation
    history = [asins[0]]
    warm_recs = hybrid_rec.recommend(history_asins=history, top_k=3, exclude_consumed=True)
    # Total catalog is 3 items, 1 consumed -> exactly 2 recommended
    assert len(warm_recs) == 2
    assert warm_recs[0].product_id not in history
    assert "content" in warm_recs[0].signals
    assert "collaborative" in warm_recs[0].signals
    assert "popularity" in warm_recs[0].signals
    assert "rating" in warm_recs[0].signals
    assert len(warm_recs[0].reasons) > 0

    # 2. Cold-start user recommendation (empty history -> 3 recommended)
    cold_recs = hybrid_rec.recommend(history_asins=[], top_k=3)
    assert len(cold_recs) == 3
    assert cold_recs[0].recommendation_type == "cold_start_popularity"


def test_mmr_diversity_reranking(sample_recommendation_data):
    """Verify MMR diversity reranker reduces redundancy and respects lambda."""
    catalog, _, embeddings, asins = sample_recommendation_data

    mmr = MMRReranker(embeddings=embeddings, doc_ids=asins, default_lambda=0.5)

    candidates = [
        RecommendationCandidate(product_id=asins[0], score=0.95, rank=1, metadata=catalog[asins[0]]),
        RecommendationCandidate(product_id=asins[1], score=0.90, rank=2, metadata=catalog[asins[1]]),
        RecommendationCandidate(product_id=asins[2], score=0.85, rank=3, metadata=catalog[asins[2]]),
    ]

    reranked = mmr.rerank(candidates, top_k=3, lambda_param=0.5)
    assert len(reranked) == 3
    assert reranked[0].rank == 1
    assert "mmr_score" in reranked[0].signals
    assert "max_intra_similarity" in reranked[1].signals


def test_recommendation_service_end_to_end(sample_recommendation_data, sample_products):
    """Verify RecommendationService handles API requests, user history lookup, and response schemas."""
    catalog_dict, interactions_df, embeddings, asins = sample_recommendation_data
    products_map = {p.asin: p for p in sample_products}

    pop_rec = PopularityRecommender(product_catalog=catalog_dict).fit(interactions_df)
    content_rec = ContentBasedRecommender(embeddings=embeddings, doc_ids=asins, product_catalog=catalog_dict)
    collab_rec = CollaborativeRecommender(product_catalog=catalog_dict).fit(interactions_df)
    hybrid_rec = HybridRecommender(
        popularity_recommender=pop_rec,
        content_recommender=content_rec,
        collaborative_recommender=collab_rec,
        product_catalog=catalog_dict,
    )

    service = RecommendationService(
        popularity_recommender=pop_rec,
        content_recommender=content_rec,
        collaborative_recommender=collab_rec,
        hybrid_recommender=hybrid_rec,
        product_catalog=products_map,
        interactions_df=interactions_df,
    )

    # 1. Request for known user "u1" (u1 consumed item 0 and 1; 1 item remaining in 3-item catalog)
    req_user = RecommendRequest(user_id="u1", top_k=2, strategy="hybrid")
    resp_user = service.recommend(req_user)

    assert resp_user.total_returned == 1
    assert resp_user.execution_time_ms >= 0.0
    assert len(resp_user.recommendations) == 1
    assert resp_user.recommendations[0].product.asin == asins[2]
    # Check that consumed items of u1 are excluded
    u1_history = service.get_user_history("u1")
    for r in resp_user.recommendations:
        assert r.product.asin not in u1_history

    # 2. Item-to-item request
    req_item = RecommendRequest(asin=asins[0], top_k=2, strategy="hybrid")
    resp_item = service.recommend(req_item)
    assert resp_item.total_returned == 2
    assert resp_item.recommendations[0].product.asin != asins[0]
