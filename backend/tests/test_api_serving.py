"""Comprehensive API serving integration tests for Phase 11 FastAPI ML serving layer."""

import pytest
from fastapi.testclient import TestClient
from backend.app.models.explanation import ProductEvidence
from backend.app.models.product import Product


def test_api_health(client: TestClient) -> None:
    """Test GET /api/health and GET /api/v1/health."""
    for path in ["/api/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app_name" in data
        assert "version" in data


def test_api_readiness(client: TestClient) -> None:
    """Test GET /api/ready and GET /api/v1/ready."""
    for path in ["/api/ready", "/api/v1/ready"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "catalog" in data["components"]
        assert "llm_explainer" in data["components"]


def test_api_search_valid_query(client: TestClient) -> None:
    """Test POST /api/search with a valid query."""
    payload = {
        "query": "wireless noise cancelling headphones",
        "top_k_retrieval": 10,
        "top_k_reranking": 5,
        "enable_reranking": False,  # Test without heavy GPU/CPU model load in mock test
        "enable_explanation": True,
    }
    for path in ["/api/search", "/api/v1/search"]:
        response = client.post(path, json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "wireless noise cancelling headphones"
        assert "query_understanding" in data
        assert "results" in data
        assert "timings" in data
        assert isinstance(data["results"], list)


def test_api_search_invalid_request(client: TestClient) -> None:
    """Test POST /api/search with invalid / missing payload."""
    response = client.post("/api/search", json={})
    assert response.status_code == 422


def test_api_recommend_by_asin(client: TestClient) -> None:
    """Test POST /api/recommend with item anchor."""
    payload = {
        "asin": "B001",
        "top_k": 5,
        "strategy": "popularity",
        "generate_explanations": True,
    }
    for path in ["/api/recommend", "/api/v1/recommend"]:
        response = client.post(path, json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["anchor_asin"] == "B001"
        assert "recommendations" in data
        assert "execution_time_ms" in data


def test_api_recommend_by_user_history(client: TestClient) -> None:
    """Test POST /api/recommend with user history."""
    payload = {
        "user_history_asins": ["B001", "B002"],
        "top_k": 3,
        "strategy": "popularity",
        "generate_explanations": True,
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data


def test_api_explain_search_match(client: TestClient) -> None:
    """Test POST /api/explain for a search query and catalog product."""
    payload = {
        "query": "Sony headphones",
        "product_id": "B001",
    }
    for path in ["/api/explain", "/api/v1/explain"]:
        response = client.post(path, json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "B001"
        assert data["grounded"] is True
        assert "summary" in data
        assert "reasons" in data
        assert len(data["reasons"]) > 0


def test_api_explain_with_preassembled_evidence(client: TestClient) -> None:
    """Test POST /api/explain directly supplying a structured ProductEvidence object."""
    evidence = ProductEvidence(
        product_id="B001",
        product_title="Sony WH-1000XM4 Wireless Noise Cancelling Headphones",
        brand="Sony",
        price=348.00,
        rating=4.7,
        rating_count=45000,
        matched_constraints={"brand": "Manufactured by Sony"},
    )
    payload = {
        "evidence": evidence.model_dump(),
    }
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "B001"
    assert data["grounded"] is True
    assert any(r["label"] == "Brand" for r in data["reasons"])


def test_api_explain_nonexistent_product(client: TestClient) -> None:
    """Test POST /api/explain for non-existent ASIN."""
    payload = {
        "query": "Sony headphones",
        "product_id": "NON_EXISTENT_ASIN_999",
    }
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 404


def test_api_explain_invalid_empty_input(client: TestClient) -> None:
    """Test POST /api/explain when neither query/product nor evidence is given."""
    payload = {}
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 400


def test_api_get_product_success(client: TestClient) -> None:
    """Test GET /api/products/{id} for existing product."""
    for path in ["/api/products/B001", "/api/v1/products/B001"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["asin"] == "B001"
        assert "Sony" in data["title"]
        assert data["price"] == 348.00


def test_api_get_product_not_found(client: TestClient) -> None:
    """Test GET /api/products/{id} for non-existent product."""
    for path in ["/api/products/NONEXISTENT_999", "/api/v1/products/NONEXISTENT_999"]:
        response = client.get(path)
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data


def test_api_get_metrics(client: TestClient) -> None:
    """Test GET /api/metrics exposes authoritative offline benchmark metrics."""
    for path in ["/api/metrics", "/api/v1/metrics"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "authoritative"
        assert "scientific_notes" in data
        assert "stage_separation" in data["scientific_notes"]
        assert "hybrid_retrieval" in data["scientific_notes"]
