"""Test health and readiness endpoints — asserts match actual health.py response shapes."""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient) -> None:
    """GET /api/v1/health returns status=healthy plus app metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data


def test_readiness_check(client: TestClient) -> None:
    """GET /api/v1/ready returns status and component breakdown."""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ready", "initializing"]
    assert "components" in data
    components = data["components"]
    assert "catalog" in components
    assert "vector_index" in components
    assert "cross_encoder" in components
    assert "llm_explainer" in components
