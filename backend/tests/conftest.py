"""Pytest fixtures and configuration."""

import pytest
from fastapi.testclient import TestClient
from backend.app.core.config import Settings
from backend.app.main import create_application
from backend.app.models.product import Product
from backend.app.services.model_registry import ModelRegistry


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        environment="test",
        debug=True,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )


@pytest.fixture(scope="session")
def sample_products() -> list[Product]:
    return [
        Product(
            asin="B001",
            title="Sony WH-1000XM4 Wireless Noise Cancelling Headphones",
            description="Industry-leading noise canceling with Dual Noise Sensor technology.",
            features=["Active Noise Cancelling", "30-hour battery life", "Touch sensor controls"],
            price=348.00,
            brand="Sony",
            categories=["Electronics", "Headphones", "Over-Ear"],
            average_rating=4.7,
            rating_count=45000,
            bought_together=["B002"],
        ),
        Product(
            asin="B002",
            title="Headphone Stand Aluminum Desk Headset Holder",
            description="Sturdy aluminum desktop headphone stand.",
            features=["Aluminum alloy", "Anti-slip base"],
            price=19.99,
            brand="AudioTech",
            categories=["Electronics", "Accessories", "Headphone Stands"],
            average_rating=4.5,
            rating_count=5200,
            bought_together=["B001"],
        ),
        Product(
            asin="B003",
            title="Anker USB C Charger 65W Fast Charger",
            description="Compact GaN fast wall charger for laptops and phones.",
            features=["GaN II Technology", "65W High Speed", "Foldable Plug"],
            price=29.99,
            brand="Anker",
            categories=["Electronics", "Accessories", "Chargers"],
            average_rating=4.8,
            rating_count=32000,
            bought_together=[],
        ),
    ]


@pytest.fixture
def client(sample_products) -> TestClient:
    """Provide a TestClient with a freshly seeded in-memory catalog for each test.

    Resets the ModelRegistry singleton before each test to prevent catalog state
    from one test leaking into another.
    """
    # Reset singleton so each test starts with a clean registry
    ModelRegistry._instance = None

    app = create_application()
    registry = ModelRegistry.get_instance()
    for p in sample_products:
        registry.catalog[p.asin] = p
    with TestClient(app) as test_client:
        yield test_client

    # Post-test cleanup: reset singleton again so subsequent tests are isolated
    ModelRegistry._instance = None
