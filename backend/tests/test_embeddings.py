"""Unit tests for SentenceTransformer Embedding Service and Exact Dense Retriever."""

import json
import os
import tempfile
import numpy as np
import pytest

from backend.app.preprocessing.product_document import (
    TextRepresentationVariant,
    build_product_text,
)
from backend.app.retrieval.embeddings import (
    EXPECTED_EMBEDDING_DIM,
    EmbeddingService,
    ExactDenseRetriever,
)


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedding service instance for fast test execution."""
    return EmbeddingService(device="cpu", normalize_embeddings=True)


class TestEmbeddingService:
    """Test suite for SentenceTransformer embedding service."""

    def test_model_initialization_and_dimension(self, embedder):
        assert embedder.embedding_dimension == EXPECTED_EMBEDDING_DIM
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.device == "cpu"

    def test_encode_single_and_batch_queries(self, embedder):
        single_vec = embedder.encode_queries("noise cancelling headphones")
        assert single_vec.shape == (EXPECTED_EMBEDDING_DIM,)
        assert single_vec.dtype == np.float32
        
        # Verify L2 normalization
        norm = np.linalg.norm(single_vec)
        assert np.isclose(norm, 1.0, atol=1e-3)

        batch_vecs = embedder.encode_queries(["mechanical keyboard", "gaming laptop"])
        assert batch_vecs.shape == (2, EXPECTED_EMBEDDING_DIM)
        assert batch_vecs.dtype == np.float32
        assert np.allclose(np.linalg.norm(batch_vecs, axis=1), 1.0, atol=1e-3)

    def test_encode_documents_batch(self, embedder):
        docs = [
            "Sony WH-1000XM5 Wireless Headphones",
            "ASUS TUF Gaming Laptop RTX 4060",
            "Anker USB-C Multiport Hub Adapter",
        ]
        vecs = embedder.encode_documents(docs, batch_size=2, show_progress_bar=False)
        assert vecs.shape == (3, EXPECTED_EMBEDDING_DIM)
        assert vecs.dtype == np.float32
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)

    def test_encode_empty_queries_and_docs(self, embedder):
        empty_vec = embedder.encode_queries([])
        assert empty_vec.shape == (0, EXPECTED_EMBEDDING_DIM)

        empty_docs = embedder.encode_documents([], batch_size=10, show_progress_bar=False)
        assert empty_docs.shape == (0, EXPECTED_EMBEDDING_DIM)


class TestExactDenseRetriever:
    """Test suite for ExactDenseRetriever cosine similarity search."""

    @pytest.fixture
    def mock_retriever(self):
        # Create synthetic normalized embeddings for 4 products
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        doc_ids = ["P001", "P002", "P003", "P004"]
        metadata = [
            {"title": "Product 1", "brand": "Sony", "price": 100.0},
            {"title": "Product 2", "brand": "Bose", "price": 200.0},
            {"title": "Product 3", "brand": "Sony", "price": 150.0},
            {"title": "Product 4", "brand": "Apple", "price": 300.0},
        ]
        
        retriever = ExactDenseRetriever()
        retriever.set_corpus(embeddings=vecs, doc_ids=doc_ids, metadata=metadata)
        return retriever

    def test_exact_cosine_similarity_ordering(self, mock_retriever):
        # Query aligned with [1, 0, 0, 0]
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = mock_retriever.search(q, top_k=3)
        
        assert len(results) == 3
        # P001 has dot product 1.0, P003 has 0.7071, others 0.0
        assert results[0].doc_id == "P001"
        assert np.isclose(results[0].score, 1.0, atol=1e-4)
        assert results[1].doc_id == "P003"
        assert np.isclose(results[1].score, 0.7071, atol=1e-3)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_metadata_filtering(self, mock_retriever):
        q = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
        
        # Filter by brand
        results = mock_retriever.search(q, top_k=4, filters={"brand": "Sony"})
        assert len(results) == 2
        for r in results:
            assert r.metadata["brand"] == "Sony"

        # Filter by max price
        cheap_results = mock_retriever.search(q, top_k=4, filters={"max_price": 120.0})
        assert len(cheap_results) == 1
        assert cheap_results[0].doc_id == "P001"

    def test_save_and_load_roundtrip(self, mock_retriever):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_emb.npy")
            mock_retriever.save(save_path)

            loaded_retriever = ExactDenseRetriever()
            loaded_retriever.load(save_path)

            assert loaded_retriever.total_documents == 4
            assert loaded_retriever.doc_ids == ["P001", "P002", "P003", "P004"]
            
            q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            res = loaded_retriever.search(q, top_k=1)
            assert res[0].doc_id == "P002"
            assert np.isclose(res[0].score, 1.0)


class TestProductRepresentationAblations:
    """Test suite verifying representation variant generators with numpy structures."""

    def test_representation_variants(self):
        sample = {
            "title": "Sony WH-1000XM5",
            "brand": "Sony",
            "categories": np.array(["Electronics", "Headphones"]),
            "features": np.array(["30h battery", "Noise Cancelling"]),
            "description": "Premium wireless headphones.",
        }

        var_a = build_product_text(sample, variant="title_brand_category")
        assert "Title: Sony WH-1000XM5" in var_a
        assert "Brand: Sony" in var_a
        assert "Category: Electronics > Headphones" in var_a
        assert "Features:" not in var_a
        assert "Description:" not in var_a

        var_b = build_product_text(sample, variant="title_brand_category_features")
        assert "Features:" in var_b
        assert "- 30h battery" in var_b
        assert "Description:" not in var_b

        var_c = build_product_text(sample, variant="title_brand_category_features_description")
        assert "Features:" in var_c
        assert "Description:" in var_c
        assert "Premium wireless headphones." in var_c
