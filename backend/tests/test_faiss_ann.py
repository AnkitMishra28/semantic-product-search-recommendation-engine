"""Unit and Integration Tests for FAISS ANN Retrieval, Index Types, and Fidelity."""

import json
import os
import tempfile
import numpy as np
import pytest

from backend.app.retrieval.base import BaseRetriever, CandidateResult
from backend.app.retrieval.embeddings import EXPECTED_EMBEDDING_DIM, EmbeddingService
from backend.app.retrieval.faiss_retriever import FAISS_AVAILABLE, FaissRetriever
from scripts.run_faiss_benchmark import compute_ann_recall_at_k


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS library is not installed.")
class TestFaissRetriever:
    """Test suite for FaissRetriever indexing, search algorithms, and persistence."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic normalized vector embeddings and document metadata."""
        dim = 64
        num_docs = 200
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((num_docs, dim)).astype(np.float32)
        # L2-normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        doc_ids = [f"PROD_{i:04d}" for i in range(num_docs)]
        metadata = [
            {
                "brand": "Sony" if i % 2 == 0 else "Bose",
                "price": float(10.0 + (i % 50) * 5.0),
                "category": "Headphones" if i % 3 == 0 else "Audio",
            }
            for i in range(num_docs)
        ]
        return vectors, doc_ids, metadata, dim

    def test_flat_ip_correctness_and_normalization(self, synthetic_data):
        """Verify IndexFlatIP computes exact inner product cosine similarities."""
        vectors, doc_ids, metadata, dim = synthetic_data
        retriever = FaissRetriever(dimension=dim, index_type="FlatIP", metric="inner_product")
        retriever.index(vectors, doc_ids, metadata)

        assert retriever.total_documents == len(doc_ids)
        assert retriever.is_trained

        # Query with exact first vector
        query_vec = vectors[0].copy()
        results = retriever.search(query_vec, top_k=5)

        assert len(results) == 5
        assert results[0].doc_id == doc_ids[0]
        assert results[0].rank == 1
        assert np.isclose(results[0].score, 1.0, atol=1e-4)

    def test_hnsw_configuration_and_search(self, synthetic_data):
        """Verify IndexHNSWFlat initialization, dynamic efSearch, and top-K search."""
        vectors, doc_ids, metadata, dim = synthetic_data
        retriever = FaissRetriever(
            dimension=dim,
            index_type="HNSW",
            m=16,
            ef_construction=100,
            ef_search=32,
            metric="inner_product",
        )
        build_time = retriever.index(vectors, doc_ids, metadata)
        assert build_time >= 0.0
        assert retriever.total_documents == len(doc_ids)
        assert retriever.is_trained

        # Test dynamic parameter update
        retriever.set_ef_search(64)
        assert retriever.ef_search == 64

        # Search
        query_vec = vectors[5]
        results = retriever.search(query_vec, top_k=10)
        assert len(results) == 10
        assert results[0].doc_id == doc_ids[5]
        assert np.isclose(results[0].score, 1.0, atol=1e-3)

    def test_ivfflat_training_and_search(self, synthetic_data):
        """Verify IndexIVFFlat requires training, executes nprobe routing, and retrieves neighbors."""
        vectors, doc_ids, metadata, dim = synthetic_data
        retriever = FaissRetriever(
            dimension=dim,
            index_type="IVFFlat",
            nlist=16,
            nprobe=4,
            metric="inner_product",
        )
        assert not retriever.is_trained

        # Indexing automatically trains
        train_time = retriever.train(vectors)
        assert train_time >= 0.0
        assert retriever.is_trained

        add_time = retriever.index(vectors, doc_ids, metadata)
        assert add_time >= 0.0
        assert retriever.total_documents == len(doc_ids)

        # Dynamic nprobe update
        retriever.set_nprobe(8)
        assert retriever.nprobe == 8

        query_vec = vectors[10]
        results = retriever.search(query_vec, top_k=5)
        assert len(results) == 5
        assert results[0].doc_id == doc_ids[10]

    def test_save_and_load_roundtrip_equivalence(self, synthetic_data):
        """Verify index serialization and deserialization produces bitwise identical rankings."""
        vectors, doc_ids, metadata, dim = synthetic_data
        retriever = FaissRetriever(
            dimension=dim,
            index_type="HNSW",
            m=16,
            ef_construction=100,
            ef_search=32,
        )
        retriever.index(vectors, doc_ids, metadata)

        query_vec = vectors[20]
        original_results = retriever.search(query_vec, top_k=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_hnsw.index")
            retriever.save(save_path)

            loaded_retriever = FaissRetriever(dimension=dim, index_type="HNSW")
            loaded_retriever.load(save_path)

            assert loaded_retriever.total_documents == retriever.total_documents
            assert loaded_retriever.dimension == dim
            assert loaded_retriever.ef_search == 32

            loaded_results = loaded_retriever.search(query_vec, top_k=10)
            assert len(loaded_results) == len(original_results)
            for orig, loaded in zip(original_results, loaded_results):
                assert orig.doc_id == loaded.doc_id
                assert np.isclose(orig.score, loaded.score, atol=1e-4)

    def test_metadata_filtering(self, synthetic_data):
        """Verify post-search metadata filtering for brand and price constraints."""
        vectors, doc_ids, metadata, dim = synthetic_data
        retriever = FaissRetriever(dimension=dim, index_type="FlatIP")
        retriever.index(vectors, doc_ids, metadata)

        query_vec = vectors[0]
        results = retriever.search(query_vec, top_k=10, filters={"brand": "Sony", "max_price": 50.0})

        for r in results:
            assert r.metadata["brand"] == "Sony"
            assert r.metadata["price"] <= 50.0

    def test_top_k_bounds(self, synthetic_data):
        """Verify top_k bounds when requested top_k exceeds total index count."""
        vectors, doc_ids, metadata, dim = synthetic_data
        small_vecs = vectors[:5]
        small_ids = doc_ids[:5]

        retriever = FaissRetriever(dimension=dim, index_type="FlatIP")
        retriever.index(small_vecs, small_ids)

        query_vec = small_vecs[0]
        results = retriever.search(query_vec, top_k=50)
        assert len(results) == 5

    def test_deterministic_query_behavior(self, synthetic_data):
        """Verify query search execution is deterministic across repeated calls."""
        vectors, doc_ids, metadata, dim = synthetic_data
        retriever = FaissRetriever(dimension=dim, index_type="HNSW", m=16, ef_construction=100, ef_search=32)
        retriever.index(vectors, doc_ids, metadata)

        query_vec = vectors[15]
        run1 = [r.doc_id for r in retriever.search(query_vec, top_k=10)]
        run2 = [r.doc_id for r in retriever.search(query_vec, top_k=10)]
        assert run1 == run2

    def test_ann_recall_calculation_utility(self):
        """Verify ANN Recall@K mathematical computation formula."""
        exact_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        approx_perfect = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        approx_partial = ["A", "B", "C", "D", "X", "Y", "G", "H", "I", "J"]  # 8 out of 10
        approx_disjoint = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

        assert compute_ann_recall_at_k(approx_perfect, exact_ids, 10) == 1.0
        assert compute_ann_recall_at_k(approx_partial, exact_ids, 10) == 0.8
        assert compute_ann_recall_at_k(approx_disjoint, exact_ids, 10) == 0.0
        assert compute_ann_recall_at_k(approx_partial, exact_ids, 4) == 1.0
