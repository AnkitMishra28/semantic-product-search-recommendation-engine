"""Test retrieval interface contract and FAISS retriever behavior."""

import numpy as np
import pytest
from backend.app.retrieval.base import BaseRetriever, CandidateResult
from backend.app.retrieval.faiss_retriever import FAISS_AVAILABLE, FaissRetriever


def test_base_retriever_subclass_contract() -> None:
    """Ensure custom retriever adheres to BaseRetriever abstract contract."""

    class MockRetriever(BaseRetriever):
        def search(self, query_vector, top_k=100, filters=None):
            return [CandidateResult(doc_id="DOC1", score=0.9, rank=1)]

        def index(self, vectors, doc_ids, metadata=None):
            pass

        def save(self, file_path):
            pass

        def load(self, file_path):
            pass

        @property
        def total_documents(self):
            return 1

    retriever = MockRetriever()
    assert isinstance(retriever, BaseRetriever)
    res = retriever.search(np.zeros(384))
    assert len(res) == 1
    assert res[0].doc_id == "DOC1"


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed in test environment")
def test_faiss_retriever_indexing_and_search() -> None:
    """Test FAISS indexing, vector normalization, and search top-k retrieval."""
    dim = 64
    retriever = FaissRetriever(dimension=dim, index_type="FlatIP")

    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10, dim)).astype(np.float32)
    doc_ids = [f"ASIN_{i}" for i in range(10)]

    retriever.index(vectors, doc_ids)
    assert retriever.total_documents == 10

    query_vec = vectors[0]  # Exact match query
    results = retriever.search(query_vec, top_k=3)

    assert len(results) == 3
    assert results[0].doc_id == "ASIN_0"
    assert results[0].rank == 1
    assert results[0].score > 0.99  # Normalized cosine self-similarity is ~1.0
