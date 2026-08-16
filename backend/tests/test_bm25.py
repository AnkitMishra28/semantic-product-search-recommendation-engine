"""Unit tests for BM25 lexical retriever and technical tokenization."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.tokenizer import tokenize_lexical
from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class TestLexicalTokenizer:
    """Test suite for domain-aware technical tokenization."""

    def test_tokenize_lowercase_and_unicode_normalization(self):
        text = "SONY WH-1000XM5 Wireless\u00a0Headphones"
        tokens = tokenize_lexical(text)
        assert "sony" in tokens
        assert "wireless" in tokens
        assert "headphones" in tokens
        assert "wh-1000xm5" in tokens

    def test_tokenize_preserves_technical_identifiers(self):
        sample = "ASUS ROG Laptop with NVIDIA RTX 4060, 16GB DDR5, 4K HDMI 2.1, and M.2 NVMe SSD for PS5"
        tokens = tokenize_lexical(sample)
        
        expected_terms = ["rtx", "4060", "ddr5", "4k", "hdmi", "2.1", "m.2", "nvme", "ssd", "ps5"]
        for term in expected_terms:
            assert term in tokens, f"Expected '{term}' in tokens: {tokens}"

    def test_tokenize_usb_c_and_wifi_compounds(self):
        sample = "Anker USB-C Multiport Hub with Wi-Fi 6 Support"
        tokens = tokenize_lexical(sample)
        
        assert "usb-c" in tokens or "usbc" in tokens
        assert "usb" in tokens
        assert "wi-fi" in tokens or "wifi" in tokens
        assert "6" in tokens

    def test_tokenize_empty_and_special_strings(self):
        assert tokenize_lexical("") == []
        assert tokenize_lexical(None) == []
        assert tokenize_lexical("   \n\t  ") == []
        assert tokenize_lexical("!@#$%^&*()") == []


class TestBM25RetrieverLifecycle:
    """Test suite for BM25 indexing, scoring, saving, and loading."""

    @pytest.fixture
    def sample_products_df(self):
        return pd.DataFrame([
            {
                "parent_asin": "B001",
                "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
                "brand": "Sony",
                "categories": ["Electronics", "Headphones"],
                "features": ["30-hour battery", "Industry leading ANC"],
                "description": "Premium travel headphones with crystal clear audio.",
                "price": 399.99,
                "average_rating": 4.7,
            },
            {
                "parent_asin": "B002",
                "title": "Bose QuietComfort 45 Bluetooth Headphones",
                "brand": "Bose",
                "categories": ["Electronics", "Headphones"],
                "features": ["Quiet and Aware Modes", "All-day comfort"],
                "description": "Iconic noise cancelling travel headphones.",
                "price": 279.00,
                "average_rating": 4.6,
            },
            {
                "parent_asin": "B003",
                "title": "ASUS TUF Gaming Laptop RTX 4060 16GB DDR5",
                "brand": "ASUS",
                "categories": ["Electronics", "Computers", "Laptops"],
                "features": ["NVIDIA RTX 4060", "144Hz FHD Display"],
                "description": "High performance gaming laptop for esports.",
                "price": 999.99,
                "average_rating": 4.5,
            },
            {
                "parent_asin": "B004",
                "title": "Anker USB C Hub 7-in-1 Multiport Adapter for MacBook Pro",
                "brand": "Anker",
                "categories": ["Electronics", "Hubs & Adapters"],
                "features": ["4K HDMI", "100W Power Delivery", "SD Card Reader"],
                "description": "Portable multiport expansion hub for laptops.",
                "price": 34.99,
                "average_rating": 4.8,
            },
        ])

    def test_index_and_search_exact_matches(self, sample_products_df):
        retriever = BM25Retriever(k1=1.5, b=0.75)
        build_time = retriever.index_corpus(sample_products_df)
        assert build_time >= 0.0
        assert retriever.total_documents == 4

        results = retriever.search_text("noise cancelling headphones", top_k=2)
        assert len(results) == 2
        retrieved_ids = [r.doc_id for r in results]
        assert "B001" in retrieved_ids
        assert "B002" in retrieved_ids
        assert results[0].score > 0.0
        assert results[0].rank == 1

    def test_search_technical_term_matches(self, sample_products_df):
        retriever = BM25Retriever()
        retriever.index_corpus(sample_products_df)

        results = retriever.search_text("RTX 4060 gaming laptop", top_k=5)
        assert len(results) >= 1
        assert results[0].doc_id == "B003"
        assert results[0].metadata["brand"] == "ASUS"

    def test_empty_and_oov_queries(self, sample_products_df):
        retriever = BM25Retriever()
        retriever.index_corpus(sample_products_df)

        # Empty query
        assert retriever.search_text("") == []
        assert retriever.search_text("   ") == []

        # Out of vocabulary query
        assert retriever.search_text("xylophone quantum telescope") == []

    def test_metadata_filtering(self, sample_products_df):
        retriever = BM25Retriever()
        retriever.index_corpus(sample_products_df)

        # Brand filter
        results = retriever.search_text(
            "noise cancelling headphones",
            top_k=5,
            filters={"brand": "Bose"},
        )
        assert len(results) == 1
        assert results[0].doc_id == "B002"

        # Price filter
        cheap_results = retriever.search_text(
            "adapter hub",
            top_k=5,
            filters={"max_price": 50.0},
        )
        assert len(cheap_results) == 1
        assert cheap_results[0].doc_id == "B004"

    def test_save_and_load_roundtrip(self, sample_products_df):
        retriever = BM25Retriever(k1=1.6, b=0.8)
        retriever.index_corpus(sample_products_df)
        orig_results = retriever.search_text("USB C hub MacBook", top_k=2)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            retriever.save(tmp_path)
            
            loaded_retriever = BM25Retriever()
            loaded_retriever.load(tmp_path)
            
            assert loaded_retriever.total_documents == 4
            assert loaded_retriever.k1 == 1.6
            assert loaded_retriever.b == 0.8
            
            loaded_results = loaded_retriever.search_text("USB C hub MacBook", top_k=2)
            assert len(loaded_results) == len(orig_results)
            assert [r.doc_id for r in loaded_results] == [r.doc_id for r in orig_results]
            assert np.isclose(loaded_results[0].score, orig_results[0].score)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestBM25EvaluationMetricsIntegration:
    """Test suite verifying metric calculations on BM25 outputs."""

    def test_metric_computation_on_retrieval_output(self):
        retrieved_ids = ["B001", "B002", "B003", "B004", "B005"]
        relevant_ids = ["B002", "B004"]
        graded_rel = {"B002": 1.0, "B004": 1.0}

        r5 = recall_at_k(retrieved_ids, relevant_ids, 5)
        r2 = recall_at_k(retrieved_ids, relevant_ids, 2)
        mrr = reciprocal_rank_at_k(retrieved_ids, relevant_ids, 5)
        ndcg = ndcg_at_k(retrieved_ids, graded_rel, 5)

        assert r5 == 1.0
        assert r2 == 0.5
        assert mrr == 0.5  # First relevant item at rank 2 -> 1/2
        assert ndcg > 0.0
