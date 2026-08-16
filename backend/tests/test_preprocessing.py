"""Unit tests for Phase 1 data ingestion, cleaning, representation, and validation modules."""

import json
import os
import tempfile
import pandas as pd
import pytest

from backend.app.preprocessing.cleaners import (
    clean_brand,
    clean_categories,
    clean_description,
    clean_features,
    clean_text,
    extract_images,
    parse_price,
)
from backend.app.preprocessing.eval_queries import (
    build_evaluation_queries,
    find_matching_products,
)
from backend.app.preprocessing.interaction_processor import (
    clean_interaction_record,
    process_interactions,
)
from backend.app.preprocessing.product_document import (
    build_product_text,
    TextRepresentationVariant,
)
from backend.app.preprocessing.profiler import profile_dataset
from backend.app.preprocessing.sampler import (
    clean_raw_product_record,
    compute_product_quality_score,
    sample_and_deduplicate_products,
)
from backend.app.preprocessing.validator import (
    ValidationError,
    validate_evaluation_queries,
    validate_interactions,
    validate_products_catalog,
)


class TestTextAndMetadataCleaners:
    """Test suite for cleaning and normalization functions."""

    def test_clean_text_html_entities_and_whitespace(self):
        raw = "   &lt;b&gt;Sony&lt;/b&gt; WH-1000XM5 &amp; Bose 700 &nbsp; \n\n  Headphones   "
        expected = "Sony WH-1000XM5 & Bose 700 Headphones"
        assert clean_text(raw) == expected

    def test_clean_text_unicode_normalization(self):
        raw = "Apple MacBook Pro \u2013 16\u201d M2 Max\u00a0Edition"
        cleaned = clean_text(raw)
        assert "MacBook Pro" in cleaned
        assert "\u00a0" not in cleaned

    def test_parse_price_valid_formats(self):
        assert parse_price(19.99) == 19.99
        assert parse_price(25) == 25.0
        assert parse_price("$29.99") == 29.99
        assert parse_price("$1,299.99") == 1299.99
        assert parse_price(" $499.00 USD ") == 499.00

    def test_parse_price_invalid_formats(self):
        assert parse_price(None) is None
        assert parse_price("") is None
        assert parse_price("N/A") is None
        assert parse_price("$") is None
        assert parse_price(-10.0) is None
        assert parse_price(200000.0) is None  # Exceeds max threshold

    def test_clean_brand_extraction(self):
        assert clean_brand("Visit the Sony Store") == "Sony"
        assert clean_brand("Apple Store") == "Apple"
        assert clean_brand("Brand: Anker") == "Anker"
        assert clean_brand("Generic", {"Brand": "Logitech"}) == "Logitech"
        assert clean_brand(None, {"Manufacturer": "Samsung"}) == "Samsung"
        assert clean_brand("Unknown") is None

    def test_clean_categories(self):
        raw_list = [" Electronics ", "Audio", "Headphones", ""]
        assert clean_categories(raw_list) == ["Electronics", "Audio", "Headphones"]
        assert clean_categories([], main_category="All Electronics") == ["All Electronics"]
        assert clean_categories("Electronics > Laptops > Gaming") == ["Electronics", "Laptops", "Gaming"]

    def test_clean_features_and_description(self):
        raw_feats = ["<p>Active Noise Cancellation</p>", "  ", "30-Hour Battery Life"]
        cleaned_feats = clean_features(raw_feats)
        assert len(cleaned_feats) == 2
        assert cleaned_feats[0] == "Active Noise Cancellation"
        assert cleaned_feats[1] == "30-Hour Battery Life"

        raw_desc = ["First paragraph with <b>HTML</b>.", "Second paragraph."]
        cleaned_desc = clean_description(raw_desc)
        assert "First paragraph with HTML." in cleaned_desc
        assert "Second paragraph." in cleaned_desc

    def test_extract_images(self):
        raw_images = [
            {"thumb": "https://img.amazon.com/thumb1.jpg", "large": "https://img.amazon.com/large1.jpg"},
            {"thumb": "https://img.amazon.com/thumb2.jpg", "large": "https://img.amazon.com/large2.jpg"},
        ]
        primary, all_imgs = extract_images(raw_images)
        assert primary == "https://img.amazon.com/large1.jpg"
        assert len(all_imgs) == 2


class TestProductDocumentRepresentations:
    """Test suite for embedding document serialization and ablation variants."""

    @pytest.fixture
    def sample_product(self):
        return {
            "title": "ASUS TUF Gaming A15 Laptop",
            "brand": "ASUS",
            "categories": ["Electronics", "Computers", "Laptops"],
            "features": ["AMD Ryzen 7 7735HS", "NVIDIA RTX 4060", "16GB DDR5 RAM"],
            "description": "High performance gaming laptop designed for smooth esports and AAA gaming.",
            "price": 999.99,
            "average_rating": 4.6,
            "rating_number": 1250,
        }

    def test_representation_variant_a_title_brand_category(self, sample_product):
        text = build_product_text(sample_product, variant="title_brand_category")
        assert "Title: ASUS TUF Gaming A15 Laptop" in text
        assert "Brand: ASUS" in text
        assert "Category: Electronics > Computers > Laptops" in text
        assert "Features:" not in text
        assert "Description:" not in text
        # Numerical business signals must NEVER be present in semantic embedding text
        assert "999.99" not in text
        assert "4.6" not in text
        assert "1250" not in text

    def test_representation_variant_b_with_features(self, sample_product):
        text = build_product_text(sample_product, variant="title_brand_category_features")
        assert "Title: ASUS TUF Gaming A15 Laptop" in text
        assert "Brand: ASUS" in text
        assert "Features:" in text
        assert "- NVIDIA RTX 4060" in text
        assert "Description:" not in text

    def test_representation_variant_c_full(self, sample_product):
        text = build_product_text(sample_product, variant="title_brand_category_features_description")
        assert "Title: ASUS TUF Gaming A15 Laptop" in text
        assert "Features:" in text
        assert "Description:\nHigh performance gaming laptop" in text


class TestSamplerAndDeduplication:
    """Test suite for quality scoring, deduplication, and sampling."""

    def test_quality_score_computation(self):
        rich_product = {
            "title": "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones",
            "brand": "Sony",
            "categories": ["Electronics", "Headphones"],
            "features": ["Industry-leading ANC", "30-hr battery", "Crystal clear hands-free"],
            "description": "Premium wireless headphones engineered for incredible sound clarity and noise isolation.",
            "price": 398.00,
            "average_rating": 4.7,
            "rating_number": 8500,
        }
        score = compute_product_quality_score(rich_product)
        assert score >= 7.0

        bare_product = {
            "title": "Cable",
            "brand": None,
            "categories": [],
            "features": [],
            "description": "",
            "price": None,
            "average_rating": None,
            "rating_number": 0,
        }
        bare_score = compute_product_quality_score(bare_product)
        assert bare_score < 2.0

    def test_sample_and_deduplicate_products(self):
        raw_stream = [
            # Low quality duplicate 1
            {"parent_asin": "B001", "title": "Headphones A", "price": None, "store": None},
            # High quality duplicate 1 (should replace previous)
            {
                "parent_asin": "B001",
                "title": "Sony Noise Cancelling Over-Ear Headphones A",
                "price": 199.99,
                "store": "Sony",
                "categories": ["Electronics", "Headphones"],
                "features": ["ANC", "Bluetooth"],
                "description": "Great sound quality with long battery life.",
                "average_rating": 4.5,
                "rating_number": 200,
            },
            # Product 2
            {
                "parent_asin": "B002",
                "title": "Logitech MX Master 3S Wireless Mouse",
                "price": 99.99,
                "store": "Logitech",
                "categories": ["Electronics", "Computers", "Mice"],
                "features": ["8K DPI sensor", "Quiet clicks"],
                "description": "Ergonomic productivity mouse.",
                "average_rating": 4.8,
                "rating_number": 500,
            },
            # Invalid product (no title)
            {"parent_asin": "B003", "title": ""},
        ]

        results = sample_and_deduplicate_products(
            products_iter=iter(raw_stream),
            target_size=2,
            min_quality_score=2.0,
            seed=42,
        )

        assert len(results) == 2
        asins = {p["parent_asin"] for p in results}
        assert asins == {"B001", "B002"}
        # Verify deduplication kept the rich version of B001
        b001 = next(p for p in results if p["parent_asin"] == "B001")
        assert b001["brand"] == "Sony"
        assert b001["price"] == 199.99
        assert "embedding_text" in b001

    def test_deterministic_sampling_reproducibility(self):
        raw_items = [
            {
                "parent_asin": f"ASIN_{i:04d}",
                "title": f"Product Item {i:04d} with Rich Metadata Details",
                "price": 10.0 + (i % 50),
                "store": f"Brand_{i % 5}",
                "categories": ["Electronics", "Gadgets"],
                "features": [f"Feature {i}-A", f"Feature {i}-B"],
                "description": f"Detailed product description for item {i} ensuring sufficient length.",
                "average_rating": 4.0 + (i % 10) / 10.0,
                "rating_number": i * 10,
            }
            for i in range(100)
        ]

        run1 = sample_and_deduplicate_products(iter(raw_items), target_size=25, seed=42)
        run2 = sample_and_deduplicate_products(iter(raw_items), target_size=25, seed=42)

        asins1 = [p["parent_asin"] for p in run1]
        asins2 = [p["parent_asin"] for p in run2]
        assert asins1 == asins2


class TestInteractionProcessingAndTemporalSplitting:
    """Test suite for interaction cleaning and temporal evaluation splitting."""

    def test_interaction_cleaning_and_referential_integrity(self):
        valid_asins = {"B001", "B002"}
        
        valid_raw = {
            "parent_asin": "B001",
            "user_id": "USER_123",
            "rating": 5.0,
            "timestamp": 1600000000000,
            "verified_purchase": True,
        }
        cleaned = clean_interaction_record(valid_raw, valid_asins)
        assert cleaned is not None
        assert cleaned["user_id"] == "USER_123"

        # Missing parent asin
        missing_asin_raw = {
            "parent_asin": "B999",  # Not in valid_asins
            "user_id": "USER_123",
            "rating": 5.0,
            "timestamp": 1600000000000,
        }
        assert clean_interaction_record(missing_asin_raw, valid_asins) is None

        # Invalid rating
        bad_rating_raw = {
            "parent_asin": "B001",
            "user_id": "USER_123",
            "rating": 6.5,
            "timestamp": 1600000000000,
        }
        assert clean_interaction_record(bad_rating_raw, valid_asins) is None

    def test_temporal_quantile_split(self):
        valid_asins = {"B001"}
        # Create 100 interactions spread over timestamps 1000 to 2000
        raw_reviews = [
            {
                "parent_asin": "B001",
                "user_id": f"USER_{i:03d}",
                "rating": 4.0,
                "timestamp": 1000 + i * 10,
                "verified_purchase": True,
            }
            for i in range(100)
        ]

        df, meta = process_interactions(
            reviews_iter=iter(raw_reviews),
            valid_parent_asins=valid_asins,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        assert len(df) == 100
        counts = df["split"].value_counts().to_dict()
        assert counts["train"] == 71 or counts["train"] == 70
        assert counts["val"] == 15
        assert counts["test"] == 15 or counts["test"] == 14

        # Verify temporal monotonicity: max(train) <= min(val) <= max(val) <= min(test)
        max_train_ts = df[df["split"] == "train"]["timestamp"].max()
        min_val_ts = df[df["split"] == "val"]["timestamp"].min()
        max_val_ts = df[df["split"] == "val"]["timestamp"].max()
        min_test_ts = df[df["split"] == "test"]["timestamp"].min()

        assert max_train_ts <= min_val_ts
        assert max_val_ts <= min_test_ts


class TestValidationSuite:
    """Test suite for automated dataset validation checks."""

    def test_validate_products_catalog_passes(self):
        df = pd.DataFrame([
            {
                "parent_asin": "B001",
                "title": "Sony Wireless Headphones",
                "brand": "Sony",
                "categories": ["Electronics", "Headphones"],
                "features": ["ANC"],
                "description": "Great headphones",
                "price": 199.99,
                "average_rating": 4.5,
                "rating_number": 100,
                "embedding_text": "Title: Sony Wireless Headphones\n\nBrand: Sony",
            }
        ])
        res = validate_products_catalog(df)
        assert res["status"] == "PASSED"

    def test_validate_products_catalog_fails_on_duplicate(self):
        df = pd.DataFrame([
            {"parent_asin": "B001", "title": "Item 1", "categories": [], "features": [], "embedding_text": "Text"},
            {"parent_asin": "B001", "title": "Item 2", "categories": [], "features": [], "embedding_text": "Text"},
        ])
        with pytest.raises(ValidationError, match="Duplicate parent_asins"):
            validate_products_catalog(df)

    def test_validate_interactions_referential_integrity(self):
        catalog = {"B001", "B002"}
        valid_df = pd.DataFrame([
            {"user_id": "U1", "parent_asin": "B001", "rating": 5.0, "timestamp": 100, "split": "train"}
        ])
        assert validate_interactions(valid_df, catalog)["status"] == "PASSED"

        invalid_df = pd.DataFrame([
            {"user_id": "U1", "parent_asin": "B999", "rating": 5.0, "timestamp": 100, "split": "train"}
        ])
        with pytest.raises(ValidationError, match="Referential integrity failure"):
            validate_interactions(invalid_df, catalog)


class TestEvaluationQueriesAndProfiling:
    """Test suite for evaluation query curation and profiling."""

    def test_eval_queries_catalog_grounding(self):
        products_df = pd.DataFrame([
            {
                "parent_asin": "B_ANC_01",
                "title": "Bose QuietComfort 45 Bluetooth Noise Cancelling Headphones",
                "categories": ["Electronics", "Headphones"],
                "price": 279.00,
                "rating_number": 500,
            },
            {
                "parent_asin": "B_HUB_01",
                "title": "Anker USB C Hub Multiport Adapter for MacBook Pro",
                "categories": ["Electronics", "Hubs & Adapters"],
                "price": 25.99,
                "rating_number": 1200,
            }
        ])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            queries = build_evaluation_queries(products_df, tmp_path)
            assert len(queries) >= 2
            
            # Check validation
            val_res = validate_evaluation_queries(tmp_path, set(products_df["parent_asin"]))
            assert val_res["status"] == "PASSED"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
