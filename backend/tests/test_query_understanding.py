"""Comprehensive test suite for Query Understanding, Currency Correction, and Edge-Case Hardening."""

import pytest
from backend.app.query_understanding.intent_classifier import QueryIntentClassifier
from backend.app.query_understanding.normalizer import QueryNormalizer
from backend.app.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.query_understanding.price_extractor import PriceExtractor


@pytest.fixture
def default_pipeline() -> QueryUnderstandingPipeline:
    return QueryUnderstandingPipeline(default_currency="USD")


@pytest.fixture
def inr_pipeline() -> QueryUnderstandingPipeline:
    return QueryUnderstandingPipeline(default_currency="INR")


class TestQueryNormalizer:
    """Test string cleaning, unicode NFKC normalization, and shorthand price parsing."""

    def test_unicode_and_punctuation(self) -> None:
        normalizer = QueryNormalizer()
        raw = "  Sony WH-1000XM4 Noise-Cancelling Headphones!! "
        norm = normalizer.normalize(raw)
        assert norm == "sony wh-1000xm4 noise-cancelling headphones"

    def test_spelling_corrections(self) -> None:
        normalizer = QueryNormalizer()
        assert normalizer.normalize("cheap blutooth head phones") == "cheap bluetooth headphones"
        assert normalizer.normalize("lap top powerbank") == "laptop power bank"
        assert normalizer.normalize("tws ear buds") == "tws earbuds"

    def test_price_shorthand_expansion(self) -> None:
        normalizer = QueryNormalizer()
        assert "80000" in normalizer.normalize("laptop under 80k")
        assert "80000" in normalizer.normalize("laptop below 80 thousand")
        assert "50000" in normalizer.normalize("budget 50,000")


class TestPriceAndCurrencyExtraction:
    """Test USD canonical default, explicit INR, EUR, ranges, and symbols."""

    def test_canonical_usd_default_without_symbol(self) -> None:
        extractor = PriceExtractor(default_currency="USD")
        pmin, pmax, curr, stripped = extractor.extract("laptop under 800")
        assert pmin is None
        assert pmax == 800.0
        assert curr == "USD"
        assert "800" not in stripped

    def test_explicit_usd_formats(self) -> None:
        extractor = PriceExtractor(default_currency="USD")
        # $800
        pmin, pmax, curr, _ = extractor.extract("gaming laptop under $800")
        assert pmax == 800.0
        assert curr == "USD"

        # 800 USD
        pmin, pmax, curr, _ = extractor.extract("monitor below 800 usd")
        assert pmax == 800.0
        assert curr == "USD"

        # up to 800
        pmin, pmax, curr, _ = extractor.extract("headphones up to 800")
        assert pmax == 800.0
        assert curr == "USD"

    def test_explicit_inr_formats(self) -> None:
        extractor = PriceExtractor(default_currency="USD")
        # ₹80000
        pmin, pmax, curr, _ = extractor.extract("laptop under ₹80000")
        assert pmax == 80000.0
        assert curr == "INR"

        # 80000 INR
        pmin, pmax, curr, _ = extractor.extract("headphones below 80000 inr")
        assert pmax == 80000.0
        assert curr == "INR"

        # 80k INR
        norm = QueryNormalizer().normalize("phone under 80k inr")
        pmin, pmax, curr, _ = extractor.extract(norm)
        assert pmax == 80000.0
        assert curr == "INR"

    def test_price_ranges_and_floors(self) -> None:
        extractor = PriceExtractor(default_currency="USD")
        # between 500 and 1000
        pmin, pmax, curr, _ = extractor.extract("monitor between 500 and 1000")
        assert pmin == 500.0
        assert pmax == 1000.0
        assert curr == "USD"

        # above 500
        pmin, pmax, curr, _ = extractor.extract("smart tv above 500")
        assert pmin == 500.0
        assert pmax is None
        assert curr == "USD"

        # over $500
        pmin, pmax, curr, _ = extractor.extract("soundbar over $500")
        assert pmin == 500.0
        assert curr == "USD"


class TestAttributeNonOverlappingExtraction:
    """Test longest-match / boundary-aware attribute extraction without substring pollution."""

    def test_usb_vs_usb_c_longest_match(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        # USB-C should NOT produce USB
        res1 = default_pipeline.extract_attributes("usb-c fast charging cable")
        assert res1.get("connectivity") == ["USB-C"]
        assert "USB" not in res1.get("connectivity", [])

        # USB charger should produce USB
        res2 = default_pipeline.extract_attributes("usb wall charger")
        assert res2.get("connectivity") == ["USB"]

        # USB 3.0 should produce USB 3.0
        res3 = default_pipeline.extract_attributes("usb 3.0 high speed hub")
        assert res3.get("connectivity") == ["USB 3.0"]
        assert "USB" not in res3.get("connectivity", [])

        # Distinct co-occurring USB-C and USB-A
        res4 = default_pipeline.extract_attributes("usb-c and usb-a multi-port hub")
        assert res4.get("connectivity") == ["USB-A", "USB-C"]

    def test_wifi_vs_wifi_6(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        # WiFi 6 should NOT produce generic WiFi
        res1 = default_pipeline.extract_attributes("wifi 6 mesh router")
        assert res1.get("connectivity") == ["WiFi 6"]
        assert "WiFi" not in res1.get("connectivity", [])

        # WiFi generic
        res2 = default_pipeline.extract_attributes("wifi extender for home")
        assert res2.get("connectivity") == ["WiFi"]

    def test_rtx_vs_rtx_4060(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        # RTX 4060 should NOT produce generic RTX
        res1 = default_pipeline.extract_attributes("laptop with rtx 4060 gpu")
        assert res1.get("gpu") == ["RTX 4060"]
        assert "RTX" not in res1.get("gpu", [])

        # Generic RTX
        res2 = default_pipeline.extract_attributes("gaming pc with rtx graphics")
        assert res2.get("gpu") == ["RTX"]

    def test_ram_and_storage_specs(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        res = default_pipeline.extract_attributes("notebook with 16gb ram and 1tb ssd")
        assert res.get("ram") == ["16GB"]
        assert res.get("storage") == ["1TB", "SSD"]


class TestCatalogExtractionAndSynonyms:
    """Test category resolution, synonyms, and brand recognition."""

    def test_category_and_synonyms(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        cat1, syn1 = default_pipeline.extract_category("notebook computer for programming")
        assert cat1 == "laptop"
        assert syn1 is True

        cat2, syn2 = default_pipeline.extract_category("tws true wireless earbuds")
        assert cat2 == "earbuds"
        assert syn2 is True

        cat3, syn3 = default_pipeline.extract_category("studio monitor headphones")
        assert cat3 == "headphones"

        cat4, syn4 = default_pipeline.extract_category("monitor arm desk mount")
        assert cat4 == "mount"

    def test_brands(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        assert default_pipeline.extract_brand("western digital 1tb ssd") == "Western Digital"
        assert default_pipeline.extract_brand("amazon basics usb cable") == "Amazon Basics"
        assert default_pipeline.extract_brand("audio-technica studio headphones") == "Audio-Technica"
        assert default_pipeline.extract_brand("tp-link mesh router") == "Tp-Link"
        assert default_pipeline.extract_brand("sony bluetooth speaker") == "Sony"


class TestIntentClassification:
    """Test query intent classification."""

    def test_intents(self) -> None:
        clf = QueryIntentClassifier()
        assert clf.classify("laptop under 800", price_max=800.0) == "price_constrained_search"
        assert clf.classify("sony xm5 vs bose qc45") == "product_comparison"
        assert clf.classify("best wireless headphones for travel") == "recommendation"
        assert clf.classify("bose official store products", brand="Bose") == "brand_search"
        assert clf.classify("rtx 4070 32gb ram", attributes={"gpu": ["RTX 4070"], "ram": ["32GB"]}) == "attribute_search"
        assert clf.classify("bluetooth headphones", category="headphones") == "product_search"


class TestFullPipelineIntegrationAndConfidence:
    """Test end-to-end extraction, confidence scoring, hard/soft filter separation."""

    def test_combined_usd_query(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        intent = default_pipeline.process_to_intent("gaming laptop under $1000 with rtx 4060")
        assert intent.category == "laptop"
        assert intent.price_max == 1000.0
        assert intent.currency == "USD"
        assert intent.intent == "price_constrained_search"
        assert intent.attributes.get("gpu") == ["RTX 4060"]
        assert intent.attributes.get("use_case") == ["gaming"]
        assert intent.hard_filters["price_max"] == 1000.0
        assert intent.hard_filters["category"] == "laptop"
        assert intent.confidence >= 0.95

    def test_heuristic_confidence_levels(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        # Multi-signal: category + brand -> 1.0
        res_multi = default_pipeline.process_to_intent("sony noise cancelling headphones")
        assert res_multi.confidence == 1.0

        # Category synonym only -> 0.85
        res_syn = default_pipeline.process_to_intent("notebook")
        assert res_syn.confidence == 0.85

        # Brand only -> 0.85
        res_brand = default_pipeline.process_to_intent("sony")
        assert res_brand.confidence == 0.85

        # Completely ungrounded -> 0.60
        res_gibberish = default_pipeline.process_to_intent("random unsupported gibberish xyz123")
        assert res_gibberish.confidence == 0.60

    def test_malformed_queries(self, default_pipeline: QueryUnderstandingPipeline) -> None:
        res_empty = default_pipeline.process_to_intent("")
        assert res_empty.normalized_query == ""
        assert res_empty.intent == "product_search"

        res_noisy = default_pipeline.process_to_intent("   ??? cheap bluetooth headphones under 50 $$$ ")
        assert res_noisy.category == "headphones"
        assert res_noisy.price_max == 50.0
        assert res_noisy.currency == "USD"
