"""Comprehensive test suite for Phase 10 Grounded LLM Product Explanations."""

import json
import pytest
from backend.app.explanations.evidence_builder import EvidenceBuilder
from backend.app.explanations.fallback_explainer import FallbackExplainer
from backend.app.explanations.grounded_explainer import GroundedExplainer
from backend.app.explanations.guardrails import HallucinationGuardrail
from backend.app.explanations.llm_client import MockLLMClient, OpenAILLMClient
from backend.app.explanations.prompt_builder import PromptBuilder
from backend.app.models.explanation import (
    ExplanationReason,
    GroundedExplanation,
    ProductEvidence,
)
from backend.app.models.product import Product
from backend.app.models.recommendation import RecommendationItem
from backend.app.models.search import (
    QueryIntent,
    QueryUnderstandingResult,
    RerankSignal,
    RetrievalSignal,
)


@pytest.fixture
def sample_laptop_product():
    return Product(
        parent_asin="B08ABC1234",
        asin="B08ABC1234",
        title="Acer Nitro 5 Gaming Laptop, 15.6 FHD 144Hz, Intel i5-12500H, NVIDIA GeForce RTX 4060, 16GB DDR4, 512GB SSD",
        description="High performance gaming laptop with advanced cooling and high refresh rate display.",
        features=[
            "NVIDIA GeForce RTX 4060 graphics card with 8GB dedicated VRAM",
            "16GB DDR4 3200MHz Memory and 512GB PCIe Gen 4 SSD",
            "15.6 inch Full HD IPS display with 144Hz refresh rate",
        ],
        price=799.99,
        brand="Acer",
        categories=["Laptops", "Computers", "Electronics"],
        average_rating=4.5,
        rating_count=1250,
    )


@pytest.fixture
def sample_headphones_no_battery():
    return Product(
        parent_asin="B09XYZ9876",
        asin="B09XYZ9876",
        title="Sony MDR-ZX110 Wired Over-Ear Headphones, Black",
        description="Lightweight 30mm neodymium dynamic driver units deliver a punchy, rhythmic response to the most demanding tracks.",
        features=[
            "30mm dynamic dome drivers for balanced sound",
            "High energy neodymium magnets deliver powerful sound",
            "Tangle-free 1.2m Y-type cord",
        ],
        price=19.99,
        brand="Sony",
        categories=["Headphones", "Audio", "Electronics"],
        average_rating=4.3,
        rating_count=45000,
    )


def test_evidence_builder_search_constraint_matches(sample_laptop_product):
    """Test that query constraints (category, brand, price, attributes) are accurately mapped to evidence."""
    intent = QueryIntent(
        raw_query="gaming laptop under 900 with RTX 4060 and 16GB RAM",
        normalized_query="gaming laptop rtx 4060 16gb",
        category="Laptops",
        brand="Acer",
        price_max=900.0,
        attributes={"gpu": ["RTX 4060"], "ram": ["16GB"]},
    )
    ret_sig = RetrievalSignal(stage="hybrid_rrf", initial_score=0.0312, initial_rank=2)
    rr_sig = RerankSignal(stage="cross_encoder", rerank_score=4.82, rerank_rank=1)

    evidence = EvidenceBuilder.build_search_evidence(
        product=sample_laptop_product,
        query="gaming laptop under 900 with RTX 4060 and 16GB RAM",
        query_intent=intent,
        retrieval_signal=ret_sig,
        rerank_signal=rr_sig,
        final_rank=1,
    )

    assert evidence.product_id == "B08ABC1234"
    assert evidence.price == 799.99
    assert evidence.brand == "Acer"
    assert "budget" in evidence.matched_constraints
    assert "brand" in evidence.matched_constraints
    assert "category" in evidence.matched_constraints
    assert "gpu" in evidence.matched_constraints
    assert "ram" in evidence.matched_constraints
    assert len(evidence.missing_constraints) == 0


def test_hallucination_safety_missing_battery_constraint(sample_headphones_no_battery):
    """CRITICAL HALLUCINATION SAFETY TEST:

    When query explicitly requests battery life, but product evidence has NO battery information,
    the explanation engine MUST NOT claim or fabricate any battery duration.
    """
    query = "Sony noise cancelling headphones with 8-hour battery life"
    evidence = EvidenceBuilder.build_search_evidence(
        product=sample_headphones_no_battery,
        query=query,
        query_intent=QueryUnderstandingResult(
            raw_query=query,
            normalized_query="sony noise cancelling headphones battery",
            brand="Sony",
            category="Headphones",
        ),
    )

    # 1. EvidenceBuilder must record missing battery constraint
    assert any("battery" in m.lower() for m in evidence.missing_constraints)

    # 2. Generate explanation via FallbackExplainer
    explanation = FallbackExplainer.generate_explanation(evidence)

    # 3. Verify no battery life is claimed
    explanation_text = (explanation.summary + " " + " ".join(r.text for r in explanation.reasons)).lower()
    assert "8-hour" not in explanation_text
    assert "8 hour" not in explanation_text
    assert "battery life of" not in explanation_text

    # 4. Verify explicit missing attribute warning exists
    assert any("battery" in w.lower() for w in explanation.warnings)
    missing_reasons = [r for r in explanation.reasons if r.type == "unsupported_constraint" or not r.is_matched]
    assert len(missing_reasons) > 0
    assert any("battery" in r.text.lower() for r in missing_reasons)


def test_evidence_builder_recommendation(sample_laptop_product):
    """Test structured evidence extraction for item-to-item and personalized recommendations."""
    anchor = Product(
        parent_asin="B07ANCHOR1",
        title="Acer Predator Gaming Mouse RGB",
        brand="Acer",
        categories=["Computers", "Laptops", "Gaming Accessories"],
        price=49.99,
    )

    evidence = EvidenceBuilder.build_recommendation_evidence(
        product=sample_laptop_product,
        anchor_product=anchor,
        strategy="hybrid",
    )

    assert evidence.product_id == "B08ABC1234"
    assert evidence.personalization_evidence["same_brand"] is True
    assert "brand_alignment" in evidence.matched_constraints
    assert "category_continuity" in evidence.matched_constraints


def test_prompt_builder_structure(sample_laptop_product):
    """Test that PromptBuilder produces strict system instructions and valid JSON payload."""
    evidence = EvidenceBuilder.build_search_evidence(
        product=sample_laptop_product,
        query="gaming laptop rtx 4060",
    )
    prompts = PromptBuilder.build_prompt(evidence)

    assert "STRICT FACTUAL GROUNDING RULES" in prompts["system"]
    assert "NEVER invent, hallucinate, or infer" in prompts["system"]
    assert "B08ABC1234" in prompts["user"]
    assert "RTX 4060" in prompts["user"]


def test_hallucination_guardrail_rejection(sample_headphones_no_battery):
    """Test that HallucinationGuardrail catches and rejects unsupported claims."""
    evidence = EvidenceBuilder.build_search_evidence(
        product=sample_headphones_no_battery,
        query="wireless headphones with 40-hour battery",
    )
    evidence.missing_constraints.append("Battery capability requested in query but not listed in product evidence")

    # Fabricated LLM response claiming 40-hour battery
    fake_hallucinated_response = json.dumps({
        "product_id": "B09XYZ9876",
        "summary": "Great match with 40-hour battery life.",
        "reasons": [
            {
                "type": "constraint_match",
                "label": "Battery",
                "text": "Offers impressive 40-hour continuous battery playback.",
                "evidence": "40-hour battery",
                "is_matched": True,
            }
        ],
        "grounded": True,
        "warnings": [],
        "generation_method": "llm",
    })

    is_valid, explanation, msg = HallucinationGuardrail.validate_llm_response(
        raw_response=fake_hallucinated_response,
        evidence=evidence,
    )

    # Must fail validation because battery was listed as missing and evidence snippet is fictitious
    assert is_valid is False
    assert "Hallucination detected" in msg or "unverified" in msg


def test_grounded_explainer_with_mock_llm(sample_laptop_product):
    """Test GroundedExplainer execution with a validated Mock LLM client."""
    mock_response = json.dumps({
        "product_id": "B08ABC1234",
        "summary": "Strong match for your requested gaming laptop specifications.",
        "reasons": [
            {
                "type": "constraint_match",
                "label": "GPU",
                "text": "Equipped with NVIDIA GeForce RTX 4060 graphics.",
                "evidence": "NVIDIA GeForce RTX 4060",
                "is_matched": True,
            },
            {
                "type": "constraint_match",
                "label": "Budget",
                "text": "Listed price ($799.99) is within your budget.",
                "evidence": "$799.99",
                "is_matched": True,
            },
        ],
        "semantic_match_score": 0.95,
        "grounded": True,
        "warnings": [],
        "generation_method": "llm",
    })

    mock_client = MockLLMClient(canned_response=mock_response)
    explainer = GroundedExplainer(llm_client=mock_client, enable_remote_llm=True)

    result = explainer.explain_search(
        query="gaming laptop rtx 4060",
        product=sample_laptop_product,
    )

    assert isinstance(result, GroundedExplanation)
    assert result.product_id == "B08ABC1234"
    assert result.grounded is True
    assert result.generation_method == "llm"
    assert len(result.reasons) == 2
    assert result.reasons[0].label == "GPU"


def test_grounded_explainer_fallback_when_disabled(sample_laptop_product):
    """Test that when remote LLM is disabled, GroundedExplainer uses deterministic fallback with 100% reliability."""
    explainer = GroundedExplainer(enable_remote_llm=False)
    result = explainer.explain_search(
        query="Acer gaming laptop",
        product=sample_laptop_product,
    )

    assert isinstance(result, GroundedExplanation)
    assert result.product_id == "B08ABC1234"
    assert result.grounded is True
    assert result.generation_method == "deterministic_fallback"
    assert len(result.reasons) >= 1
    assert any(r.label == "Brand" for r in result.reasons)


def test_batch_explain_search(sample_laptop_product, sample_headphones_no_battery):
    """Test batch explanation generation for top-N displayed results."""
    explainer = GroundedExplainer(enable_remote_llm=False)
    products = [sample_laptop_product, sample_headphones_no_battery]

    explanations = explainer.batch_explain_search(
        query="electronics",
        products=products,
        max_to_explain=2,
    )

    assert len(explanations) == 2
    assert explanations[0].product_id == "B08ABC1234"
    assert explanations[1].product_id == "B09XYZ9876"
