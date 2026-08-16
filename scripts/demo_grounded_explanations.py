#!/usr/bin/env python3
"""Interactive demonstration of Phase 10 Grounded LLM Product Explanations.

Shows the complete pipeline flow:
User Query / Recommendation Request
  ↓
Query Understanding / Personalization
  ↓
Hybrid Retrieval / Reranking
  ↓
Structured Product Evidence
  ↓
Grounded Explanation Generation (LLM / Guardrail / Deterministic Fallback)
"""

import json
import logging
from pathlib import Path
from pprint import pprint
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.explanations.evidence_builder import EvidenceBuilder
from backend.app.explanations.fallback_explainer import FallbackExplainer
from backend.app.explanations.grounded_explainer import GroundedExplainer
from backend.app.explanations.prompt_builder import PromptBuilder
from backend.app.models.explanation import GroundedExplanation, ProductEvidence
from backend.app.models.product import Product
from backend.app.models.recommendation import RecommendationItem
from backend.app.models.search import (
    QueryUnderstandingResult,
    RerankSignal,
    RetrievalSignal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def demo_search_explanation():
    print("\n" + "=" * 80)
    print("DEMO 1: SEARCH EXPLANATION WITH SPECIFICATION & BUDGET CONSTRAINTS")
    print("=" * 80)

    query = "gaming laptop under 85000 with RTX GPU and 16GB RAM"
    print(f"User Query: \"{query}\"")

    # 1. Product in Catalog
    product = Product(
        parent_asin="B08N5WRWNW",
        asin="B08N5WRWNW",
        title="Acer Nitro 5 AN515 Gaming Laptop | Intel Core i5-12500H | NVIDIA GeForce RTX 4060 GPU | 15.6 FHD 144Hz IPS | 16GB DDR4 | 512GB Gen 4 SSD",
        description="Dominate the competition with the powerful combination of 12th Gen Intel Core i5 processor and NVIDIA GeForce RTX 4060 laptop GPU.",
        features=[
            "NVIDIA GeForce RTX 4060 Laptop GPU with 8GB dedicated GDDR6 VRAM",
            "16GB 3200MHz DDR4 Memory (2 DDR4 Slots Total, Maximum 32GB)",
            "512GB PCIe Gen 4 SSD (2 x PCIe M.2 Slots)",
            "15.6 Full HD (1920 x 1080) widescreen LED-backlit IPS display with 144Hz refresh rate",
        ],
        price=799.99,
        brand="Acer",
        categories=["Laptops", "Computers & Tablets", "Electronics"],
        average_rating=4.5,
        rating_count=2380,
    )

    # 2. Query Understanding Result
    qu = QueryUnderstandingResult(
        raw_query=query,
        normalized_query="gaming laptop rtx gpu 16gb ram",
        category="Laptops",
        price_max=850.0,
        currency="USD",
        attributes={"gpu": ["RTX"], "ram": ["16GB"]},
    )

    # 3. Pipeline Signals
    ret_sig = RetrievalSignal(stage="hybrid_rrf", initial_score=0.0315, initial_rank=3)
    rerank_sig = RerankSignal(stage="cross_encoder", rerank_score=5.4120, rerank_rank=1)

    # 4. Extract Structured Evidence
    evidence = EvidenceBuilder.build_search_evidence(
        product=product,
        query=query,
        query_intent=qu,
        retrieval_signal=ret_sig,
        rerank_signal=rerank_sig,
        final_rank=1,
    )

    print("\n--- [Structured Evidence Object] ---")
    pprint(evidence.model_dump(exclude_none=True))

    # 5. Generate Grounded Explanation
    explainer = GroundedExplainer(enable_remote_llm=False)
    explanation = explainer.explain_evidence(evidence)

    print("\n--- [Grounded Explanation Response] ---")
    print(json.dumps(explanation.model_dump(), indent=2))


def demo_hallucination_safety():
    print("\n" + "=" * 80)
    print("DEMO 2: HALLUCINATION SAFETY — MISSING ATTRIBUTE GUARDRAIL")
    print("=" * 80)

    query = "Sony wired studio headphones with 10-hour battery life"
    print(f"User Query: \"{query}\" (Requested '10-hour battery life' on a wired headphone)")

    # Product in Catalog (Wired headphones, NO battery info exists)
    product = Product(
        parent_asin="B00001W0DI",
        asin="B00001W0DI",
        title="Sony MDR-7506 Professional Large Diaphragm Headphone",
        description="Neodymium magnets and 40mm drivers for powerful, detailed sound. Closed-ear design provides comfort and outstanding reduction of external noises.",
        features=[
            "Rugged design: reliable in the most demanding situations",
            "Folds up for storage or transport in provided soft case",
            "9.8-foot coiled cord with gold-plated plug",
        ],
        price=99.99,
        brand="Sony",
        categories=["Headphones", "Professional Audio", "Electronics"],
        average_rating=4.7,
        rating_count=31200,
    )

    qu = QueryUnderstandingResult(
        raw_query=query,
        normalized_query="sony wired studio headphones battery",
        brand="Sony",
        category="Headphones",
    )

    evidence = EvidenceBuilder.build_search_evidence(
        product=product,
        query=query,
        query_intent=qu,
    )

    print("\n--- [Detected Missing Constraints] ---")
    print(f"Missing Constraints in Evidence: {evidence.missing_constraints}")

    explainer = GroundedExplainer(enable_remote_llm=False)
    explanation = explainer.explain_evidence(evidence)

    print("\n--- [Grounded Explanation with Explicit Safety Disclaimers] ---")
    print(json.dumps(explanation.model_dump(), indent=2))
    print("\nVerification: Notice that the engine explicitly listed 'Missing Info' and did NOT hallucinate battery runtime.")


def demo_recommendation_explanation():
    print("\n" + "=" * 80)
    print("DEMO 3: ITEM-TO-ITEM & USER PERSONALIZATION EXPLANATION")
    print("=" * 80)

    anchor = Product(
        parent_asin="B08N5WRWNW",
        asin="B08N5WRWNW",
        title="Acer Nitro 5 AN515 Gaming Laptop",
        brand="Acer",
        categories=["Laptops", "Computers & Tablets", "Electronics"],
        price=799.99,
    )

    rec_product = Product(
        parent_asin="B084G3K539",
        asin="B084G3K539",
        title="Acer Nitro Gaming Mouse II with 6 DPI Levels & RGB Backlight",
        description="Ergonomically designed optical gaming mouse engineered for Acer Nitro systems.",
        features=[
            "Adjustable 6-level DPI setting up to 4200 DPI",
            "4-color LED backlight breathing mode",
            "Ergonomic comfort grip for long gaming sessions",
        ],
        price=29.99,
        brand="Acer",
        categories=["Computer Accessories", "Gaming Mice", "Electronics"],
        average_rating=4.6,
        rating_count=5420,
    )

    rec_item = RecommendationItem(
        product=rec_product,
        score=0.892,
        recommendation_type="multi_signal_hybrid",
        signals={
            "content_similarity": 0.81,
            "collaborative_co_occurrence": 0.94,
            "popularity": 0.72,
            "rating_prior": 0.88,
        },
    )

    evidence = EvidenceBuilder.build_recommendation_evidence(
        product=rec_product,
        rec_item=rec_item,
        anchor_product=anchor,
        strategy="hybrid",
    )

    explainer = GroundedExplainer(enable_remote_llm=False)
    explanation = explainer.explain_evidence(evidence)

    print("\n--- [Personalized Grounded Explanation] ---")
    print(json.dumps(explanation.model_dump(), indent=2))


if __name__ == "__main__":
    demo_search_explanation()
    demo_hallucination_safety()
    demo_recommendation_explanation()
    print("\n" + "=" * 80)
    print("ALL PHASE 10 DEMONSTRATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 80)
