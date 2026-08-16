"""Structured evidence extraction and constraint matching from search & recommendation pipelines."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
from backend.app.models.explanation import ProductEvidence
from backend.app.models.product import Product
from backend.app.models.recommendation import RecommendationItem
from backend.app.models.search import (
    QueryIntent,
    QueryUnderstandingResult,
    RerankSignal,
    RetrievalSignal,
)
from backend.app.recommendation.base import RecommendationCandidate

logger = logging.getLogger(__name__)


class EvidenceBuilder:
    """Extracts, verifies, and packages multi-stage pipeline signals into a structured ProductEvidence payload."""

    # Common technical specifications and features to check for query-to-product alignment
    STANDARD_SPEC_KEYWORDS = {
        "battery": ["battery", "mah", "runtime", "hours of battery", "hour battery", "battery life"],
        "wireless": ["wireless", "bluetooth", "2.4ghz", "wifi", "wi-fi", "cordless"],
        "anc": ["noise cancelling", "active noise cancellation", "anc", "noise isolation"],
        "waterproof": ["waterproof", "ipx", "water resistant", "water-resistant", "splashproof"],
        "gpu": ["rtx", "gtx", "radeon", "geforce", "gpu", "graphics"],
        "ram": ["gb ram", "ram", "ddr4", "ddr5", "memory"],
        "storage": ["ssd", "nvme", "hdd", "storage", "tb ssd", "gb ssd"],
        "display": ["4k", "oled", "144hz", "120hz", "ips", "retina", "screen"],
        "fast_charging": ["fast charging", "quick charge", "pd charge", "power delivery"],
    }

    @classmethod
    def build_search_evidence(
        cls,
        product: Product,
        query: str,
        query_intent: Optional[Union[QueryIntent, QueryUnderstandingResult]] = None,
        retrieval_signal: Optional[RetrievalSignal] = None,
        rerank_signal: Optional[RerankSignal] = None,
        final_rank: Optional[int] = None,
        candidate_provenance: Optional[Dict[str, Any]] = None,
    ) -> ProductEvidence:
        """Build structured evidence package for a search result."""
        # 1. Base product attributes
        doc_text = " ".join([
            product.title or "",
            product.description or "",
            " ".join(product.features or []),
            product.brand or "",
            " ".join(product.categories or []),
        ]).lower()

        # 2. Extract structured query constraints
        q_constraints: Dict[str, Any] = {}
        if query_intent:
            if query_intent.category:
                q_constraints["category"] = query_intent.category
            if query_intent.brand:
                q_constraints["brand"] = query_intent.brand
            if query_intent.price_min is not None:
                q_constraints["price_min"] = query_intent.price_min
            if query_intent.price_max is not None:
                q_constraints["price_max"] = query_intent.price_max
            if query_intent.currency:
                q_constraints["currency"] = query_intent.currency
            if hasattr(query_intent, "attributes") and query_intent.attributes:
                q_constraints["attributes"] = dict(query_intent.attributes)

        # 3. Deterministic constraint verification & missing attribute detection
        matched_constraints: Dict[str, str] = {}
        missing_constraints: List[str] = []

        # Check Category constraint
        if "category" in q_constraints:
            req_cat = q_constraints["category"].lower()
            prod_cats = [c.lower() for c in product.categories]
            if any(req_cat in c for c in prod_cats) or req_cat in product.title.lower():
                matched_constraints["category"] = f"Categorized under '{product.categories[0] if product.categories else req_cat}'"
            else:
                missing_constraints.append(f"Category: {q_constraints['category']}")
        elif any(c.lower() in query.lower() for c in product.categories):
            matched_cat = next(c for c in product.categories if c.lower() in query.lower())
            matched_constraints["category"] = f"Categorized under '{matched_cat}'"

        # Check Brand constraint
        if "brand" in q_constraints:
            req_brand = q_constraints["brand"].lower()
            if product.brand and (req_brand in product.brand.lower() or product.brand.lower() in req_brand):
                matched_constraints["brand"] = f"Manufactured by {product.brand}"
            elif req_brand in product.title.lower():
                matched_constraints["brand"] = f"Brand token '{q_constraints['brand']}' confirmed in title"
            else:
                missing_constraints.append(f"Brand: {q_constraints['brand']}")
        elif product.brand and product.brand.lower() in query.lower():
            matched_constraints["brand"] = f"Manufactured by {product.brand}"

        # Check Price constraints
        currency_sym = "$" if q_constraints.get("currency", "USD") == "USD" else q_constraints.get("currency", "$")
        if "price_max" in q_constraints:
            p_max = q_constraints["price_max"]
            if product.price is not None:
                if product.price <= p_max:
                    matched_constraints["budget"] = f"Listed price ({currency_sym}{product.price:.2f}) is within the requested {currency_sym}{p_max:.2f} limit"
                else:
                    missing_constraints.append(f"Price budget ({currency_sym}{product.price:.2f} exceeds max {currency_sym}{p_max:.2f})")
            else:
                missing_constraints.append(f"Price budget (Price not listed in catalog)")

        if "price_min" in q_constraints:
            p_min = q_constraints["price_min"]
            if product.price is not None and product.price >= p_min:
                matched_constraints["min_price"] = f"Price ({currency_sym}{product.price:.2f}) satisfies minimum requirement of {currency_sym}{p_min:.2f}"

        # Check extracted attributes
        if "attributes" in q_constraints and isinstance(q_constraints["attributes"], dict):
            for attr_key, attr_vals in q_constraints["attributes"].items():
                val_list = attr_vals if isinstance(attr_vals, list) else [attr_vals]
                for val in val_list:
                    val_str = str(val).lower()
                    if val_str in doc_text:
                        matched_constraints[attr_key] = f"Matches '{val}' specification"
                    else:
                        missing_constraints.append(f"{attr_key}: {val}")

        # Check query keywords for requested capabilities (e.g. battery life, waterproof, etc.)
        query_lower = query.lower()
        for spec_key, spec_terms in cls.STANDARD_SPEC_KEYWORDS.items():
            if any(term in query_lower for term in spec_terms):
                found_in_doc = any(term in doc_text for term in spec_terms)
                if found_in_doc:
                    # Find exact feature snippet
                    matched_feat = next((f for f in product.features if any(term in f.lower() for term in spec_terms)), None)
                    if matched_feat:
                        matched_constraints[spec_key] = f"Feature verified: '{matched_feat}'"
                    else:
                        matched_constraints[spec_key] = f"Mentions {spec_key} capabilities"
                else:
                    label = spec_key.replace("_", " ").title()
                    if f"{label} information" not in [m for m in missing_constraints]:
                        missing_constraints.append(f"{label} capability requested in query but not listed in product evidence")

        # 4. Retrieval & Cross-Encoder provenance
        retrieval_prov: Dict[str, Any] = {}
        if retrieval_signal:
            retrieval_prov["stage"] = retrieval_signal.stage
            retrieval_prov["score"] = retrieval_signal.initial_score
            retrieval_prov["rank"] = retrieval_signal.initial_rank
        if candidate_provenance:
            retrieval_prov.update(candidate_provenance)

        ce_evidence: Dict[str, Any] = {}
        if rerank_signal:
            ce_evidence["score"] = rerank_signal.rerank_score
            ce_evidence["rank"] = rerank_signal.rerank_rank
            if final_rank is not None:
                ce_evidence["final_rank"] = final_rank

        return ProductEvidence(
            query=query,
            product_id=product.parent_asin or product.asin or "UNKNOWN",
            product_title=product.title,
            price=product.price,
            currency=q_constraints.get("currency", "USD"),
            rating=product.average_rating,
            rating_count=product.rating_count or product.rating_number,
            brand=product.brand,
            category=product.categories[0] if product.categories else None,
            categories=product.categories,
            features=product.features[:5] if product.features else [],
            description=(product.description[:300] + "...") if product.description and len(product.description) > 300 else (product.description or ""),
            attributes=product.metadata or {},
            query_constraints=q_constraints,
            retrieval_provenance=retrieval_prov,
            cross_encoder_evidence=ce_evidence,
            matched_constraints=matched_constraints,
            missing_constraints=missing_constraints,
        )

    @classmethod
    def build_recommendation_evidence(
        cls,
        product: Product,
        rec_item: Optional[Union[RecommendationItem, RecommendationCandidate]] = None,
        anchor_product: Optional[Product] = None,
        user_history_products: Optional[List[Product]] = None,
        strategy: str = "hybrid",
    ) -> ProductEvidence:
        """Build structured evidence package for a recommendation item."""
        matched_constraints: Dict[str, str] = {}
        missing_constraints: List[str] = []
        personalization: Dict[str, Any] = {
            "strategy": strategy,
        }

        if rec_item:
            if isinstance(rec_item, RecommendationItem):
                personalization["score"] = rec_item.score
                personalization["signals"] = rec_item.signals
                personalization["recommendation_type"] = rec_item.recommendation_type
            elif isinstance(rec_item, RecommendationCandidate):
                personalization["score"] = rec_item.score
                personalization["signals"] = rec_item.signals
                personalization["recommendation_type"] = rec_item.model_name

        # Item-to-Item Anchor evidence
        if anchor_product:
            personalization["anchor_product_id"] = anchor_product.parent_asin or anchor_product.asin
            personalization["anchor_title"] = anchor_product.title
            
            # Shared categories
            shared_cats = [c for c in product.categories if c in anchor_product.categories]
            if shared_cats:
                personalization["shared_categories"] = shared_cats
                matched_constraints["category_continuity"] = f"Shares category '{shared_cats[0]}' with currently viewed product"
            
            # Brand alignment
            if product.brand and anchor_product.brand and product.brand.lower() == anchor_product.brand.lower():
                personalization["same_brand"] = True
                matched_constraints["brand_alignment"] = f"From the same trusted brand ({product.brand})"
            
            # Feature overlap
            shared_features = [f for f in product.features if f in anchor_product.features]
            if shared_features:
                personalization["shared_features"] = shared_features[:3]

        # User History / Profile evidence
        if user_history_products:
            history_brands = {p.brand.lower() for p in user_history_products if p.brand}
            history_cats = {c.lower() for p in user_history_products for c in p.categories}
            
            personalization["user_history_count"] = len(user_history_products)
            
            if product.brand and product.brand.lower() in history_brands:
                matched_constraints["user_brand_preference"] = f"Aligns with your past affinity for {product.brand} products"
            
            matched_user_cats = [c for c in product.categories if c.lower() in history_cats]
            if matched_user_cats:
                matched_constraints["user_category_preference"] = f"Matches your interest in {matched_user_cats[0]}"

        # Rating / popularity credibility
        if product.average_rating and product.average_rating >= 4.3:
            matched_constraints["rating_credibility"] = f"Highly rated ({product.average_rating:.1f}★ with {product.rating_count:,} reviews)"
        elif product.rating_count and product.rating_count >= 1000:
            matched_constraints["popularity_credibility"] = f"Widely chosen item with {product.rating_count:,} customer reviews"

        return ProductEvidence(
            product_id=product.parent_asin or product.asin or "UNKNOWN",
            product_title=product.title,
            price=product.price,
            currency="USD",
            rating=product.average_rating,
            rating_count=product.rating_count or product.rating_number,
            brand=product.brand,
            category=product.categories[0] if product.categories else None,
            categories=product.categories,
            features=product.features[:5] if product.features else [],
            description=(product.description[:300] + "...") if product.description and len(product.description) > 300 else (product.description or ""),
            attributes=product.metadata or {},
            personalization_evidence=personalization,
            matched_constraints=matched_constraints,
            missing_constraints=missing_constraints,
        )
