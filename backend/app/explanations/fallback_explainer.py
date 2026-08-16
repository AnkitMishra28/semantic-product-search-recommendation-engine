"""Deterministic, rule-grounded explanation generator without LLM dependencies."""

import logging
from typing import List, Optional
from backend.app.models.explanation import (
    ExplanationReason,
    GroundedExplanation,
    ProductEvidence,
)

logger = logging.getLogger(__name__)


class FallbackExplainer:
    """Generates structured, factually grounded explanations deterministically from ProductEvidence."""

    @classmethod
    def generate_explanation(cls, evidence: ProductEvidence) -> GroundedExplanation:
        """Generate a fully structured GroundedExplanation from verified ProductEvidence."""
        reasons: List[ExplanationReason] = []
        warnings: List[str] = []

        # 1. Constraint Matches (Price/Budget, Brand, Category, Attributes)
        for key, desc in evidence.matched_constraints.items():
            reason_type = "constraint_match"
            label = key.replace("_", " ").title()
            
            if "budget" in key.lower() or "price" in key.lower():
                label = "Budget"
                quoted_ev = f"${evidence.price:.2f}" if evidence.price is not None else "Price verified"
            elif "brand" in key.lower():
                label = "Brand"
                quoted_ev = evidence.brand or "Brand verified"
            elif "category" in key.lower():
                label = "Category"
                quoted_ev = evidence.category or "Category verified"
            elif "rating" in key.lower():
                reason_type = "rating_credibility"
                label = "Customer Rating"
                quoted_ev = f"{evidence.rating}★ ({evidence.rating_count:,} reviews)"
            else:
                label = "Specification"
                quoted_ev = desc

            reasons.append(
                ExplanationReason(
                    type=reason_type,
                    label=label,
                    text=desc,
                    evidence=quoted_ev,
                    is_matched=True,
                )
            )

        # 2. Semantic & Cross-Encoder Relevance
        score = None
        if evidence.cross_encoder_evidence and "score" in evidence.cross_encoder_evidence:
            ce_score = evidence.cross_encoder_evidence["score"]
            ce_rank = evidence.cross_encoder_evidence.get("rank", 1)
            score = round(float(ce_score), 4)
            reasons.append(
                ExplanationReason(
                    type="ranking_strength",
                    label="Neural Relevance",
                    text=f"Ranked #{ce_rank} with high query-product cross-attention relevance ({ce_score:.2f}).",
                    evidence=f"Cross-Encoder Score: {ce_score:.4f}",
                    is_matched=True,
                )
            )
        elif evidence.retrieval_provenance and "score" in evidence.retrieval_provenance:
            ret_score = evidence.retrieval_provenance["score"]
            score = round(float(ret_score), 4)
            reasons.append(
                ExplanationReason(
                    type="semantic_relevance",
                    label="Vector Relevance",
                    text=f"High semantic vector proximity to your search intent.",
                    evidence=f"Similarity: {ret_score:.4f}",
                    is_matched=True,
                )
            )

        # 3. Personalization & Recommendation Continuity
        if evidence.personalization_evidence:
            p_ev = evidence.personalization_evidence
            strat = p_ev.get("strategy", "recommendation")
            
            if "anchor_title" in p_ev:
                reasons.append(
                    ExplanationReason(
                        type="personalization",
                        label="Complementary Match",
                        text=f"Frequently paired or related to '{p_ev['anchor_title'][:40]}...'",
                        evidence=f"Anchor: {p_ev.get('anchor_product_id', 'Item')}",
                        is_matched=True,
                    )
                )
            elif p_ev.get("user_history_count", 0) > 0:
                reasons.append(
                    ExplanationReason(
                        type="personalization",
                        label="User Affinity",
                        text=f"Personalized based on your past browsing interest across {p_ev['user_history_count']} viewed products.",
                        evidence=f"History items: {p_ev['user_history_count']}",
                        is_matched=True,
                    )
                )

        # 4. Feature Snippet Evidence (if no specific constraint matched yet)
        if len(reasons) < 2 and evidence.features:
            reasons.append(
                ExplanationReason(
                    type="constraint_match",
                    label="Key Feature",
                    text=f"Key product highlight: {evidence.features[0]}",
                    evidence=evidence.features[0][:80],
                    is_matched=True,
                )
            )

        # 5. Explicit Missing / Unsupported Constraint Disclaimers (Crucial Hallucination Safeguard)
        for missing in evidence.missing_constraints:
            warning_msg = f"{missing} is not specified in the verified product catalog metadata."
            warnings.append(warning_msg)
            reasons.append(
                ExplanationReason(
                    type="unsupported_constraint",
                    label="Missing Info",
                    text=warning_msg,
                    evidence="Not available in catalog",
                    is_matched=False,
                )
            )

        # 6. Compose User-Facing Summary Sentence
        matched_points = [r.text for r in reasons if r.is_matched]
        if matched_points:
            summary = f"Recommended product: {matched_points[0]}"
            if len(matched_points) > 1:
                summary += f" and {matched_points[1].lower()}."
            else:
                summary += "."
        elif evidence.query:
            summary = f"Retrieved as a relevant match for '{evidence.query}' based on verified catalog attributes."
        else:
            summary = f"Recommended from the catalog based on relevance and quality ratings."

        return GroundedExplanation(
            product_id=evidence.product_id,
            summary=summary,
            reasons=reasons,
            semantic_match_score=score,
            grounded=True,
            warnings=warnings,
            generation_method="deterministic_fallback",
        )
