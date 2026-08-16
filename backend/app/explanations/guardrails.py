"""Hallucination detection, factual guardrails, and validation for LLM explanations."""

import json
import logging
from typing import Optional, Tuple
from pydantic import ValidationError
from backend.app.models.explanation import GroundedExplanation, ProductEvidence

logger = logging.getLogger(__name__)


class HallucinationGuardrail:
    """Validates that LLM generated explanations are factually anchored to the input ProductEvidence."""

    @classmethod
    def validate_llm_response(
        cls,
        raw_response: str,
        evidence: ProductEvidence,
    ) -> Tuple[bool, Optional[GroundedExplanation], str]:
        """Validate an LLM response string against the source evidence.

        Returns:
            Tuple of (is_valid, parsed_explanation_or_none, failure_reason_if_any)
        """
        # 1. Parse JSON
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed_dict = json.loads(cleaned)
            explanation = GroundedExplanation.model_validate(parsed_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            msg = f"Failed to parse LLM response into GroundedExplanation schema: {str(e)}"
            logger.warning(msg)
            return False, None, msg

        # 2. Check factual grounding against evidence text
        evidence_corpus = " ".join([
            evidence.product_title or "",
            evidence.description or "",
            " ".join(evidence.features or []),
            evidence.brand or "",
            evidence.category or "",
            " ".join(evidence.categories or []),
            str(evidence.price or ""),
            str(evidence.rating or ""),
            str(evidence.rating_count or ""),
            str(evidence.cross_encoder_evidence or ""),
            str(evidence.personalization_evidence or ""),
        ]).lower()

        # 3. Check for forbidden hallucination on missing constraints
        for missing in evidence.missing_constraints:
            missing_tokens = [
                w for w in missing.lower().replace(":", " ").replace("-", " ").split()
                if len(w) > 3 and w not in (
                    "requested", "query", "listed", "product", "evidence", "catalog",
                    "metadata", "information", "specified", "capability",
                )
            ]
            # If a missing constraint (e.g. "battery") is claimed as matched, reject
            for r in explanation.reasons:
                if r.is_matched and any(tok in r.label.lower() or tok in r.text.lower() for tok in missing_tokens):
                    ev_clean = r.evidence.lower().strip()
                    if ev_clean not in evidence_corpus:
                        msg = f"Hallucination detected: Reason claims match on missing constraint '{missing}' with unverified evidence '{r.evidence}'"
                        logger.warning(msg)
                        return False, None, msg

        # 4. Check that reason evidence snippets are present in evidence corpus
        for r in explanation.reasons:
            if r.is_matched and r.evidence and len(r.evidence.strip()) > 2:
                ev_clean = r.evidence.lower().replace("$", "").replace("★", "").strip()
                if ev_clean not in evidence_corpus and not all(w in evidence_corpus for w in ev_clean.split() if len(w) > 3):
                    if any(char.isdigit() for char in ev_clean) and ev_clean not in evidence_corpus:
                        msg = f"Hallucination detected: Unverified numerical evidence '{r.evidence}' not found in catalog evidence."
                        logger.warning(msg)
                        return False, None, msg

        # Mark confirmed grounded
        explanation.grounded = True
        explanation.generation_method = "llm"

        return True, explanation, "Validation successful"
