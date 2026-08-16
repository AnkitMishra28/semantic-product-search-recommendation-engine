"""Grounded LLM explanation service with strict factual boundaries and deterministic fallback."""

import logging
from typing import Any, Dict, List, Optional, Union
from backend.app.explanations.base import BaseExplainer
from backend.app.explanations.evidence_builder import EvidenceBuilder
from backend.app.explanations.fallback_explainer import FallbackExplainer
from backend.app.explanations.guardrails import HallucinationGuardrail
from backend.app.explanations.llm_client import BaseLLMClient, OpenAILLMClient
from backend.app.explanations.prompt_builder import PromptBuilder
from backend.app.models.explanation import (
    ExplanationReason,
    GroundedExplanation,
    ProductEvidence,
)
from backend.app.models.product import Product
from backend.app.models.recommendation import ExplanationItem, RecommendationItem
from backend.app.models.search import (
    QueryIntent,
    QueryUnderstandingResult,
    RerankSignal,
    RetrievalSignal,
)
from backend.app.recommendation.base import RecommendationCandidate

logger = logging.getLogger(__name__)


class GroundedExplainer(BaseExplainer):
    """Orchestrates evidence extraction, LLM grounding prompt execution, and deterministic fallback."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        enable_remote_llm: bool = False,
    ) -> None:
        self.enable_remote_llm = enable_remote_llm
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAILLMClient(api_key=api_key, model_name=model_name)

    def explain_search_result(
        self,
        query: str,
        product: Product,
        matched_features: Optional[List[str]] = None,
    ) -> str:
        """Backward-compatible string explanation method."""
        grounded = self.explain_search(query=query, product=product)
        return grounded.summary

    def explain_recommendation(
        self,
        anchor_product: Product,
        recommended_product: Product,
    ) -> ExplanationItem:
        """Backward-compatible Recommendation ExplanationItem method."""
        grounded = self.explain_recommendation_grounded(
            product=recommended_product,
            anchor_product=anchor_product,
        )
        return ExplanationItem(
            summary=grounded.summary,
            key_features_matched=[r.evidence for r in grounded.reasons if r.is_matched][:3],
            shared_categories=recommended_product.categories,
            confidence=0.92 if grounded.generation_method == "llm" else 0.85,
        )

    def explain_evidence(self, evidence: ProductEvidence) -> GroundedExplanation:
        """Generate a grounded explanation directly from a pre-assembled ProductEvidence object."""
        # 1. Try remote LLM generation if enabled
        if self.enable_remote_llm:
            try:
                prompts = PromptBuilder.build_prompt(evidence)
                raw_response = self.llm_client.generate(
                    system_prompt=prompts["system"],
                    user_prompt=prompts["user"],
                )
                is_valid, explanation, msg = HallucinationGuardrail.validate_llm_response(
                    raw_response=raw_response,
                    evidence=evidence,
                )
                if is_valid and explanation is not None:
                    return explanation
                logger.warning(f"LLM explanation failed guardrail validation ({msg}). Falling back to deterministic generation.")
            except Exception as e:
                logger.warning(f"LLM explanation call failed ({str(e)}). Falling back to deterministic generation.")

        # 2. Deterministic Fallback Generation
        return FallbackExplainer.generate_explanation(evidence)

    def explain_search(
        self,
        query: str,
        product: Product,
        query_intent: Optional[Union[QueryIntent, QueryUnderstandingResult]] = None,
        retrieval_signal: Optional[RetrievalSignal] = None,
        rerank_signal: Optional[RerankSignal] = None,
        final_rank: Optional[int] = None,
        candidate_provenance: Optional[Dict[str, Any]] = None,
    ) -> GroundedExplanation:
        """Generate a structured GroundedExplanation for a search result."""
        evidence = EvidenceBuilder.build_search_evidence(
            product=product,
            query=query,
            query_intent=query_intent,
            retrieval_signal=retrieval_signal,
            rerank_signal=rerank_signal,
            final_rank=final_rank,
            candidate_provenance=candidate_provenance,
        )
        return self.explain_evidence(evidence)

    def explain_recommendation_grounded(
        self,
        product: Product,
        rec_item: Optional[Union[RecommendationItem, RecommendationCandidate]] = None,
        anchor_product: Optional[Product] = None,
        user_history_products: Optional[List[Product]] = None,
        strategy: str = "hybrid",
    ) -> GroundedExplanation:
        """Generate a structured GroundedExplanation for a recommendation item."""
        evidence = EvidenceBuilder.build_recommendation_evidence(
            product=product,
            rec_item=rec_item,
            anchor_product=anchor_product,
            user_history_products=user_history_products,
            strategy=strategy,
        )
        return self.explain_evidence(evidence)

    def batch_explain_search(
        self,
        query: str,
        products: List[Product],
        query_intent: Optional[Union[QueryIntent, QueryUnderstandingResult]] = None,
        retrieval_signals: Optional[Dict[str, RetrievalSignal]] = None,
        rerank_signals: Optional[Dict[str, RerankSignal]] = None,
        max_to_explain: int = 5,
    ) -> List[GroundedExplanation]:
        """Efficiently generate explanations only for the top-N items displayed to the user."""
        explanations: List[GroundedExplanation] = []
        ret_sigs = retrieval_signals or {}
        rr_sigs = rerank_signals or {}

        for idx, prod in enumerate(products[:max_to_explain]):
            asin = prod.parent_asin or prod.asin
            exp = self.explain_search(
                query=query,
                product=prod,
                query_intent=query_intent,
                retrieval_signal=ret_sigs.get(asin),
                rerank_signal=rr_sigs.get(asin),
                final_rank=idx + 1,
            )
            explanations.append(exp)

        return explanations
