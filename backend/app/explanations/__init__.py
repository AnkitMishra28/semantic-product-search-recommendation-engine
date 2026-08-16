"""Explanations package for search and recommendation rationale."""

from backend.app.explanations.base import BaseExplainer
from backend.app.explanations.evidence_builder import EvidenceBuilder
from backend.app.explanations.fallback_explainer import FallbackExplainer
from backend.app.explanations.grounded_explainer import GroundedExplainer
from backend.app.explanations.guardrails import HallucinationGuardrail
from backend.app.explanations.llm_client import BaseLLMClient, MockLLMClient, OpenAILLMClient
from backend.app.explanations.llm_explainer import LLMExplainer
from backend.app.explanations.prompt_builder import PromptBuilder

__all__ = [
    "BaseExplainer",
    "LLMExplainer",
    "GroundedExplainer",
    "EvidenceBuilder",
    "PromptBuilder",
    "HallucinationGuardrail",
    "FallbackExplainer",
    "BaseLLMClient",
    "OpenAILLMClient",
    "MockLLMClient",
]
