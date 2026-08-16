"""Prompt construction for evidence-grounded, zero-hallucination LLM product explanations."""

import json
from typing import Dict, Any
from backend.app.models.explanation import ProductEvidence


class PromptBuilder:
    """Constructs strict, factual grounding prompts for LLM explanation generation."""

    SYSTEM_INSTRUCTIONS = """You are a strictly grounded AI product explanation engine for an e-commerce platform.
Your task is to generate a concise, user-friendly explanation answering: "Why is this product recommended for this query or user?"

STRICT FACTUAL GROUNDING RULES:
1. Use ONLY the factual evidence provided in the input JSON.
2. NEVER invent, hallucinate, or infer product specifications, battery life, performance claims, warranty, discounts, or shipping information not explicitly present in the evidence.
3. If a requested query constraint or feature is absent or listed under 'missing_constraints', you MUST explicitly acknowledge that the information is unavailable in the catalog evidence. DO NOT fabricate a value.
4. Output MUST be valid JSON adhering strictly to the schema below.

OUTPUT JSON SCHEMA:
{
  "product_id": "<string>",
  "summary": "<1-2 sentence user-facing summary>",
  "reasons": [
    {
      "type": "<constraint_match | semantic_relevance | ranking_strength | personalization | rating_credibility | unsupported_constraint>",
      "label": "<short label like 'GPU', 'Budget', 'Brand', 'Category', 'User History', 'Missing Info'>",
      "text": "<concise explanation sentence>",
      "evidence": "<exact snippet quoted from product features or metadata>",
      "is_matched": <true or false>
    }
  ],
  "semantic_match_score": <float or null>,
  "grounded": true,
  "warnings": ["<list of missing constraint warnings if any>"],
  "generation_method": "llm"
}
"""

    @classmethod
    def build_prompt(cls, evidence: ProductEvidence) -> Dict[str, str]:
        """Build system and user messages from a ProductEvidence payload."""
        evidence_dict = evidence.model_dump(exclude_none=True)
        evidence_json = json.dumps(evidence_dict, indent=2)

        user_content = f"""Please generate a grounded explanation for the following product evidence:

```json
{evidence_json}
```

Remember: Use ONLY the provided evidence. Do NOT invent missing attributes or specs. Output ONLY the valid JSON response."""

        return {
            "system": cls.SYSTEM_INSTRUCTIONS,
            "user": user_content,
        }
