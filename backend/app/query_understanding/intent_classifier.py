"""Rule-based query intent classifier with extensibility for future ML/LLM models."""

import re
from typing import Dict, List, Optional


class QueryIntentClassifier:
    """Classifies user search intent into standard e-commerce interaction archetypes."""

    COMPARISON_PATTERNS = [
        r"\bvs\b",
        r"\bversus\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifference between\b",
        r"\bor\b",
    ]

    RECOMMENDATION_PATTERNS = [
        r"\bbest\b",
        r"\btop\b",
        r"\btop rated\b",
        r"\brecommend\b",
        r"\brecommended\b",
        r"\bsuggest\b",
        r"\badvice\b",
        r"\bideal\b",
        r"\bguide\b",
        r"\bgift\b",
    ]

    BRAND_SEARCH_PATTERNS = [
        r"\bstore\b",
        r"\bofficial\b",
        r"\bproducts\b",
        r"\baccessories\b",
        r"\bbrand\b",
    ]

    def classify(
        self,
        query: str,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        attributes: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Classify search intent based on structured entities and lexical signals."""
        attributes = attributes or {}
        q_lower = query.lower()

        # 1. Product comparison (e.g., 'sony xm5 vs bose qc45')
        for pat in self.COMPARISON_PATTERNS:
            if re.search(pat, q_lower):
                return "product_comparison"

        # 2. Price constrained search (e.g., 'laptop under 80000')
        if price_min is not None or price_max is not None:
            return "price_constrained_search"

        # 3. Recommendation query (e.g., 'best wireless headphones for travel')
        for pat in self.RECOMMENDATION_PATTERNS:
            if re.search(pat, q_lower):
                return "recommendation"

        # 4. Brand search (e.g., 'bose official store products', 'sony store')
        if brand and not category:
            for pat in self.BRAND_SEARCH_PATTERNS:
                if re.search(pat, q_lower):
                    return "brand_search"
            if len(q_lower.split()) <= 4:
                return "brand_search"

        # 5. Technical Attribute Search (e.g., 'rtx 4060 16gb ram 1tb ssd', 'usb-c and usb-a hub with usb 3.0')
        has_tech_specs = bool(
            attributes.get("gpu") or attributes.get("ram") or attributes.get("storage") or len(attributes.get("connectivity", [])) >= 2
        )
        if has_tech_specs and not category:
            return "attribute_search"

        # 6. Default Product Search
        return "product_search"
