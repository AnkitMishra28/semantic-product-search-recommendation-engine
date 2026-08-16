"""Production-Grade Hybrid Query Understanding Pipeline for E-Commerce Search."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.models.search import QueryIntent, QueryUnderstandingResult
from backend.app.query_understanding.base import BaseQueryProcessor
from backend.app.query_understanding.intent_classifier import QueryIntentClassifier
from backend.app.query_understanding.normalizer import QueryNormalizer
from backend.app.query_understanding.price_extractor import DEFAULT_CURRENCY, PriceExtractor
from backend.app.query_understanding.vocabulary import (
    ATTRIBUTE_PATTERNS,
    CATALOG_BRANDS,
    CATEGORY_SYNONYMS,
    MULTI_WORD_BRANDS,
)

logger = logging.getLogger(__name__)


class QueryUnderstandingPipeline(BaseQueryProcessor):
    """Hybrid, catalog-aware query understanding pipeline.
    
    Orchestrates:
    1. Deterministic normalization & spelling correction
    2. Budget & currency extraction (defaulting to USD canonical catalog currency)
    3. Multi-word and single-word brand matching
    4. Catalog category and synonym resolution
    5. Boundary-aware, longest-match non-overlapping technical attribute extraction
    6. Intent classification
    7. Hard filter vs. soft ranking signal separation
    8. Heuristic parsing confidence scoring
    """

    def __init__(self, default_currency: str = DEFAULT_CURRENCY) -> None:
        self.default_currency = default_currency
        self.normalizer = QueryNormalizer()
        self.price_extractor = PriceExtractor(default_currency=default_currency)
        self.intent_classifier = QueryIntentClassifier()

        # Pre-sort category synonyms by descending string length
        self.sorted_category_phrases: List[Tuple[str, str]] = sorted(
            CATEGORY_SYNONYMS.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

    def extract_brand(self, text: str) -> Optional[str]:
        """Extract recognized catalog brand with boundary awareness."""
        # Check multi-word brands first (e.g. 'amazon basics', 'western digital', 'audio-technica')
        for m_brand in MULTI_WORD_BRANDS:
            pattern = r"(?<![\w\-])" + re.escape(m_brand) + r"(?![\w\-])"
            if re.search(pattern, text, re.IGNORECASE):
                return m_brand.title()

        # Check single-word brands
        tokens = text.lower().split()
        for token in tokens:
            if token in CATALOG_BRANDS:
                return token.title()

        return None

    def extract_category(self, text: str) -> Tuple[Optional[str], bool]:
        """Extract canonical catalog category via synonym resolution, prioritizing earliest head noun.
        
        Returns:
            Tuple of (canonical_category, is_synonym_match)
        """
        text_lower = text.lower()
        matches: List[Tuple[int, int, str, str]] = []  # (start_index, -phrase_length, phrase, canonical)

        for phrase, canonical in self.sorted_category_phrases:
            pattern = r"(?<![\w\-])" + re.escape(phrase) + r"(?![\w\-])"
            m = re.search(pattern, text_lower)
            if m:
                matches.append((m.start(), -len(phrase), phrase, canonical))

        if matches:
            # Sort by earliest start position in query, then longest phrase match
            matches.sort(key=lambda x: (x[0], x[1]))
            best_phrase = matches[0][2]
            best_canonical = matches[0][3]
            is_synonym = (best_phrase != best_canonical)
            return best_canonical, is_synonym

        return None, False

    def extract_attributes(self, text: str) -> Dict[str, List[str]]:
        """Extract technical specifications using boundary-aware, longest-match non-overlapping parsing."""
        text_lower = text.lower()
        extracted: Dict[str, List[str]] = {}

        for attr_category, patterns in ATTRIBUTE_PATTERNS.items():
            # Sort keys by descending length to match specific specs first (e.g. 'rtx 4060' before 'rtx', 'usb-c' before 'usb')
            sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)
            matched_vals: Set[str] = set()
            consumed_spans: List[Tuple[int, int]] = []

            for pat_key, canonical_val in sorted_patterns:
                pattern = re.compile(r"(?<![\w\-])" + re.escape(pat_key) + r"(?![\w\-])", re.IGNORECASE)
                for m in pattern.finditer(text_lower):
                    s, e = m.start(), m.end()
                    # Check if this match overlaps with an already consumed longer match
                    overlap = any(not (e <= cs or s >= ce) for cs, ce in consumed_spans)
                    if not overlap:
                        consumed_spans.append((s, e))
                        matched_vals.add(canonical_val)

            if matched_vals:
                extracted[attr_category] = sorted(list(matched_vals))

        return extracted

    def compute_confidence(
        self,
        category: Optional[str],
        is_synonym: bool,
        brand: Optional[str],
        price_min: Optional[float],
        price_max: Optional[float],
        attributes: Dict[str, List[str]],
        raw_query: str,
    ) -> float:
        """Calculate documented heuristic parsing confidence score.
        
        Rules:
        - Completely empty query: 1.0 (trivial parsing)
        - Multi-signal match (e.g. Category + Brand, Category + Price, or Category + Attributes): 1.0 (or 0.95 with synonym)
        - Single primary entity (Category only): 0.90 (or 0.85 with synonym)
        - Single brand only: 0.85
        - Price constraint only: 0.85
        - Attributes only without Category/Brand: 0.80
        - Completely ungrounded / unsupported terms: 0.60
        """
        if not raw_query or not raw_query.strip():
            return 1.0

        signals = 0
        if category:
            signals += 1
        if brand:
            signals += 1
        if price_min is not None or price_max is not None:
            signals += 1
        if attributes:
            signals += len(attributes)

        if signals >= 2:
            base_conf = 1.0
        elif category:
            base_conf = 0.90
        elif brand:
            base_conf = 0.85
        elif price_min is not None or price_max is not None:
            base_conf = 0.85
        elif attributes:
            base_conf = 0.80
        else:
            base_conf = 0.60

        if is_synonym and base_conf > 0.60:
            base_conf -= 0.05

        return float(round(base_conf, 2))

    def process_to_intent(self, raw_query: str) -> QueryIntent:
        """Process natural language query into structured QueryIntent."""
        if not raw_query or not raw_query.strip():
            return QueryIntent(
                raw_query="",
                normalized_query="",
                intent="product_search",
                currency=self.default_currency,
                confidence=1.0,
            )

        # 1. Normalization
        normalized = self.normalizer.normalize(raw_query)

        # 2. Price Extraction
        price_min, price_max, currency, stripped_query = self.price_extractor.extract(normalized)

        # 3. Brand Extraction
        brand = self.extract_brand(normalized)

        # 4. Category Extraction
        category, is_synonym = self.extract_category(normalized)

        # 5. Attribute Extraction (boundary-aware non-overlapping)
        attributes = self.extract_attributes(normalized)

        # 6. Intent Classification
        intent = self.intent_classifier.classify(
            query=normalized,
            category=category,
            brand=brand,
            price_min=price_min,
            price_max=price_max,
            attributes=attributes,
        )

        # 7. Hard vs. Soft Filter Separation
        hard_filters: Dict[str, Any] = {}
        if price_max is not None:
            hard_filters["price_max"] = price_max
        if price_min is not None:
            hard_filters["price_min"] = price_min
        if brand is not None:
            hard_filters["brand"] = brand
        if category is not None:
            hard_filters["category"] = category

        soft_signals: Dict[str, Any] = {}
        for attr_cat, values in attributes.items():
            if attr_cat in ("use_case", "features", "connectivity"):
                soft_signals[attr_cat] = values

        # 8. Heuristic Confidence
        confidence = self.compute_confidence(
            category=category,
            is_synonym=is_synonym,
            brand=brand,
            price_min=price_min,
            price_max=price_max,
            attributes=attributes,
            raw_query=raw_query,
        )

        # 9. Form grouped entities for diagnostics
        entities: Dict[str, List[str]] = {
            "brands": [brand.lower()] if brand else [],
            "categories": [category] if category else [],
            "modifiers": soft_signals.get("use_case", []) + soft_signals.get("features", []),
        }

        return QueryIntent(
            raw_query=raw_query,
            normalized_query=normalized,
            intent=intent,
            category=category,
            brand=brand,
            price_min=price_min,
            price_max=price_max,
            currency=currency,
            attributes=attributes,
            entities=entities,
            hard_filters=hard_filters,
            soft_signals=soft_signals,
            confidence=confidence,
        )

    def process(self, raw_query: str) -> QueryUnderstandingResult:
        """Process natural language query and return QueryUnderstandingResult for API compatibility."""
        intent_obj = self.process_to_intent(raw_query)

        # Build query expansions if useful
        expansions: List[str] = []
        if intent_obj.category and intent_obj.brand:
            expansions.append(f"{intent_obj.brand.lower()} {intent_obj.category}")
        if intent_obj.attributes.get("features"):
            for f in intent_obj.attributes["features"]:
                if f == "noise_cancelling" and "anc" not in intent_obj.normalized_query:
                    expansions.append(intent_obj.normalized_query.replace("noise cancelling", "anc"))

        return QueryUnderstandingResult(
            raw_query=intent_obj.raw_query,
            normalized_query=intent_obj.normalized_query,
            intent=intent_obj.intent,
            category=intent_obj.category,
            brand=intent_obj.brand,
            price_min=intent_obj.price_min,
            price_max=intent_obj.price_max,
            currency=intent_obj.currency,
            attributes=intent_obj.attributes,
            detected_entities=intent_obj.entities,
            expanded_queries=expansions,
            hard_filters=intent_obj.hard_filters,
            soft_signals=intent_obj.soft_signals,
            confidence=intent_obj.confidence,
        )
