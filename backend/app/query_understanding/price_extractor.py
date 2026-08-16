"""Deterministic price constraint extractor supporting ranges, ceilings, floors, and currency detection."""

import re
from typing import Optional, Tuple

DEFAULT_CURRENCY = "USD"


class PriceExtractor:
    """Extracts min/max budget boundaries and currency from normalized search queries."""

    def __init__(self, default_currency: str = DEFAULT_CURRENCY) -> None:
        self.default_currency = default_currency

    def extract(self, normalized_query: str) -> Tuple[Optional[float], Optional[float], str, str]:
        """Extract price_min, price_max, currency, and query string with price clause stripped.
        
        Returns:
            Tuple of (price_min, price_max, currency, stripped_query)
        """
        price_min: Optional[float] = None
        price_max: Optional[float] = None
        currency = self.default_currency

        # Detect currency mentions explicitly
        if re.search(r"\b(?:inr|rupees?|rs)\b|₹", normalized_query, re.IGNORECASE):
            currency = "INR"
        elif re.search(r"\b(?:usd|dollars?|bucks)\b|\$", normalized_query, re.IGNORECASE):
            currency = "USD"
        elif re.search(r"\b(?:eur|euros?)\b|€", normalized_query, re.IGNORECASE):
            currency = "EUR"
        elif re.search(r"\b(?:gbp|pounds?)\b|£", normalized_query, re.IGNORECASE):
            currency = "GBP"

        cleaned_query = normalized_query

        # Pattern 1: Between X and Y / From X to Y / X - Y range
        # e.g., 'between 500 and 1000', 'from 30000 to 50000 inr', '500 to 1000 usd', '500-1000'
        range_match = re.search(
            r"\b(?:between|from)?\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\s*(\d+(?:\.\d+)?)\s*(?:and|to|\-|–)\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\s*(\d+(?:\.\d+)?)\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\b",
            cleaned_query,
            re.IGNORECASE,
        )
        if range_match:
            val1 = float(range_match.group(1))
            val2 = float(range_match.group(2))
            price_min = min(val1, val2)
            price_max = max(val1, val2)
            cleaned_query = cleaned_query[:range_match.start()] + " " + cleaned_query[range_match.end():]
            cleaned_query = re.sub(r"\b(?:inr|usd|eur|gbp|rupees?|rs|dollars?)\b|\$|₹|€|£", " ", cleaned_query, flags=re.IGNORECASE)
            return price_min, price_max, currency, " ".join(cleaned_query.split())

        # Pattern 2: Max price / Price ceiling
        # e.g. 'under 800', 'under $800', 'below 50000 inr', 'less than 400', 'up to 600', 'max 800', 'budget 500'
        max_match = re.search(
            r"\b(?:under|below|less than|up to|maximum|max|within|budget(?:\s+of)?)\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\s*(\d+(?:\.\d+)?)\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\b",
            cleaned_query,
            re.IGNORECASE,
        )
        if max_match:
            price_max = float(max_match.group(1))
            cleaned_query = cleaned_query[:max_match.start()] + " " + cleaned_query[max_match.end():]

        # Pattern 3: Min price / Price floor
        # e.g. 'above 500', 'above $500', 'over 300', 'more than 15000 inr', 'minimum 250', 'min 100', 'at least 200'
        min_match = re.search(
            r"\b(?:above|over|more than|minimum|min|at least)\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\s*(\d+(?:\.\d+)?)\s*(?:inr|usd|eur|gbp|rupees?|rs|dollars?|\$|₹|€|£)?\b",
            cleaned_query,
            re.IGNORECASE,
        )
        if min_match:
            price_min = float(min_match.group(1))
            cleaned_query = cleaned_query[:min_match.start()] + " " + cleaned_query[min_match.end():]

        # Clean leftover currency tokens if isolated
        cleaned_query = re.sub(r"\b(?:inr|usd|eur|gbp|rupees?|rs|dollars?)\b|\$|₹|€|£", " ", cleaned_query, flags=re.IGNORECASE)
        cleaned_query = " ".join(cleaned_query.split())

        return price_min, price_max, currency, cleaned_query
