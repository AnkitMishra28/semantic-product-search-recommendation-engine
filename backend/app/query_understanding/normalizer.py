"""Query Normalization routines for cleaning, punctuation handling, and shorthand expansion."""

import re
import unicodedata

# Common spelling variants and synonym mappings
SPELLING_CORRECTIONS = {
    "head phones": "headphones",
    "head phone": "headphones",
    "ear buds": "earbuds",
    "ear bud": "earbuds",
    "lap top": "laptop",
    "lap tops": "laptop",
    "blutooth": "bluetooth",
    "blue tooth": "bluetooth",
    "wi-fi": "wifi",
    "sound bar": "soundbar",
    "powerbank": "power bank",
    "powerbanks": "power bank",
    "powerstrip": "power strip",
    "powerstrips": "power strip",
    "moniter": "monitor",
    "key board": "keyboard",
    "web cam": "webcam",
    "web cams": "webcam",
    "smart watch": "smartwatch",
    "smart watches": "smartwatch",
    "mic": "microphone",
}


class QueryNormalizer:
    """Normalizes raw natural language queries for deterministic parsing and embedding."""

    def __init__(self) -> None:
        self.spelling_map = SPELLING_CORRECTIONS

    def normalize_price_shorthand(self, text: str) -> str:
        """Convert numeric shorthand like '80k', '80 thousand', '₹80k', '$800' to standard digits."""
        # 1. Handle currency symbols
        text = text.replace("₹", " inr ").replace("$", " usd ").replace("€", " eur ").replace("£", " gbp ")

        # 2. Convert 'X thousand' -> 'X000' (e.g., '80 thousand' -> '80000')
        text = re.sub(
            r"\b(\d+(?:\.\d+)?)\s*thousand\b",
            lambda m: str(int(float(m.group(1)) * 1000)),
            text,
            flags=re.IGNORECASE,
        )

        # 3. Convert 'Xk' -> 'X000' (e.g., '80k' -> '80000', '1.5k' -> '1500')
        text = re.sub(
            r"\b(\d+(?:\.\d+)?)\s*k\b",
            lambda m: str(int(float(m.group(1)) * 1000)),
            text,
            flags=re.IGNORECASE,
        )

        # 4. Strip commas in numbers (e.g., '50,000' -> '50000')
        text = re.sub(r"(?<=\d),(?=\d)", "", text)

        return text

    def normalize(self, query: str) -> str:
        """Fully normalize input query."""
        if not query:
            return ""

        # 1. Unicode NFKC normalization
        norm = unicodedata.normalize("NFKC", str(query).strip())

        # 2. Lowercase
        norm = norm.lower()

        # 3. Expand price shorthands
        norm = self.normalize_price_shorthand(norm)

        # 4. Clean punctuation while preserving alphanumeric, hyphens, and whitespace
        norm = re.sub(r"[^\w\s\-\.]", " ", norm)

        # 5. Fix isolated periods (keep decimal numbers like 14.5, remove trailing/isolated dots)
        norm = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", norm)

        # 6. Apply spelling corrections
        for wrong, right in self.spelling_map.items():
            pattern = r"\b" + re.escape(wrong) + r"\b"
            norm = re.sub(pattern, right, norm)

        # 7. Collapse whitespace
        norm = " ".join(norm.split())

        return norm
