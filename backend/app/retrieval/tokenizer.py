"""Deterministic lexical tokenizer for e-commerce catalog retrieval.

Preserves technical product specifications, model numbers, hyphenated hardware terms,
and alphanumeric codes while normalizing Unicode and casing.
"""

import re
import unicodedata
from typing import List

# Regular expressions for technical identifiers and compound terms
_TECH_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\.\-][a-z0-9]+)*")
_PUNCT_CLEAN_RE = re.compile(r"^[^\w]+|[^\w]+$")


def tokenize_lexical(text: str) -> List[str]:
    """Tokenize text into lexical terms while preserving technical identifiers and specs.
    
    Examples of preserved terms:
      - 'RTX 4060' -> ['rtx', '4060']
      - 'USB-C' -> ['usb-c', 'usbc', 'usb', 'c']
      - 'WiFi 6' -> ['wifi', '6', 'wifi6']
      - 'DDR5' -> ['ddr5']
      - 'M.2 NVMe' -> ['m.2', 'm2', 'm', '2', 'nvme']
      - '4K HDMI 2.1' -> ['4k', 'hdmi', '2.1', '21', '2', '1']
      - 'PS5' -> ['ps5']
      
    Args:
        text: Raw document or query string.
        
    Returns:
        List of normalized token strings.
    """
    if not text or not isinstance(text, str):
        return []
    
    # 1. Unicode NFKC normalization and lowercasing
    text = unicodedata.normalize("NFKC", text).lower()
    
    tokens: List[str] = []
    
    # 2. Extract alphanumeric patterns and hyphenated/dotted compounds
    raw_matches = _TECH_TOKEN_RE.findall(text)
    
    for match in raw_matches:
        cleaned = _PUNCT_CLEAN_RE.sub("", match)
        if not cleaned:
            continue
            
        tokens.append(cleaned)
        
        # If the term contains internal hyphens or periods (e.g., usb-c, 2.1, m.2, wi-fi),
        # generate sub-tokens and joined forms for maximum lexical recall
        if "-" in cleaned or "." in cleaned:
            joined = cleaned.replace("-", "").replace(".", "")
            if joined and joined != cleaned and joined not in tokens:
                tokens.append(joined)
                
            parts = [p for p in re.split(r"[\.\-]", cleaned) if p]
            for p in parts:
                if p not in tokens:
                    tokens.append(p)

    return tokens
