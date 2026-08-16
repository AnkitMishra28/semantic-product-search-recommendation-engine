"""Robust text, price, brand, category, and metadata cleaning utilities."""

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# Regex patterns for HTML tag stripping and whitespace normalization
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPACING_RE = re.compile(r"\s+([.,!?:;])")
_PRICE_RE = re.compile(r"[\$£€¥]?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)")
_BRAND_PREFIX_RE = re.compile(r"^(?:visit the|brand:\s*|manufacturer:\s*)", re.IGNORECASE)
_BRAND_SUFFIX_RE = re.compile(r"\s+store$", re.IGNORECASE)

# Generic boilerplate phrases to suppress in descriptions
_BOILERPLATE_PHRASES = {
    "no description available",
    "no description",
    "n/a",
    "none",
    "null",
    "[null]",
    "see description for details",
}


def clean_text(text: Optional[str]) -> str:
    """Normalize raw text: unescape HTML entities, strip tags, NFC normalize, and collapse whitespace.
    
    Note: Does NOT perform stemming or aggressive stop-word removal to preserve
    semantic context for downstream transformer embeddings.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML entities (&amp;, &quot;, &#39;, &nbsp;, etc.)
    text = html.unescape(text)
    
    # Strip HTML tags (replace with space to avoid fusing adjacent words)
    text = _HTML_TAG_RE.sub(" ", text)
    
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Fix whitespace preceding punctuation (e.g. "word ." -> "word.")
    text = _PUNCT_SPACING_RE.sub(r"\1", text)

    # Collapse multiple whitespace characters into single space
    text = _WHITESPACE_RE.sub(" ", text).strip()
    
    return text


def parse_price(raw_price: Any) -> Optional[float]:
    """Parse raw price from numeric, currency string, or range into float USD.
    
    Returns None if missing, malformed, non-positive, or unreasonably large (> 100,000).
    """
    if raw_price is None:
        return None
    
    if isinstance(raw_price, (int, float)):
        val = float(raw_price)
        return val if 0.0 < val < 100000.0 else None
    
    if isinstance(raw_price, str):
        cleaned = raw_price.strip()
        if not cleaned or cleaned.lower() in ("n/a", "none", "null", "$"):
            return None
        
        match = _PRICE_RE.search(cleaned)
        if match:
            try:
                num_str = match.group(1).replace(",", "")
                val = float(num_str)
                return val if 0.0 < val < 100000.0 else None
            except ValueError:
                return None
                
    return None


def clean_brand(store: Optional[str], details: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extract and normalize brand name from store field or metadata details dictionary."""
    candidate = None
    
    # 1. Check store field
    if store and isinstance(store, str):
        s = store.strip()
        s = _BRAND_PREFIX_RE.sub("", s).strip()
        s = _BRAND_SUFFIX_RE.sub("", s).strip()
        if s and s.lower() not in ("generic", "unknown", "n/a", "none", "null", ""):
            candidate = s
            
    # 2. Check details dictionary if store is empty or generic
    if not candidate and isinstance(details, dict):
        for key in ("Brand", "brand", "Manufacturer", "manufacturer", "Brand Name"):
            val = details.get(key)
            if val and isinstance(val, str):
                s = clean_text(val)
                s = _BRAND_PREFIX_RE.sub("", s).strip()
                s = _BRAND_SUFFIX_RE.sub("", s).strip()
                if s and s.lower() not in ("generic", "unknown", "n/a", "none", "null", ""):
                    candidate = s
                    break

    if candidate:
        candidate = clean_text(candidate)
        return candidate if len(candidate) > 0 else None
    return None


def clean_categories(categories: Any, main_category: Optional[str] = None) -> List[str]:
    """Clean and normalize category hierarchy list."""
    cleaned_cats: List[str] = []
    
    if isinstance(categories, list):
        for c in categories:
            if isinstance(c, str):
                cleaned_c = clean_text(c)
                if cleaned_c and cleaned_c not in cleaned_cats:
                    cleaned_cats.append(cleaned_c)
    elif isinstance(categories, str):
        cleaned_c = clean_text(categories)
        if cleaned_c:
            parts = [p.strip() for p in re.split(r"[>,/]", cleaned_c) if p.strip()]
            cleaned_cats = parts if parts else [cleaned_c]

    # Fallback to main_category if categories list is empty
    if not cleaned_cats and main_category and isinstance(main_category, str):
        cleaned_main = clean_text(main_category)
        if cleaned_main:
            cleaned_cats.append(cleaned_main)
            
    return cleaned_cats


def clean_features(features: Any) -> List[str]:
    """Clean bullet point features, stripping HTML and blank/noise bullets."""
    cleaned: List[str] = []
    if isinstance(features, list):
        for f in features:
            if isinstance(f, str):
                cf = clean_text(f)
                if cf and len(cf) > 2 and cf.lower() not in _BOILERPLATE_PHRASES:
                    cleaned.append(cf)
    elif isinstance(features, str):
        cf = clean_text(features)
        if cf and len(cf) > 2 and cf.lower() not in _BOILERPLATE_PHRASES:
            cleaned.append(cf)
    return cleaned


def clean_description(description: Any) -> str:
    """Clean description, joining multiple paragraph blocks and stripping boilerplate."""
    if not description:
        return ""
    
    if isinstance(description, list):
        paragraphs = []
        for p in description:
            if isinstance(p, str):
                cp = clean_text(p)
                if cp and len(cp) > 5 and cp.lower() not in _BOILERPLATE_PHRASES:
                    paragraphs.append(cp)
        return "\n\n".join(paragraphs).strip()
    
    if isinstance(description, str):
        cd = clean_text(description)
        if cd.lower() in _BOILERPLATE_PHRASES:
            return ""
        return cd
        
    return ""


def extract_images(images: Any) -> Tuple[Optional[str], List[str]]:
    """Extract primary thumbnail/large image URL and list of all valid image URLs."""
    primary_url = None
    all_urls: List[str] = []
    
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                # Prefer large/hi_res, fallback to thumb
                u = img.get("large") or img.get("hi_res") or img.get("thumb")
                if u and isinstance(u, str) and u.startswith("http"):
                    all_urls.append(u)
                    if not primary_url:
                        primary_url = u
            elif isinstance(img, str) and img.startswith("http"):
                all_urls.append(img)
                if not primary_url:
                    primary_url = img
    elif isinstance(images, str) and images.startswith("http"):
        primary_url = images
        all_urls.append(images)
        
    return primary_url, all_urls
