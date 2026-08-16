"""Product document representation and embedding text ablation variants.

Critical Design Decision:
Numerical business signals such as price, average star ratings, review counts, and sales rank
are deliberately EXCLUDED from the primary semantic embedding text.
Embedding spaces should represent intrinsic product semantic identity, functional use-case,
and physical attributes rather than dynamic business/popularity signals. Numerical signals
are reserved for downstream multi-stage hybrid ranking (Stage 3).
"""

from typing import Any, Dict, List, Optional
from enum import Enum
import numpy as np


class TextRepresentationVariant(str, Enum):
    """Supported text representation ablation variants for semantic embedding."""
    
    TITLE_BRAND_CATEGORY = "title_brand_category"
    TITLE_BRAND_CATEGORY_FEATURES = "title_brand_category_features"
    TITLE_BRAND_CATEGORY_FEATURES_DESCRIPTION = "title_brand_category_features_description"
    FULL = "title_brand_category_features_description"


def build_product_text(
    product: Dict[str, Any],
    variant: str = "title_brand_category_features_description"
) -> str:
    """Deterministically serializes a product record into an embedding-ready document string.
    
    Supports ablation variants:
      - 'title_brand_category': Minimal concise representation.
      - 'title_brand_category_features': Adds structured feature bullets.
      - 'title_brand_category_features_description': Full detailed specification text.
    
    Args:
        product: Cleaned product dictionary containing title, brand, categories, features, description.
        variant: String identifier of text ablation representation.
        
    Returns:
        Structured, label-delimited product document representation string.
    """
    variant_norm = variant.lower().strip()
    
    # 1. Title (always included)
    title = str(product.get("title") or "").strip()
    if not title:
        title = "Unknown Product"
        
    sections: List[str] = [f"Title: {title}"]
    
    # 2. Brand
    brand = product.get("brand")
    if brand and str(brand).strip():
        sections.append(f"Brand: {str(brand).strip()}")
        
    # 3. Category Hierarchy
    categories = product.get("categories")
    if categories is not None:
        if isinstance(categories, (list, tuple, np.ndarray)):
            cat_str = " > ".join([str(c).strip() for c in categories if str(c).strip()])
        else:
            cat_str = str(categories).strip()
        if cat_str:
            sections.append(f"Category: {cat_str}")

    # If minimal variant A requested, return early
    if variant_norm in ("title_brand_category", "variant_a", "minimal"):
        return "\n\n".join(sections).strip()

    # 4. Features (Variants B & C)
    features = product.get("features")
    if features is not None:
        if isinstance(features, (list, tuple, np.ndarray)):
            valid_feats = [str(f).strip() for f in features if str(f).strip()]
            if valid_feats:
                feat_text = "\n".join(f"- {f}" for f in valid_feats)
                sections.append(f"Features:\n{feat_text}")
        elif isinstance(features, str) and features.strip():
            sections.append(f"Features:\n{features.strip()}")

    # If variant B requested, return here
    if variant_norm in ("title_brand_category_features", "variant_b"):
        return "\n\n".join(sections).strip()

    # 5. Description (Variant C / Full)
    description = product.get("description")
    if description and str(description).strip():
        sections.append(f"Description:\n{str(description).strip()}")

    return "\n\n".join(sections).strip()


# Alias for backwards compatibility
build_product_document = build_product_text

