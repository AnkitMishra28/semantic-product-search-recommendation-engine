"""Deterministic product quality scoring, deduplication, and stratified sampling."""

import random
from typing import Any, Dict, Iterator, List, Optional, Tuple
from backend.app.preprocessing.cleaners import (
    clean_brand,
    clean_categories,
    clean_description,
    clean_features,
    clean_text,
    extract_images,
    parse_price,
)
from backend.app.preprocessing.product_document import build_product_text


def compute_product_quality_score(product: Dict[str, Any]) -> float:
    """Compute a deterministic information-richness score (0.0 to 8.0) for a product.
    
    Weights emphasize fields critical for semantic indexing and multi-stage ranking:
      - Title presence and depth: 1.0
      - Category hierarchy: 1.0
      - Brand presence: 1.0
      - Bullet features: 1.5
      - Detailed description: 1.5
      - Valid price: 1.0
      - Review/Rating volume: 1.0
    """
    score = 0.0
    
    title = product.get("title", "")
    if title:
        if len(title) >= 15:
            score += 1.0
        elif len(title) >= 5:
            score += 0.5
            
    cats = product.get("categories", [])
    if cats and len(cats) >= 1:
        score += 1.0
        
    brand = product.get("brand")
    if brand and str(brand).strip():
        score += 1.0
        
    features = product.get("features", [])
    if features and len(features) >= 1:
        if len(features) >= 3:
            score += 1.5
        else:
            score += 1.0
            
    description = product.get("description", "")
    if description:
        if len(description) >= 50:
            score += 1.5
        elif len(description) >= 15:
            score += 1.0
        else:
            score += 0.5
            
    price = product.get("price")
    if price is not None and isinstance(price, (int, float)) and price > 0:
        score += 1.0
        
    rating_count = product.get("rating_number") or product.get("rating_count") or 0
    avg_rating = product.get("average_rating")
    if rating_count > 0 and avg_rating is not None and avg_rating > 0:
        score += 1.0
        
    return score


def clean_raw_product_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean a single raw Amazon 2023 metadata record into standardized schema.
    
    Returns None if missing required canonical parent_asin or title.
    """
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not parent_asin or not isinstance(parent_asin, str):
        return None
    parent_asin = parent_asin.strip()
    if not parent_asin:
        return None
        
    raw_title = raw.get("title")
    title = clean_text(raw_title)
    if not title or len(title) < 3:
        return None
        
    store = raw.get("store")
    details = raw.get("details") if isinstance(raw.get("details"), dict) else None
    brand = clean_brand(store, details)
    
    categories = clean_categories(raw.get("categories"), raw.get("main_category"))
    features = clean_features(raw.get("features"))
    description = clean_description(raw.get("description"))
    price = parse_price(raw.get("price"))
    
    # Rating numbers
    avg_rating = raw.get("average_rating")
    try:
        avg_rating = float(avg_rating) if avg_rating is not None else None
        if avg_rating is not None and not (1.0 <= avg_rating <= 5.0):
            avg_rating = None
    except (ValueError, TypeError):
        avg_rating = None
        
    rating_num = raw.get("rating_number") or raw.get("rating_count") or 0
    try:
        rating_num = int(rating_num) if rating_num is not None else 0
        if rating_num < 0:
            rating_num = 0
    except (ValueError, TypeError):
        rating_num = 0
        
    # Images
    primary_img, all_imgs = extract_images(raw.get("images"))
    
    # Co-purchase bought_together
    bought_together = []
    bt_raw = raw.get("bought_together")
    if isinstance(bt_raw, list):
        bought_together = [str(x).strip() for x in bt_raw if str(x).strip()]
    elif isinstance(bt_raw, str) and bt_raw.strip():
        bought_together = [bt_raw.strip()]

    cleaned = {
        "parent_asin": parent_asin,
        "title": title,
        "brand": brand,
        "categories": categories,
        "description": description,
        "features": features,
        "price": price,
        "average_rating": avg_rating,
        "rating_number": rating_num,
        "image_url": primary_img,
        "images": all_imgs,
        "bought_together": bought_together,
    }
    
    # Calculate quality score for deduplication and ranking
    cleaned["quality_score"] = compute_product_quality_score(cleaned)
    return cleaned


def sample_and_deduplicate_products(
    products_iter: Iterator[Dict[str, Any]],
    target_size: int = 60000,
    min_quality_score: float = 2.5,
    text_variant: str = "title_brand_category_features_description",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deduplicate raw products on canonical parent_asin, keeping highest quality entry,
    and deterministically select a target subset prioritizing rich metadata.
    
    Args:
        products_iter: Stream/iterator of raw product dictionaries.
        target_size: Desired number of products in development subset (e.g. 50,000 - 100,000).
        min_quality_score: Minimum information-richness score threshold.
        text_variant: Text representation variant for embedding_text.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        List of cleaned, validated, embedding-ready product dictionaries.
    """
    seen_products: Dict[str, Dict[str, Any]] = {}
    
    for raw in products_iter:
        cleaned = clean_raw_product_record(raw)
        if not cleaned:
            continue
        
        asin = cleaned["parent_asin"]
        if asin in seen_products:
            # Keep the record with higher quality score or higher rating count
            prev = seen_products[asin]
            if (cleaned["quality_score"], cleaned["rating_number"]) > (prev["quality_score"], prev["rating_number"]):
                seen_products[asin] = cleaned
        else:
            seen_products[asin] = cleaned

    all_products = list(seen_products.values())
    
    # Filter by minimum quality score if we have sufficient records
    high_quality = [p for p in all_products if p["quality_score"] >= min_quality_score]
    candidates = high_quality if len(high_quality) >= target_size else all_products

    # Deterministic selection:
    # Sort deterministically by (quality_score DESC, rating_number DESC, parent_asin ASC)
    candidates.sort(
        key=lambda p: (
            -p["quality_score"],
            -p["rating_number"],
            p["parent_asin"]
        )
    )
    
    selected = candidates[:target_size]
    
    # Shuffle deterministically with fixed seed so ordering is not biased by source order
    rng = random.Random(seed)
    rng.shuffle(selected)
    
    # Generate final embedding_text for each selected product
    for p in selected:
        p["embedding_text"] = build_product_text(p, variant=text_variant)
        # Drop temporary quality_score before storing if desired or keep in metadata
        
    return selected
