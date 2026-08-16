import json
import re
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


# Curated list of realistic evaluation search query specifications across diverse intent types
CURATED_QUERY_TEMPLATES = [
    {
        "query_id": "q_001",
        "query": "noise cancelling bluetooth headphones for travel",
        "intent_type": "semantic_use_case",
        "category": "Headphones",
        "keywords": ["noise cancelling", "headphones", "bluetooth", "wireless"],
        "min_matches": 3,
    },
    {
        "query_id": "q_002",
        "query": "mechanical keyboard for programming with quiet switches",
        "intent_type": "attribute_use_case",
        "category": "Keyboards",
        "keywords": ["mechanical keyboard", "keyboard"],
        "min_matches": 2,
    },
    {
        "query_id": "q_003",
        "query": "wireless earbuds with charging case for running",
        "intent_type": "semantic_use_case",
        "category": "Earbuds",
        "keywords": ["wireless earbuds", "earbuds", "earphones"],
        "min_matches": 3,
    },
    {
        "query_id": "q_004",
        "query": "4K HDMI 2.1 cable high speed for gaming console",
        "intent_type": "attribute_compatibility",
        "category": "Cables",
        "keywords": ["hdmi", "cable", "4k"],
        "min_matches": 3,
    },
    {
        "query_id": "q_005",
        "query": "USB C multiport hub adapter for MacBook Pro",
        "intent_type": "compatibility_hardware",
        "category": "Hubs & Adapters",
        "keywords": ["usb c hub", "usb c adapter", "hub", "adapter"],
        "min_matches": 3,
    },
    {
        "query_id": "q_006",
        "query": "portable bluetooth speaker waterproof with deep bass",
        "intent_type": "semantic_attribute",
        "category": "Speakers",
        "keywords": ["bluetooth speaker", "portable speaker", "speaker"],
        "min_matches": 3,
    },
    {
        "query_id": "q_007",
        "query": "high capacity power bank fast charging 20000mAh",
        "intent_type": "attribute_spec",
        "category": "Power Banks",
        "keywords": ["power bank", "portable charger", "battery pack"],
        "min_matches": 2,
    },
    {
        "query_id": "q_008",
        "query": "ergonomic wireless vertical mouse for wrist pain",
        "intent_type": "semantic_use_case",
        "category": "Mice",
        "keywords": ["ergonomic", "mouse", "wireless"],
        "min_matches": 2,
    },
    {
        "query_id": "q_009",
        "query": "gaming headset with microphone for PC and console",
        "intent_type": "category_use_case",
        "category": "Headsets",
        "keywords": ["gaming headset", "headset", "microphone"],
        "min_matches": 3,
    },
    {
        "query_id": "q_010",
        "query": "webcam with ring light and microphone for streaming",
        "intent_type": "attribute_use_case",
        "category": "Webcams",
        "keywords": ["webcam", "camera", "streaming"],
        "min_matches": 2,
    },
    {
        "query_id": "q_011",
        "query": "laptop cooling pad with quiet fans",
        "intent_type": "category_use_case",
        "category": "Laptop Accessories",
        "keywords": ["cooling pad", "laptop cooler", "cooling"],
        "min_matches": 2,
    },
    {
        "query_id": "q_012",
        "query": "ultra high speed micro SD card for 4K action camera",
        "intent_type": "attribute_spec",
        "category": "Memory Cards",
        "keywords": ["micro sd", "sd card", "memory card"],
        "min_matches": 2,
    },
    {
        "query_id": "q_013",
        "query": "surge protector power strip with USB ports",
        "intent_type": "attribute_spec",
        "category": "Power & Surge",
        "keywords": ["surge protector", "power strip", "strip"],
        "min_matches": 3,
    },
    {
        "query_id": "q_014",
        "query": "smart home indoor wifi security camera night vision",
        "intent_type": "semantic_attribute",
        "category": "Security Cameras",
        "keywords": ["security camera", "camera", "wifi"],
        "min_matches": 2,
    },
    {
        "query_id": "q_015",
        "query": "clip-on lapel lavalier microphone for smartphone video recording",
        "intent_type": "long_tail_intent",
        "category": "Microphones",
        "keywords": ["lavalier", "lapel", "microphone", "mic"],
        "min_matches": 2,
    },
    {
        "query_id": "q_016",
        "query": "cat8 ethernet cable high speed for gigabit network router",
        "intent_type": "attribute_spec",
        "category": "Networking Cables",
        "keywords": ["ethernet cable", "cat8", "cat7", "network cable"],
        "min_matches": 2,
    },
    {
        "query_id": "q_017",
        "query": "dual monitor arm desk mount adjustable",
        "intent_type": "category_attribute",
        "category": "Monitor Mounts",
        "keywords": ["monitor arm", "monitor mount", "desk mount"],
        "min_matches": 2,
    },
    {
        "query_id": "q_018",
        "query": "magnetic wireless car charger mount for iPhone",
        "intent_type": "compatibility_use_case",
        "category": "Car Electronics",
        "keywords": ["car charger", "wireless car", "car mount"],
        "min_matches": 2,
    },
    {
        "query_id": "q_019",
        "query": "bluetooth audio transmitter receiver for TV and airplane",
        "intent_type": "long_tail_intent",
        "category": "Audio Adapters",
        "keywords": ["bluetooth transmitter", "bluetooth adapter", "audio receiver"],
        "min_matches": 2,
    },
    {
        "query_id": "q_020",
        "query": "laptop stand aluminum foldable portable for desk",
        "intent_type": "category_use_case",
        "category": "Laptop Stands",
        "keywords": ["laptop stand", "foldable laptop", "stand"],
        "min_matches": 2,
    },
    {
        "query_id": "q_021",
        "query": "electronic digital luggage scale for travel suitcase",
        "intent_type": "long_tail_intent",
        "category": "Gadgets",
        "keywords": ["luggage scale", "digital scale", "scale"],
        "min_matches": 2,
    },
    {
        "query_id": "q_022",
        "query": "wireless barcode scanner handheld for inventory",
        "intent_type": "professional_use_case",
        "category": "Office Electronics",
        "keywords": ["barcode scanner", "scanner"],
        "min_matches": 2,
    },
    {
        "query_id": "q_023",
        "query": "stylus pen for touch screen tablet high precision",
        "intent_type": "category_compatibility",
        "category": "Tablet Accessories",
        "keywords": ["stylus", "stylus pen", "touch pen"],
        "min_matches": 2,
    },
    {
        "query_id": "q_024",
        "query": "headphone stand with USB charger ports desktop",
        "intent_type": "attribute_use_case",
        "category": "Audio Accessories",
        "keywords": ["headphone stand", "headset stand", "stand"],
        "min_matches": 2,
    },
    {
        "query_id": "q_025",
        "query": "external DVD drive USB 3.0 portable optical drive",
        "intent_type": "attribute_compatibility",
        "category": "Optical Drives",
        "keywords": ["external dvd", "dvd drive", "cd drive", "optical drive"],
        "min_matches": 2,
    },
    {
        "query_id": "q_026",
        "query": "cable management box organizer for desk cords",
        "intent_type": "use_case",
        "category": "Cable Management",
        "keywords": ["cable management", "cable organizer", "cord box"],
        "min_matches": 2,
    },
    {
        "query_id": "q_027",
        "query": "anti blue light screen protector for computer monitor",
        "intent_type": "attribute_use_case",
        "category": "Screen Protectors",
        "keywords": ["blue light", "screen protector", "privacy filter"],
        "min_matches": 2,
    },
    {
        "query_id": "q_028",
        "query": "wifi range extender booster for home coverage",
        "intent_type": "category_use_case",
        "category": "Networking",
        "keywords": ["wifi extender", "range extender", "wifi booster"],
        "min_matches": 2,
    },
    {
        "query_id": "q_029",
        "query": "headphones under 50",
        "intent_type": "budget_constraint",
        "category": "Headphones",
        "keywords": ["headphones", "earphones", "headset"],
        "structured_constraints": {"max_price": 50.0},
        "min_matches": 3,
    },
    {
        "query_id": "q_030",
        "query": "usb c hub under 30",
        "intent_type": "budget_constraint",
        "category": "Hubs & Adapters",
        "keywords": ["usb c", "hub", "adapter"],
        "structured_constraints": {"max_price": 30.0},
        "min_matches": 3,
    },
]


def find_matching_products(
    products_df: pd.DataFrame,
    keywords: List[str],
    category_hint: Optional[str] = None,
    max_price: Optional[float] = None,
    max_results: int = 10,
) -> List[str]:
    """Find ground-truth matching parent_asins from processed products catalog."""
    titles = products_df["title"].fillna("").astype(str).str.lower()
    cats = products_df["categories"].apply(lambda c: " ".join([str(x) for x in c]) if isinstance(c, (list, tuple, np.ndarray)) else str(c)).str.lower()
    
    mask = pd.Series(False, index=products_df.index)
    for kw in keywords:
        kw_lower = kw.lower()
        match_title = titles.str.contains(re.escape(kw_lower), regex=True)
        match_cat = cats.str.contains(re.escape(kw_lower), regex=True)
        mask = mask | match_title | match_cat

    if max_price is not None and "price" in products_df.columns:
        valid_price = products_df["price"].notna() & (products_df["price"] <= max_price)
        mask = mask & valid_price

    matched_df = products_df[mask]
    
    # Sort by rating count / rating number descending to prefer prominent matching items
    if "rating_number" in matched_df.columns:
        matched_df = matched_df.sort_values(by=["rating_number"], ascending=False)
        
    return matched_df["parent_asin"].head(max_results).tolist()


def build_evaluation_queries(
    products_df: pd.DataFrame,
    output_path: str,
) -> List[Dict[str, Any]]:
    """Build, match, validate, and serialize ground-truth evaluation queries against products.parquet.
    
    Guarantees:
      - Every relevant_product_id exists in the actual products catalog.
      - No fabricated or synthetic product IDs.
      - Unique query IDs.
      - Non-empty relevant_product_ids list.
    """
    valid_asins: Set[str] = set(products_df["parent_asin"].unique())
    queries_output: List[Dict[str, Any]] = []
    
    for item in CURATED_QUERY_TEMPLATES:
        qid = item["query_id"]
        qtext = item["query"]
        keywords = item.get("keywords", [qtext])
        constraints = item.get("structured_constraints", {})
        max_p = constraints.get("max_price") if constraints else None
        
        matched_asins = find_matching_products(
            products_df=products_df,
            keywords=keywords,
            category_hint=item.get("category"),
            max_price=max_p,
            max_results=8,
        )
        
        # Verify all matched IDs strictly exist in catalog
        verified_asins = [a for a in matched_asins if a in valid_asins]
        
        if verified_asins:
            query_record = {
                "query_id": qid,
                "query": qtext,
                "relevant_product_ids": verified_asins,
                "category": item.get("category"),
                "intent_type": item.get("intent_type"),
                "structured_constraints": constraints if constraints else None,
            }
            queries_output.append(query_record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries_output, f, indent=2, ensure_ascii=False)
        
    return queries_output
