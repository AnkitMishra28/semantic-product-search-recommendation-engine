"""Preprocessing and dataset preparation package."""

from backend.app.preprocessing.cleaners import (
    clean_brand,
    clean_categories,
    clean_description,
    clean_features,
    clean_text,
    extract_images,
    parse_price,
)
from backend.app.preprocessing.eval_queries import (
    build_evaluation_queries,
    CURATED_QUERY_TEMPLATES,
)
from backend.app.preprocessing.interaction_processor import (
    clean_interaction_record,
    process_interactions,
)
from backend.app.preprocessing.product_document import (
    build_product_text,
    TextRepresentationVariant,
)
from backend.app.preprocessing.profiler import profile_dataset
from backend.app.preprocessing.sampler import (
    clean_raw_product_record,
    compute_product_quality_score,
    sample_and_deduplicate_products,
)
from backend.app.preprocessing.validator import (
    ValidationError,
    validate_evaluation_queries,
    validate_interactions,
    validate_products_catalog,
)

__all__ = [
    "clean_text",
    "parse_price",
    "clean_brand",
    "clean_categories",
    "clean_features",
    "clean_description",
    "extract_images",
    "build_product_text",
    "TextRepresentationVariant",
    "compute_product_quality_score",
    "clean_raw_product_record",
    "sample_and_deduplicate_products",
    "clean_interaction_record",
    "process_interactions",
    "build_evaluation_queries",
    "CURATED_QUERY_TEMPLATES",
    "profile_dataset",
    "validate_products_catalog",
    "validate_interactions",
    "validate_evaluation_queries",
    "ValidationError",
]
