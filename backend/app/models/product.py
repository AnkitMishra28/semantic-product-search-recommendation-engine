"""Product domain schemas and metadata representations."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Product(BaseModel):
    """Standardized product representation across search, recommendation, and indexing."""

    asin: Optional[str] = Field(default=None, description="Unique Amazon Standard Identification Number")
    parent_asin: Optional[str] = Field(default=None, description="Canonical product identifier (parent ASIN)")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(default="", description="Product description text")
    features: List[str] = Field(default_factory=list, description="Bullet point features and specifications")
    price: Optional[float] = Field(default=None, description="Product price in USD")
    brand: Optional[str] = Field(default=None, description="Brand name")
    categories: List[str] = Field(default_factory=list, description="Hierarchical category path")
    average_rating: Optional[float] = Field(default=None, description="Average review star rating (0-5)")
    rating_number: int = Field(default=0, description="Total count of customer ratings/reviews")
    rating_count: int = Field(default=0, description="Alias for rating_number for backward compatibility")
    image_url: Optional[str] = Field(default=None, description="Product thumbnail / primary image URL")
    images: List[str] = Field(default_factory=list, description="List of product image URLs")
    bought_together: List[str] = Field(
        default_factory=list,
        description="List of ASINs frequently co-purchased with this product",
    )
    embedding_text: Optional[str] = Field(default=None, description="Pre-computed semantic document text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary raw metadata fields")

    def model_post_init(self, __context: Any) -> None:
        if self.parent_asin is None and self.asin is not None:
            self.parent_asin = self.asin
        elif self.asin is None and self.parent_asin is not None:
            self.asin = self.parent_asin
        elif self.parent_asin is None and self.asin is None:
            raise ValueError("Either parent_asin or asin must be provided")

        if self.rating_count == 0 and self.rating_number > 0:
            self.rating_count = self.rating_number
        elif self.rating_number == 0 and self.rating_count > 0:
            self.rating_number = self.rating_count


class ProductFilter(BaseModel):
    """Faceted search filtering parameters."""

    min_price: Optional[float] = Field(default=None, description="Minimum price filter")
    max_price: Optional[float] = Field(default=None, description="Maximum price filter")
    brand: Optional[str] = Field(default=None, description="Exact brand filter")
    categories: Optional[List[str]] = Field(default=None, description="Category filter match")
    min_rating: Optional[float] = Field(default=None, description="Minimum star rating filter")
