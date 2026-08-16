"""User-Product interaction domain schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class Interaction(BaseModel):
    """User interaction record (review/rating) for recommendation and evaluation."""

    user_id: str = Field(..., description="Unique customer/user identifier")
    parent_asin: str = Field(..., description="Canonical product identifier")
    rating: float = Field(..., ge=1.0, le=5.0, description="Star rating (1.0 to 5.0)")
    timestamp: int = Field(..., description="Interaction timestamp (epoch milliseconds)")
    verified_purchase: bool = Field(default=False, description="Whether the interaction is a verified purchase")
    split: Optional[str] = Field(default=None, description="Evaluation split assignment (train, val, test)")
