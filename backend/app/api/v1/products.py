"""Product retrieval and catalog lookup API endpoints."""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from backend.app.models.product import Product
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/products", tags=["Product Catalog"])


@router.get(
    "/{product_id}",
    response_model=Product,
    summary="Get verified product by ASIN",
    responses={
        200: {"description": "Product found in catalog"},
        404: {"description": "Product not found"},
    },
)
async def get_product(product_id: str) -> Product:
    """Retrieve verified product metadata from the catalog repository."""
    registry = ModelRegistry.get_instance()
    
    # 1. Direct key lookup (ASIN or parent_asin)
    if product_id in registry.catalog:
        return registry.catalog[product_id]

    # 2. Linear scan for matching asin or parent_asin if key format differs
    for p in registry.catalog.values():
        if p.asin == product_id or p.parent_asin == product_id:
            return p

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Product with ID '{product_id}' was not found in the catalog.",
    )
