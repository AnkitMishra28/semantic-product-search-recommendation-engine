"""Main API router combining all sub-version routers and direct /api endpoints."""

from fastapi import APIRouter
from backend.app.api.v1 import api_direct_router, api_v1_router

main_router = APIRouter()
main_router.include_router(api_direct_router)
main_router.include_router(api_v1_router)

__all__ = ["main_router"]
