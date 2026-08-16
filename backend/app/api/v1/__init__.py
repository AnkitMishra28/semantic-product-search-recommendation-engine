"""API v1 endpoints registration."""

from fastapi import APIRouter
from backend.app.api.v1.evaluate import router as evaluate_router
from backend.app.api.v1.explain import router as explain_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.metrics import router as metrics_router
from backend.app.api.v1.products import router as products_router
from backend.app.api.v1.recommend import router as recommend_router
from backend.app.api.v1.search import router as search_router

# Router with prefix /api/v1
api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health_router)
api_v1_router.include_router(search_router)
api_v1_router.include_router(recommend_router)
api_v1_router.include_router(explain_router)
api_v1_router.include_router(products_router)
api_v1_router.include_router(metrics_router)
api_v1_router.include_router(evaluate_router)

# Direct /api router for un-versioned root access (/api/search, /api/recommend, /api/explain, /api/products, /api/health, /api/metrics)
api_direct_router = APIRouter(prefix="/api")

api_direct_router.include_router(health_router)
api_direct_router.include_router(search_router)
api_direct_router.include_router(recommend_router)
api_direct_router.include_router(explain_router)
api_direct_router.include_router(products_router)
api_direct_router.include_router(metrics_router)
api_direct_router.include_router(evaluate_router)

__all__ = ["api_v1_router", "api_direct_router"]
