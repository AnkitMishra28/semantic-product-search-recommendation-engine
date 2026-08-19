"""Health, readiness, and system status API endpoints."""

from typing import Any, Dict
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.app.core.config import get_settings
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(tags=["Health & Diagnostics"])


@router.api_route(
    "/health",
    methods=["GET", "HEAD"],
    summary="Basic liveness health check",
    include_in_schema=True,
)
async def health_check() -> JSONResponse:
    """Return lightweight application alive status."""
    settings = get_settings()
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        },
    )


@router.api_route(
    "/ready",
    methods=["GET", "HEAD"],
    summary="System readiness and model status check",
    include_in_schema=True,
)
async def readiness_check() -> JSONResponse:
    """Check if ML models, catalog, and vector indexes are loaded and ready to serve."""
    settings = get_settings()
    registry = ModelRegistry.get_instance(settings)

    indexed_count = registry.retriever.total_documents if registry.retriever else 0
    catalog_count = len(registry.catalog)

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready" if registry._is_initialized else "initializing",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "components": {
                "model_registry": "ready" if registry._is_initialized else "not_initialized",
                "catalog": {
                    "status": "available" if catalog_count > 0 else "empty",
                    "product_count": catalog_count,
                },
                "vector_index": {
                    "status": "available" if registry.retriever is not None else "unavailable",
                    "backend": settings.vector_store_backend,
                    "indexed_documents": indexed_count,
                },
                "cross_encoder": {
                    "status": "available" if registry.reranker is not None else "unavailable",
                    "model": settings.reranker_model_name,
                    "device": settings.device,
                },
                "llm_explainer": {
                    "status": "available",
                    "mode": "remote_llm" if settings.enable_llm_explanations and settings.openai_api_key else "deterministic_fallback",
                    "model": settings.llm_model_name,
                },
            },
        },
    )
