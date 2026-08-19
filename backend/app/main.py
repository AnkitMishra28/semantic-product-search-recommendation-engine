"""FastAPI application entrypoint with startup/shutdown lifespan lifecycle."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.router import main_router
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.services.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to load ML models once at startup and clean up on shutdown."""
    setup_logging()
    settings = get_settings()

    # Initialize Singleton ModelRegistry once at server startup
    registry = ModelRegistry.get_instance(settings)
    registry.initialize()

    yield

    # Graceful shutdown
    registry.shutdown()


def create_application() -> FastAPI:
    """Factory function to configure and instantiate the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Research-oriented, production-style prototype inspired by Amazon-scale "
            "multi-stage semantic product search, cross-encoder reranking, and recommendations."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:[0-9]+)?|https://.*\.vercel\.app)$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Main API Router
    app.include_router(main_router)

    # Root & Direct Health Check Endpoints (for root-level health probes)
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    @app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
    async def root_health_probe() -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "app_name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
            },
        )

    # HTTP Exception Handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "message": exc.detail,
                "path": str(request.url),
            },
        )

    # Validation Exception Handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "message": "Invalid request payload",
                "details": exc.errors(),
                "path": str(request.url),
            },
        )

    # Global Uncaught Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": str(exc),
                "path": str(request.url),
            },
        )

    return app


app = create_application()
