"""Semantic product search API endpoints."""

from fastapi import APIRouter, Depends
from backend.app.models.search import SearchRequest, SearchResponse
from backend.app.services.engine import SearchEngine
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/search", tags=["Semantic Search"])


def get_search_engine() -> SearchEngine:
    """Dependency provider for SearchEngine orchestrator."""
    registry = ModelRegistry.get_instance()
    return SearchEngine(registry=registry)


@router.post("", response_model=SearchResponse, summary="Execute multi-stage semantic product search")
async def execute_search(
    request: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine),
) -> SearchResponse:
    """Run full query understanding, dense FAISS retrieval, Cross-Encoder reranking, and hybrid scoring."""
    return engine.search(request)
