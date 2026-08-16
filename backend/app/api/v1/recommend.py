"""Product recommendation API endpoints."""

from fastapi import APIRouter, Depends
from backend.app.models.recommendation import RecommendRequest, RecommendResponse
from backend.app.services.engine import SearchEngine
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/recommend", tags=["Product Recommendations"])


def get_search_engine() -> SearchEngine:
    registry = ModelRegistry.get_instance()
    return SearchEngine(registry=registry)


@router.post("", response_model=RecommendResponse, summary="Get item-to-item or personalized recommendations")
async def execute_recommendation(
    request: RecommendRequest,
    engine: SearchEngine = Depends(get_search_engine),
) -> RecommendResponse:
    """Generate recommendations using item co-purchase graph and dense semantic similarity."""
    return engine.recommend(request)
