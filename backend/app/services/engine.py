"""Search and Recommendation Engine Orchestrator."""

import logging
import time
from typing import Dict, List, Optional
from backend.app.core.config import Settings, get_settings
from backend.app.explanations.grounded_explainer import GroundedExplainer
from backend.app.models.product import Product
from backend.app.models.recommendation import (
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
)
from backend.app.models.search import (
    PipelineStageTiming,
    QueryUnderstandingResult,
    RerankSignal,
    RetrievalSignal,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from backend.app.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.ranking.base import RankedCandidate
from backend.app.ranking.hybrid_ranker import HybridRanker
from backend.app.recommendation.item_to_item import ItemToItemRecommender
from backend.app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class SearchEngine:
    """End-to-End Orchestrator for Query Understanding, Retrieval, Reranking, and Recommendations."""

    def __init__(self, registry: Optional[ModelRegistry] = None, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry.get_instance(self.settings)
        self.query_pipeline = QueryUnderstandingPipeline()
        self.hybrid_ranker = HybridRanker(
            relevance_weight=self.settings.hybrid_alpha,
            rating_weight=0.2,
            popularity_weight=0.1,
        )
        self.recommender = ItemToItemRecommender(
            retriever=self.registry.retriever,
            product_catalog=self.registry.catalog,
        )
        self.explainer = GroundedExplainer(
            api_key=self.settings.openai_api_key,
            model_name=self.settings.llm_model_name,
            enable_remote_llm=self.settings.enable_llm_explanations,
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute full multi-stage search workflow with timing instrumentation."""
        total_start = time.perf_counter()
        timings = PipelineStageTiming()

        # Stage 1: Query Understanding
        t0 = time.perf_counter()
        qu_result = self.query_pipeline.process(request.query)
        timings.query_understanding_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Dense Semantic Vector Retrieval
        t0 = time.perf_counter()
        query_vec = self.registry.encode_text(qu_result.normalized_query)
        top_k_ret = request.top_k_retrieval or self.settings.default_retrieval_top_k
        candidates = []
        if self.registry.retriever is not None:
            candidates = self.registry.retriever.search(query_vec, top_k=top_k_ret)
        timings.dense_retrieval_ms = (time.perf_counter() - t0) * 1000.0

        # Map candidate IDs to full Product domain objects
        retrieved_products: List[Product] = []
        retrieval_signals: Dict[str, RetrievalSignal] = {}
        for cand in candidates:
            if cand.doc_id in self.registry.catalog:
                prod = self.registry.catalog[cand.doc_id]
                retrieved_products.append(prod)
                retrieval_signals[cand.doc_id] = RetrievalSignal(
                    stage="dense_retrieval",
                    initial_score=cand.score,
                    initial_rank=cand.rank,
                )

        # Stage 3: Neural Cross-Encoder Reranking
        top_k_rerank = request.top_k_reranking or self.settings.default_reranking_top_k
        t0 = time.perf_counter()
        ranked_candidates: List[RankedCandidate] = []
        rerank_signals: Dict[str, RerankSignal] = {}

        if request.enable_reranking and self.registry.reranker is not None and retrieved_products:
            ranked_candidates = self.registry.reranker.rerank(
                query=qu_result.normalized_query,
                products=retrieved_products,
                top_k=top_k_rerank,
            )
            for r in ranked_candidates:
                rerank_signals[r.product.asin] = RerankSignal(
                    stage="cross_encoder",
                    rerank_score=r.score,
                    rerank_rank=r.rank,
                )
        else:
            # Dense only fallback ranking
            ranked_candidates = [
                RankedCandidate(
                    product=p,
                    score=retrieval_signals[p.asin].initial_score if p.asin in retrieval_signals else 0.0,
                    rank=idx + 1,
                )
                for idx, p in enumerate(retrieved_products[:top_k_rerank])
            ]
        timings.cross_encoder_rerank_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Hybrid Business & Rating Ranking
        t0 = time.perf_counter()
        if request.ranking_strategy == "hybrid":
            final_ranked = self.hybrid_ranker.rank(
                query=qu_result.normalized_query,
                candidates=ranked_candidates,
                top_k=top_k_rerank,
            )
        else:
            final_ranked = ranked_candidates
        timings.business_ranking_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 5: Optional Explanation Generation
        t0 = time.perf_counter()
        results: List[SearchResultItem] = []
        for idx, item in enumerate(final_ranked):
            explanation = None
            grounded_exp = None
            if request.enable_explanation:
                grounded_exp = self.explainer.explain_search(
                    query=request.query,
                    product=item.product,
                    query_intent=qu_result,
                    retrieval_signal=retrieval_signals.get(item.product.asin),
                    rerank_signal=rerank_signals.get(item.product.asin),
                    final_rank=idx + 1,
                )
                explanation = grounded_exp.summary
            results.append(
                SearchResultItem(
                    product=item.product,
                    final_score=item.score,
                    retrieval_signal=retrieval_signals.get(item.product.asin),
                    rerank_signal=rerank_signals.get(item.product.asin),
                    explanation=explanation,
                    grounded_explanation=grounded_exp,
                )
            )
        timings.explanation_generation_ms = (time.perf_counter() - t0) * 1000.0
        timings.total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        return SearchResponse(
            query=request.query,
            query_understanding=qu_result,
            total_retrieved=len(retrieved_products),
            total_returned=len(results),
            results=results,
            timings=timings,
        )

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Execute recommendation generation.

        Prefers the registry's unified RecommendationService (popularity /
        content-based / collaborative / hybrid / MMR diversity, built from the real
        catalog + interactions + embeddings — see ModelRegistry._build_recommendation_service).
        Falls back to the lightweight co-purchase-only ItemToItemRecommender when no
        RecommendationService was built (e.g. under test fixtures with a minimal catalog
        and no matching real interaction/embedding data).
        """
        rec_service = self.registry.get_recommendation_service()
        if rec_service is not None:
            rec_service.set_catalog(self.registry.catalog)
            response = rec_service.recommend(request)
        else:
            t0 = time.perf_counter()
            self.recommender.set_catalog(self.registry.catalog)

            recs: List[RecommendationItem] = []
            if request.asin:
                recs = self.recommender.recommend_for_item(request.asin, top_k=request.top_k)
            elif request.user_history_asins:
                recs = self.recommender.recommend_for_user(request.user_history_asins, top_k=request.top_k)

            response = RecommendResponse(
                anchor_asin=request.asin,
                strategy=request.strategy,
                total_returned=len(recs),
                recommendations=recs,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if request.generate_explanations:
            anchor_prod = self.registry.catalog.get(request.asin) if request.asin else None
            user_prods = [
                self.registry.catalog[a]
                for a in (request.user_history_asins or [])
                if a in self.registry.catalog
            ]
            for rec in response.recommendations:
                rec.grounded_explanation = self.explainer.explain_recommendation_grounded(
                    product=rec.product,
                    rec_item=rec,
                    anchor_product=anchor_prod,
                    user_history_products=user_prods,
                    strategy=request.strategy,
                )
                if not rec.reasons:
                    rec.reasons = [r.text for r in rec.grounded_explanation.reasons if r.is_matched]

        return response
