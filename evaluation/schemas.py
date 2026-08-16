"""Schemas for reproducible evaluation experiments and metric reporting."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    """Ground-truth evaluation query item."""

    query_id: str = Field(..., description="Unique query identifier")
    query_text: str = Field(..., description="The query string")
    relevant_doc_ids: List[str] = Field(
        default_factory=list,
        description="List of ground-truth relevant product ASINs/IDs",
    )
    relevance_grades: Optional[Dict[str, int]] = Field(
        default=None,
        description="Graded relevance map {doc_id: grade} for NDCG calculation (e.g. 0-3)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional query metadata (category, difficulty, length, etc.)",
    )


class MetricScores(BaseModel):
    """Aggregate metrics computed across the evaluation query set."""

    recall_at_k: Dict[str, float] = Field(
        default_factory=dict,
        description="Recall@K for various K values (e.g., {'recall@10': 0.82})",
    )
    mrr_at_k: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean Reciprocal Rank@K (e.g., {'mrr@10': 0.74})",
    )
    ndcg_at_k: Dict[str, float] = Field(
        default_factory=dict,
        description="NDCG@K for graded relevance (e.g., {'ndcg@10': 0.78})",
    )
    precision_at_k: Dict[str, float] = Field(
        default_factory=dict,
        description="Precision@K",
    )
    mean_average_precision: Optional[float] = Field(
        default=None,
        description="MAP score across queries",
    )


class LatencyMetrics(BaseModel):
    """Query latency profiling metrics in milliseconds."""

    p50_ms: float = Field(..., description="50th percentile latency in ms")
    p90_ms: float = Field(..., description="90th percentile latency in ms")
    p95_ms: float = Field(..., description="95th percentile latency in ms")
    p99_ms: float = Field(..., description="99th percentile latency in ms")
    mean_ms: float = Field(..., description="Mean latency in ms")
    min_ms: float = Field(..., description="Minimum latency in ms")
    max_ms: float = Field(..., description="Maximum latency in ms")
    total_queries: int = Field(..., description="Number of evaluated queries")


class SystemProvenance(BaseModel):
    """Hardware and environment provenance for experimental reproducibility."""

    platform: str
    python_version: str
    torch_version: Optional[str] = None
    cuda_available: bool = False
    device_name: Optional[str] = None
    git_commit: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Full experiment configuration schema."""

    experiment_id: str
    description: str
    track: str = Field(..., description="baseline | retrieval | reranking | recommendation")
    random_seed: int = 42
    dataset_name: str
    dataset_subset: str
    dataset_path: str
    test_queries_path: str
    model_name: Optional[str] = None
    reranker_name: Optional[str] = None
    index_type: Optional[str] = None
    top_k_retrieval: int = 100
    top_k_reranking: Optional[int] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Persisted output record of an evaluation experiment."""

    experiment_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: ExperimentConfig
    metrics: MetricScores
    latency: LatencyMetrics
    system_provenance: SystemProvenance
    num_queries_evaluated: int
    notes: Optional[str] = None
