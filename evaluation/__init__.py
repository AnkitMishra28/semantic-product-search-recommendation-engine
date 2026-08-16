"""Evaluation package for IR metrics, latency profiling, and experiment reproducibility."""

from evaluation.evaluator import Evaluator, collect_system_provenance
from evaluation.metrics import (
    LatencyTracker,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from evaluation.schemas import (
    EvaluationQuery,
    ExperimentConfig,
    ExperimentResult,
    LatencyMetrics,
    MetricScores,
    SystemProvenance,
)

__all__ = [
    "Evaluator",
    "collect_system_provenance",
    "LatencyTracker",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    "EvaluationQuery",
    "ExperimentConfig",
    "ExperimentResult",
    "MetricScores",
    "LatencyMetrics",
    "SystemProvenance",
]
