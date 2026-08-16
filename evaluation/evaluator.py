"""Experiment Evaluator and Reproducibility Runner.

Orchestrates running a collection of evaluation queries against a search or ranking pipeline,
computes aggregate metrics, measures latency percentiles, gathers hardware/software provenance,
and exports reproducible experiment results.
"""

import json
import logging
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from evaluation.metrics import (
    LatencyTracker,
    average_precision,
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

logger = logging.getLogger(__name__)


def get_git_commit() -> Optional[str]:
    """Retrieve the current Git commit hash if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return None


def collect_system_provenance() -> SystemProvenance:
    """Collect host system and ML library environment details."""
    torch_version = None
    cuda_available = False
    device_name = None

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return SystemProvenance(
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        device_name=device_name,
        git_commit=get_git_commit(),
    )


class Evaluator:
    """Evaluates search/ranking pipelines against ground-truth query sets."""

    def __init__(
        self,
        k_values: Sequence[int] = (5, 10, 20, 50, 100),
    ) -> None:
        self.k_values = sorted(list(set(k_values)))

    def evaluate_pipeline(
        self,
        config: ExperimentConfig,
        queries: List[EvaluationQuery],
        search_fn: Callable[[str, int], List[str]],
    ) -> ExperimentResult:
        """Execute evaluation queries, measure latency, and compute metrics.

        Args:
            config: The experiment configuration details.
            queries: List of evaluation query objects with ground truth annotations.
            search_fn: Callable taking (query_text, top_k) and returning ordered doc IDs.

        Returns:
            ExperimentResult containing all computed metrics, latency profile, and provenance.
        """
        max_k = max(self.k_values)
        if config.top_k_retrieval > max_k:
            max_k = config.top_k_retrieval

        latency_tracker = LatencyTracker()

        # Metric accumulators
        recalls: dict[int, list[float]] = {k: [] for k in self.k_values}
        precisions: dict[int, list[float]] = {k: [] for k in self.k_values}
        mrrs: dict[int, list[float]] = {k: [] for k in self.k_values}
        ndcgs: dict[int, list[float]] = {k: [] for k in self.k_values}
        aps: list[float] = []

        logger.info(f"Starting evaluation of {len(queries)} queries for experiment '{config.experiment_id}'")

        for query in queries:
            t0 = time.perf_counter()
            retrieved_ids = search_fn(query.query_text, max_k)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latency_tracker.record(latency_ms)

            # Compute standard metrics per query
            for k in self.k_values:
                recalls[k].append(recall_at_k(retrieved_ids, query.relevant_doc_ids, k))
                precisions[k].append(precision_at_k(retrieved_ids, query.relevant_doc_ids, k))
                mrrs[k].append(reciprocal_rank_at_k(retrieved_ids, query.relevant_doc_ids, k))

                if query.relevance_grades:
                    ndcgs[k].append(ndcg_at_k(retrieved_ids, query.relevance_grades, k))

            aps.append(average_precision(retrieved_ids, query.relevant_doc_ids))

        # Compute aggregate averages
        recall_at_k_dict = {f"recall@{k}": float(sum(recalls[k]) / max(len(recalls[k]), 1)) for k in self.k_values}
        precision_at_k_dict = {f"precision@{k}": float(sum(precisions[k]) / max(len(precisions[k]), 1)) for k in self.k_values}
        mrr_at_k_dict = {f"mrr@{k}": float(sum(mrrs[k]) / max(len(mrrs[k]), 1)) for k in self.k_values}
        ndcg_at_k_dict = {
            f"ndcg@{k}": float(sum(ndcgs[k]) / max(len(ndcgs[k]), 1))
            for k in self.k_values
            if len(ndcgs[k]) > 0
        }

        map_score = float(sum(aps) / max(len(aps), 1)) if aps else None

        metrics = MetricScores(
            recall_at_k=recall_at_k_dict,
            mrr_at_k=mrr_at_k_dict,
            ndcg_at_k=ndcg_at_k_dict,
            precision_at_k=precision_at_k_dict,
            mean_average_precision=map_score,
        )

        latency_summary = latency_tracker.summary()
        latency_metrics = LatencyMetrics(
            p50_ms=latency_summary["p50_ms"],
            p90_ms=latency_summary["p90_ms"],
            p95_ms=latency_summary["p95_ms"],
            p99_ms=latency_summary["p99_ms"],
            mean_ms=latency_summary["mean_ms"],
            min_ms=latency_summary["min_ms"],
            max_ms=latency_summary["max_ms"],
            total_queries=int(latency_summary["total_queries"]),
        )

        provenance = collect_system_provenance()

        return ExperimentResult(
            experiment_id=config.experiment_id,
            timestamp=datetime.utcnow(),
            config=config,
            metrics=metrics,
            latency=latency_metrics,
            system_provenance=provenance,
            num_queries_evaluated=len(queries),
        )

    def save_results(self, result: ExperimentResult, output_dir: str = "experiments/results") -> Path:
        """Persist the experiment result to a formatted JSON file."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.experiment_id}_{timestamp_str}.json"
        file_path = out_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        logger.info(f"Experiment results saved to {file_path}")
        return file_path
