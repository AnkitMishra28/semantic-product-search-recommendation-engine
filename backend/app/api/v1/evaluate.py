"""Evaluation metadata and experiment runs inspection API."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from backend.app.core.config import get_settings

router = APIRouter(prefix="/evaluate", tags=["Evaluation & Reproducibility"])


@router.get("/experiments", summary="List tracked experiment results")
async def list_experiments() -> List[Dict[str, Any]]:
    """Enumerate all persisted experiment results from the results/ directory."""
    settings = get_settings()
    results_dir = Path(settings.experiments_dir) / "results"
    if not results_dir.exists():
        return []

    experiments = []
    for file_path in sorted(results_dir.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Determine track name
                track = (
                    data.get("track")
                    or data.get("method")
                    or data.get("config", {}).get("track")
                    or file_path.stem.replace("_", " ").title()
                )

                # Extract concise summary of metrics
                metrics_summary: Optional[Dict[str, Any]] = None
                if "metrics" in data and isinstance(data["metrics"], dict):
                    metrics_summary = {
                        k: v for k, v in data["metrics"].items()
                        if isinstance(v, (int, float, str)) and not isinstance(v, bool)
                    }
                elif "master_comparison_pipelines" in data:
                    metrics_summary = {"pipelines_count": len(data["master_comparison_pipelines"])}
                elif "methods_comparison" in data:
                    metrics_summary = {"methods_count": len(data["methods_comparison"])}
                elif "master_test_benchmark" in data:
                    metrics_summary = {"strategies_count": len(data["master_test_benchmark"])}

                experiments.append({
                    "filename": file_path.name,
                    "experiment_id": data.get("experiment_id") or data.get("benchmark_id") or file_path.stem,
                    "timestamp": data.get("timestamp"),
                    "track": track,
                    "dataset": data.get("dataset", {}).get("name") if isinstance(data.get("dataset"), dict) else data.get("dataset"),
                    "metrics_summary": metrics_summary,
                    "latency": data.get("latency") or data.get("latency_ms") or data.get("latency_benchmarks"),
                    "file_size_bytes": file_path.stat().st_size,
                })
        except Exception:
            continue

    return experiments


@router.get("/experiments/{experiment_name_or_id}", summary="Get detailed experiment artifact payload")
async def get_experiment_detail(experiment_name_or_id: str) -> Dict[str, Any]:
    """Retrieve the full raw experiment JSON artifact for inspection."""
    settings = get_settings()
    results_dir = Path(settings.experiments_dir) / "results"
    
    # Try direct filename match or filename with .json
    target_file = results_dir / experiment_name_or_id
    if not target_file.exists() and not experiment_name_or_id.endswith(".json"):
        target_file = results_dir / f"{experiment_name_or_id}.json"

    # If not found directly, search by experiment_id or benchmark_id
    if not target_file.exists():
        for file_path in results_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if (
                        data.get("experiment_id") == experiment_name_or_id
                        or data.get("benchmark_id") == experiment_name_or_id
                        or file_path.stem == experiment_name_or_id
                    ):
                        return data
            except Exception:
                continue
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name_or_id}' not found.")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read experiment artifact: {exc}")

