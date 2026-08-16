"""CLI entrypoint to load an experiment configuration and run reproducible evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import Evaluator
from evaluation.schemas import ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("experiment_runner")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML or JSON file."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Simple parser fallback for basic key-value YAML if PyYAML is not installed in current Python env
        logger.warning("PyYAML not installed in environment. Attempting basic line parser fallback.")
        content = config_path.read_text(encoding="utf-8")
        data: dict = {}
        curr_dict = data
        curr_key = None

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if not val:
                    data[key] = {}
                    curr_key = key
                else:
                    # Strip quotes / brackets
                    val_clean = val.strip('"\'')
                    if val.startswith("[") and val.endswith("]"):
                        items = [x.strip().strip('"\'') for x in val[1:-1].split(",") if x.strip()]
                        val_clean = items
                    elif val.lower() == "true":
                        val_clean = True
                    elif val.lower() == "false":
                        val_clean = False
                    elif val_clean.isdigit():
                        val_clean = int(val_clean)

                    if raw_line.startswith("  ") and curr_key and isinstance(data[curr_key], dict):
                        data[curr_key][key] = val_clean
                    else:
                        data[key] = val_clean
                        curr_key = None
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible semantic search and ranking experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file (e.g. experiments/retrieval/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate experiment configuration without executing search operations",
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)

    raw_cfg = load_config(config_file)
    logger.info(f"Loaded configuration for experiment: {raw_cfg.get('experiment_id')}")

    # Build Pydantic experiment configuration
    exp_cfg = ExperimentConfig(
        experiment_id=raw_cfg.get("experiment_id", "unknown_experiment"),
        description=raw_cfg.get("description", ""),
        track=raw_cfg.get("track", "retrieval"),
        random_seed=int(raw_cfg.get("random_seed", 42)),
        dataset_name=raw_cfg.get("dataset", {}).get("name", "amazon_reviews_2023_electronics"),
        dataset_subset=raw_cfg.get("dataset", {}).get("subset", "dev_sample"),
        dataset_path=raw_cfg.get("dataset", {}).get("path", "data/processed/electronics_sample.jsonl"),
        test_queries_path=raw_cfg.get("dataset", {}).get("test_queries_path", ""),
        model_name=raw_cfg.get("model", {}).get("name") or raw_cfg.get("stage1_retriever", {}).get("model_name"),
        reranker_name=raw_cfg.get("stage2_reranker", {}).get("model_name"),
        index_type=raw_cfg.get("index", {}).get("type"),
        top_k_retrieval=int(raw_cfg.get("retrieval", {}).get("top_k", 100)),
        top_k_reranking=int(raw_cfg.get("stage2_reranker", {}).get("top_k", 20)) if raw_cfg.get("stage2_reranker", {}).get("top_k") else None,
        parameters=raw_cfg,
    )

    logger.info(f"Experiment verified successfully: {exp_cfg.experiment_id} (Track: {exp_cfg.track})")

    if args.dry_run:
        logger.info("Dry-run complete. Configuration schema is valid.")
        return

    logger.info(
        "Phase 0 Skeleton: Ready for Phase 1 data ingestion and index building. "
        "Run Phase 1 to generate index artifacts before executing full evaluations."
    )


if __name__ == "__main__":
    main()
