# Experiments Directory

This directory contains declarative experiment configurations and tracked results for reproducible evaluation.

## Reproducibility Contract

Every experiment:
1. Is driven strictly by a YAML configuration file.
2. Must log all parameters (dataset version, model hashes, index parameters, search K, random seeds).
3. Produces a machine-readable JSON result in `experiments/results/<experiment_name>_<timestamp>.json`.
4. Never contains hardcoded or fabricated metric scores.

## Experiment Tracks

- `baseline/`: Lexical BM25 and popularity baselines.
- `retrieval/`: Dense vector retrieval with varying embedding models and FAISS index types.
- `reranking/`: Cross-encoder second-stage reranking benchmarks.
- `recommendation/`: Hybrid and session-based recommendation benchmarks.
- `results/`: JSON outputs containing metrics, latency profiles, and environment metadata.
