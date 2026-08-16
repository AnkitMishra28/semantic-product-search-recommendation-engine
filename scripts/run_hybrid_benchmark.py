#!/usr/bin/env python3
"""Convenience runner script for Phase 7 Hybrid Retrieval Benchmark."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.hybrid_retrieval.run_hybrid_benchmark import run_hybrid_benchmark

if __name__ == "__main__":
    run_hybrid_benchmark()
