#!/usr/bin/env python3
"""Convenience runner script for Phase 8 Hybrid Personalized Recommendation Benchmark."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.recommendation.run_recommendation_benchmark import run_recommendation_benchmark

if __name__ == "__main__":
    run_recommendation_benchmark()
