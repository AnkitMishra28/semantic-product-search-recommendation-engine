#!/usr/bin/env python3
"""Convenience runner script for Phase 9 Cross-Encoder Reranking & Latency Optimization Benchmark."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cross_encoder.run_cross_encoder_benchmark import run_cross_encoder_benchmark

if __name__ == "__main__":
    run_cross_encoder_benchmark()
