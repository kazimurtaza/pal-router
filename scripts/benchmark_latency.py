"""Benchmark routing latency."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.router import TernaryRouter


def benchmark_latency(n_iterations: int = 100):
    """Benchmark routing latency for both methods."""
    queries = [
        "What is the capital of France?",
        "Calculate 15% of $200",
        "Explain the trolley problem and its ethical implications",
    ]

    print("Initializing routers...")
    heuristic_router = TernaryRouter(use_trained_classifier=False, use_routellm=False)
    trained_router = TernaryRouter(use_trained_classifier=True, use_routellm=False)

    # Warm up
    print("Warming up...")
    for q in queries:
        heuristic_router.route(q)
        trained_router.route(q)

    # Benchmark heuristic
    print(f"\nBenchmarking {n_iterations} iterations...")
    start = time.perf_counter()
    for _ in range(n_iterations):
        for q in queries:
            heuristic_router.route(q)
    heuristic_time = (time.perf_counter() - start) / (n_iterations * len(queries)) * 1000

    # Benchmark trained
    start = time.perf_counter()
    for _ in range(n_iterations):
        for q in queries:
            trained_router.route(q)
    trained_time = (time.perf_counter() - start) / (n_iterations * len(queries)) * 1000

    print(f"\nResults (per query):")
    print(f"Heuristic: {heuristic_time:.2f}ms")
    print(f"Trained:   {trained_time:.2f}ms")
    print(f"Overhead:  {trained_time - heuristic_time:.2f}ms ({trained_time/heuristic_time:.1f}x slower)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--iterations", type=int, default=100)
    args = parser.parse_args()

    benchmark_latency(args.iterations)
