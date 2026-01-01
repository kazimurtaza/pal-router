#!/usr/bin/env python3
"""Calibration script for tuning RouteLLM threshold.

This script analyzes the test suite and finds the optimal RouteLLM threshold
to achieve a target percentage of queries routed to the strong model.

Usage:
    python eval/calibrate_complexity.py --strong-model-pct 0.3
    python eval/calibrate_complexity.py --visualize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.complexity import estimate_complexity
from pal_router.router import ROUTELLM_AVAILABLE, TernaryRouter


def load_test_suite(path: Path | None = None) -> list[dict]:
    """Load the test suite from JSON file."""
    if path is None:
        path = Path(__file__).parent / "test_suite.json"

    with open(path) as f:
        data = json.load(f)

    # Exclude holdout queries
    holdout_ids = set(data.get("holdout_ids", []))
    queries = [q for q in data["queries"] if q["id"] not in holdout_ids]

    return queries


def get_routellm_scores(queries: list[dict]) -> list[tuple[str, float, str]]:
    """Get RouteLLM scores for all queries.

    Returns:
        List of (query_id, routellm_score, expected_lane) tuples.
    """
    if not ROUTELLM_AVAILABLE:
        print("WARNING: RouteLLM not available. Using complexity-based fallback.")
        return []

    # Create router with RouteLLM enabled
    router = TernaryRouter(use_routellm=True)

    if not router.use_routellm:
        print("WARNING: RouteLLM failed to initialize. Check OPENAI_API_KEY for embeddings.")
        return []

    results = []
    for query in queries:
        prompt = query["prompt"]
        expected = query["expected_lane"]

        # Skip queries that should be AGENTIC (handled by complexity scorer)
        if expected == "AGENTIC":
            continue

        score = router._get_routellm_score(prompt)
        if score is not None:
            results.append((query["id"], score, expected))

    return results


def find_optimal_threshold(
    scores: list[tuple[str, float, str]],
    target_strong_pct: float = 0.3,
) -> tuple[float, dict]:
    """Find the threshold that achieves the target strong model percentage.

    Args:
        scores: List of (query_id, routellm_score, expected_lane) tuples.
        target_strong_pct: Target percentage of queries to route to strong model.

    Returns:
        Tuple of (optimal_threshold, stats_dict).
    """
    if not scores:
        return 0.12, {"error": "No scores available"}

    # Sort by score
    sorted_scores = sorted(scores, key=lambda x: x[1])

    # Find threshold that gives target percentage to strong model
    n = len(sorted_scores)
    target_idx = int(n * (1 - target_strong_pct))

    if target_idx >= n:
        threshold = sorted_scores[-1][1] + 0.01
    elif target_idx <= 0:
        threshold = sorted_scores[0][1] - 0.01
    else:
        threshold = sorted_scores[target_idx][1]

    # Calculate actual stats at this threshold
    strong_count = sum(1 for _, score, _ in sorted_scores if score > threshold)
    actual_strong_pct = strong_count / n

    # Calculate accuracy (FAST should have low score, REASONING should have high score)
    correct = 0
    for query_id, score, expected in sorted_scores:
        if expected == "FAST" and score <= threshold:
            correct += 1
        elif expected == "REASONING" and score > threshold:
            correct += 1

    accuracy = correct / n if n > 0 else 0

    stats = {
        "threshold": threshold,
        "target_strong_pct": target_strong_pct,
        "actual_strong_pct": actual_strong_pct,
        "total_queries": n,
        "routed_to_strong": strong_count,
        "accuracy": accuracy,
        "score_min": sorted_scores[0][1],
        "score_max": sorted_scores[-1][1],
        "score_median": sorted_scores[n // 2][1],
    }

    return threshold, stats


def visualize_distribution(scores: list[tuple[str, float, str]], threshold: float = 0.12):
    """Visualize the score distribution with matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    # Separate by expected lane
    fast_scores = [s for _, s, lane in scores if lane == "FAST"]
    reasoning_scores = [s for _, s, lane in scores if lane == "REASONING"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    bins = 20
    ax.hist(fast_scores, bins=bins, alpha=0.7, label="FAST (expected)", color="green")
    ax.hist(reasoning_scores, bins=bins, alpha=0.7, label="REASONING (expected)", color="blue")

    # Plot threshold line
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("RouteLLM Score")
    ax.set_ylabel("Count")
    ax.set_title("RouteLLM Score Distribution by Expected Lane")
    ax.legend()

    # Save figure
    output_path = Path(__file__).parent / "results" / "routellm_distribution.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")

    plt.close()


def analyze_complexity_scores(queries: list[dict]):
    """Analyze complexity score distribution for all queries."""
    results = []

    for query in queries:
        prompt = query["prompt"]
        expected = query["expected_lane"]
        score, signals = estimate_complexity(prompt)

        results.append({
            "id": query["id"],
            "expected_lane": expected,
            "complexity_score": score,
            "numeric_density": signals.numeric_density,
            "logic_density": signals.logic_density,
            "constraint_count": signals.constraint_count,
            "estimated_steps": signals.estimated_steps,
        })

    # Print summary by lane
    print("\n=== Complexity Score Analysis ===\n")

    for lane in ["FAST", "REASONING", "AGENTIC"]:
        lane_results = [r for r in results if r["expected_lane"] == lane]
        if not lane_results:
            continue

        scores = [r["complexity_score"] for r in lane_results]
        numeric = [r["numeric_density"] for r in lane_results]

        print(f"{lane} ({len(lane_results)} queries):")
        print(f"  Complexity: min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")
        print(f"  Numeric density: min={min(numeric):.3f}, max={max(numeric):.3f}, mean={sum(numeric)/len(numeric):.3f}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibrate RouteLLM threshold")
    parser.add_argument(
        "--strong-model-pct",
        type=float,
        default=0.3,
        help="Target percentage of queries to route to strong model (default: 0.3)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of score distribution",
    )
    parser.add_argument(
        "--analyze-complexity",
        action="store_true",
        help="Analyze complexity scores (no RouteLLM needed)",
    )
    parser.add_argument(
        "--test-suite",
        type=Path,
        default=None,
        help="Path to test suite JSON file",
    )

    args = parser.parse_args()

    # Load test suite
    queries = load_test_suite(args.test_suite)
    print(f"Loaded {len(queries)} queries (excluding holdout)")

    # Analyze complexity scores (always works, no API needed)
    if args.analyze_complexity:
        analyze_complexity_scores(queries)
        return

    # Get RouteLLM scores
    print("\nGetting RouteLLM scores...")
    scores = get_routellm_scores(queries)

    if not scores:
        print("\nNo RouteLLM scores available. Running complexity analysis instead.")
        analyze_complexity_scores(queries)
        return

    print(f"Got scores for {len(scores)} queries (excluding AGENTIC)")

    # Find optimal threshold
    threshold, stats = find_optimal_threshold(scores, args.strong_model_pct)

    print("\n=== Calibration Results ===\n")
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Target strong model %: {stats['target_strong_pct']:.1%}")
    print(f"Actual strong model %: {stats['actual_strong_pct']:.1%}")
    print(f"Routing accuracy: {stats['accuracy']:.1%}")
    print(f"\nScore distribution:")
    print(f"  Min: {stats['score_min']:.4f}")
    print(f"  Median: {stats['score_median']:.4f}")
    print(f"  Max: {stats['score_max']:.4f}")

    # Generate visualization
    if args.visualize:
        visualize_distribution(scores, threshold)

    # Save results
    output_path = Path(__file__).parent / "results" / "calibration.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    # Print usage suggestion
    print(f"\n=== Usage ===")
    print(f"router = create_fast_router(routellm_threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
