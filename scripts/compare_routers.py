"""Compare static heuristic router vs trained classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.router import TernaryRouter


def load_test_suite() -> list[dict]:
    """Load test suite queries."""
    test_suite_path = Path(__file__).parent.parent / "eval" / "test_suite.json"
    with open(test_suite_path) as f:
        return json.load(f)["queries"]


def compare_on_test_suite():
    """Run both routers on test suite and compare."""
    queries = load_test_suite()

    # Initialize routers
    print("Initializing routers...")
    heuristic_router = TernaryRouter(use_trained_classifier=False, use_routellm=False)
    trained_router = TernaryRouter(use_trained_classifier=True, use_routellm=False)

    results = []
    for query in queries:
        heuristic_decision = heuristic_router.route(query["prompt"])
        trained_decision = trained_router.route(query["prompt"])

        results.append({
            "query": query["prompt"][:60] + "..." if len(query["prompt"]) > 60 else query["prompt"],
            "expected": query["expected_lane"],
            "heuristic": heuristic_decision.lane.value,
            "trained": trained_decision.lane.value,
        })

    # Compute accuracies
    heuristic_correct = sum(r["heuristic"] == r["expected"] for r in results)
    trained_correct = sum(r["trained"] == r["expected"] for r in results)
    total = len(results)

    print(f"\nResults on {total} queries:")
    print(f"Heuristic accuracy: {heuristic_correct}/{total} ({heuristic_correct/total:.1%})")
    print(f"Trained accuracy:   {trained_correct}/{total} ({trained_correct/total:.1%})")

    # Show disagreements
    print("\nDisagreements (trained vs heuristic):")
    disagreements = [r for r in results if r["heuristic"] != r["trained"]]
    if not disagreements:
        print("  None - both routers agree on all queries")
    else:
        for r in disagreements:
            print(f"  Query: {r['query']}")
            print(f"    Expected: {r['expected']}, Heuristic: {r['heuristic']}, Trained: {r['trained']}")

    # Show errors
    print("\nHeuristic errors:")
    heuristic_errors = [r for r in results if r["heuristic"] != r["expected"]]
    if not heuristic_errors:
        print("  None - 100% accuracy")
    else:
        for r in heuristic_errors[:5]:
            print(f"  {r['query']}: expected {r['expected']}, got {r['heuristic']}")
        if len(heuristic_errors) > 5:
            print(f"  ... and {len(heuristic_errors) - 5} more")

    print("\nTrained errors:")
    trained_errors = [r for r in results if r["trained"] != r["expected"]]
    if not trained_errors:
        print("  None - 100% accuracy")
    else:
        for r in trained_errors[:5]:
            print(f"  {r['query']}: expected {r['expected']}, got {r['trained']}")
        if len(trained_errors) > 5:
            print(f"  ... and {len(trained_errors) - 5} more")


if __name__ == "__main__":
    compare_on_test_suite()
