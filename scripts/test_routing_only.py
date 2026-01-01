"""Test routing logic without initializing model clients."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.trained_router import TrainedRouter
from pal_router.complexity import estimate_complexity
from pal_router.router import Lane


def load_test_suite() -> list[dict]:
    """Load test suite queries."""
    test_suite_path = Path(__file__).parent.parent / "eval" / "test_suite.json"
    with open(test_suite_path) as f:
        return json.load(f)["queries"]


def test_trained_router():
    """Test the trained router on all queries."""
    queries = load_test_suite()

    print("Initializing trained router...")
    try:
        trained_router = TrainedRouter(
            model_dir="models/router_classifier",
        )
    except Exception as e:
        print(f"Failed to load trained router: {e}")
        return

    results = []
    for query in queries:
        prediction = trained_router.predict(query["prompt"])
        results.append({
            "query": query["prompt"][:60] + "..." if len(query["prompt"]) > 60 else query["prompt"],
            "expected": query["expected_lane"],
            "predicted": str(prediction.lane),
            "confidence": prediction.confidence,
        })

    # Compute accuracy
    correct = sum(r["predicted"] == r["expected"] for r in results)
    total = len(results)

    print(f"\nResults on {total} queries:")
    print(f"Trained accuracy: {correct}/{total} ({correct/total:.1%})")

    # Show errors
    print("\nTrained errors:")
    errors = [r for r in results if r["predicted"] != r["expected"]]
    if not errors:
        print("  None - 100% accuracy")
    else:
        for r in errors:
            print(f"  {r['query']}")
            print(f"    Expected: {r['expected']}, Got: {r['predicted']} (conf: {r['confidence']:.2f})")

    # Show low confidence predictions
    print("\nLow confidence predictions (<0.6):")
    low_conf = [r for r in results if r["confidence"] < 0.6]
    if not low_conf:
        print("  None")
    else:
        for r in low_conf[:10]:
            correct_marker = "✓" if r["predicted"] == r["expected"] else "✗"
            print(f"  {correct_marker} {r['query']}")
            print(f"      {r['predicted']} ({r['confidence']:.2f}), expected {r['expected']}")


if __name__ == "__main__":
    test_trained_router()
