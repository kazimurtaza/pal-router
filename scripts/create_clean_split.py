"""Create clean train/test split without contamination.

The test suite (eval/test_suite.json) should NEVER be in training data.
This script:
1. Removes all test suite queries from training data
2. Creates a clean training set
3. Reports the split
"""

from __future__ import annotations

import json
from pathlib import Path


def main():
    # Load test suite (this is our held-out evaluation set)
    test_path = Path("eval/test_suite.json")
    with open(test_path) as f:
        test_suite = json.load(f)

    test_queries = {q["prompt"].lower().strip() for q in test_suite["queries"]}
    print(f"Test suite queries (held-out): {len(test_queries)}")

    # Load current training data
    train_path = Path("data/training_queries.json")
    with open(train_path) as f:
        train_data = json.load(f)

    original_count = len(train_data["queries"])
    print(f"Original training queries: {original_count}")

    # Remove any queries that are in the test suite
    clean_queries = [
        q for q in train_data["queries"]
        if q["text"].lower().strip() not in test_queries
    ]

    removed_count = original_count - len(clean_queries)
    print(f"Removed (test contamination): {removed_count}")
    print(f"Clean training queries: {len(clean_queries)}")

    # Show distribution
    from collections import Counter
    lane_counts = Counter(q["lane"] for q in clean_queries)
    print(f"Distribution: {dict(lane_counts)}")

    # Save clean training data
    clean_data = {
        "version": "2.0.0",
        "note": "Test suite queries removed to prevent train/test contamination",
        "queries": clean_queries,
    }

    with open(train_path, "w") as f:
        json.dump(clean_data, f, indent=2)

    print(f"\nSaved clean training data to {train_path}")

    # Verify no overlap
    clean_texts = {q["text"].lower().strip() for q in clean_queries}
    overlap = clean_texts & test_queries
    print(f"Verification - overlap with test set: {len(overlap)}")

    if overlap:
        print("WARNING: Still have overlap!")
        for text in list(overlap)[:5]:
            print(f"  - {text[:60]}...")
    else:
        print("SUCCESS: No overlap with test set")


if __name__ == "__main__":
    main()
