"""Convert test_suite.json to training_queries.json format."""

from __future__ import annotations

import json
from pathlib import Path


def main():
    # Load test suite
    test_suite_path = Path(__file__).parent.parent / "eval" / "test_suite.json"
    with open(test_suite_path) as f:
        test_suite = json.load(f)

    # Convert to training format
    queries = []
    for item in test_suite["queries"]:
        queries.append({
            "text": item["prompt"],
            "lane": item["expected_lane"],
        })

    training_data = {
        "version": "1.0.0",
        "queries": queries,
    }

    # Save
    output_path = Path(__file__).parent.parent / "data" / "training_queries.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Converted {len(queries)} queries to {output_path}")

    # Show distribution
    from collections import Counter
    lane_counts = Counter(q["lane"] for q in queries)
    print(f"Distribution: {dict(lane_counts)}")


if __name__ == "__main__":
    main()
