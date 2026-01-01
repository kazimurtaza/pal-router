#!/usr/bin/env python3
"""Test the router with Groq (free tier)."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY not set. Add it to .env file.")
    sys.exit(1)

from pal_router import Config, TernaryRouter

# Test queries for each lane
TEST_QUERIES = [
    # FAST - simple factual
    ("What is the capital of France?", "FAST"),
    # AGENTIC - math computation
    ("If I invest $10,000 at 5% interest for 3 years, how much will I have?", "AGENTIC"),
    # REASONING - complex reasoning
    ("Explain the trolley problem and its ethical implications.", "REASONING"),
]


def main():
    print("Testing Ternary Router with Groq\n")
    print("=" * 60)

    config = Config(provider="groq")
    router = TernaryRouter(config=config)

    for query, expected in TEST_QUERIES:
        print(f"\nQuery: {query[:50]}...")
        print(f"Expected Lane: {expected}")

        # Route
        decision = router.route(query)
        print(f"Actual Lane: {decision.lane.value}")
        print(f"Complexity: {decision.complexity_score:.2f}")
        print(f"Reason: {decision.reason}")

        # Execute
        print("\nExecuting...")
        result = router.execute(query)
        print(f"Latency: {result.total_latency_ms:.0f}ms")
        print(f"Answer: {result.answer[:200]}...")

        if result.agentic_result and result.agentic_result.code:
            print(f"\nGenerated Code:\n{result.agentic_result.code[:300]}...")

        print("-" * 60)


if __name__ == "__main__":
    main()
