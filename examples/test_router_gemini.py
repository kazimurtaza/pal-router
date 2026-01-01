#!/usr/bin/env python3
"""Test the router with Gemini (free tier)."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not set. Add it to .env file.")
    sys.exit(1)

from pal_router import Config, TernaryRouter

# Test queries
TEST_QUERIES = [
    ("What is the capital of Japan?", "FAST"),
    ("Calculate 15% tip on a $85 dinner bill.", "AGENTIC"),
    ("What are the pros and cons of remote work?", "REASONING"),
]


def main():
    print("Testing Ternary Router with Gemini")
    print("  Weak:   gemini-2.5-flash")
    print("  Strong: gemini-3-flash-preview")
    print("=" * 60)

    config = Config(provider="gemini")
    router = TernaryRouter(config=config)

    for query, expected in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")

        decision = router.route(query)
        print(f"Routed:   {decision.lane.value}")

        print("Executing...")
        result = router.execute(query)
        print(f"Latency:  {result.total_latency_ms:.0f}ms")
        print(f"Answer:   {result.answer[:150]}...")

        if result.agentic_result and result.agentic_result.code:
            print(f"Code:\n{result.agentic_result.code[:200]}")

        print("-" * 60)


if __name__ == "__main__":
    main()
