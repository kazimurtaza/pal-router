#!/usr/bin/env python3
"""Test PAL-Router with real API calls - no mocks.

This script validates the implementation works end-to-end with real LLM APIs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router import TernaryRouter
from pal_router.router import Lane


def test_real_apis():
    """Test all three lanes with real API calls."""

    print("=" * 70)
    print("PAL-ROUTER REAL API TEST")
    print("=" * 70)

    # Check API keys
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set")
        return False

    print(f"✓ GROQ_API_KEY configured")

    # Initialize router with Groq (free tier)
    print("\nInitializing router with Groq provider...")
    try:
        router = TernaryRouter()
        print(f"✓ Router initialized")
        print(f"  Weak model:  {router.weak_model.model_name}")
        print(f"  Strong model: {router.strong_model.model_name}")
    except Exception as e:
        print(f"ERROR initializing router: {e}")
        return False

    # Test queries for each lane
    test_cases = [
        # FAST lane - simple factual query
        {
            "query": "What is the capital of France?",
            "expected_lane": Lane.FAST,
            "validate": lambda ans: "paris" in ans.lower(),
        },
        # REASONING lane - complex analysis
        {
            "query": "Compare the advantages and disadvantages of solar power versus nuclear power for large-scale electricity generation.",
            "expected_lane": Lane.REASONING,
            "validate": lambda ans: len(ans) > 100,  # Should be substantive
        },
        # AGENTIC lane - requires calculation
        {
            "query": "Calculate 18% tip on a $67.50 restaurant bill.",
            "expected_lane": Lane.AGENTIC,
            "validate": lambda ans: "12.15" in ans or "12.1" in ans,  # 67.50 * 0.18 = 12.15
        },
    ]

    results = []
    total_cost = 0.0
    total_latency = 0.0

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected_lane = test["expected_lane"]

        print(f"\n{'─' * 70}")
        print(f"TEST {i}: {expected_lane.value} Lane")
        print(f"{'─' * 70}")
        print(f"Query: {query[:60]}...")

        try:
            # Route and execute
            result = router.execute(query)

            # Check routing
            routed_correctly = result.decision.lane == expected_lane
            lane_status = "✓" if routed_correctly else "✗"
            print(f"\nRouting: {lane_status} {result.decision.lane.value} (expected {expected_lane.value})")
            print(f"Confidence: {result.decision.confidence:.2f}")

            # Check answer
            answer_valid = test["validate"](result.answer)
            answer_status = "✓" if answer_valid else "✗"
            print(f"\nAnswer: {answer_status}")
            print(f"  {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")

            # Show metrics
            print(f"\nMetrics:")
            print(f"  Cost:    ${result.total_cost_usd:.6f}")
            print(f"  Latency: {result.total_latency_ms:.0f}ms")

            # For AGENTIC, show code if available
            if result.agentic_result and result.agentic_result.code:
                print(f"\nGenerated code:")
                for line in result.agentic_result.code.split("\n")[:5]:
                    print(f"    {line}")
                if len(result.agentic_result.code.split("\n")) > 5:
                    print("    ...")

            total_cost += result.total_cost_usd
            total_latency += result.total_latency_ms

            results.append({
                "lane": expected_lane.value,
                "routed_correctly": routed_correctly,
                "answer_valid": answer_valid,
                "passed": routed_correctly and answer_valid,
            })

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "lane": expected_lane.value,
                "routed_correctly": False,
                "answer_valid": False,
                "passed": False,
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {r['lane']}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Total latency: {total_latency:.0f}ms")

    if passed == total:
        print("\n✓ ALL REAL API TESTS PASSED - Implementation is NOT mocked")
        return True
    else:
        print(f"\n✗ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = test_real_apis()
    sys.exit(0 if success else 1)
