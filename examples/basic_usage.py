#!/usr/bin/env python3
"""Basic usage example for the Ternary LLM Router."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router import Config, TernaryRouter


def main():
    """Demonstrate basic router usage."""
    load_dotenv()

    # Initialize with default configuration
    config = Config()
    router = TernaryRouter(config=config)

    # Example queries for each lane
    examples = [
        # Simple factual → FAST
        "What is the capital of France?",
        # Complex reasoning → REASONING
        "Compare and contrast the philosophical positions of utilitarianism and deontological ethics.",
        # Math computation → AGENTIC
        "If I invest $10,000 at 5% annual interest compounded yearly, how much will I have after 3 years?",
    ]

    for query in examples:
        print(f"\n{'='*60}")
        print(f"Query: {query[:50]}...")
        print("=" * 60)

        # Get routing decision
        decision = router.route(query)
        print(f"\nRouting Decision:")
        print(f"  Lane: {decision.lane.value}")
        print(f"  Complexity: {decision.complexity_score:.3f}")
        print(f"  Reason: {decision.reason}")

        # Execute
        result = router.execute(query)
        print(f"\nResult:")
        print(f"  Cost: ${result.total_cost_usd:.6f}")
        print(f"  Latency: {result.total_latency_ms:.0f}ms")
        print(f"\nAnswer: {result.answer[:200]}...")

        if result.agentic_result and result.agentic_result.code:
            print(f"\nGenerated Code:\n{result.agentic_result.code}")


if __name__ == "__main__":
    main()
