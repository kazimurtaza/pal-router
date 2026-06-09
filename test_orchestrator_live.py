#!/usr/bin/env python3
"""Quick integration test for OrchestratorRouter with live llama.cpp server."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pal_router import (
    OrchestratorRouter,
    OrchestratorConfig,
    ToolRegistry,
)


def test_orchestrator_live():
    """Test the orchestrator with the live llama.cpp server."""
    config = OrchestratorConfig(
        backend="llamacpp",
        model_url="http://10.0.0.169:8080/v1",
        model_path="nvidia_Orchestrator-8B-Q8_0.gguf",  # Match server's loaded model
        max_rounds=3,
        max_cost_usd=0.10,
    )

    # Create router with no infrastructure (will use defaults)
    router = OrchestratorRouter(config=config, existing_infra=None)

    print("=" * 60)
    print("Orchestrator Router Live Test")
    print("=" * 60)
    print(f"Server: http://10.0.0.169:8080")
    print(f"Model: nvidia_Orchestrator-8B-Q8_0.gguf")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Max cost: ${config.max_cost_usd}")
    print("=" * 60)

    # Test a simple query
    query = "What is 2+2?"
    print(f"\nQuery: {query}")
    print("-" * 60)

    try:
        result = router.route_and_execute(query)

        print(f"\nResult:")
        print(f"  Answer: {result.answer[:200]}...")
        print(f"  Cost: ${result.total_cost_usd:.4f}")
        print(f"  Latency: {result.total_latency_ms:.0f}ms")
        print(f"  Lane: {result.decision.lane.value}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_orchestrator_live()
