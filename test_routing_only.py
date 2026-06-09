#!/usr/bin/env python3
"""Test OrchestratorRouter routing decisions only (no tool execution)."""

import sys
sys.path.insert(0, 'src')

from pal_router import OrchestratorRouter, OrchestratorConfig
from pal_router.conversation import ToolResult

# Mock infrastructure that doesn't need API keys
class MockInfrastructure:
    def get_client(self, model_name: str, tier: str = "fast"):
        """Return a mock client that returns fake responses."""
        return MockClient()

class MockClient:
    def complete(self, query: str):
        """Return a mock response."""
        from types import SimpleNamespace
        return SimpleNamespace(
            content=f"Mock answer to: {query}",
            cost_usd=0.0001,
        )

# Patch ToolRegistry to use mock
import pal_router.orchestrator as orch_module

original_execute = orch_module.ToolRegistry.execute

def mock_execute(self, tool_call):
    """Mock execute that returns fake results."""
    print(f"  [MOCK] Executing: {tool_call.name} with {tool_call.parameters}")
    return ToolResult(
        success=True,
        output=f"Mock result from {tool_call.name}",
        metadata={"cost_usd": 0.0001, "latency_ms": 100}
    )

orch_module.ToolRegistry.execute = mock_execute

config = OrchestratorConfig(
    backend="llamacpp",
    model_url="http://10.0.0.169:8080/v1",
    model_path="nvidia_Orchestrator-8B-Q8_0.gguf",
    max_rounds=3,
    max_cost_usd=0.10,
)

router = OrchestratorRouter(config=config, existing_infra=MockInfrastructure())

print("=" * 70)
print("Orchestrator Router Test (Mock Execution)")
print("=" * 70)

queries = [
    ("Simple factual", "What is the capital of France?"),
    ("Computation", "What is 2+2?"),
    ("Complex", "Analyze the themes in Hamlet."),
]

for category, query in queries:
    print(f"\n{category}: {query}")
    print("-" * 70)

    try:
        result = router.route_and_execute(query)
        print(f"  Answer: {result.answer[:150]}...")
        print(f"  Cost: ${result.total_cost_usd:.4f}")
        print(f"  Latency: {result.total_latency_ms:.0f}ms")
        print(f"  Lane: {result.decision.lane.value}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
