#!/usr/bin/env python3
"""Test Tavily web search integration."""

import os
import sys

# Set API key for testing
os.environ["TAVILY_API_KEY"] = "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"

sys.path.insert(0, 'src')

from pal_router import OrchestratorRouter, OrchestratorConfig

# Mock infrastructure (only testing web_search, not other tools)
class MockInfrastructure:
    def get_client(self, model_name: str, tier: str = "fast"):
        """Return a mock client."""
        return MockClient()

class MockClient:
    def complete(self, query: str):
        from types import SimpleNamespace
        return SimpleNamespace(
            content=f"Mock answer for: {query}",
            cost_usd=0.0001,
        )

# Patch to only test web_search
import pal_router.orchestrator as orch_module
from pal_router.conversation import ToolResult, ToolCall

original_execute = orch_module.ToolRegistry.execute

def selective_execute(self, tool_call):
    """Only execute web_search for real, mock others."""
    if tool_call.name == "web_search":
        return self._execute_search(tool_call.parameters)
    elif tool_call.name == "final_answer":
        return self._execute_final_answer(tool_call.parameters)
    else:
        print(f"  [MOCK] Skipping: {tool_call.name}")
        return ToolResult(
            success=True,
            output=f"Mock result from {tool_call.name}",
            metadata={"cost_usd": 0.0001, "latency_ms": 100}
        )

orch_module.ToolRegistry.execute = selective_execute

config = OrchestratorConfig(
    backend="llamacpp",
    model_url="http://10.0.0.169:8080/v1",
    model_path="nvidia_Orchestrator-8B-Q8_0.gguf",
    max_rounds=3,
)

router = OrchestratorRouter(config=config, existing_infra=MockInfrastructure())

print("=" * 70)
print("Web Search Integration Test")
print("=" * 70)

# Direct test of web_search tool
print("\n[Direct Test] Calling web_search directly...")
print("-" * 70)

from pal_router.tools import ToolRegistry
tool_registry = ToolRegistry(config, MockInfrastructure())

result = tool_registry.execute(ToolCall(
    name="web_search",
    parameters={"query": "capital of France", "max_results": 3}
))

print(f"Success: {result.success}")
print(f"Output:\n{result.output[:500]}...")
print(f"Sources: {result.metadata.get('sources', [])}")

# Test through orchestrator
print("\n\n[Orchestrator Test] Query that triggers web_search...")
print("-" * 70)

query = "What is the current population of Paris, France?"
print(f"Query: {query}")

result = router.route_and_execute(query)
print(f"\nAnswer: {result.answer[:300]}...")
print(f"Cost: ${result.total_cost_usd:.4f}")
print(f"Latency: {result.total_latency_ms:.0f}ms")

print("\n" + "=" * 70)
