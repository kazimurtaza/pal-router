#!/usr/bin/env python3
"""Debug test for OrchestratorRouter to trace execution."""

import sys
sys.path.insert(0, 'src')

from pal_router import OrchestratorRouter, OrchestratorConfig

# Patch to add debugging
import pal_router.orchestrator as orch_module

original_get_decision = orch_module.OrchestratorRouter._get_orchestrator_decision
original_execute = orch_module.ToolRegistry.execute

def debug_get_decision(self, context):
    print("  [DEBUG] Calling _get_orchestrator_decision...")
    try:
        result = original_get_decision(self, context)
        print(f"  [DEBUG] Got decision: is_final={result.is_final}, tool_calls={[tc.name for tc in result.tool_calls]}")
        return result
    except Exception as e:
        print(f"  [DEBUG] Error getting decision: {e}")
        import traceback
        traceback.print_exc()
        raise

def debug_execute(self, tool_call):
    print(f"  [DEBUG] Executing tool: {tool_call.name}")
    try:
        result = original_execute(self, tool_call)
        print(f"  [DEBUG] Result: success={result.success}, output={result.output[:100]}...")
        return result
    except Exception as e:
        print(f"  [DEBUG] Error executing tool: {e}")
        import traceback
        traceback.print_exc()
        raise

orch_module.OrchestratorRouter._get_orchestrator_decision = debug_get_decision
orch_module.ToolRegistry.execute = debug_execute

config = OrchestratorConfig(
    backend="llamacpp",
    model_url="http://10.0.0.169:8080/v1",
    model_path="nvidia_Orchestrator-8B-Q8_0.gguf",
    max_rounds=3,
    max_cost_usd=0.10,
)

router = OrchestratorRouter(config=config, existing_infra=None)

print("=" * 70)
print("Orchestrator Router Debug Test")
print("=" * 70)

query = "What is 2+2?"
print(f"\nQuery: {query}")
print("-" * 70)

try:
    result = router.route_and_execute(query)
    print(f"\nFinal Answer: {result.answer[:200]}...")
    print(f"Cost: ${result.total_cost_usd:.4f}")
    print(f"Latency: {result.total_latency_ms:.0f}ms")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
