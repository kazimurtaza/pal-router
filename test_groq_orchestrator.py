#!/usr/bin/env python3
"""Test Groq + Orchestrator integration."""

import os
import sys

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'src')

from pal_router import create_groq_orchestrator_router

print("=" * 70)
print("Groq + Orchestrator Integration Test")
print("=" * 70)

# Check API keys
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
llamacpp_url = os.getenv("LLAMACPP_URL")

print(f"\nConfiguration:")
print(f"  GROQ_API_KEY: {'✓ Set' if groq_key else '✗ Missing'}")
print(f"  TAVILY_API_KEY: {'✓ Set' if tavily_key else '✗ Missing'}")
print(f"  LLAMACPP_URL: {llamacpp_url}")

try:
    router = create_groq_orchestrator_router(
        model_url=llamacpp_url,
        model_path="nvidia_Orchestrator-8B-Q8_0.gguf"
    )

    print("\n" + "=" * 70)
    print("Testing Queries")
    print("=" * 70)

    queries = [
        ("Simple factual", "What is the capital of France?"),
        ("Math", "What is 15% of 200?"),
        ("Complex", "Compare Python and JavaScript."),
    ]

    for category, query in queries:
        print(f"\n[{category}] {query}")
        print("-" * 70)

        result = router.route_and_execute(query)

        print(f"Answer: {result.answer[:200]}...")
        print(f"Cost: ${result.total_cost_usd:.4f}")
        print(f"Latency: {result.total_latency_ms:.0f}ms")
        print(f"Lane: {result.decision.lane.value}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
