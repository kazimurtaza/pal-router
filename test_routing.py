#!/usr/bin/env python3
"""Test the Orchestrator-8B routing decisions directly."""

import sys
import json
import requests

sys.path.insert(0, 'src')
from pal_router import OrchestratorConfig

config = OrchestratorConfig(
    backend='llamacpp',
    model_url='http://10.0.0.169:8080/v1',
    model_path='nvidia_Orchestrator-8B-Q4_K_M.gguf',
    max_rounds=1,
)

# Test the orchestrator's routing decisions directly
url = f"{config.model_url}/chat/completions"

queries = [
    ("Simple factual", "What is the capital of France?"),
    ("Complex reasoning", "Analyze the themes in Hamlet."),
    ("Computation", "Calculate 15% of $200."),
]

print("=" * 70)
print("Testing Orchestrator-8B Routing Decisions")
print("=" * 70)

for category, query in queries:
    # System prompt is REQUIRED for Orchestrator-8B to respond
    system_prompt = """You are an orchestrator that routes queries to appropriate tools and models.

Available tools:
- fast_model: Simple facts, definitions. Fast & cheap.
- strong_model: Complex reasoning, analysis. Slower & expensive.
- code_executor: Math, computation. Runs Python.
- final_answer: Deliver your answer (REQUIRED to complete).

Your job:
1. Analyze the user's query
2. Choose the best tool for the job
3. Reformulate the query for that tool if needed
4. Continue until you have a complete answer
5. Use final_answer when done

Guidelines:
- Use fast_model for simple questions (facts, definitions)
- Use strong_model for complex analysis (themes, implications)
- Use code_executor for computation, math, logic problems
- Use final_answer when you have a satisfactory answer

Response format:
<tool_calls>
{{"tool": "tool_name", "parameters": {{"key": "value"}}}}
</tool_calls>"""

    response = requests.post(
        url,
        json={
            "model": config.model_path,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 1.0,  # NVIDIA uses temp=1
            "max_tokens": 500,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Extract the tool call
    import re
    match = re.search(r'"tool":\s*"(\w+)"', content)
    tool = match.group(1) if match else "unknown"

    print(f"\n{category}: {query}")
    print(f"  → Chosen tool: {tool}")
    print(f"  → Raw response: {content[:150]}...")

print("\n" + "=" * 70)
