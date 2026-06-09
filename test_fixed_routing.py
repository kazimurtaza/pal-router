#!/usr/bin/env python3
"""Fixed test for Orchestrator-8B with proper prompt format."""

import requests
import json

config = {
    "url": "http://10.0.0.169:8080/v1/chat/completions",
    "model": "nvidia_Orchestrator-8B-Q8_0.gguf"  # Use actual loaded model
}

print("=" * 70)
print("Orchestrator-8B Fixed Routing Test")
print("=" * 70)

# Test queries
queries = [
    ("Complete question", "What is the capital of France?"),
    ("Short fragment", "Paris"),  # This will still fail likely
    ("Complete question 2", "What is 2+2?"),
    ("Complex", "Analyze the themes in Hamlet."),
]

for category, query in queries:
    # IMPORTANT: System message is REQUIRED!
    system_prompt = """You are an orchestrator that routes queries to appropriate tools.

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

For simple factual questions, use fast_model directly.
For math/computation, use code_executor.
For complex analysis, use strong_model.
When done, use final_answer with your final response."""

    response = requests.post(
        config["url"],
        json={
            "model": config["model"],
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

    print(f"\n{category}: {query}")
    print(f"  → Response: {content[:200]}...")
    if not content or content.strip() == "":
        print(f"  → EMPTY RESPONSE!")

print("\n" + "=" * 70)
