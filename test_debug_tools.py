#!/usr/bin/env python3
"""Debug test to check if tools parameter works with llama.cpp."""

import sys
sys.path.insert(0, 'src')

from pal_router import OrchestratorConfig
from pal_router.tools import DEFAULT_TOOLS
import requests

config = OrchestratorConfig(
    backend='llamacpp',
    model_url='http://10.0.0.169:8080/v1',
    model_path='nvidia_Orchestrator-8B-Q8_0.gguf',
    max_rounds=1,
)

url = f"{config.model_url}/chat/completions"

system_prompt = """You are an orchestrator that solves tasks by calling tools.

## Available Tools
- fast_model: Simple facts, definitions. Fast & cheap.
- strong_model: Complex reasoning, analysis. Slower & expensive.
- code_executor: Math, computation. Runs Python.
- web_search: Current information, facts to verify.
- final_answer: Deliver your answer (REQUIRED to complete).

## Instructions
1. Analyze what's still needed to answer the question
2. Choose ONE tool that makes progress
3. Do NOT repeat failed approaches
4. When you have enough information, use final_answer"""

print("=" * 70)
print("Debug: Testing tools parameter with llama.cpp")
print("=" * 70)

# Test 1: Without tools (baseline)
print("\n[Test 1] Without tools parameter")
print("-" * 70)
response = requests.post(
    url,
    json={
        "model": config.model_path,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 1.0,
        "max_tokens": 500,
    },
    timeout=60,
)
response.raise_for_status()
data = response.json()
msg = data["choices"][0]["message"]
print(f"Content: {msg.get('content', 'None')[:200]}...")
print(f"Tool calls: {msg.get('tool_calls')}")

# Test 2: With tools parameter (DEFAULT_TOOLS)
print("\n[Test 2] With DEFAULT_TOOLS parameter")
print("-" * 70)
print(f"Number of tools: {len(DEFAULT_TOOLS)}")
print(f"First tool: {DEFAULT_TOOLS[0]['function']['name']}")

response = requests.post(
    url,
    json={
        "model": config.model_path,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "tools": DEFAULT_TOOLS,
        "temperature": 1.0,
        "max_tokens": 500,
    },
    timeout=60,
)
response.raise_for_status()
data = response.json()
msg = data["choices"][0]["message"]
print(f"Content: {msg.get('content', 'None')[:200]}...")
print(f"Tool calls: {msg.get('tool_calls')}")

# Test 3: Check raw response
print("\n[Test 3] Raw response dump")
print("-" * 70)
import json
print(json.dumps(data, indent=2)[:1000])

print("\n" + "=" * 70)
