#!/usr/bin/env python3
"""Diagnostic test for Orchestrator-8B to identify setup issues."""

import requests
import json

config = {
    "url": "http://10.0.0.169:8080/v1/chat/completions",
    "model": "nvidia_Orchestrator-8B-Q4_K_M.gguf"
}

print("=" * 70)
print("Orchestrator-8B Diagnostic Test")
print("=" * 70)

# Test 1: Simple prompt without tools (should always work)
print("\n[Test 1] Simple completion (no tools)")
print("-" * 70)

response = requests.post(
    config["url"],
    json={
        "model": config["model"],
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    },
    timeout=60,
)
response.raise_for_status()
data = response.json()
content = data["choices"][0]["message"]["content"]
print(f"Response: {content}")

# Test 2: With ChatML format manually
print("\n[Test 2] Manual ChatML format")
print("-" * 70)

chatml_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
"""

response = requests.post(
    config["url"],
    json={
        "model": config["model"],
        "messages": [
            {"role": "user", "content": chatml_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    },
    timeout=60,
)
response.raise_for_status()
data = response.json()
content = data["choices"][0]["message"]["content"]
print(f"Response: {content}")

# Test 3: Check if llama.cpp supports tools parameter
print("\n[Test 3] Testing OpenAI-style tools parameter")
print("-" * 70)

tools = [{
    "type": "function",
    "function": {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "arg": {"type": "string"}
            }
        }
    }
}]

response = requests.post(
    config["url"],
    json={
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "You are an orchestrator. Use tools when needed."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "tools": tools,
        "temperature": 1.0,
        "max_tokens": 500,
    },
    timeout=60,
)
response.raise_for_status()
data = response.json()
msg = data["choices"][0]["message"]
print(f"Content: {msg.get('content', 'None')}")
print(f"Tool calls: {msg.get('tool_calls', 'None')}")

# Test 4: Get model info from server
print("\n[Test 4] Server model info")
print("-" * 70)

try:
    response = requests.get("http://10.0.0.169:8080/v1/models", timeout=10)
    response.raise_for_status()
    models = response.json()
    print(json.dumps(models, indent=2))
except Exception as e:
    print(f"Could not get model info: {e}")

print("\n" + "=" * 70)
