#!/usr/bin/env python3
"""Quick test of all available providers."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.config import (
    GROQ_LLAMA_8B,
    GEMINI_FLASH,
    LLAMACPP_QWEN,
    ModelConfig,
)
from pal_router.models import (
    GroqClient,
    GeminiClient,
    LlamaCppClient,
)

TEST_PROMPT = "What is 2 + 2? Answer with just the number."


def test_groq():
    """Test Groq API."""
    print("\n=== Testing Groq (Llama 3.1 8B) ===")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("SKIPPED: GROQ_API_KEY not set")
        return None
    try:
        client = GroqClient(GROQ_LLAMA_8B, api_key=api_key)
        result = client.complete(TEST_PROMPT)
        print(f"Response: {result.content}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print("SUCCESS")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_gemini():
    """Test Google Gemini."""
    print("\n=== Testing Gemini (Flash) ===")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("SKIPPED: GOOGLE_API_KEY not set")
        return None
    try:
        client = GeminiClient(GEMINI_FLASH, api_key=api_key)
        result = client.complete(TEST_PROMPT)
        print(f"Response: {result.content}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print("SUCCESS")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_llamacpp():
    """Test local llama.cpp server."""
    print("\n=== Testing llama.cpp (Qwen2.5-Coder) ===")
    try:
        client = LlamaCppClient(
            LLAMACPP_QWEN,
            base_url="http://10.3.0.163:8080/v1",
        )
        result = client.complete(TEST_PROMPT)
        print(f"Response: {result.content}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print("SUCCESS")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("Testing all providers with prompt:", TEST_PROMPT)

    results = {
        "Groq": test_groq(),
        "Gemini": test_gemini(),
        "LlamaCpp": test_llamacpp(),
    }

    print("\n=== Summary ===")
    for name, success in results.items():
        if success is None:
            status = "SKIP"
        elif success:
            status = "OK"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
