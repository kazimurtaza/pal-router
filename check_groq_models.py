#!/usr/bin/env python3
"""Check what models are available on Groq."""

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Get Groq API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment")
    print("Get a free key at: https://console.groq.com/")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

print("=" * 70)
print("Groq Available Models")
print("=" * 70)

# List models endpoint
print("\nCalling Groq models endpoint...")
print("-" * 70)

try:
    models = client.models.list()
    print(f"\nFound {len(models.data)} models:\n")

    # Group by model family
    families = {}
    for model in models.data:
        id = model.id
        # Extract family name (e.g., "llama-3.1-8b-instruct" -> "llama")
        family = id.split("-")[0] if "-" in id else id

        if family not in families:
            families[family] = []
        families[family].append(id)

    # Print by family
    for family, model_ids in sorted(families.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{family.upper()} ({len(model_ids)} models):")
        for model_id in sorted(model_ids):
            print(f"  - {model_id}")

    # Show Llama models specifically
    print("\n" + "=" * 70)
    print("LLAMA MODELS (most relevant for us):")
    print("=" * 70)

    llama_models = [m for m in models.data if m.id.startswith("llama")]
    for model in llama_models:
        print(f"\n{model.id}")
        if hasattr(model, "max_tokens"):
            print(f"  Context: {model.max_tokens:,} tokens")
        print(f"  Created: {model.created}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
