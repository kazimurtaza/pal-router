#!/usr/bin/env python3
"""Test different Gemini models."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY not set. Add it to .env file.")
    sys.exit(1)
genai.configure(api_key=API_KEY)

# List available models
print("=== Available Gemini Models ===\n")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"  {model.name}")

# Test specific models (from available list)
TEST_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

print("\n=== Testing Models ===\n")

for model_name in TEST_MODELS:
    print(f"Testing {model_name}...", end=" ")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("What is 2+2? Just the number.")
        print(f"OK - Response: {response.text.strip()}")
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"FAILED - {error_msg}")
