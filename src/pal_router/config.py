"""Configuration for the Ternary LLM Router."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# Environment variable for local llama.cpp server URL
DEFAULT_LLAMACPP_URL = "http://localhost:8080/v1"


def get_llamacpp_url() -> str:
    """Get llama.cpp server URL from environment or default."""
    return os.getenv("LLAMACPP_URL", DEFAULT_LLAMACPP_URL)


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD


# Default model configurations with current pricing (as of 2024)
OPENAI_GPT4O_MINI = ModelConfig(
    name="gpt-4o-mini",
    cost_per_1k_input=0.00015,
    cost_per_1k_output=0.0006,
)

OPENAI_GPT4O = ModelConfig(
    name="gpt-4o",
    cost_per_1k_input=0.0025,
    cost_per_1k_output=0.01,
)

ANTHROPIC_HAIKU = ModelConfig(
    name="claude-3-haiku-20240307",
    cost_per_1k_input=0.00025,
    cost_per_1k_output=0.00125,
)

ANTHROPIC_SONNET = ModelConfig(
    name="claude-3-5-sonnet-20241022",
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
)

# Groq models (free tier) - FASTEST provider!
GROQ_LLAMA_8B = ModelConfig(
    name="llama-3.1-8b-instant",  # Weak - very fast
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
)

GROQ_LLAMA_70B = ModelConfig(
    name="llama-3.3-70b-versatile",  # Strong - best quality
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
)

GROQ_LLAMA4_MAVERICK = ModelConfig(
    name="meta-llama/llama-4-maverick-17b-128e-instruct",  # FASTEST! 91ms
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
)

GROQ_LLAMA4_SCOUT = ModelConfig(
    name="meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
)

GROQ_GPT_OSS_120B = ModelConfig(
    name="openai/gpt-oss-120b",  # 120B params, very capable
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
)

GROQ_QWEN3_32B = ModelConfig(
    name="qwen/qwen3-32b",  # Good for code, has thinking mode
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
)

# Google Gemini models (free tier)
GEMINI_FLASH = ModelConfig(
    name="gemini-2.5-flash",  # Weak model
    cost_per_1k_input=0.0,  # Free tier
    cost_per_1k_output=0.0,
)

GEMINI_PRO = ModelConfig(
    name="gemini-3-flash-preview",  # Strong model (Gemini 3!)
    cost_per_1k_input=0.0,  # Free tier
    cost_per_1k_output=0.0,
)

# Local llama.cpp (Qwen3-8B-128K at 10.3.0.163)
LLAMACPP_QWEN = ModelConfig(
    name="/models/gguf/Qwen3-8B-128K-Q8_0.gguf",  # Local model
    cost_per_1k_input=0.0,  # Free - local
    cost_per_1k_output=0.0,
)


@dataclass
class Config:
    """Main configuration for the Ternary Router."""

    # Routing thresholds (calibrated based on test suite distribution)
    agentic_threshold: float = 0.20  # Complexity score above this → Agentic lane
    reasoning_threshold: float = 0.15  # Score above this with low numeric → Reasoning lane
    routellm_threshold: float = 0.12  # RouteLLM score (if used)

    # Model selection
    weak_model: ModelConfig = field(default_factory=lambda: GROQ_LLAMA_8B)
    strong_model: ModelConfig = field(default_factory=lambda: GROQ_LLAMA_70B)
    provider: Literal["openai", "anthropic", "groq", "gemini", "llamacpp"] = "groq"

    # Agentic workflow settings
    agentic_max_retries: int = 3
    agentic_timeout_seconds: int = 30
    agentic_memory_limit_mb: int = 512
    agentic_fallback_to_strong: bool = True  # Escalate to strong model on failure

    # Complexity scorer weights
    complexity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "syntactic_grade": 0.15,
            "logic_density": 0.30,
            "numeric_density": 0.25,
            "constraint_count": 0.15,
            "question_depth": 0.15,
        }
    )

    def get_model_config(self, strength: Literal["weak", "strong"]) -> ModelConfig:
        """Get the appropriate model config based on provider."""
        defaults = {
            "openai": (OPENAI_GPT4O_MINI, OPENAI_GPT4O),
            "anthropic": (ANTHROPIC_HAIKU, ANTHROPIC_SONNET),
            "groq": (GROQ_LLAMA_8B, GROQ_LLAMA_70B),
            "gemini": (GEMINI_FLASH, GEMINI_PRO),
            "llamacpp": (LLAMACPP_QWEN, LLAMACPP_QWEN),  # Same model for both
        }
        weak, strong = defaults.get(self.provider, (self.weak_model, self.strong_model))
        return weak if strength == "weak" else strong
