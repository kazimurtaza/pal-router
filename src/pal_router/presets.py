"""Pre-configured router setups for common use cases."""

from __future__ import annotations

import os

from pal_router.config import (
    GEMINI_PRO,
    GROQ_LLAMA4_MAVERICK,
    GROQ_LLAMA_70B,
    GROQ_LLAMA_8B,
    LLAMACPP_QWEN,
    get_llamacpp_url,
)
from pal_router.models import (
    FallbackClient,
    GeminiClient,
    GroqClient,
    LlamaCppClient,
)
from pal_router.router import TernaryRouter


def create_fast_router(
    google_api_key: str | None = None,
    groq_api_key: str | None = None,
    llamacpp_url: str | None = None,
) -> TernaryRouter:
    """Create a fast router with 3-way fallback chain.

    Weak chain:  Gemini 3 Flash → Groq Llama 8B → Local Qwen3
    Strong:      Llama 4 Maverick on Groq (91ms!)

    Args:
        google_api_key: Google AI API key (or set GOOGLE_API_KEY env var)
        groq_api_key: Groq API key (or set GROQ_API_KEY env var)
        llamacpp_url: Local llama.cpp server URL (or set LLAMACPP_URL env var)
    """
    google_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    llama_url = llamacpp_url or get_llamacpp_url()

    # 3-way fallback chain: Gemini → Groq → Local
    gemini = GeminiClient(GEMINI_PRO, api_key=google_key)
    groq_weak = GroqClient(GROQ_LLAMA_8B, api_key=groq_key)
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llama_url)

    weak = FallbackClient(gemini, groq_weak, local)

    # Strong: Llama 4 Maverick (fastest on Groq!)
    strong = GroqClient(GROQ_LLAMA4_MAVERICK, api_key=groq_key)

    return TernaryRouter(
        weak_model=weak,
        strong_model=strong,
    )


def create_quality_router(
    google_api_key: str | None = None,
    groq_api_key: str | None = None,
    llamacpp_url: str | None = None,
) -> TernaryRouter:
    """Create a quality-focused router with best models.

    Weak chain:  Gemini 3 Flash → Groq Llama 8B → Local Qwen3
    Strong:      Llama 3.3 70B on Groq (best quality)

    Args:
        google_api_key: Google AI API key
        groq_api_key: Groq API key
        llamacpp_url: Local llama.cpp server URL (or set LLAMACPP_URL env var)
    """
    google_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    llama_url = llamacpp_url or get_llamacpp_url()

    gemini = GeminiClient(GEMINI_PRO, api_key=google_key)
    groq_weak = GroqClient(GROQ_LLAMA_8B, api_key=groq_key)
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llama_url)

    weak = FallbackClient(gemini, groq_weak, local)
    strong = GroqClient(GROQ_LLAMA_70B, api_key=groq_key)

    return TernaryRouter(
        weak_model=weak,
        strong_model=strong,
    )


def create_groq_only_router(
    api_key: str | None = None,
    llamacpp_url: str | None = None,
) -> TernaryRouter:
    """Create a Groq-only router with local fallback.

    Weak chain:  Llama 4 Maverick → Local Qwen3
    Strong:      Llama 3.3 70B (best quality)

    Args:
        api_key: Groq API key
        llamacpp_url: Local llama.cpp server URL (or set LLAMACPP_URL env var)
    """
    groq_key = api_key or os.getenv("GROQ_API_KEY")
    llama_url = llamacpp_url or get_llamacpp_url()

    groq_weak = GroqClient(GROQ_LLAMA4_MAVERICK, api_key=groq_key)
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llama_url)

    weak = FallbackClient(groq_weak, local)
    strong = GroqClient(GROQ_LLAMA_70B, api_key=groq_key)

    return TernaryRouter(
        weak_model=weak,
        strong_model=strong,
    )


def create_local_only_router(
    llamacpp_url: str | None = None,
) -> TernaryRouter:
    """Create a fully local router (no API keys needed, no rate limits).

    Uses local Qwen3-8B for both weak and strong.

    Args:
        llamacpp_url: Local llama.cpp server URL (or set LLAMACPP_URL env var)
    """
    llama_url = llamacpp_url or get_llamacpp_url()
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llama_url)

    return TernaryRouter(
        weak_model=local,
        strong_model=local,
    )
