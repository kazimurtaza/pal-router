"""Pre-configured router setups for common use cases."""

import os

from pal_router.config import (
    GEMINI_PRO,
    GROQ_LLAMA4_MAVERICK,
    GROQ_LLAMA_70B,
    GROQ_LLAMA_8B,
    LLAMACPP_QWEN,
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
    llamacpp_url: str = "http://10.3.0.163:8080/v1",
    routellm_threshold: float = 0.12,
    use_routellm: bool = True,
) -> TernaryRouter:
    """Create a fast router with 3-way fallback chain.

    Weak chain:  Gemini 3 Flash → Groq Llama 8B → Local Qwen3
    Strong:      Llama 4 Maverick on Groq (91ms!)

    Args:
        google_api_key: Google AI API key (or set GOOGLE_API_KEY env var)
        groq_api_key: Groq API key (or set GROQ_API_KEY env var)
        llamacpp_url: Local llama.cpp server URL
        routellm_threshold: Threshold for RouteLLM MF router (default 0.12)
        use_routellm: Whether to use RouteLLM for weak vs strong routing
    """
    google_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")

    # 3-way fallback chain: Gemini → Groq → Local
    gemini = GeminiClient(GEMINI_PRO, api_key=google_key)
    groq_weak = GroqClient(GROQ_LLAMA_8B, api_key=groq_key)
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llamacpp_url)

    weak = FallbackClient(gemini, groq_weak, local)

    # Strong: Llama 4 Maverick (fastest on Groq!)
    strong = GroqClient(GROQ_LLAMA4_MAVERICK, api_key=groq_key)

    return TernaryRouter(
        weak_model=weak,
        strong_model=strong,
        routellm_threshold=routellm_threshold,
        use_routellm=use_routellm,
    )


def create_quality_router(
    google_api_key: str | None = None,
    groq_api_key: str | None = None,
    llamacpp_url: str = "http://10.3.0.163:8080/v1",
    routellm_threshold: float = 0.12,
    use_routellm: bool = True,
) -> TernaryRouter:
    """Create a quality-focused router with best models.

    Weak chain:  Gemini 3 Flash → Groq Llama 8B → Local Qwen3
    Strong:      Llama 3.3 70B on Groq (best quality)

    Args:
        google_api_key: Google AI API key
        groq_api_key: Groq API key
        llamacpp_url: Local llama.cpp server URL
        routellm_threshold: Threshold for RouteLLM MF router (default 0.12)
        use_routellm: Whether to use RouteLLM for weak vs strong routing
    """
    google_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")

    gemini = GeminiClient(GEMINI_PRO, api_key=google_key)
    groq_weak = GroqClient(GROQ_LLAMA_8B, api_key=groq_key)
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llamacpp_url)

    weak = FallbackClient(gemini, groq_weak, local)
    strong = GroqClient(GROQ_LLAMA_70B, api_key=groq_key)

    return TernaryRouter(
        weak_model=weak,
        strong_model=strong,
        routellm_threshold=routellm_threshold,
        use_routellm=use_routellm,
    )


def create_groq_only_router(
    api_key: str | None = None,
    llamacpp_url: str = "http://10.3.0.163:8080/v1",
    routellm_threshold: float = 0.12,
    use_routellm: bool = True,
) -> TernaryRouter:
    """Create a Groq-only router with local fallback.

    Weak chain:  Llama 4 Maverick → Local Qwen3
    Strong:      Llama 3.3 70B (best quality)

    Args:
        api_key: Groq API key
        llamacpp_url: Local llama.cpp server URL
        routellm_threshold: Threshold for RouteLLM MF router (default 0.12)
        use_routellm: Whether to use RouteLLM for weak vs strong routing
    """
    groq_key = api_key or os.getenv("GROQ_API_KEY")

    groq_weak = GroqClient(GROQ_LLAMA4_MAVERICK, api_key=groq_key)
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llamacpp_url)

    weak = FallbackClient(groq_weak, local)
    strong = GroqClient(GROQ_LLAMA_70B, api_key=groq_key)

    return TernaryRouter(
        weak_model=weak,
        strong_model=strong,
        routellm_threshold=routellm_threshold,
        use_routellm=use_routellm,
    )


def create_local_only_router(
    llamacpp_url: str = "http://10.3.0.163:8080/v1",
    use_routellm: bool = False,
) -> TernaryRouter:
    """Create a fully local router (no API keys needed, no rate limits).

    Uses local Qwen3-8B for both weak and strong.
    RouteLLM disabled by default (same model for both lanes).

    Args:
        llamacpp_url: Local llama.cpp server URL
        use_routellm: Whether to use RouteLLM (default False for local)
    """
    local = LlamaCppClient(LLAMACPP_QWEN, base_url=llamacpp_url)

    return TernaryRouter(
        weak_model=local,
        strong_model=local,
        use_routellm=use_routellm,
    )
