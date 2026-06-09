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
from pal_router.orchestrator import OrchestratorRouter
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


def create_orchestrator_router(
    model_url: str | None = None,
    existing_infra = None,
) -> OrchestratorRouter:
    """Create an orchestrator-based router using Nemotron-Orchestrator-8B.

    The orchestrator router uses an LLM to intelligently route queries
    and orchestrate tool calls for complex multi-step tasks.

    Args:
        model_url: URL for the orchestrator model (llama.cpp server).
                   Defaults to LLAMACPP_URL env var or http://localhost:8080/v1
        existing_infra: PAL-Router infrastructure (ModelClient, AgenticWorkflow, etc.)

    Returns:
        Configured OrchestratorRouter instance
    """
    from pal_router.conversation import OrchestratorConfig

    url = model_url or get_llamacpp_url()
    config = OrchestratorConfig(model_url=url)

    return OrchestratorRouter(
        config=config,
        existing_infra=existing_infra,
    )


def create_groq_orchestrator_router(
    groq_api_key: str | None = None,
    model_url: str | None = None,
    model_path: str = "nvidia_Orchestrator-8B-Q8_0.gguf",
) -> OrchestratorRouter:
    """Create an Orchestrator-8B router with Groq-powered tools (RECOMMENDED).

    This is the recommended configuration for production use:
    - Orchestrator-8B: Routes queries to appropriate tools (local llama.cpp)
    - fast_model: Groq Llama 3.1 8B (FREE, ~50ms latency)
    - strong_model: Groq Llama 3.3 70B (FREE, ~100ms latency)
    - code_executor: Groq + Python execution
    - web_search: Tavily Search API (FREE tier available)

    Total cost: $0 (all free tiers!)

    Requires:
        - GROQ_API_KEY environment variable (get free key at https://groq.com/)
        - TAVILY_API_KEY environment variable (get free key at https://tavily.com/)
        - Local llama.cpp server running Orchestrator-8B model

    Args:
        groq_api_key: Groq API key. If None, uses GROQ_API_KEY env var.
        model_url: llama.cpp server URL for Orchestrator-8B model.
                   Defaults to LLAMACPP_URL env var or http://localhost:8080/v1
        model_path: Name/path of the Orchestrator-8B GGUF model

    Returns:
        Configured OrchestratorRouter instance with Groq infrastructure

    Example:
        from pal_router import create_groq_orchestrator_router

        router = create_groq_orchestrator_router()
        result = router.route_and_execute("What is the capital of France?")
        print(result.answer)  # "The capital of France is Paris..."
    """
    from pal_router.conversation import OrchestratorConfig
    from pal_router.orchestrator import _ExistingInfrastructure

    url = model_url or get_llamacpp_url()
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it as environment variable or pass groq_api_key parameter. "
            "Get a free key at: https://groq.com/"
        )

    config = OrchestratorConfig(
        model_url=url,
        model_path=model_path,
    )

    # Use Groq-powered infrastructure
    return OrchestratorRouter(
        config=config,
        existing_infra=_ExistingInfrastructure(provider="groq"),
    )


__all__ = [
    "create_fast_router",
    "create_groq_only_router",
    "create_local_only_router",
    "create_orchestrator_router",
    "create_quality_router",
    "create_groq_orchestrator_router",  # NEW: Recommended Groq + Orchestrator setup
]
