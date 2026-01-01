"""Ternary LLM Router - Routes queries to Fast, Reasoning, or Agentic lanes."""

from pal_router.agentic import AgenticResult, AgenticWorkflow
from pal_router.complexity import ComplexitySignals, estimate_complexity
from pal_router.config import Config
from pal_router.models import CompletionResult, ModelClient, get_client, FallbackClient
from pal_router.presets import create_fast_router, create_quality_router, create_groq_only_router, create_local_only_router
from pal_router.router import Lane, RoutingDecision, RouterResult, TernaryRouter

__all__ = [
    "AgenticResult",
    "AgenticWorkflow",
    "ComplexitySignals",
    "CompletionResult",
    "Config",
    "FallbackClient",
    "Lane",
    "ModelClient",
    "RoutingDecision",
    "RouterResult",
    "TernaryRouter",
    "create_fast_router",
    "create_groq_only_router",
    "create_local_only_router",
    "create_quality_router",
    "estimate_complexity",
    "get_client",
]
__version__ = "0.1.0"
