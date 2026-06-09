"""PAL-Router conversation and configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def get_llamacpp_url() -> str:
    """Get llama.cpp server URL from environment or default."""
    import os

    return os.getenv("LLAMACPP_URL", "http://localhost:8080/v1")


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator router."""
    # Model settings
    model_path: str = "nvidia/Nemotron-Orchestrator-8B"
    backend: Literal["llamacpp", "vllm", "transformers"] = "llamacpp"
    model_url: str | None = None  # For llama.cpp server

    # Budget limits
    max_rounds: int = 5
    max_cost_usd: float = 0.10
    max_latency_ms: float = 10000

    # API keys for external tools
    tavily_api_key: str | None = None  # Tavily Search API key (or set TAVILY_API_KEY env var)

    # Tool settings - populated in tools.py
    tools: list[dict] = field(default_factory=list)

    # Groq models - using actual Groq model names directly
    # Orchestrator uses these exact names in tool_calls
    fast_models: dict[str, str] = field(default_factory=lambda: {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",  # Fast tier
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",  # Strong tier
    })

    strong_models: dict[str, str] = field(default_factory=lambda: {
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",  # Best quality
    })

    # Code models - Groq models for code execution
    # Uses same names as fast_models for consistency
    code_models: dict[str, str] = field(default_factory=lambda: {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",  # Fast code execution
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",  # Stronger code/complex tasks
    })


@dataclass
class OrchestratorDecision:
    """Decision from orchestrator model."""

    reasoning: str  # Orchestrator's reasoning
    tool_calls: list[ToolCall]  # Tools to execute
    is_final: bool = False  # Whether this ends with final_answer
    final_answer: str | None = None  # Final answer when is_final=True
    sources: list[str] | None = None  # Sources/tools used


@dataclass
class ToolCall:
    """Represents a tool call from orchestrator."""

    name: str  # Tool name (fast_model, strong_model, code_executor, web_search, final_answer)
    parameters: dict  # Tool parameters
    reasoning: str | None = None  # Orchestrator's reasoning for this call


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: str
    metadata: dict  # Cost, latency, sources, etc.


@dataclass
class ConversationTurn:
    """Single turn in conversation."""

    tool_call: ToolCall
    result: ToolResult
    reasoning: str | None = None


@dataclass
class ConversationContext:
    """Accumulated context across conversation."""
    original_query: str
    turns: list[ConversationTurn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    has_retried_no_tool: bool = False  # Track if we've tried answering without tools

    @property
    def total_cost(self) -> float:
        """Calculate total cost from all turns."""
        return sum(
            turn.result.metadata.get("cost_usd", 0)
            for turn in self.turns
            if turn.result.success
        )

    def build_prompt_context(self, config: OrchestratorConfig | None = None) -> str:
        """Build prompt context for the orchestrator from conversation history.

        Args:
            config: Configuration with tool definitions
            For backward compatibility, config can be None

        Returns:
            Formatted prompt string for orchestrator
        """
        from pal_router.tools import DEFAULT_TOOLS

        tools = config.tools if config else DEFAULT_TOOLS
        tool_list = "\n".join([
            f"- **{tool['function']['name']}**: {tool['function']['description']}"
            for tool in tools
        ])

        turn_summary = ""
        if self.turns:
            for i, turn in enumerate(self.turns[-2:], 1):  # Last 2 turns
                status = "✓" if turn.result.success else "✗"
                turn_summary += f"**Turn {i}** - {turn.tool_call.name}: {status}\n"

        steps_so_far = len(self.turns)
        return (
            f"## Original Question\n{self.original_query}\n\n"
            f"## Recent Conversation\n{turn_summary}\n\n"
            f"## Available Tools\n{tool_list}\n\n"
            f"## Current Status\n"
            f"We've taken {steps_so_far} step(s) so far. To solve this, I recommend:\n"
            "1. Use fast_model for simple factual questions\n"
            "2. Choose ONE tool that makes progress\n"
            "3. Do NOT repeat failed approaches\n"
            "4. When you have enough information, use final_answer\n\n"
            "What tool do you want to use?\n"
        )

    def budget_exceeded(self, config: OrchestratorConfig) -> bool:
        """Check if budget limits have been exceeded.

        Args:
            config: Orchestrator configuration with max_rounds, max_cost, max_latency

        Returns:
            True if any budget limit exceeded
        """
        # Check round limit
        if len(self.turns) >= config.max_rounds:
            return True

        # Check if we've tried answering without tools (fallback prevention)
        if self.has_retried_no_tool:
            return True

        # Check cost accumulation
        total_cost = sum(
            turn.result.metadata.get("cost_usd", 0) for turn in self.turns
            if turn.result.success
        )
        if total_cost > config.max_cost_usd:
            return True

        # Check latency
        total_latency = sum(
            turn.result.metadata.get("latency_ms", 0) for turn in self.turns
            if turn.result.success
        )
        if total_latency > config.max_latency_ms:
            return True

        return False

    def is_stuck(self) -> bool:
        """Check if orchestrator is stuck in a loop.

        Returns:
            True if stuck (same tool failing repeatedly or no progress)
        """
        # Check if we have recent errors
        if self.errors and len(self.errors) > 2:
            return True

        # Check if we've made multiple rounds without final answer
        if len(self.turns) > 3 and not any(
            turn.result.success and turn.tool_call.name == "final_answer"
            for turn in self.turns[-3:]
        ):
            return True

        # Check if same tool keeps failing
        recent_tools = [turn.tool_call.name for turn in self.turns[-3:]]
        if len(recent_tools) >= 3 and len(set(recent_tools)) == 1:
            # Same tool used 3+ times in a row
            return True

        return False
