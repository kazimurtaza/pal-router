"""Conversation context and data structures for orchestrator router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ToolCall:
    """Single tool invocation request from orchestrator."""
    name: str  # "fast_model", "code_executor", etc.
    parameters: dict  # {"query": "..."} or {"problem": "...", "model": "..."}
    reasoning: str | None = None  # Why this tool (if provided)


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: str
    error: str | None = None
    metadata: dict = field(default_factory=dict)  # cost, latency, sources, etc.


@dataclass
class ConversationTurn:
    """One round of orchestrator -> tool execution -> result."""
    tool_call: ToolCall
    result: ToolResult
    cost_usd: float
    latency_ms: float


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator router."""
    # Model settings
    model_path: str = "nvidia/Nemotron-Orchestrator-8B"
    backend: Literal["llamacpp", "vllm", "transformers"] = "llamacpp"
    model_url: str | None = None  # For llama.cpp server

    # Budget limits
    max_rounds: int = 5
    max_cost_usd: float = 0.10
    max_latency_ms: float = 10000

    # Tool settings - populated in tools.py
    tools: list[dict] = field(default_factory=list)

    # Model mappings (tier -> actual model)
    fast_models: dict[str, str] = field(default_factory=lambda: {
        "gpt-4o-mini": "gpt-4o-mini",
        "claude-haiku": "claude-3-haiku-20240307",
        "llama-8b": "llama-3.1-8b-instant",
    })
    strong_models: dict[str, str] = field(default_factory=lambda: {
        "gpt-4o": "gpt-4o",
        "claude-sonnet": "claude-3-5-sonnet-20241022",
        "llama-70b": "llama-3.3-70b-versatile",
    })
    code_models: dict[str, str] = field(default_factory=lambda: {
        "gpt-4o-mini": "gpt-4o-mini",
        "claude-haiku": "claude-3-haiku-20240307",
        "qwen-coder": "qwen/qwen2.5-coder-32b-instruct",
    })


@dataclass
class ConversationContext:
    """Accumulated context across the conversation."""
    original_query: str
    turns: list[ConversationTurn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    has_retried_no_tool: bool = False

    @property
    def total_cost(self) -> float:
        """Total cost across all turns."""
        return sum(t.cost_usd for t in self.turns)

    @property
    def total_latency(self) -> float:
        """Total latency across all turns."""
        return sum(t.latency_ms for t in self.turns)

    def is_stuck(self) -> bool:
        """Detect stuck-in-loop patterns."""
        if len(self.turns) < 3:
            return False
        last_3_tools = [t.tool_call.name for t in self.turns[-3:]]
        return len(set(last_3_tools)) == 1

    def budget_exceeded(self, config: OrchestratorConfig) -> bool:
        """Check if budgets exceeded."""
        return (
            self.total_cost > config.max_cost_usd
            or self.total_latency > config.max_latency_ms
        )

    def build_prompt_context(self, config: OrchestratorConfig) -> str:
        """Build context string for next orchestrator call."""
        parts = []

        # Original query
        parts.append(f"## Query\n{self.original_query}")

        # Budget status
        remaining_cost = config.max_cost_usd - self.total_cost
        remaining_rounds = config.max_rounds - len(self.turns)
        parts.append(f"## Budget\nRemaining: ${remaining_cost:.4f}, {remaining_rounds} rounds")

        # What's been tried
        if self.turns:
            tried = []
            for t in self.turns:
                params = t.tool_call.parameters
                key = params.get("query") or params.get("problem") or str(params)
                tried.append(f"{t.tool_call.name}({key[:50]}...)")
            parts.append("## Already Tried\n" + "\n".join(f"- {t}" for t in tried))

        # Results by type
        code_results = [
            t for t in self.turns
            if t.tool_call.name == "code_executor" and t.result.success
        ]
        if code_results:
            outputs = "\n".join(f"```\n{t.result.output}\n```" for t in code_results[-2:])
            parts.append("## Code Results\n" + outputs)

        search_results = [
            t for t in self.turns
            if t.tool_call.name == "web_search" and t.result.success
        ]
        if search_results:
            outputs = "\n".join(f"- {t.result.output[:200]}" for t in search_results[-2:])
            parts.append("## Search Results\n" + outputs)

        model_results = [
            t for t in self.turns
            if t.tool_call.name in ("fast_model", "strong_model") and t.result.success
        ]
        if model_results:
            outputs = "\n".join(f"- {t.result.output[:300]}" for t in model_results[-2:])
            parts.append("## Model Responses\n" + outputs)

        # Errors
        if self.errors:
            error_list = "\n".join(f"- {e}" for e in self.errors[-2:])
            parts.append("## Errors (avoid these)\n" + error_list)

        return "\n\n".join(parts)


@dataclass
class OrchestratorDecision:
    """Full output from orchestrator for one round."""
    reasoning: str  # Free-form thinking before tool calls
    tool_calls: list[ToolCall]  # Tools to execute (usually 1)
    is_final: bool  # Is final_answer included?
    final_answer: str | None = None  # Answer if is_final=True
    sources: list[str] | None = None  # Sources if final_answer
    raw_response: str | None = None  # Raw LLM output for debugging
