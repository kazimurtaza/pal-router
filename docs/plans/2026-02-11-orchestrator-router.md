# OrchestratorRouter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace PAL-Router's embedding+MLP classifier with NVIDIA's Nemotron-Orchestrator-8B for intelligent multi-tool orchestration.

**Architecture:** LLM-based orchestrator (via llama.cpp) that outputs OpenAI-style tool calls to coordinate fast_model, strong_model, code_executor, web_search, and final_answer. Reuses existing `ModelClient`, `AgenticWorkflow` infrastructure.

**Tech Stack:** Python 3.10+, llama.cpp server, OpenAI-compatible API, pydantic, existing PAL-Router infrastructure

---

## Overview

This adds a new `OrchestratorRouter` class alongside (not replacing) the existing `TernaryRouter`. The orchestrator:

1. Uses Nemotron-Orchestrator-8B (GGUF) running on llama.cpp server
2. Outputs structured tool calls (OpenAI function calling format)
3. Executes tools through existing PAL-Router infrastructure
4. Maintains conversation context across multiple rounds
5. Handles errors, budgets, and loop detection

### File Structure

```
src/pal_router/
├── orchestrator.py          # NEW: Main OrchestratorRouter class
├── tools.py                 # NEW: Tool definitions and registry
├── conversation.py          # NEW: ConversationContext, dataclasses
├── __init__.py              # MODIFY: Export new classes
```

```
tests/
├── test_orchestrator.py     # NEW: OrchestratorRouter tests
├── test_tools.py            # NEW: Tool registry tests
├── test_conversation.py     # NEW: Context management tests
```

---

## Task 1: Create Core Data Structures

**Files:**
- Create: `src/pal_router/conversation.py`
- Test: `tests/test_conversation.py`

**Step 1: Write the failing test**

Create `tests/test_conversation.py`:

```python
"""Tests for conversation context data structures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.conversation import (
    ConversationTurn,
    ConversationContext,
    OrchestratorConfig,
    ToolCall,
    ToolResult,
)

def test_conversation_turn_creation():
    """ConversationTurn should store all fields."""
    tool_call = ToolCall(
        name="fast_model",
        parameters={"query": "What is 2+2?", "model": "llama-8b"}
    )
    result = ToolResult(
        success=True,
        output="4",
        metadata={"cost_usd": 0.0001, "latency_ms": 100}
    )

    turn = ConversationTurn(
        tool_call=tool_call,
        result=result,
        cost_usd=0.0001,
        latency_ms=100
    )

    assert turn.tool_call.name == "fast_model"
    assert turn.result.output == "4"
    assert turn.cost_usd == 0.0001
    assert turn.latency_ms == 100

def test_conversation_context_empty():
    """Empty context should have zero totals."""
    context = ConversationContext(original_query="Test query")

    assert context.total_cost == 0.0
    assert context.total_latency == 0.0
    assert len(context.turns) == 0
    assert context.is_stuck() is False

def test_conversation_context_with_turns():
    """Context should aggregate totals from turns."""
    from pal_router.conversation import ConversationTurn, ToolCall, ToolResult

    context = ConversationContext(original_query="Test")

    # Add a turn
    tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
    result = ToolResult(success=True, output="answer", metadata={"cost_usd": 0.001})
    turn = ConversationTurn(
        tool_call=tool_call,
        result=result,
        cost_usd=0.001,
        latency_ms=200
    )
    context.turns.append(turn)

    assert context.total_cost == 0.001
    assert context.total_latency == 200

def test_is_stuck_detection():
    """Should detect when same tool called 3x in a row."""
    context = ConversationContext(original_query="Test")

    # Add 3 turns with same tool
    for _ in range(3):
        tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
        result = ToolResult(success=True, output="answer")
        turn = ConversationTurn(
            tool_call=tool_call,
            result=result,
            cost_usd=0.001,
            latency_ms=100
        )
        context.turns.append(turn)

    assert context.is_stuck() is True

def test_is_stuck_false_with_different_tools():
    """Should not be stuck when different tools used."""
    context = ConversationContext(original_query="Test")

    tools = ["fast_model", "strong_model", "code_executor"]
    for tool_name in tools:
        tool_call = ToolCall(name=tool_name, parameters={"query": "test"})
        result = ToolResult(success=True, output="answer")
        turn = ConversationTurn(
            tool_call=tool_call,
            result=result,
            cost_usd=0.001,
            latency_ms=100
        )
        context.turns.append(turn)

    assert context.is_stuck() is False

def test_budget_exceeded():
    """Should detect when budgets exceeded."""
    config = OrchestratorConfig(max_cost_usd=0.01, max_latency_ms=1000)
    context = ConversationContext(original_query="Test")

    # Add turn that exceeds cost
    tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
    result = ToolResult(success=True, output="answer")
    turn = ConversationTurn(
        tool_call=tool_call,
        result=result,
        cost_usd=0.02,  # Exceeds max
        latency_ms=500
    )
    context.turns.append(turn)

    assert context.budget_exceeded(config) is True

def test_build_prompt_context_empty():
    """Empty context should build minimal prompt."""
    context = ConversationContext(original_query="What is 2+2?")
    config = OrchestratorConfig()

    prompt = context.build_prompt_context(config)

    assert "What is 2+2?" in prompt
    assert "Budget" in prompt
    assert "Remaining:" in prompt

def test_orchestrator_config_defaults():
    """Config should have sensible defaults."""
    config = OrchestratorConfig()

    assert config.max_rounds == 5
    assert config.max_cost_usd == 0.10
    assert config.max_latency_ms == 10000
    assert config.backend == "llamacpp"
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/murtaza/projects/pal-router
pytest tests/test_conversation.py -v
```

Expected: `ModuleNotFoundError: No module named 'pal_router.conversation'`

**Step 3: Create the data structures**

Create `src/pal_router/conversation.py`:

```python
"""Conversation context and data structures for orchestrator router."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ToolCall:
    """Single tool invocation request from orchestrator."""
    name: str  # "fast_model", "code_executor", etc.
    parameters: dict  # {"query": "..."} or {"problem": "...", "model": "..."}
    reasoning: Optional[str] = None  # Why this tool (if provided)


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: str
    error: Optional[str] = None
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
    model_url: Optional[str] = None  # For llama.cpp server

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
        code_results = [t for t in self.turns if t.tool_call.name == "code_executor" and t.result.success]
        if code_results:
            outputs = "\n".join(f"```\n{t.result.output}\n```" for t in code_results[-2:])
            parts.append("## Code Results\n" + outputs)

        search_results = [t for t in self.turns if t.tool_call.name == "web_search" and t.result.success]
        if search_results:
            outputs = "\n".join(f"- {t.result.output[:200]}" for t in search_results[-2:])
            parts.append("## Search Results\n" + outputs)

        model_results = [t for t in self.turns if t.tool_call.name in ("fast_model", "strong_model") and t.result.success]
        if model_results:
            outputs = "\n".join(f"- {t.result.output[:300]}" for t in model_results[-2:])
            parts.append("## Model Responses\n" + outputs)

        # Errors
        if self.errors:
            parts.append("## Errors (avoid these)\n" + "\n".join(f"- {e}" for e in self.errors[-2:]))

        return "\n\n".join(parts)


@dataclass
class OrchestratorDecision:
    """Full output from orchestrator for one round."""
    reasoning: str  # Free-form thinking before tool calls
    tool_calls: list[ToolCall]  # Tools to execute (usually 1)
    is_final: bool  # Is final_answer included?
    final_answer: Optional[str] = None  # Answer if is_final=True
    sources: Optional[list[str]] = None  # Sources if final_answer
    raw_response: Optional[str] = None  # Raw LLM output for debugging
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/murtaza/projects/pal-router
pytest tests/test_conversation.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_conversation.py src/pal_router/conversation.py
git commit -m "feat: add conversation context data structures for orchestrator"
```

---

## Task 2: Create Tool Definitions and Registry

**Files:**
- Create: `src/pal_router/tools.py`
- Test: `tests/test_tools.py`

**Step 1: Write the failing test**

Create `tests/test_tools.py`:

```python
"""Tests for tool definitions and registry."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import Mock, MagicMock
from pal_router.tools import (
    ToolRegistry,
    DEFAULT_TOOLS,
    parse_orchestrator_response,
)
from pal_router.conversation import (
    OrchestratorConfig,
    ToolCall,
    ToolResult,
    OrchestratorDecision,
)


def test_default_tools_structure():
    """Default tools should have correct structure."""
    assert len(DEFAULT_TOOLS) == 5  # fast_model, strong_model, code_executor, web_search, final_answer

    tool_names = [t["function"]["name"] for t in DEFAULT_TOOLS]
    assert "fast_model" in tool_names
    assert "strong_model" in tool_names
    assert "code_executor" in tool_names
    assert "web_search" in tool_names
    assert "final_answer" in tool_names


def test_tool_schema_has_required_fields():
    """Each tool should have required OpenAI function calling fields."""
    for tool in DEFAULT_TOOLS:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool

        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func


def test_tool_registry_init():
    """ToolRegistry should initialize with config and infrastructure."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    registry = ToolRegistry(config, mock_infra)

    assert registry.config == config
    assert registry.infra == mock_infra
    assert len(registry._tools) == 5


def test_execute_fast_model():
    """Should execute fast_model using infrastructure."""
    from pal_router.tools import ToolRegistry

    config = OrchestratorConfig()
    mock_infra = Mock()
    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="The answer is 42",
        cost_usd=0.0001,
    )
    mock_infra.get_client.return_value = mock_client

    registry = ToolRegistry(config, mock_infra)
    tool_call = ToolCall(name="fast_model", parameters={"query": "What is the meaning?", "model": "llama-8b"})

    result = registry.execute(tool_call)

    assert result.success is True
    assert "42" in result.output
    assert result.metadata["cost_usd"] >= 0


def test_execute_code_executor():
    """Should execute code using AgenticWorkflow."""
    from pal_router.tools import ToolRegistry

    config = OrchestratorConfig()
    mock_infra = Mock()
    mock_client = Mock()

    # Mock AgenticWorkflow result
    mock_workflow = Mock()
    mock_workflow.execute.return_value = Mock(
        success=True,
        answer="30",
        total_cost_usd=0.001,
        total_latency_ms=500,
        code="print(15 * 2)",
        attempts=1,
    )
    mock_infra.get_agentic_workflow.return_value = mock_workflow

    registry = ToolRegistry(config, mock_infra)
    tool_call = ToolCall(
        name="code_executor",
        parameters={"problem": "Calculate 15 * 2", "model": "gpt-4o-mini"}
    )

    result = registry.execute(tool_call)

    assert result.success is True
    assert result.output == "30"
    assert result.metadata["code"] == "print(15 * 2)"


def test_execute_final_answer():
    """Final answer should just return the answer."""
    from pal_router.tools import ToolRegistry

    config = OrchestratorConfig()
    mock_infra = Mock()
    registry = ToolRegistry(config, mock_infra)

    tool_call = ToolCall(
        name="final_answer",
        parameters={"answer": "The final answer is 42", "sources": ["source1", "source2"]}
    )

    result = registry.execute(tool_call)

    assert result.success is True
    assert result.output == "The final answer is 42"
    assert result.metadata["sources"] == ["source1", "source2"]


def test_execute_unknown_tool_raises_error():
    """Unknown tools should raise ValueError."""
    from pal_router.tools import ToolRegistry

    config = OrchestratorConfig()
    mock_infra = Mock()
    registry = ToolRegistry(config, mock_infra)

    tool_call = ToolCall(name="unknown_tool", parameters={})

    try:
        registry.execute(tool_call)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown tool" in str(e)


def test_parse_orchestrator_response_with_tool_call():
    """Should parse OpenAI-style response with tool calls."""
    # Mock response with tool call
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I need to search for information."

    mock_tc = Mock()
    mock_tc.function.name = "web_search"
    mock_tc.function.arguments = '{"query": "latest AI news"}'

    mock_message.tool_calls = [mock_tc]
    mock_response.choices = [Mock(message=mock_message)]

    config = OrchestratorConfig()
    decision = parse_orchestrator_response(mock_response, config.tools)

    assert decision.reasoning == "I need to search for information."
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "web_search"
    assert decision.tool_calls[0].parameters["query"] == "latest AI news"
    assert decision.is_final is False


def test_parse_orchestrator_response_with_final_answer():
    """Should detect final_answer tool call."""
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I have the answer."

    mock_tc = Mock()
    mock_tc.function.name = "final_answer"
    mock_tc.function.arguments = '{"answer": "The answer is 42", "sources": []}'

    mock_message.tool_calls = [mock_tc]
    mock_response.choices = [Mock(message=mock_message)]

    config = OrchestratorConfig()
    decision = parse_orchestrator_response(mock_response, config.tools)

    assert decision.is_final is True
    assert decision.final_answer == "The answer is 42"


def test_parse_orchestrator_response_no_tool_calls():
    """Should handle response with no tool calls."""
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "Let me think about this..."
    mock_message.tool_calls = None
    mock_response.choices = [Mock(message=mock_message)]

    config = OrchestratorConfig()
    decision = parse_orchestrator_response(mock_response, config.tools)

    assert decision.reasoning == "Let me think about this..."
    assert len(decision.tool_calls) == 0
    assert decision.is_final is False
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/murtaza/projects/pal-router
pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'pal_router.tools'`

**Step 3: Create the tools module**

Create `src/pal_router/tools.py`:

```python
"""Tool definitions and registry for orchestrator router."""

from __future__ import annotations

import json
import time
from typing import Optional

from pal_router.conversation import (
    OrchestratorConfig,
    ToolCall,
    ToolResult,
    OrchestratorDecision,
)
from pal_router.config import Config as PalRouterConfig


# Tool definitions matching NVIDIA's approach
DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fast_model",
            "description": "Simple factual queries. Cost: ~$0.0001. Latency: <500ms. Use for: definitions, facts, simple Q&A, translations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to answer"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gpt-4o-mini", "claude-haiku", "llama-8b"],
                        "default": "llama-8b"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "strong_model",
            "description": "Complex reasoning tasks. Cost: ~$0.01. Latency: 1-3s. Use for: analysis, comparisons, nuanced questions, creative writing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to answer"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gpt-4o", "claude-sonnet", "llama-70b"],
                        "default": "llama-70b"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "Math and computation via Python code. Cost: ~$0.0001 + compute. Use for: arithmetic, formulas, data processing, anything with numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The problem to solve with code"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gpt-4o-mini", "claude-haiku", "qwen-coder"],
                        "default": "gpt-4o-mini"
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 30,
                        "description": "Max execution time in seconds"
                    }
                },
                "required": ["problem"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search web for current/missing information. Cost: ~$0.001. Use for: recent events, real-time data, facts you're unsure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer when task is complete. Always use this to deliver results to user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer to the user's query"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools/sources used to derive answer"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


class ToolRegistry:
    """Registry and executor for available tools."""

    def __init__(self, config: OrchestratorConfig, existing_infra):
        """Initialize tool registry.

        Args:
            config: Orchestrator configuration
            existing_infra: PAL-Router's existing infrastructure (ModelClient, AgenticWorkflow, etc.)
        """
        self.config = config
        self.infra = existing_infra
        self._tools = {t["function"]["name"]: t for t in config.tools or DEFAULT_TOOLS}
        self.config.tools = self.config.tools or DEFAULT_TOOLS

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return result.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with success status and output

        Raises:
            ValueError: If tool name is unknown
        """
        tool_name = tool_call.name
        params = tool_call.parameters

        if tool_name == "fast_model":
            return self._execute_fast_model(params)
        elif tool_name == "strong_model":
            return self._execute_strong_model(params)
        elif tool_name == "code_executor":
            return self._execute_code(params)
        elif tool_name == "web_search":
            return self._execute_search(params)
        elif tool_name == "final_answer":
            return self._execute_final_answer(params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_fast_model(self, params: dict) -> ToolResult:
        """Execute fast_model tool."""
        model_key = params.get("model", "llama-8b")
        model_name = self.config.fast_models.get(model_key, model_key)
        client = self.infra.get_client(model_name, tier="fast")

        start = time.perf_counter()
        response = client.complete(params["query"])
        latency_ms = (time.perf_counter() - start) * 1000

        return ToolResult(
            success=True,
            output=response.content,
            metadata={"cost_usd": response.cost_usd, "latency_ms": latency_ms}
        )

    def _execute_strong_model(self, params: dict) -> ToolResult:
        """Execute strong_model tool."""
        model_key = params.get("model", "llama-70b")
        model_name = self.config.strong_models.get(model_key, model_key)
        client = self.infra.get_client(model_name, tier="strong")

        start = time.perf_counter()
        response = client.complete(params["query"])
        latency_ms = (time.perf_counter() - start) * 1000

        return ToolResult(
            success=True,
            output=response.content,
            metadata={"cost_usd": response.cost_usd, "latency_ms": latency_ms}
        )

    def _execute_code(self, params: dict) -> ToolResult:
        """Execute code_executor tool using existing AgenticWorkflow."""
        from pal_router.agentic import AgenticWorkflow

        model_key = params.get("model", "gpt-4o-mini")
        model_name = self.config.code_models.get(model_key, model_key)
        client = self.infra.get_client(model_name, tier="fast")

        timeout = params.get("timeout", 30)

        # Reuse existing AgenticWorkflow
        workflow = AgenticWorkflow(
            weak_model=client,
            config=PalRouterConfig(agentic_timeout_seconds=timeout)
        )

        result = workflow.execute(params["problem"])

        return ToolResult(
            success=result.success,
            output=result.answer,
            metadata={
                "cost_usd": result.total_cost_usd,
                "latency_ms": result.total_latency_ms,
                "code": result.code,
                "attempts": result.attempts
            }
        )

    def _execute_search(self, params: dict) -> ToolResult:
        """Execute web_search tool.

        TODO: Implement actual search provider (Tavily, SerpAPI, Brave, etc.)
        """
        # Placeholder implementation
        query = params["query"]
        max_results = params.get("max_results", 5)

        # TODO: Implement actual search
        # For now, return a placeholder response
        return ToolResult(
            success=True,
            output=f"Search results for: {query} (TODO: implement search provider)",
            metadata={"sources": [], "query": query}
        )

    def _execute_final_answer(self, params: dict) -> ToolResult:
        """Final answer is just returned, no execution needed."""
        return ToolResult(
            success=True,
            output=params["answer"],
            metadata={"sources": params.get("sources", [])}
        )


def parse_orchestrator_response(
    response,
    tools: list[dict]
) -> OrchestratorDecision:
    """Parse OpenAI-style response into structured decision.

    Args:
        response: OpenAI chat completion response
        tools: List of tool definitions

    Returns:
        OrchestratorDecision with parsed data
    """
    message = response.choices[0].message
    tool_calls = []
    is_final = False
    final_answer = None
    sources = None

    # Parse tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                parameters = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                parameters = {}

            tc_obj = ToolCall(
                name=tc.function.name,
                parameters=parameters
            )
            tool_calls.append(tc_obj)

            # Check if this is final_answer
            if tc.function.name == "final_answer":
                is_final = True
                final_answer = parameters.get("answer")
                sources = parameters.get("sources")

    return OrchestratorDecision(
        reasoning=message.content or "",
        tool_calls=tool_calls,
        is_final=is_final,
        final_answer=final_answer,
        sources=sources,
        raw_response=str(message)
    )
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/murtaza/projects/pal-router
pytest tests/test_tools.py -v
```

Expected: Most tests PASS, but some may fail due to infrastructure mocking. Fix as needed.

**Step 5: Commit**

```bash
git add tests/test_tools.py src/pal_router/tools.py
git commit -m "feat: add tool definitions and registry for orchestrator"
```

---

## Task 3: Create OrchestratorRouter Class

**Files:**
- Create: `src/pal_router/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Step 1: Write the failing test**

Create `tests/test_orchestrator.py`:

```python
"""Tests for OrchestratorRouter class."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import Mock, MagicMock, patch
from pal_router.orchestrator import OrchestratorRouter, ORCHESTRATOR_PROMPT
from pal_router.conversation import (
    OrchestratorConfig,
    ConversationContext,
    OrchestratorDecision,
    ToolCall,
)
from pal_router.router import RouterResult


def test_orchestrator_router_init():
    """OrchestratorRouter should initialize with config and infrastructure."""
    config = OrchestratorConfig(model_url="http://localhost:8080/v1")
    mock_infra = Mock()
    mock_infra.get_client.return_value = Mock()

    with patch('pal_router.orchestrator.OpenAI'):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        assert router.config == config
        assert router.infra == mock_infra
        assert router.tool_registry is not None


def test_orchestrator_prompt_includes_all_sections():
    """Prompt should include all required sections."""
    assert "## Original Question" in ORCHESTRATOR_PROMPT
    assert "## Available Tools" in ORCHESTRATOR_PROMPT
    assert "fast_model" in ORCHESTRATOR_PROMPT
    assert "strong_model" in ORCHESTRATOR_PROMPT
    assert "code_executor" in ORCHESTRATOR_PROMPT
    assert "web_search" in ORCHESTRATOR_PROMPT
    assert "final_answer" in ORCHESTRATOR_PROMPT


def test_build_orchestrator_prompt():
    """Should build prompt with original query and context."""
    from pal_router.orchestrator import OrchestratorRouter

    config = OrchestratorConfig()
    mock_infra = Mock()

    with patch('pal_router.orchestrator.OpenAI'):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        context = ConversationContext(original_query="What is 2+2?")
        prompt = router._build_orchestrator_prompt(context)

        assert "What is 2+2?" in prompt
        assert "## Budget" in prompt


def test_synthesize_from_context():
    """Should create decision from accumulated context."""
    from pal_router.orchestrator import OrchestratorRouter

    config = OrchestratorConfig()
    mock_infra = Mock()

    with patch('pal_router.orchestrator.OpenAI'):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        context = ConversationContext(original_query="Test query")
        # Add a turn with result
        from pal_router.conversation import ToolCall, ToolResult, ConversationTurn
        turn = ConversationTurn(
            tool_call=ToolCall(name="fast_model", parameters={"query": "test"}),
            result=ToolResult(success=True, output="Test answer"),
            cost_usd=0.001,
            latency_ms=100
        )
        context.turns.append(turn)

        decision = router._synthesize_from_context(context)

        assert decision.is_final is True
        assert decision.final_answer is not None


def test_force_final_answer_when_stuck():
    """Should force final answer when stuck in loop."""
    from pal_router.orchestrator import OrchestratorRouter

    config = OrchestratorConfig()
    mock_infra = Mock()

    with patch('pal_router.orchestrator.OpenAI'):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        context = ConversationContext(original_query="Test")
        # Create stuck condition - 3 same tool calls
        from pal_router.conversation import ToolCall, ToolResult, ConversationTurn
        for _ in range(3):
            turn = ConversationTurn(
                tool_call=ToolCall(name="fast_model", parameters={"query": "test"}),
                result=ToolResult(success=True, output="answer"),
                cost_usd=0.001,
                latency_ms=100
            )
            context.turns.append(turn)

        assert context.is_stuck() is True

        decision = router._force_final_answer(context)

        assert decision.is_final is True
        assert "unable to complete" in decision.final_answer.lower() or "answer" in decision.final_answer.lower()


@patch('pal_router.orchestrator.OpenAI')
def test_route_and_execute_simple_query(mock_openai):
    """Should handle a simple query that routes to fast_model."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    # Mock model client
    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="The capital of France is Paris.",
        cost_usd=0.0001,
    )
    mock_infra.get_client.return_value = mock_client

    # Mock orchestrator response
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "This is a simple factual query."

    mock_tc = Mock()
    mock_tc.function.name = "fast_model"
    mock_tc.function.arguments = '{"query": "What is the capital of France?"}'

    mock_message.tool_calls = [mock_tc]
    mock_response.choices = [Mock(message=mock_message)]

    mock_openai.return_value.chat.completions.create.return_value = mock_response

    router = OrchestratorRouter(config=config, existing_infra=mock_infra)

    result = router.route_and_execute("What is the capital of France?")

    assert isinstance(result, RouterResult)
    assert "Paris" in result.answer
    assert result.total_cost_usd >= 0


@patch('pal_router.orchestrator.OpenAI')
def test_route_and_execute_with_final_answer(mock_openai):
    """Should handle orchestrator that directly gives final_answer."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    # Mock orchestrator response with final_answer
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I know this answer."

    mock_tc = Mock()
    mock_tc.function.name = "final_answer"
    mock_tc.function.arguments = '{"answer": "42", "sources": ["calculation"]}'

    mock_message.tool_calls = [mock_tc]
    mock_response.choices = [Mock(message=mock_message)]

    mock_openai.return_value.chat.completions.create.return_value = mock_response

    router = OrchestratorRouter(config=config, existing_infra=mock_infra)

    result = router.route_and_execute("What is the meaning of life?")

    assert result.answer == "42"
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/murtaza/projects/pal-router
pytest tests/test_orchestrator.py -v
```

Expected: `ModuleNotFoundError: No module named 'pal_router.orchestrator'`

**Step 3: Create the orchestrator module**

Create `src/pal_router/orchestrator.py`:

```python
"""Orchestrator-based router using Nemotron-Orchestrator-8B."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI
from transformers import AutoTokenizer

from pal_router.conversation import (
    ConversationContext,
    ConversationTurn,
    OrchestratorConfig,
    OrchestratorDecision,
    ToolCall,
)
from pal_router.router import RouterResult, RoutingDecision
from pal_router.tools import ToolRegistry, DEFAULT_TOOLS, parse_orchestrator_response
from pal_router.types import Lane


# Orchestrator prompt template
ORCHESTRATOR_PROMPT = """You are an orchestrator that solves tasks by calling tools.

## Original Question
{original_query}

## Available Tools
- fast_model: Simple facts, definitions. Fast & cheap.
- strong_model: Complex reasoning, analysis. Slower & expensive.
- code_executor: Math, computation. Runs Python.
- web_search: Current information, facts to verify.
- final_answer: Deliver your answer (REQUIRED to complete).

{context_string}

## Instructions
1. Analyze what's still needed to answer the question
2. Choose ONE tool that makes progress
3. Do NOT repeat failed approaches
4. When you have enough information, use final_answer

What tool do you want to use?"""


class OrchestratorRouter:
    """Main router using Nemotron-Orchestrator-8B.

    Replaces the embedding+MLP classifier with an LLM-based orchestrator.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        existing_infra = None,  # PAL-Router's existing infrastructure
    ):
        """Initialize the orchestrator router.

        Args:
            config: Orchestrator configuration
            existing_infra: PAL-Router infrastructure (ModelClient, AgenticWorkflow, etc.)
        """
        self.config = config or OrchestratorConfig()
        self.infra = existing_infra or _ExistingInfrastructure()
        self.tool_registry = ToolRegistry(self.config, self.infra)

        # Load orchestrator model
        self._orchestrator = self._load_orchestrator()
        self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def _load_orchestrator(self):
        """Load the orchestrator model based on backend."""
        if self.config.backend == "llamacpp":
            url = self.config.model_url or os.getenv("LLAMACPP_URL", "http://localhost:8080/v1")
            return OpenAI(base_url=url, api_key="not-needed")
        else:
            raise NotImplementedError(f"Backend {self.config.backend} not yet implemented")

    def route_and_execute(self, query: str) -> RouterResult:
        """Main entry point - route and execute a query.

        Replaces old TernaryRouter.execute()

        Args:
            query: The user's query

        Returns:
            RouterResult with decision, answer, cost, and latency
        """
        context = ConversationContext(original_query=query)
        start_time = time.perf_counter()

        decision = None

        for round_num in range(self.config.max_rounds):
            # Check budgets
            if context.budget_exceeded(self.config):
                decision = self._synthesize_from_context(context)
                break

            if context.is_stuck():
                decision = self._force_final_answer(context)
                break

            # Get orchestrator decision
            try:
                decision = self._get_orchestrator_decision(context)
            except Exception as e:
                context.errors.append(f"Orchestrator error: {e}")
                if len(context.turns) == 0:
                    # First round failed - fallback immediately
                    decision = self._fallback_to_strong_model(query)
                    break
                # Otherwise try to continue
                continue

            # Handle no tool calls
            if not decision.tool_calls and not decision.is_final:
                context.errors.append("No tool selected")
                if hasattr(context, 'has_retried_no_tool') and context.has_retried_no_tool:
                    decision = self._fallback_to_strong_model(query)
                    break
                context.has_retried_no_tool = True
                continue

            # Check if done
            if decision.is_final:
                break

            # Execute tool calls
            for tool_call in decision.tool_calls:
                try:
                    result = self.tool_registry.execute(tool_call)
                    context.turns.append(ConversationTurn(
                        tool_call=tool_call,
                        result=result,
                        cost_usd=result.metadata.get("cost_usd", 0),
                        latency_ms=result.metadata.get("latency_ms", 0)
                    ))
                except Exception as e:
                    error_msg = self._format_error_for_context(tool_call, e)
                    context.errors.append(error_msg)
                    context.turns.append(ConversationTurn(
                        tool_call=tool_call,
                        result=Mock(success=False, output="", error=str(e)),
                        cost_usd=0,
                        latency_ms=0
                    ))

            # Check if should continue
            if len(context.turns) >= self.config.max_rounds:
                decision = self._synthesize_from_context(context)
                break

        # Build result
        total_latency = (time.perf_counter() - start_time) * 1000
        total_cost = context.total_cost

        # Extract final answer
        if decision and decision.is_final:
            final_answer = decision.final_answer or ""
        elif decision and decision.tool_calls:
            final_answer = decision.tool_calls[0].parameters.get("answer", "")
        else:
            final_answer = "Unable to generate answer"

        return RouterResult(
            decision=self._build_routing_decision(context, decision),
            answer=final_answer,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            completions=[],  # Could populate from context.turns
        )

    def _get_orchestrator_decision(self, context: ConversationContext) -> OrchestratorDecision:
        """Get decision from orchestrator model.

        Args:
            context: Current conversation context

        Returns:
            OrchestratorDecision with tool calls and reasoning
        """
        prompt = self._build_orchestrator_prompt(context)

        response = self._orchestrator.chat.completions.create(
            model=self.config.model_path,
            messages=[{"role": "user", "content": prompt}],
            tools=self.config.tools,
            temperature=1.0,  # NVIDIA uses temp=1 for orchestrator
        )

        return parse_orchestrator_response(response, self.config.tools)

    def _build_orchestrator_prompt(self, context: ConversationContext) -> str:
        """Build prompt for orchestrator.

        Args:
            context: Current conversation context

        Returns:
            Formatted prompt string
        """
        context_string = context.build_prompt_context(self.config)
        return ORCHESTRATOR_PROMPT.format(
            original_query=context.original_query,
            context_string=context_string
        )

    def _synthesize_from_context(self, context: ConversationContext) -> OrchestratorDecision:
        """Create final decision from accumulated context.

        Args:
            context: Conversation context with accumulated turns

        Returns:
            OrchestratorDecision with synthesized final answer
        """
        # Gather all outputs
        outputs = []
        for turn in context.turns:
            if turn.result.success:
                outputs.append(turn.result.output)

        if outputs:
            answer = "\n\n".join(outputs[-2:])  # Last 2 outputs
        else:
            answer = "Unable to complete the task. The orchestrator encountered errors or exceeded budget limits."

        return OrchestratorDecision(
            reasoning="Synthesized from accumulated context",
            tool_calls=[],
            is_final=True,
            final_answer=answer,
            sources=[str(t.tool_call.name) for t in context.turns]
        )

    def _force_final_answer(self, context: ConversationContext) -> OrchestratorDecision:
        """Force final answer when stuck in loop.

        Args:
            context: Conversation context

        Returns:
            OrchestratorDecision with forced final answer
        """
        # Try to use the last successful result
        for turn in reversed(context.turns):
            if turn.result.success:
                return OrchestratorDecision(
                    reasoning="Using last successful result",
                    tool_calls=[],
                    is_final=True,
                    final_answer=turn.result.output,
                    sources=[turn.tool_call.name]
                )

        return OrchestratorDecision(
            reasoning="No successful results available",
            tool_calls=[],
            is_final=True,
            final_answer="I was unable to complete this task after multiple attempts. The task may require additional information or capabilities.",
            sources=[]
        )

    def _fallback_to_strong_model(self, query: str) -> OrchestratorDecision:
        """Fallback to strong model when orchestrator fails.

        Args:
            query: Original query

        Returns:
            OrchestratorDecision with strong model's answer
        """
        try:
            client = self.infra.get_client("claude-sonnet", tier="strong")
            response = client.complete(query)

            return OrchestratorDecision(
                reasoning="Fallback to strong model",
                tool_calls=[],
                is_final=True,
                final_answer=response.content,
                sources=["strong_model_fallback"]
            )
        except Exception:
            return OrchestratorDecision(
                reasoning="All fallbacks failed",
                tool_calls=[],
                is_final=True,
                final_answer="I'm sorry, but I encountered an error while processing your request.",
                sources=[]
            )

    def _format_error_for_context(self, tool_call: ToolCall, error: Exception) -> str:
        """Format error for inclusion in context.

        Args:
            tool_call: The tool call that failed
            error: The exception that occurred

        Returns:
            Formatted error string
        """
        return f"""Tool: {tool_call.name}
Params: {tool_call.parameters}
Error: {type(error).__name__}: {str(error)[:200]}

Consider: Try different parameters, use a different tool, or use strong_model as fallback."""

    def _build_routing_decision(self, context: ConversationContext, orchestrator_decision: OrchestratorDecision) -> RoutingDecision:
        """Build RoutingDecision for compatibility with existing API.

        Args:
            context: Conversation context
            orchestrator_decision: The orchestrator's final decision

        Returns:
            RoutingDecision compatible with existing RouterResult
        """
        # Determine which lane was primarily used
        if context.turns:
            last_tool = context.turns[-1].tool_call.name
            if last_tool == "fast_model":
                lane = Lane.FAST
            elif last_tool == "strong_model":
                lane = Lane.REASONING
            elif last_tool == "code_executor":
                lane = Lane.AGENTIC
            else:
                lane = Lane.REASONING  # Default
        else:
            lane = Lane.REASONING  # Default for orchestrator

        return RoutingDecision(
            lane=lane,
            complexity_score=0.5,  # Could compute from context
            signals=None,  # Could extract from context
            reason=orchestrator_decision.reasoning if orchestrator_decision else "Orchestrator routing",
            confidence=0.8,  # Could compute from tool success rates
        )


class _ExistingInfrastructure:
    """Default adapter for PAL-Router infrastructure when none provided."""

    def __init__(self):
        """Initialize with default clients."""
        from pal_router.models import get_client
        self._get_client = get_client

    def get_client(self, model_name: str, tier: str = "fast"):
        """Get a model client.

        Args:
            model_name: Name of the model
            tier: "fast" or "strong"

        Returns:
            ModelClient instance
        """
        return self._get_client(
            {"name": model_name, "cost_per_1k_input": 0.001, "cost_per_1k_output": 0.001},
            provider="openai"  # Default
        )
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/murtaza/projects/pal-router
pytest tests/test_orchestrator.py -v
```

Expected: Most tests PASS. Fix any failures related to mocking.

**Step 5: Commit**

```bash
git add tests/test_orchestrator.py src/pal_router/orchestrator.py
git commit -m "feat: add OrchestratorRouter class with LLM-based routing"
```

---

## Task 4: Update Package Exports

**Files:**
- Modify: `src/pal_router/__init__.py`

**Step 1: Add new exports**

Add to `src/pal_router/__init__.py`:

```python
"""Ternary LLM Router - Routes queries to Fast, Reasoning, or Agentic lanes."""

from pal_router.agentic import AgenticResult, AgenticWorkflow
from pal_router.complexity import ComplexitySignals, estimate_complexity
from pal_router.config import Config
from pal_router.models import CompletionResult, ModelClient, get_client, FallbackClient
from pal_router.presets import create_fast_router, create_quality_router, create_groq_only_router, create_local_only_router
from pal_router.router import RouterResult, TernaryRouter
from pal_router.types import Lane, RoutingDecision

# NEW: Orchestrator exports
from pal_router.orchestrator import OrchestratorRouter, OrchestratorConfig
from pal_router.tools import ToolRegistry, DEFAULT_TOOLS
from pal_router.conversation import (
    ConversationContext,
    ConversationTurn,
    OrchestratorDecision,
    ToolCall,
    ToolResult,
)

__all__ = [
    # Existing exports
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
    # NEW: Orchestrator exports
    "OrchestratorRouter",
    "OrchestratorConfig",
    "ToolRegistry",
    "DEFAULT_TOOLS",
    "ConversationContext",
    "ConversationTurn",
    "OrchestratorDecision",
    "ToolCall",
    "ToolResult",
]
__version__ = "0.2.0"  # Bump version
```

**Step 2: Verify exports work**

```bash
cd /home/murtaza/projects/pal-router
python -c "from pal_router import OrchestratorRouter, OrchestratorConfig; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/pal_router/__init__.py
git commit -m "feat: export orchestrator classes from package"
```

---

## Task 5: Add Integration Test

**Files:**
- Create: `examples/test_orchestrator.py`

**Step 1: Create integration test example**

Create `examples/test_orchestrator.py`:

```python
"""Integration test for OrchestratorRouter with real llama.cpp server.

This requires:
1. llama.cpp server running with Nemotron-Orchestrator-8B GGUF model
2. Set LLAMACPP_URL environment variable (default: http://localhost:8080/v1)
3. API keys for providers used in tools
"""

import os
from pal_router import OrchestratorRouter, OrchestratorConfig


def main():
    """Run integration tests against real orchestrator model."""

    # Check if server URL is set
    url = os.getenv("LLAMACPP_URL", "http://localhost:8080/v1")
    print(f"Using llama.cpp server at: {url}")

    # Create config
    config = OrchestratorConfig(
        model_url=url,
        max_rounds=3,
        max_cost_usd=0.05,
    )

    # Create router
    print("Initializing OrchestratorRouter...")
    router = OrchestratorRouter(config=config)

    # Test queries
    test_queries = [
        ("What is the capital of France?", "fast_model"),
        ("Analyze the themes in Hamlet.", "strong_model"),
        ("Calculate 15% of $200.", "code_executor"),
    ]

    for query, expected_tool in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Expected tool: {expected_tool}")
        print(f"{'='*60}")

        try:
            result = router.route_and_execute(query)

            print(f"Answer: {result.answer[:200]}...")
            print(f"Cost: ${result.total_cost_usd:.4f}")
            print(f"Latency: {result.total_latency_ms:.0f}ms")
            print(f"Lane: {result.decision.lane}")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'='*60}")
    print("Integration tests complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

**Step 2: Document setup requirements**

Create `docs/ORCHESTRATOR_SETUP.md`:

```markdown
# OrchestratorRouter Setup Guide

## Prerequisites

1. **llama.cpp server** with Nemotron-Orchestrator-8B GGUF model

### Download GGUF Model

```bash
# Option 1: MaziyarPanahi's GGUF (recommended)
huggingface-cli download MaziyarPanahi/Nemotron-Orchestrator-8B-GGUF \
  Nemotron-Orchestrator-8B.Q4_K_M.gguf \
  --local-dir ./models

# Option 2: bartowski's GGUF
huggingface-cli download bartowski/nvidia_Orchestrator-8B-GGUF \
  Orchestrator-8B-Q4_K_M.gguf \
  --local-dir ./models
```

### Start llama.cpp Server

```bash
./llama-server \
  --model ./models/Nemotron-Orchestrator-8B.Q4_K_M.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  --threads 8 \
  --n-gpu-layers 32 \
  --ctx-size 8192 \
  --metrics
```

### Verify Server

```bash
curl http://localhost:8080/v1/models
```

Should return model info.

## Running Integration Tests

```bash
# Set server URL
export LLAMACPP_URL=http://localhost:8080/v1

# Set API keys (for tools that use external models)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

# Run test
python examples/test_orchestrator.py
```

## Usage Example

```python
from pal_router import OrchestratorRouter, OrchestratorConfig

config = OrchestratorConfig(
    model_url="http://localhost:8080/v1",
    max_rounds=5,
    max_cost_usd=0.10,
)

router = OrchestratorRouter(config=config)

result = router.route_and_execute("Calculate compound interest on $10,000 at 5% for 3 years.")

print(result.answer)
print(f"Cost: ${result.total_cost_usd:.4f}")
print(f"Latency: {result.total_latency_ms:.0f}ms")
```

## Troubleshooting

### "Connection refused"
- Ensure llama.cpp server is running
- Check LLAMACPP_URL environment variable

### "Model not found"
- Verify GGUF file exists and is loaded by llama.cpp
- Check llama.cpp server logs

### Low tool call accuracy
- Ensure you're using the correct GGUF model (Orchestrator-8B, not base Qwen3)
- Check temperature is set to 1.0 (as per NVIDIA paper)
- Verify tool definitions match expected format
```

**Step 3: Commit**

```bash
git add examples/test_orchestrator.py docs/ORCHESTRATOR_SETUP.md
git commit -m "docs: add integration test and setup guide for orchestrator"
```

---

## Task 6: Create Preset for OrchestratorRouter

**Files:**
- Modify: `src/pal_router/presets.py`

**Step 1: Add orchestrator preset**

Add to `src/pal_router/presets.py`:

```python
"""Pre-configured router presets for common use cases."""

from __future__ import annotations

from pal_router.config import Config, ModelConfig
from pal_router.models import get_client
from pal_router.router import TernaryRouter

# NEW: Import orchestrator classes
from pal_router.orchestrator import OrchestratorRouter, OrchestratorConfig
from pal_router.tools import DEFAULT_TOOLS


# Existing presets (unchanged)
def create_fast_router() -> TernaryRouter:
    """Create a fast router with 3-way fallback."""
    config = Config(
        weak_model=ModelConfig("gpt-4o-mini", 0.00015, 0.0006),
        strong_model=ModelConfig("gpt-4o", 0.0025, 0.01),
    )
    weak_client = get_client(config.weak_model)
    strong_client = get_client(config.strong_model)
    return TernaryRouter(weak_model=weak_client, strong_model=strong_client)


# ... (other existing presets unchanged) ...

# NEW: Orchestrator presets
def create_orchestrator_router(
    model_url: str | None = None,
    max_rounds: int = 5,
    max_cost: float = 0.10,
) -> OrchestratorRouter:
    """Create an OrchestratorRouter with sensible defaults.

    Args:
        model_url: URL for llama.cpp server (default: from LLAMACPP_URL env var)
        max_rounds: Maximum number of tool-calling rounds
        max_cost: Maximum cost in USD before fallback

    Returns:
        Configured OrchestratorRouter instance
    """
    import os

    config = OrchestratorConfig(
        model_url=model_url or os.getenv("LLAMACPP_URL", "http://localhost:8080/v1"),
        max_rounds=max_rounds,
        max_cost_usd=max_cost,
        tools=DEFAULT_TOOLS,
    )

    return OrchestratorRouter(config=config)


def create_local_orchestrator() -> OrchestratorRouter:
    """Create an orchestrator router for fully-local usage.

    All tools use local models via llama.cpp. Requires:
    - Orchestrator-8B GGUF for routing
    - Qwen3-8B GGUF for fast_model
    - Optional: Larger models for strong_model
    """
    import os

    config = OrchestratorConfig(
        model_url=os.getenv("LLAMACPP_URL", "http://localhost:8080/v1"),
        max_rounds=5,
        max_cost_usd=0.0,  # All local = no cost
        tools=DEFAULT_TOOLS,
    )

    # Override model mappings to use local models only
    config.fast_models = {
        "llama-8b": "llama-3.1-8b-instant",  # Via Groq or local
    }
    config.strong_models = {
        "llama-70b": "llama-3.3-70b-versatile",  # Via Groq or local
    }
    config.code_models = {
        "gpt-4o-mini": "llama-3.1-8b-instant",  # Use local for code too
    }

    return OrchestratorRouter(config=config)
```

**Step 2: Update exports in __init__.py**

Modify `src/pal_router/__init__.py`:

```python
from pal_router.presets import (
    create_fast_router,
    create_quality_router,
    create_groq_only_router,
    create_local_only_router,
    create_orchestrator_router,  # NEW
    create_local_orchestrator,  # NEW
)
```

And update `__all__`:

```python
__all__ = [
    # ... existing exports ...
    "create_orchestrator_router",
    "create_local_orchestrator",
]
```

**Step 3: Commit**

```bash
git add src/pal_router/presets.py src/pal_router/__init__.py
git commit -m "feat: add orchestrator router presets"
```

---

## Task 7: Add Benchmark Comparison

**Files:**
- Create: `scripts/benchmark_orchestrator.py`

**Step 1: Create benchmark script**

Create `scripts/benchmark_orchestrator.py`:

```python
"""Benchmark comparing TernaryRouter vs OrchestratorRouter."""

import time
from pal_router import (
    TernaryRouter,
    OrchestratorRouter,
    create_orchestrator_router,
)

# Test queries from evaluation suite
TEST_QUERIES = [
    ("What is the capital of France?", "FAST", "Simple factual"),
    ("Analyze the causes of World War I.", "REASONING", "Complex analysis"),
    ("Calculate 15% of $200.", "AGENTIC", "Math computation"),
    ("What is 2^15?", "AGENTIC", "Math with exponent"),
    ("Explain quantum entanglement.", "REASONING", "Technical explanation"),
    ("Name three primary colors.", "FAST", "Simple list"),
    ("A farmer has chickens and cows. 20 heads, 56 legs. How many of each?", "AGENTIC", "Word problem"),
    ("Compare democracy and autocracy.", "REASONING", "Comparison"),
]


def benchmark_router(router, name: str):
    """Benchmark a single router."""
    results = {
        "name": name,
        "correct": 0,
        "total": 0,
        "total_cost": 0.0,
        "total_latency": 0.0,
        "errors": 0,
    }

    for query, expected_lane, description in TEST_QUERIES:
        try:
            start = time.perf_counter()
            result = router.execute(query) if name == "TernaryRouter" else router.route_and_execute(query)
            latency = (time.perf_counter() - start) * 1000

            results["total"] += 1
            results["total_cost"] += result.total_cost_usd
            results["total_latency"] += latency

            # Check if routing matches expected (simplified)
            actual_lane = result.decision.lane.value
            if actual_lane == expected_lane:
                results["correct"] += 1

            print(f"  [{description}] {actual_lane} - ${result.total_cost_usd:.4f} - {latency:.0f}ms")

        except Exception as e:
            results["errors"] += 1
            print(f"  [{description}] ERROR: {e}")

    return results


def main():
    """Run benchmark comparison."""
    print("="*70)
    print("PAL-Router Benchmark: TernaryRouter vs OrchestratorRouter")
    print("="*70)

    # Benchmark TernaryRouter
    print("\n[TernaryRouter]")
    try:
        ternary = TernaryRouter()
        ternary_results = benchmark_router(ternary, "TernaryRouter")
    except Exception as e:
        print(f"TernaryRouter failed: {e}")
        ternary_results = None

    # Benchmark OrchestratorRouter
    print("\n[OrchestratorRouter]")
    try:
        orchestrator = create_orchestrator_router()
        orchestrator_results = benchmark_router(orchestrator, "OrchestratorRouter")
    except Exception as e:
        print(f"OrchestratorRouter failed: {e}")
        orchestrator_results = None

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if ternary_results:
        print(f"\nTernaryRouter:")
        print(f"  Accuracy: {ternary_results['correct']}/{ternary_results['total']} ({100*ternary_results['correct']/ternary_results['total']:.1f}%)")
        print(f"  Avg Cost: ${ternary_results['total_cost']/ternary_results['total']:.4f}")
        print(f"  Avg Latency: {ternary_results['total_latency']/ternary_results['total']:.0f}ms")
        print(f"  Errors: {ternary_results['errors']}")

    if orchestrator_results:
        print(f"\nOrchestratorRouter:")
        print(f"  Accuracy: {orchestrator_results['correct']}/{orchestrator_results['total']} ({100*orchestrator_results['correct']/orchestrator_results['total']:.1f}%)")
        print(f"  Avg Cost: ${orchestrator_results['total_cost']/orchestrator_results['total']:.4f}")
        print(f"  Avg Latency: {orchestrator_results['total_latency']/orchestrator_results['total']:.0f}ms")
        print(f"  Errors: {orchestrator_results['errors']}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/benchmark_orchestrator.py
git commit -m "feat: add benchmark script comparing routers"
```

---

## Summary

This implementation plan creates a complete `OrchestratorRouter` that:

1. **Reuses existing infrastructure** - `ModelClient`, `AgenticWorkflow` unchanged
2. **Matches NVIDIA's approach** - OpenAI-style tool calling with 5 tools
3. **Handles edge cases** - Budget limits, loop detection, error recovery
4. **Is fully tested** - Unit tests for each component
5. **Is documented** - Setup guide and integration examples

### File Structure Summary

```
src/pal_router/
├── orchestrator.py          # NEW: Main OrchestratorRouter class
├── tools.py                 # NEW: Tool definitions and registry
├── conversation.py          # NEW: ConversationContext, dataclasses
├── __init__.py              # MODIFY: Export new classes
└── presets.py               # MODIFY: Add orchestrator presets

tests/
├── test_conversation.py     # NEW: Context management tests
├── test_tools.py            # NEW: Tool registry tests
└── test_orchestrator.py     # NEW: OrchestratorRouter tests

docs/
└── ORCHESTRATOR_SETUP.md    # NEW: Setup guide

examples/
└── test_orchestrator.py     # NEW: Integration test

scripts/
└── benchmark_orchestrator.py # NEW: Benchmark comparison
```

### Next Steps After Implementation

1. **Download GGUF model** from Hugging Face
2. **Start llama.cpp server** with the model
3. **Run integration tests** to verify functionality
4. **Benchmark** against existing TernaryRouter
5. **Iterate** on tool definitions and prompts based on results
