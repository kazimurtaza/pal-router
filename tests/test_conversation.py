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
