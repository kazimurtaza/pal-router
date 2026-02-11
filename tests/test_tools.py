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
