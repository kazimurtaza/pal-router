"""Tests for OrchestratorRouter class."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.conversation import (
    ConversationContext,
    ConversationTurn,
    OrchestratorConfig,
    ToolCall,
    ToolResult,
)
from pal_router.orchestrator import ORCHESTRATOR_PROMPT, OrchestratorRouter
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
    config = OrchestratorConfig()
    mock_infra = Mock()

    with patch('pal_router.orchestrator.OpenAI'):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        context = ConversationContext(original_query="Test query")
        # Add a turn with result
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
    config = OrchestratorConfig()
    mock_infra = Mock()

    with patch('pal_router.orchestrator.OpenAI'):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        context = ConversationContext(original_query="Test")
        # Create stuck condition - 3 same tool calls
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
        final_lower = decision.final_answer.lower()
        assert "unable to complete" in final_lower or "answer" in final_lower


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
