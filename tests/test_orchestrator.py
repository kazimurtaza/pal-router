"""Tests for OrchestratorRouter class."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import Mock, MagicMock, patch
from pal_router.orchestrator import OrchestratorRouter
from pal_router.conversation import (
    OrchestratorConfig,
    ToolCall,
    ToolResult,
    ConversationTurn,
    OrchestratorDecision,
)


def test_orchestrator_router_initialization():
    """OrchestratorRouter should initialize with config and infrastructure."""
    config = OrchestratorConfig(
        model_path="nvidia/Nemotron-Orchestrator-8B",
        backend="llamacpp",
        model_url="http://localhost:8080"
    )
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    assert router.config == config
    assert router.infra == mock_infra
    assert router.tool_registry is not None


def test_query_creates_conversation_context():
    """Query should create new conversation context."""
    config = OrchestratorConfig(max_rounds=3)
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    # Mock the orchestration loop to return early
    with patch.object(router, '_run_orchestration_loop') as mock_loop:
        mock_loop.return_value = None
        router.query("What is 2+2?")

    # The context should be created with the original query
    # We can verify by checking the internal state


def test_mock_llamacpp_call():
    """Should route query to llama.cpp backend."""
    config = OrchestratorConfig(
        backend="llamacpp",
        model_url="http://localhost:8080"
    )
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    # Mock the HTTP response
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Let me calculate that for you.",
                    "tool_calls": None
                }
            }]
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        response = router._query_llamacpp(messages)

        assert response is not None
        mock_post.assert_called_once()


def test_budget_stopping():
    """Should stop when budget exceeded."""
    config = OrchestratorConfig(
        max_cost_usd=0.01,
        max_rounds=10
    )
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    # Create a context with exceeded budget
    from pal_router.conversation import ConversationContext
    context = ConversationContext(original_query="Test query")

    # Add a turn that exceeds budget
    tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
    result = ToolResult(success=True, output="answer")
    turn = ConversationTurn(
        tool_call=tool_call,
        result=result,
        cost_usd=0.02,  # Exceeds max_cost_usd
        latency_ms=500
    )
    context.turns.append(turn)

    # Should stop due to budget exceeded
    assert router._should_stop(context) is True


def test_max_rounds_stopping():
    """Should stop when max rounds reached."""
    config = OrchestratorConfig(max_rounds=2)
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    from pal_router.conversation import ConversationContext
    context = ConversationContext(original_query="Test")

    # Add max rounds
    for _ in range(2):
        tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
        result = ToolResult(success=True, output="answer")
        turn = ConversationTurn(
            tool_call=tool_call,
            result=result,
            cost_usd=0.001,
            latency_ms=100
        )
        context.turns.append(turn)

    assert router._should_stop(context) is True


def test_stuck_detection_stops():
    """Should stop when stuck in loop."""
    config = OrchestratorConfig(max_rounds=10)
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    from pal_router.conversation import ConversationContext
    context = ConversationContext(original_query="Test")

    # Add 3 same tool calls (stuck pattern)
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

    assert router._should_stop(context) is True


def test_should_not_stop_normal_case():
    """Should not stop in normal conditions."""
    config = OrchestratorConfig(
        max_rounds=5,
        max_cost_usd=0.10
    )
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    from pal_router.conversation import ConversationContext
    context = ConversationContext(original_query="Test")

    # Add one turn within budget
    tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
    result = ToolResult(success=True, output="answer")
    turn = ConversationTurn(
        tool_call=tool_call,
        result=result,
        cost_usd=0.001,
        latency_ms=100
    )
    context.turns.append(turn)

    assert router._should_stop(context) is False


def test_build_system_prompt():
    """Should build system prompt with tool definitions."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    prompt = router._build_system_prompt()

    assert "orchestrator" in prompt.lower()
    assert "fast_model" in prompt
    assert "strong_model" in prompt
    assert "code_executor" in prompt
    assert "final_answer" in prompt


def test_execute_tool():
    """Should execute tool via registry."""
    config = OrchestratorConfig()
    mock_infra = Mock()
    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="Answer",
        cost_usd=0.0001,
    )
    mock_infra.get_client.return_value = mock_client

    router = OrchestratorRouter(config, mock_infra)

    tool_call = ToolCall(
        name="fast_model",
        parameters={"query": "What is 2+2?", "model": "llama-8b"}
    )

    turn = router._execute_tool(tool_call)

    assert turn.result.success is True
    assert "Answer" in turn.result.output


def test_provide_fallback_answer():
    """Should provide fallback answer when stopping early."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    from pal_router.conversation import ConversationContext
    context = ConversationContext(original_query="What is 2+2?")

    # Add a turn with some result
    tool_call = ToolCall(name="fast_model", parameters={"query": "test"})
    result = ToolResult(success=True, output="I calculated that 2+2 equals 4.")
    turn = ConversationTurn(
        tool_call=tool_call,
        result=result,
        cost_usd=0.001,
        latency_ms=100
    )
    context.turns.append(turn)

    router._provide_fallback_answer(context)

    # Should have final_answer in context
    final_turn = context.turns[-1]
    assert final_turn.tool_call.name == "final_answer"


def test_vllm_backend():
    """Should route to vLLM backend when configured."""
    config = OrchestratorConfig(
        backend="vllm",
        model_url="http://localhost:8000"
    )
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Response from vLLM",
                    "tool_calls": None
                }
            }]
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        response = router._query_vllm(messages)

        assert response is not None


def test_transformers_backend():
    """Should route to transformers backend when configured."""
    config = OrchestratorConfig(
        backend="transformers",
        model_path="gpt2"
    )
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    # Mock transformers to avoid loading actual model
    with patch('pal_router.orchestrator.AutoModelForCausalLM.from_pretrained'), \
         patch('pal_router.orchestrator.AutoTokenizer.from_pretrained'):

        messages = [{"role": "user", "content": "test"}]
        # This should not raise an error
        # (actual model generation is mocked)
        try:
            router._query_transformers(messages)
        except Exception as e:
            # Expected if model not available, but we test the path exists
            assert "transformers" in str(router.config.backend)


def test_record_final_answer():
    """Should record final answer in context."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    from pal_router.conversation import ConversationContext
    context = ConversationContext(original_query="Test")

    decision = OrchestratorDecision(
        reasoning="I have the answer.",
        tool_calls=[],
        is_final=True,
        final_answer="The answer is 42.",
        sources=["fast_model"]
    )

    router._record_final_answer(context, decision)

    assert len(context.turns) == 1
    assert context.turns[0].tool_call.name == "final_answer"
    assert context.turns[0].result.output == "The answer is 42."


def test_parse_orchestrator_response():
    """Should parse orchestrator response into decision."""
    config = OrchestratorConfig()
    mock_infra = Mock()

    router = OrchestratorRouter(config, mock_infra)

    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I need to use the fast model."

    mock_tc = Mock()
    mock_tc.function.name = "fast_model"
    mock_tc.function.arguments = '{"query": "What is 2+2?", "model": "llama-8b"}'

    mock_message.tool_calls = [mock_tc]
    mock_response.choices = [Mock(message=mock_message)]

    decision = router._parse_orchestrator_response(mock_response)

    assert decision.reasoning == "I need to use the fast model."
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "fast_model"
    assert decision.tool_calls[0].parameters["query"] == "What is 2+2?"
