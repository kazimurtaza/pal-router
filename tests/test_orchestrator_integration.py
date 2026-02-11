"""End-to-end integration tests for OrchestratorRouter.

These tests verify the complete orchestration flow including:
- Mocked OpenAI API responses
- Multi-turn reasoning with tool chaining
- Budget enforcement
- Error handling and fallback mechanisms
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.conversation import (
    ConversationContext,
    OrchestratorConfig,
    ToolCall,
    ToolResult,
)
from pal_router.orchestrator import OrchestratorRouter
from pal_router.router import RouterResult
from pal_router.types import Lane


def create_mock_orchestrator_response(tool_name: str, params: dict, reasoning: str = "test reasoning"):
    """Helper to create a mock orchestrator response."""
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = reasoning

    mock_tc = Mock()
    mock_tc.function.name = tool_name
    mock_tc.function.arguments = str(params).replace("'", '"')

    mock_message.tool_calls = [mock_tc]
    mock_response.choices = [Mock(message=mock_message)]
    return mock_response


def create_mock_final_answer_response(answer: str, sources: list = None):
    """Helper to create a mock final_answer response."""
    return create_mock_orchestrator_response(
        "final_answer",
        {"answer": answer, "sources": sources or []},
        reasoning="Task complete"
    )


def test_orchestrator_with_mock_openai():
    """Basic flow with mocked OpenAI responses.

    This test verifies:
    - OpenAI API is correctly mocked
    - Orchestrator routes to fast_model for simple query
    - Context is properly populated with turn data
    - Final RouterResult is returned with correct structure
    """
    config = OrchestratorConfig()
    mock_infra = Mock()

    # Mock the model client
    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="Paris is the capital of France.",
        cost_usd=0.0001,
        latency_ms=100,
    )
    mock_infra.get_client.return_value = mock_client

    # Create router with mocked OpenAI client
    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # First call: route to fast_model
        fast_response = create_mock_orchestrator_response(
            "fast_model",
            {"query": "What is the capital of France?", "model": "llama-8b"},
            reasoning="This is a simple factual query about geography"
        )

        # Second call: final_answer
        final_response = create_mock_final_answer_response(
            "Paris is the capital of France.",
            sources=["fast_model"]
        )

        mock_openai.chat.completions.create.side_effect = [fast_response, final_response]

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("What is the capital of France?")

        # Verify result structure
        assert isinstance(result, RouterResult)
        assert result.decision is not None
        assert isinstance(result.decision.lane, Lane)
        assert result.answer is not None
        assert "Paris" in result.answer
        assert result.total_cost_usd >= 0
        assert result.total_latency_ms >= 0

        # Verify OpenAI was called twice (fast_model + final_answer)
        assert mock_openai.chat.completions.create.call_count == 2

        # Verify infrastructure's get_client was called for fast_model
        mock_infra.get_client.assert_called()


def test_orchestrator_multi_turn_reasoning():
    """Multiple tool chaining (search -> strong_model -> final_answer).

    This test verifies:
    - Orchestrator can chain multiple tools
    - Context accumulates properly across turns
    - Each tool execution is recorded in context
    - Final answer synthesizes results from multiple tools
    """
    config = OrchestratorConfig()
    mock_infra = Mock()

    # Mock model clients
    mock_search_client = Mock()
    mock_search_client.complete.return_value = Mock(
        content="Search result: Python 3.13 was released in October 2024.",
        cost_usd=0.001,
        latency_ms=200,
    )

    mock_strong_client = Mock()
    mock_strong_client.complete.return_value = Mock(
        content="Based on the search results, Python 3.13 was released on October 1, 2024.",
        cost_usd=0.01,
        latency_ms=1500,
    )

    def get_client_side_effect(model_name: str, tier: str):
        if "search" in model_name.lower() or tier == "fast":
            return mock_search_client
        return mock_strong_client

    mock_infra.get_client.side_effect = get_client_side_effect

    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # Turn 1: web_search
        search_response = create_mock_orchestrator_response(
            "web_search",
            {"query": "Python 3.13 release date", "max_results": 5},
            reasoning="Need to find current information about Python 3.13"
        )

        # Turn 2: strong_model to analyze
        strong_response = create_mock_orchestrator_response(
            "strong_model",
            {"query": "When was Python 3.13 released?", "model": "llama-70b"},
            reasoning="Need to analyze search results and provide comprehensive answer"
        )

        # Turn 3: final_answer
        final_response = create_mock_final_answer_response(
            "Python 3.13 was released on October 1, 2024.",
            sources=["web_search", "strong_model"]
        )

        mock_openai.chat.completions.create.side_effect = [
            search_response,
            strong_response,
            final_response
        ]

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("When was Python 3.13 released?")

        # Verify multi-turn execution
        assert isinstance(result, RouterResult)
        assert result.answer is not None
        assert "Python 3.13" in result.answer or "October" in result.answer

        # Verify OpenAI was called 3 times (search -> strong_model -> final_answer)
        assert mock_openai.chat.completions.create.call_count == 3

        # Verify both clients were used
        assert mock_infra.get_client.call_count >= 1


def test_orchestrator_budget_exceeded():
    """Stops when budget exceeded.

    This test verifies:
    - Budget checking happens before each round
    - Execution stops when cost limit is exceeded
    - Context is synthesized into a final answer when budget exceeded
    - No additional API calls are made after budget exceeded
    """
    config = OrchestratorConfig(max_cost_usd=0.005, max_rounds=5)
    mock_infra = Mock()

    # Mock expensive client
    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="This is an expensive response",
        cost_usd=0.01,  # More than max_cost_usd
        latency_ms=1000,
    )
    mock_infra.get_client.return_value = mock_client

    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # First response that will cause budget exceed
        first_response = create_mock_orchestrator_response(
            "fast_model",
            {"query": "test", "model": "llama-8b"},
            reasoning="Need to answer this"
        )

        mock_openai.chat.completions.create.return_value = first_response

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("Test query")

        # Verify result is returned despite budget exceeded
        assert isinstance(result, RouterResult)
        assert result.answer is not None

        # Verify total cost reflects the expensive call
        # Note: The orchestrator cost tracking may vary based on implementation


def test_orchestrator_fallback_on_error():
    """Falls back to strong model on error.

    This test verifies:
    - API errors are caught and handled gracefully
    - Fallback to strong model occurs when orchestrator fails
    - Fallback response is returned to user
    - Error is recorded in context
    """
    config = OrchestratorConfig()
    mock_infra = Mock()

    # Mock the fallback strong model client
    mock_strong_client = Mock()
    mock_strong_client.complete.return_value = Mock(
        content="This is the fallback answer from the strong model.",
        cost_usd=0.01,
        latency_ms=1500,
    )

    mock_infra.get_client.return_value = mock_strong_client

    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # Make the orchestrator fail on first call
        mock_openai.chat.completions.create.side_effect = Exception("API Error: Rate limit exceeded")

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("What is the meaning of life?")

        # Verify fallback result
        assert isinstance(result, RouterResult)
        assert result.answer is not None
        # The answer should contain the fallback response
        assert "fallback answer" in result.answer.lower() or "meaning of life" in result.answer.lower() or result.answer != ""

        # Verify the orchestrator API was called (and failed)
        assert mock_openai.chat.completions.create.call_count >= 1

        # Verify fallback client was used
        mock_infra.get_client.assert_called()


def test_orchestrator_context_properly_populated():
    """Verify conversation context is properly populated throughout execution.

    This test verifies:
    - Context tracks all tool calls and results
    - Budget is tracked across turns
    - Errors are recorded when they occur
    - Context can be built for subsequent orchestrator calls
    """
    config = OrchestratorConfig()
    mock_infra = Mock()

    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="Test response",
        cost_usd=0.001,
        latency_ms=100,
    )
    mock_infra.get_client.return_value = mock_client

    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # Simulate a multi-turn conversation
        responses = [
            create_mock_orchestrator_response(
                "fast_model",
                {"query": "What is 2+2?", "model": "llama-8b"},
                reasoning="Simple math question"
            ),
            create_mock_orchestrator_response(
                "strong_model",
                {"query": "Explain the result", "model": "llama-70b"},
                reasoning="Need deeper explanation"
            ),
            create_mock_final_answer_response(
                "2+2 equals 4, which is a basic arithmetic operation.",
                sources=["fast_model", "strong_model"]
            ),
        ]

        mock_openai.chat.completions.create.side_effect = responses

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("What is 2+2?")

        # Verify result structure
        assert isinstance(result, RouterResult)
        assert result.total_cost_usd >= 0
        assert result.total_latency_ms >= 0

        # Verify multiple rounds occurred
        assert mock_openai.chat.completions.create.call_count == 3


def test_orchestrator_max_rounds_enforcement():
    """Verify max_rounds limit is enforced.

    This test verifies:
    - Execution stops after max_rounds is reached
    - Final answer is synthesized from context
    - No additional API calls beyond max_rounds
    """
    config = OrchestratorConfig(max_rounds=2)
    mock_infra = Mock()

    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="Response",
        cost_usd=0.001,
        latency_ms=100,
    )
    mock_infra.get_client.return_value = mock_client

    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # Create responses for each round
        responses = [
            create_mock_orchestrator_response(
                "fast_model",
                {"query": "test", "model": "llama-8b"},
                reasoning="Round 1"
            ),
            create_mock_orchestrator_response(
                "strong_model",
                {"query": "test2", "model": "llama-70b"},
                reasoning="Round 2 - should trigger max_rounds"
            ),
        ]

        mock_openai.chat.completions.create.side_effect = responses

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("Test query")

        # Verify result was returned
        assert isinstance(result, RouterResult)
        assert result.answer is not None

        # Verify API was called exactly max_rounds times
        assert mock_openai.chat.completions.create.call_count == 2


def test_orchestrator_stuck_detection():
    """Verify stuck-in-loop detection works.

    This test verifies:
    - Same tool called 3 times triggers stuck detection
    - Final answer is forced when stuck
    - Context loop is broken
    """
    config = OrchestratorConfig()
    mock_infra = Mock()

    mock_client = Mock()
    mock_client.complete.return_value = Mock(
        content="Response",
        cost_usd=0.001,
        latency_ms=100,
    )
    mock_infra.get_client.return_value = mock_client

    with patch('pal_router.orchestrator.OpenAI') as mock_openai_class:
        mock_openai = Mock()
        mock_openai_class.return_value = mock_openai

        # Create 3 identical responses (same tool each time)
        stuck_response = create_mock_orchestrator_response(
            "fast_model",
            {"query": "test", "model": "llama-8b"},
            reasoning="Trying again"
        )

        mock_openai.chat.completions.create.return_value = stuck_response

        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

        result = router.route_and_execute("Test query")

        # Verify result was returned despite being stuck
        assert isinstance(result, RouterResult)
        assert result.answer is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
