"""Tests for the ternary router module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.router import Lane, RoutingDecision, TernaryRouter
from pal_router.config import Config
from pal_router.complexity import ComplexitySignals


class TestLane:
    """Tests for Lane enum."""

    def test_lane_values(self):
        """Lane enum should have expected values."""
        assert Lane.FAST.value == "FAST"
        assert Lane.REASONING.value == "REASONING"
        assert Lane.AGENTIC.value == "AGENTIC"


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """RoutingDecision should store all fields."""
        signals = ComplexitySignals(
            syntactic_grade=0.5,
            logic_density=0.3,
            numeric_density=0.2,
            constraint_count=1,
            question_depth=1,
            estimated_steps=2,
        )

        decision = RoutingDecision(
            lane=Lane.FAST,
            complexity_score=0.25,
            signals=signals,
            reason="Test reason",
        )

        assert decision.lane == Lane.FAST
        assert decision.complexity_score == 0.25
        assert decision.signals == signals
        assert decision.reason == "Test reason"
        assert decision.timestamp is not None


class TestTernaryRouterInit:
    """Tests for TernaryRouter initialization."""

    def test_with_mock_clients(self):
        """Router should initialize with mock clients."""
        weak_mock = Mock()
        strong_mock = Mock()
        router = TernaryRouter(
            weak_model=weak_mock,
            strong_model=strong_mock,
            use_routellm=False,
        )
        assert router.weak_model == weak_mock
        assert router.strong_model == strong_mock

    def test_custom_config(self):
        """Router should accept custom config."""
        config = Config(reasoning_threshold=0.5)
        weak_mock = Mock()
        strong_mock = Mock()
        router = TernaryRouter(
            config=config,
            weak_model=weak_mock,
            strong_model=strong_mock,
            use_routellm=False,
        )
        assert router.config.reasoning_threshold == 0.5


@pytest.fixture
def router():
    """Create a router with mock clients for testing."""
    weak_mock = Mock()
    strong_mock = Mock()
    return TernaryRouter(
        weak_model=weak_mock,
        strong_model=strong_mock,
        use_routellm=False,
    )


class TestRoutingToFast:
    """Tests for queries that should route to FAST lane."""

    def test_simple_factual(self, router):
        """Simple factual queries should go to FAST."""
        queries = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What color is the sky?",
            "Name three primary colors.",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.lane == Lane.FAST, f"Failed for: {query}"

    def test_simple_greeting(self, router):
        """Simple greetings should go to FAST."""
        queries = [
            "Hello",
            "Hi there",
            "Good morning",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.lane == Lane.FAST, f"Failed for: {query}"

    def test_trivial_math(self, router):
        """Trivial math that LLMs handle easily should go to FAST."""
        queries = [
            "What is 2 + 2?",
            "What is 10 - 5?",
            "What is 3 * 4?",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.lane == Lane.FAST, f"Failed for: {query}"

    def test_trivial_algebra(self, router):
        """Trivial algebra should go to FAST."""
        query = "If x = 5, what is x + 3?"
        decision = router.route(query)
        assert decision.lane == Lane.FAST


class TestRoutingToReasoning:
    """Tests for queries that should route to REASONING lane."""

    def test_conceptual_analysis(self, router):
        """Conceptual/analytical queries should go to REASONING."""
        queries = [
            "Analyze the themes in Hamlet.",
            "Discuss the causes of World War I.",
            "Explain quantum entanglement in simple terms.",
            "Compare and contrast democracy and autocracy.",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.lane == Lane.REASONING, f"Failed for: {query}"

    def test_ethical_implications(self, router):
        """Ethics questions should go to REASONING."""
        query = "What are the ethical implications of AI in healthcare?"
        decision = router.route(query)
        assert decision.lane == Lane.REASONING

    def test_pros_and_cons(self, router):
        """Comparative analysis should go to REASONING."""
        query = "What are the pros and cons of remote work?"
        decision = router.route(query)
        assert decision.lane == Lane.REASONING


class TestRoutingToAgentic:
    """Tests for queries that should route to AGENTIC lane."""

    def test_percentage_calculation(self, router):
        """Percentage calculations should go to AGENTIC."""
        query = "Calculate 15% of $200."
        decision = router.route(query)
        assert decision.lane == Lane.AGENTIC

    def test_compound_interest(self, router):
        """Financial calculations should go to AGENTIC."""
        query = "Calculate compound interest on $10,000 at 5% for 3 years."
        decision = router.route(query)
        assert decision.lane == Lane.AGENTIC

    def test_constraint_problem(self, router):
        """Constraint problems should go to AGENTIC."""
        query = "I have 3 boxes. Box A must weigh twice as much as Box B. Total weight is 35kg."
        decision = router.route(query)
        assert decision.lane == Lane.AGENTIC

    def test_word_problem(self, router):
        """Word problems should go to AGENTIC."""
        query = "A farmer has chickens and cows. There are 20 heads and 56 legs total. How many of each?"
        decision = router.route(query)
        assert decision.lane == Lane.AGENTIC

    def test_syllogism(self, router):
        """Syllogistic logic should go to AGENTIC."""
        query = "If all A are B, and all B are C, then are all A also C?"
        decision = router.route(query)
        assert decision.lane == Lane.AGENTIC

    def test_ordering_problem(self, router):
        """Ordering problems should go to AGENTIC."""
        query = "Alice finished before Bob but after Carol. What is the order?"
        decision = router.route(query)
        assert decision.lane == Lane.AGENTIC

    def test_combinatorics(self, router):
        """Combinatorics problems should go to AGENTIC."""
        queries = [
            "How many ways can you arrange the letters in HELLO?",
            "Calculate 5 choose 2.",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.lane == Lane.AGENTIC, f"Failed for: {query}"


class TestEdgeCases:
    """Tests for edge cases in routing."""

    def test_empty_string(self, router):
        """Empty string should route to FAST."""
        decision = router.route("")
        assert decision.lane == Lane.FAST

    def test_very_long_query(self, router):
        """Very long queries should not crash."""
        query = "Calculate the sum of " + ", ".join(str(i) for i in range(100))
        decision = router.route(query)
        assert decision.lane is not None

    def test_non_ascii(self, router):
        """Non-ASCII characters should be handled."""
        query = "What is 日本 in English?"
        decision = router.route(query)
        assert decision.lane is not None

    def test_mixed_content(self, router):
        """Mixed content should be routed appropriately."""
        # Has numbers but is primarily conceptual
        query = "Discuss the implications of Moore's Law for the next 10 years."
        decision = router.route(query)
        # Should go to REASONING because it's conceptual
        assert decision.lane == Lane.REASONING


class TestRoutingDecisionMetadata:
    """Tests for routing decision metadata."""

    def test_decision_has_reason(self, router):
        """Routing decision should include reason."""
        decision = router.route("What is 2 + 2?")
        assert decision.reason is not None
        assert len(decision.reason) > 0

    def test_decision_has_signals(self, router):
        """Routing decision should include complexity signals."""
        decision = router.route("Calculate 15% of $200.")
        assert decision.signals is not None
        assert decision.signals.numeric_density >= 0
