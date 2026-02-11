"""Benchmark comparison between TernaryRouter and OrchestratorRouter.

This test compares routing decisions, latency, and cost between:
1. TernaryRouter - Uses trained embedding classifier for routing
2. OrchestratorRouter - Uses LLM-based orchestrator for routing

Both routers are tested with mock clients to avoid real API calls.
The benchmark measures:
- Routing decision consistency (which lane each query goes to)
- Relative latency (orchestrator expected to be slower due to LLM overhead)
- Relative cost (orchestrator includes routing LLM cost)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.complexity import ComplexitySignals
from pal_router.conversation import OrchestratorConfig, ToolCall, ToolResult
from pal_router.orchestrator import OrchestratorRouter
from pal_router.router import RoutingDecision, TernaryRouter
from pal_router.types import Lane


# Sample queries for benchmarking
SAMPLE_QUERIES = {
    "simple": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What color is the sky?",
        "Name three primary colors.",
        "Hello, how are you?",
    ],
    "complex": [
        "Explain the trolley problem and its implications for autonomous vehicle ethics.",
        "Compare and contrast utilitarianism and deontological ethics.",
        "Analyze the potential long-term societal impacts of AI adoption in healthcare.",
        "Discuss the trade-offs between privacy and security in modern digital systems.",
        "Evaluate the arguments for and against universal basic income.",
    ],
    "computational": [
        "Calculate 15% of $200.",
        "If I invest $10,000 at 5% annual interest compounded yearly, how much will I have after 3 years?",
        "A train travels at 60 mph for 2.5 hours, then 45 mph for 1.5 hours. What is the total distance?",
        "A farmer has chickens and cows. There are 50 heads and 140 legs total. How many of each?",
        "Find the sum of all prime numbers between 1 and 100.",
    ],
}


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single router."""

    router_name: str
    query: str
    lane: Lane
    latency_ms: float
    cost_usd: float
    reason: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison of two routers on a single query."""

    query: str
    ternary_lane: Lane
    orchestrator_lane: Lane
    ternary_latency_ms: float
    orchestrator_latency_ms: float
    ternary_cost_usd: float
    orchestrator_cost_usd: float
    lanes_match: bool
    latency_overhead_ratio: float
    cost_overhead_ratio: float


class MockModelClient:
    """Mock model client for testing without API calls."""

    def __init__(self, name: str = "mock-model", latency_ms: float = 50.0):
        self._name = name
        self._latency_ms = latency_ms
        self._cost_per_1k_input = 0.0001
        self._cost_per_1k_output = 0.0002

    @property
    def model_name(self) -> str:
        return self._name

    def complete(self, prompt: str, system: str | None = None) -> Any:
        """Return a mock completion result."""
        # Simulate latency
        time.sleep(self._latency_ms / 1000.0)

        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = 100  # Fixed response size

        cost_usd = (
            input_tokens * self._cost_per_1k_input / 1000
            + output_tokens * self._cost_per_1k_output / 1000
        )

        result = Mock()
        result.content = f"Mock response to: {prompt[:50]}..."
        result.model = self._name
        result.input_tokens = int(input_tokens)
        result.output_tokens = output_tokens
        result.cost_usd = cost_usd
        result.latency_ms = self._latency_ms
        return result


class MockOrchestratorClient:
    """Mock orchestrator client for testing."""

    def __init__(self, latency_ms: float = 200.0):
        self._latency_ms = latency_ms
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = self._create_completion

    def _create_completion(self, **kwargs):
        """Simulate orchestrator API call with latency."""
        time.sleep(self._latency_ms / 1000.0)

        # Determine which tool to call based on the prompt
        prompt = kwargs.get("messages", [{}])[0].get("content", "")

        # Simple heuristic for tool selection based on query type
        if any(q in prompt for q in SAMPLE_QUERIES["simple"]):
            tool_name = "fast_model"
        elif any(q in prompt for q in SAMPLE_QUERIES["complex"]):
            tool_name = "strong_model"
        else:
            tool_name = "code_executor"

        response = Mock()
        message = Mock()
        message.content = f"I need to use {tool_name} to answer this."

        tool_call = Mock()
        tool_call.function.name = tool_name
        tool_call.function.arguments = '{"query": "' + prompt[:50] + '"}'

        message.tool_calls = [tool_call]
        response.choices = [Mock(message=message)]

        return response


class MockTrainedRouter:
    """Mock trained router for testing without model files."""

    def __init__(self, model_dir=None, embedding_model="fast"):
        self.model_dir = model_dir
        self.embedding_model = embedding_model

    def route(self, query: str) -> RoutingDecision:
        """Return a routing decision based on simple heuristics."""
        # Simple heuristic for lane selection
        if any(q in query for q in SAMPLE_QUERIES["simple"]):
            lane = Lane.FAST
            complexity = 0.2
            reason = "Simple factual query"
        elif any(q in query for q in SAMPLE_QUERIES["complex"]):
            lane = Lane.REASONING
            complexity = 0.7
            reason = "Complex reasoning query"
        else:
            lane = Lane.AGENTIC
            complexity = 0.6
            reason = "Computational query requiring tools"

        return RoutingDecision(
            lane=lane,
            complexity_score=complexity,
            signals=ComplexitySignals(
                syntactic_grade=0.5,
                logic_density=0.3,
                numeric_density=0.2 if lane != Lane.AGENTIC else 0.7,
                constraint_count=1,
                question_depth=1,
                estimated_steps=2 if lane == Lane.AGENTIC else 1,
            ),
            reason=reason,
            confidence=0.85,
        )


class MockAgenticWorkflow:
    """Mock agentic workflow for testing."""

    def __init__(self, weak_model=None, strong_model=None, config=None):
        self.weak_model = weak_model
        self.strong_model = strong_model
        self.config = config

    def execute(self, query: str) -> Any:
        """Return a mock agentic result."""
        result = Mock()
        result.answer = f"Mock agentic answer for: {query[:50]}..."
        result.total_cost_usd = 0.005
        result.total_latency_ms = 300.0
        result.completions = []
        return result


def create_mock_ternary_router() -> TernaryRouter:
    """Create a TernaryRouter with mock clients."""
    weak_client = MockModelClient("mock-weak", latency_ms=50.0)
    strong_client = MockModelClient("mock-strong", latency_ms=150.0)

    # Create mock module and add to sys.modules before importing router
    import sys

    # Create a mock module for trained_router
    mock_trained_router_module = Mock()
    mock_trained_router_module.TrainedRouter = MockTrainedRouter
    sys.modules['pal_router.trained_router'] = mock_trained_router_module

    # Create a mock module for agentic
    mock_agentic_module = Mock()
    mock_agentic_module.AgenticWorkflow = MockAgenticWorkflow
    # Keep the original imports for other things
    from pal_router import agentic as original_agentic
    mock_agentic_module.AgenticResult = original_agentic.AgenticResult
    sys.modules['pal_router.agentic'] = mock_agentic_module

    try:
        router = TernaryRouter(
            weak_model=weak_client,
            strong_model=strong_client,
            config=None,  # Use default config
        )
    finally:
        # Clean up sys.modules
        sys.modules.pop('pal_router.trained_router', None)
        sys.modules.pop('pal_router.agentic', None)

    return router


def create_mock_orchestrator_router() -> OrchestratorRouter:
    """Create an OrchestratorRouter with mock clients."""
    config = OrchestratorConfig(
        max_rounds=3,
        max_cost_usd=0.10,
        max_latency_ms=5000,
    )

    # Mock infrastructure
    mock_infra = Mock()
    mock_infra.get_client.return_value = MockModelClient("mock-model", latency_ms=50.0)

    # Mock the OpenAI client and tool registry
    with patch("pal_router.orchestrator.OpenAI", return_value=MockOrchestratorClient(latency_ms=200.0)):
        router = OrchestratorRouter(config=config, existing_infra=mock_infra)

    return router


def benchmark_ternary_router(router: TernaryRouter, queries: list[str]) -> list[BenchmarkResult]:
    """Benchmark TernaryRouter on a list of queries.

    Args:
        router: TernaryRouter instance
        queries: List of query strings

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for query in queries:
        start_time = time.perf_counter()

        # Get routing decision
        decision = router.route(query)

        # Execute the query (uses mock client, so minimal latency)
        result = router.execute(query)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        results.append(
            BenchmarkResult(
                router_name="TernaryRouter",
                query=query,
                lane=decision.lane,
                latency_ms=latency_ms,
                cost_usd=result.total_cost_usd,
                reason=decision.reason,
                confidence=decision.confidence,
            )
        )

    return results


def benchmark_orchestrator_router(router: OrchestratorRouter, queries: list[str]) -> list[BenchmarkResult]:
    """Benchmark OrchestratorRouter on a list of queries.

    Args:
        router: OrchestratorRouter instance
        queries: List of query strings

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for query in queries:
        start_time = time.perf_counter()

        # Route and execute
        result = router.route_and_execute(query)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        results.append(
            BenchmarkResult(
                router_name="OrchestratorRouter",
                query=query,
                lane=result.decision.lane,
                latency_ms=latency_ms,
                cost_usd=result.total_cost_usd,
                reason=result.decision.reason,
                confidence=result.decision.confidence,
            )
        )

    return results


def compare_routers(
    ternary_results: list[BenchmarkResult],
    orchestrator_results: list[BenchmarkResult],
) -> list[ComparisonResult]:
    """Compare results from two routers.

    Args:
        ternary_results: Results from TernaryRouter
        orchestrator_results: Results from OrchestratorRouter

    Returns:
        List of ComparisonResult objects
    """
    comparisons = []

    for t_result, o_result in zip(ternary_results, orchestrator_results):
        lanes_match = t_result.lane == o_result.lane

        # Calculate overhead ratios
        latency_overhead = (
            o_result.latency_ms / t_result.latency_ms if t_result.latency_ms > 0 else 1.0
        )
        cost_overhead = (
            o_result.cost_usd / t_result.cost_usd if t_result.cost_usd > 0 else 1.0
        )

        comparisons.append(
            ComparisonResult(
                query=t_result.query,
                ternary_lane=t_result.lane,
                orchestrator_lane=o_result.lane,
                ternary_latency_ms=t_result.latency_ms,
                orchestrator_latency_ms=o_result.latency_ms,
                ternary_cost_usd=t_result.cost_usd,
                orchestrator_cost_usd=o_result.cost_usd,
                lanes_match=lanes_match,
                latency_overhead_ratio=latency_overhead,
                cost_overhead_ratio=cost_overhead,
            )
        )

    return comparisons


def print_benchmark_summary(
    comparisons: list[ComparisonResult],
    category: str,
) -> None:
    """Print a summary of benchmark comparisons.

    Args:
        comparisons: List of ComparisonResult objects
        category: Category name for display
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS: {category.upper()} QUERIES")
    print(f"{'='*70}")

    # Calculate statistics
    total_queries = len(comparisons)
    matching_lanes = sum(1 for c in comparisons if c.lanes_match)

    avg_ternary_latency = sum(c.ternary_latency_ms for c in comparisons) / total_queries
    avg_orchestrator_latency = sum(c.orchestrator_latency_ms for c in comparisons) / total_queries
    avg_latency_overhead = sum(c.latency_overhead_ratio for c in comparisons) / total_queries

    avg_ternary_cost = sum(c.ternary_cost_usd for c in comparisons) / total_queries
    avg_orchestrator_cost = sum(c.orchestrator_cost_usd for c in comparisons) / total_queries
    avg_cost_overhead = sum(c.cost_overhead_ratio for c in comparisons) / total_queries

    print(f"\nQuery Count: {total_queries}")
    print(f"Routing Agreement: {matching_lanes}/{total_queries} ({matching_lanes/total_queries:.1%})")

    print(f"\nLatency:")
    print(f"  TernaryRouter:      {avg_ternary_latency:.2f} ms")
    print(f"  OrchestratorRouter: {avg_orchestrator_latency:.2f} ms")
    print(f"  Overhead:           {avg_latency_overhead:.2f}x")

    print(f"\nCost:")
    print(f"  TernaryRouter:      ${avg_ternary_cost:.6f}")
    print(f"  OrchestratorRouter: ${avg_orchestrator_cost:.6f}")
    print(f"  Overhead:           {avg_cost_overhead:.2f}x")

    # Detailed comparison table
    print(f"\n{'Query':<60} {'Ternary':<10} {'Orchestrator':<12} {'Lat Overhead':<12} {'Cost Overhead':<13}")
    print("-" * 110)

    for c in comparisons:
        query_short = c.query[:57] + "..." if len(c.query) > 60 else c.query
        match_marker = "" if c.lanes_match else " *"
        print(
            f"{query_short:<60} {c.ternary_lane.value:<10} "
            f"{c.orchestrator_lane.value:<12} "
            f"{c.latency_overhead_ratio:.2f}x{'':<8} "
            f"{c.cost_overhead_ratio:.2f}x{match_marker:<9}"
        )

    if not all(c.lanes_match for c in comparisons):
        print("\n* = Routing decision mismatch")


# Test functions
def test_benchmark_simple_queries():
    """Benchmark both routers on simple queries."""
    ternary_router = create_mock_ternary_router()
    orchestrator_router = create_mock_orchestrator_router()

    queries = SAMPLE_QUERIES["simple"]

    ternary_results = benchmark_ternary_router(ternary_router, queries)
    orchestrator_results = benchmark_orchestrator_router(orchestrator_router, queries)

    comparisons = compare_routers(ternary_results, orchestrator_results)
    print_benchmark_summary(comparisons, "simple")

    # Assertions
    assert len(comparisons) == len(queries)
    assert all(c.ternary_latency_ms > 0 for c in comparisons)
    assert all(c.orchestrator_latency_ms > 0 for c in comparisons)
    assert all(c.latency_overhead_ratio > 0 for c in comparisons)


def test_benchmark_complex_queries():
    """Benchmark both routers on complex queries."""
    ternary_router = create_mock_ternary_router()
    orchestrator_router = create_mock_orchestrator_router()

    queries = SAMPLE_QUERIES["complex"]

    ternary_results = benchmark_ternary_router(ternary_router, queries)
    orchestrator_results = benchmark_orchestrator_router(orchestrator_router, queries)

    comparisons = compare_routers(ternary_results, orchestrator_results)
    print_benchmark_summary(comparisons, "complex")

    # Assertions
    assert len(comparisons) == len(queries)
    assert all(c.ternary_latency_ms > 0 for c in comparisons)
    assert all(c.orchestrator_latency_ms > 0 for c in comparisons)


def test_benchmark_computational_queries():
    """Benchmark both routers on computational queries."""
    ternary_router = create_mock_ternary_router()
    orchestrator_router = create_mock_orchestrator_router()

    queries = SAMPLE_QUERIES["computational"]

    ternary_results = benchmark_ternary_router(ternary_router, queries)
    orchestrator_results = benchmark_orchestrator_router(orchestrator_router, queries)

    comparisons = compare_routers(ternary_results, orchestrator_results)
    print_benchmark_summary(comparisons, "computational")

    # Assertions
    assert len(comparisons) == len(queries)
    assert all(c.ternary_latency_ms > 0 for c in comparisons)
    assert all(c.orchestrator_latency_ms > 0 for c in comparisons)


def test_benchmark_all_queries():
    """Run comprehensive benchmark on all query types."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BENCHMARK: TernaryRouter vs OrchestratorRouter")
    print("=" * 70)

    ternary_router = create_mock_ternary_router()
    orchestrator_router = create_mock_orchestrator_router()

    all_queries = (
        SAMPLE_QUERIES["simple"]
        + SAMPLE_QUERIES["complex"]
        + SAMPLE_QUERIES["computational"]
    )

    ternary_results = benchmark_ternary_router(ternary_router, all_queries)
    orchestrator_results = benchmark_orchestrator_router(orchestrator_router, all_queries)

    comparisons = compare_routers(ternary_results, orchestrator_results)
    print_benchmark_summary(comparisons, "all")

    # Overall statistics
    total_queries = len(comparisons)
    matching_lanes = sum(1 for c in comparisons if c.lanes_match)

    avg_latency_overhead = sum(c.latency_overhead_ratio for c in comparisons) / total_queries
    avg_cost_overhead = sum(c.cost_overhead_ratio for c in comparisons) / total_queries

    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total Queries: {total_queries}")
    print(f"Routing Agreement: {matching_lanes}/{total_queries} ({matching_lanes/total_queries:.1%})")
    print(f"Average Latency Overhead: {avg_latency_overhead:.2f}x")
    print(f"Average Cost Overhead: {avg_cost_overhead:.2f}x")
    print(f"{'='*70}\n")

    # Key assertion: Orchestrator should have higher latency due to LLM overhead
    # This is expected and documented behavior
    assert avg_latency_overhead > 1.0, "OrchestratorRouter expected to be slower than TernaryRouter"

    # Both routers should produce valid routing decisions
    assert all(c.ternary_lane in Lane for c in comparisons)
    assert all(c.orchestrator_lane in Lane for c in comparisons)


def test_benchmark_consistency():
    """Test that benchmark results are consistent across runs."""
    ternary_router = create_mock_ternary_router()
    queries = SAMPLE_QUERIES["simple"][:2]

    # Run benchmark twice
    results1 = benchmark_ternary_router(ternary_router, queries)
    results2 = benchmark_ternary_router(ternary_router, queries)

    # Results should be consistent
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.router_name == r2.router_name
        assert r1.query == r2.query
        assert r1.lane == r2.lane


if __name__ == "__main__":
    # Run benchmarks with verbose output
    test_benchmark_simple_queries()
    test_benchmark_complex_queries()
    test_benchmark_computational_queries()
    test_benchmark_all_queries()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON COMPLETE")
    print("=" * 70)
