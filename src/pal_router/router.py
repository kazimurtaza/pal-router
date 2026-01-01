"""Ternary routing logic for query classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from pal_router.agentic import AgenticResult, AgenticWorkflow
from pal_router.complexity import ComplexitySignals, estimate_complexity
from pal_router.config import Config
from pal_router.models import CompletionResult, ModelClient, get_client


class Lane(str, Enum):
    """The three routing lanes."""

    FAST = "FAST"
    REASONING = "REASONING"
    AGENTIC = "AGENTIC"


@dataclass
class RoutingDecision:
    """Complete routing decision with metadata."""

    lane: Lane
    complexity_score: float
    signals: ComplexitySignals
    reason: str
    confidence: float = 1.0
    probabilities: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RouterResult:
    """Result from routing and executing a query."""

    decision: RoutingDecision
    answer: str
    total_cost_usd: float
    total_latency_ms: float
    completions: list[CompletionResult] = field(default_factory=list)
    agentic_result: AgenticResult | None = None


class TernaryRouter:
    """Routes queries to Fast, Reasoning, or Agentic lanes using trained classifier.

    Based on research showing trained classifiers outperform heuristics:
    - BERT classifier achieved >50% APGR improvement over random baseline
    - Augmented training data significantly improves routing accuracy
    """

    def __init__(
        self,
        config: Config | None = None,
        weak_model: ModelClient | None = None,
        strong_model: ModelClient | None = None,
        weak_provider: str | None = None,
        strong_provider: str | None = None,
        classifier_model_dir: str | Path = "models/router_classifier",
        embedding_model: str = "fast",
    ):
        """Initialize the ternary router.

        Args:
            config: Configuration with thresholds and model settings.
            weak_model: Optional pre-configured weak model client.
            strong_model: Optional pre-configured strong model client.
            weak_provider: Override provider for weak model (e.g., "gemini").
            strong_provider: Override provider for strong model (e.g., "groq").
            classifier_model_dir: Directory containing trained classifier model.
            embedding_model: Which embedding model to use ("fast", "balanced", "accurate").
        """
        self.config = config or Config()

        # Load trained classifier (required)
        from pal_router.trained_router import TrainedRouter
        self._classifier = TrainedRouter(
            model_dir=classifier_model_dir,
            embedding_model=embedding_model,
        )

        # Initialize model clients (supports mixing providers)
        if weak_model:
            self.weak_model = weak_model
        else:
            provider = weak_provider or self.config.provider
            self.weak_model = get_client(
                self.config.get_model_config("weak"),
                provider,
            )

        if strong_model:
            self.strong_model = strong_model
        else:
            provider = strong_provider or self.config.provider
            self.strong_model = get_client(
                self.config.get_model_config("strong"),
                provider,
            )

        # Initialize agentic workflow
        self.agentic_workflow = AgenticWorkflow(
            weak_model=self.weak_model,
            strong_model=self.strong_model,
            config=self.config,
        )

    def route(self, prompt: str) -> RoutingDecision:
        """Determine which lane to route the query to using trained classifier.

        Args:
            prompt: The user's query.

        Returns:
            RoutingDecision with lane, confidence, and metadata.
        """
        return self._classifier.route(prompt)

    def execute(self, prompt: str) -> RouterResult:
        """Route and execute a query.

        Args:
            prompt: The user's query.

        Returns:
            RouterResult with decision, answer, cost, and latency.
        """
        decision = self.route(prompt)

        if decision.lane == Lane.FAST:
            completion = self.weak_model.complete(prompt)
            return RouterResult(
                decision=decision,
                answer=completion.content,
                total_cost_usd=completion.cost_usd,
                total_latency_ms=completion.latency_ms,
                completions=[completion],
            )

        elif decision.lane == Lane.REASONING:
            completion = self.strong_model.complete(prompt)
            return RouterResult(
                decision=decision,
                answer=completion.content,
                total_cost_usd=completion.cost_usd,
                total_latency_ms=completion.latency_ms,
                completions=[completion],
            )

        else:  # AGENTIC
            agentic_result = self.agentic_workflow.execute(prompt)
            return RouterResult(
                decision=decision,
                answer=agentic_result.answer,
                total_cost_usd=agentic_result.total_cost_usd,
                total_latency_ms=agentic_result.total_latency_ms,
                completions=agentic_result.completions,
                agentic_result=agentic_result,
            )

    def compare_baselines(
        self,
        prompt: str,
    ) -> dict[str, RouterResult]:
        """Run query through all three lanes for comparison.

        Useful for evaluation and debugging.

        Args:
            prompt: The user's query.

        Returns:
            Dict with results from each lane.
        """
        # Get actual routing decision
        decision = self.route(prompt)

        results = {}

        # Fast lane (weak model)
        fast_completion = self.weak_model.complete(prompt)
        results["FAST"] = RouterResult(
            decision=RoutingDecision(
                lane=Lane.FAST,
                complexity_score=decision.complexity_score,
                signals=decision.signals,
                reason="Baseline comparison",
            ),
            answer=fast_completion.content,
            total_cost_usd=fast_completion.cost_usd,
            total_latency_ms=fast_completion.latency_ms,
            completions=[fast_completion],
        )

        # Reasoning lane (strong model)
        strong_completion = self.strong_model.complete(prompt)
        results["REASONING"] = RouterResult(
            decision=RoutingDecision(
                lane=Lane.REASONING,
                complexity_score=decision.complexity_score,
                signals=decision.signals,
                reason="Baseline comparison",
            ),
            answer=strong_completion.content,
            total_cost_usd=strong_completion.cost_usd,
            total_latency_ms=strong_completion.latency_ms,
            completions=[strong_completion],
        )

        # Agentic lane
        agentic_result = self.agentic_workflow.execute(prompt)
        results["AGENTIC"] = RouterResult(
            decision=RoutingDecision(
                lane=Lane.AGENTIC,
                complexity_score=decision.complexity_score,
                signals=decision.signals,
                reason="Baseline comparison",
            ),
            answer=agentic_result.answer,
            total_cost_usd=agentic_result.total_cost_usd,
            total_latency_ms=agentic_result.total_latency_ms,
            completions=agentic_result.completions,
            agentic_result=agentic_result,
        )

        # Also run with actual routing
        routed_result = self.execute(prompt)
        results["ROUTED"] = routed_result

        return results
