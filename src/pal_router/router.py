"""Ternary routing logic for query classification."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from pal_router.agentic import AgenticResult, AgenticWorkflow
from pal_router.complexity import ComplexitySignals, estimate_complexity
from pal_router.config import Config
from pal_router.models import CompletionResult, ModelClient, get_client

# Optional RouteLLM import - graceful fallback if not available
# Note: RouteLLM requires OPENAI_API_KEY even at import time for embeddings
try:
    from routellm.routers.routers import ROUTER_CLS
    ROUTELLM_AVAILABLE = True
except (ImportError, Exception):
    # Catches ImportError and OpenAI API key errors
    ROUTELLM_AVAILABLE = False
    ROUTER_CLS = None


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
    """Routes queries to Fast, Reasoning, or Agentic lanes."""

    def __init__(
        self,
        config: Config | None = None,
        weak_model: ModelClient | None = None,
        strong_model: ModelClient | None = None,
        weak_provider: str | None = None,
        strong_provider: str | None = None,
        routellm_threshold: float = 0.12,
        use_routellm: bool = True,
    ):
        """Initialize the ternary router.

        Args:
            config: Configuration with thresholds and model settings.
            weak_model: Optional pre-configured weak model client.
            strong_model: Optional pre-configured strong model client.
            weak_provider: Override provider for weak model (e.g., "gemini").
            strong_provider: Override provider for strong model (e.g., "groq").
            routellm_threshold: Threshold for RouteLLM MF router (default 0.12).
            use_routellm: Whether to use RouteLLM for weak vs strong routing.
        """
        self.config = config or Config()
        self.routellm_threshold = routellm_threshold
        self.use_routellm = use_routellm and ROUTELLM_AVAILABLE
        self._routellm_router = None

        # Initialize RouteLLM MF router if available and enabled
        if self.use_routellm:
            try:
                self._routellm_router = ROUTER_CLS["mf"](
                    checkpoint_path="routellm/mf_gpt4_augmented",
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize RouteLLM: {e}. Falling back to threshold-based routing.")
                self.use_routellm = False

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

    def _get_routellm_score(self, prompt: str) -> float | None:
        """Get routing score from RouteLLM MF router.

        Returns:
            Score between 0 and 1, where higher means route to strong model.
            Returns None if RouteLLM is not available.
        """
        if not self.use_routellm or self._routellm_router is None:
            return None

        try:
            # RouteLLM MF router returns probability of needing strong model
            score = self._routellm_router.calculate_strong_win_rate(prompt)
            return float(score)
        except Exception:
            return None

    def route(self, prompt: str) -> RoutingDecision:
        """Determine which lane to route the query to.

        The routing logic (per original plan):
        1. AGENTIC: Queries with numeric content that benefit from code execution
        2. RouteLLM MF router decides between REASONING and FAST

        Args:
            prompt: The user's query.

        Returns:
            RoutingDecision with lane, score, signals, and reason.
        """
        score, signals = estimate_complexity(
            prompt,
            weights=self.config.complexity_weights,
        )

        # Step 1: AGENTIC - Numeric-heavy queries benefit from code execution
        # Key insight: weak model + code > weak model alone for math

        # Base threshold for numeric content
        is_numeric_heavy = signals.numeric_density > 0.04
        has_constraints = signals.constraint_count >= 2

        # Word problems often have lower numeric density but still benefit from code
        # Detect by: longer text (>15 words) + some numerics + story patterns
        word_count = len(prompt.split())
        is_word_problem = (
            word_count > 15 and
            signals.numeric_density > 0.03 and
            any(kw in prompt.lower() for kw in ['per day', 'each', 'every', 'total', 'how many', 'how much'])
        )

        # Conceptual/analytical queries should go to REASONING, not AGENTIC
        # These are marked by verbs like "analyze", "discuss", "explain", "critique"
        # Also includes "implications", "consequences", "impact" for ethics/philosophy queries
        conceptual_pattern = r'\b(analyze|discuss|explain|compare|evaluate|consider|describe|outline|examine|critique|implications|consequences|impact)\b'
        is_conceptual = bool(re.search(conceptual_pattern, prompt.lower()))

        # Skip AGENTIC for trivial queries that LLMs handle easily
        has_direct_arithmetic = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', prompt))
        is_trivial_math = (
            has_direct_arithmetic and
            word_count <= 8 and
            '%' not in prompt  # Percentages aren't trivial
        )

        # Skip AGENTIC for pure factual queries with word-form numbers
        # e.g., "Name three primary colors" - the "three" is just a count
        math_keywords = ['calculate', 'compute', 'solve', 'how many', 'how much', 'sum of', 'product of']
        is_factual_with_count = (
            word_count <= 10 and
            not any(op in prompt for op in ['+', '-', '*', '/', '%', '=']) and
            not any(kw in prompt.lower() for kw in math_keywords)
        )

        # Algebraic expressions with variables should go to AGENTIC
        # But skip trivial substitution like "If x = 5, what is x + 3?"
        has_algebraic = bool(re.search(r'\b[a-z]\s*=\s*\d+', prompt.lower()))
        is_trivial_algebra = has_algebraic and word_count <= 10

        # Detect syllogistic/ordering/causal logic that benefits from code
        # e.g., "If all A are B and all B are C..." or "X finished before Y but after Z"
        # or causal chains like "Given X, and X→Y..."
        syllogism_pattern = r'\b(all\s+\w+\s+are|if\s+all|before\s+\w+\s+but\s+after|finished\s+(before|after)|given\s+that\b.*\band\b.*\band\b)\b'
        has_syllogism = bool(re.search(syllogism_pattern, prompt.lower()))

        # Combinatorics problems benefit from code even without explicit numbers
        # Be specific to avoid false positives like "choose a city"
        combinatorics_pattern = r'\b(arrange\s+the|permut|combin|ways\s+(can|to)\s+(arrange|select|choose)|factorial|n\s+choose\s+r|\d+\s+choose\s+\d+)\b'
        is_combinatorics = bool(re.search(combinatorics_pattern, prompt.lower()))

        # Route to AGENTIC if:
        # - Has significant numerics OR constraints OR is a word problem OR has non-trivial algebra
        # - OR has syllogistic logic OR is combinatorics
        # - Not trivial math that LLMs handle easily
        # - Not a conceptual/analytical question
        # - Not a factual query with incidental word-form numbers
        # - Not trivial algebra (simple substitution)
        has_code_benefit = (
            (is_numeric_heavy and not is_trivial_algebra) or
            has_constraints or
            is_word_problem or
            (has_algebraic and not is_trivial_algebra) or
            has_syllogism or
            is_combinatorics
        )
        should_use_agentic = (
            has_code_benefit and
            not is_trivial_math and
            not is_conceptual and
            not is_factual_with_count
        )

        if should_use_agentic:
            return RoutingDecision(
                lane=Lane.AGENTIC,
                complexity_score=score,
                signals=signals,
                reason=f"Numeric/constraint content (numeric={signals.numeric_density:.2f}, constraints={signals.constraint_count}, word_problem={is_word_problem}); routing to code execution",
            )

        # Step 2: RouteLLM decides weak vs strong
        routellm_score = self._get_routellm_score(prompt)

        if routellm_score is not None:
            # Use RouteLLM's decision
            if routellm_score > self.routellm_threshold:
                return RoutingDecision(
                    lane=Lane.REASONING,
                    complexity_score=score,
                    signals=signals,
                    reason=f"RouteLLM score {routellm_score:.3f} > threshold {self.routellm_threshold}; routing to strong model",
                )
            else:
                return RoutingDecision(
                    lane=Lane.FAST,
                    complexity_score=score,
                    signals=signals,
                    reason=f"RouteLLM score {routellm_score:.3f} <= threshold {self.routellm_threshold}; routing to fast model",
                )

        # Fallback: Use complexity-based routing if RouteLLM unavailable

        # Conceptual/analytical queries always go to REASONING
        # Also catch "pros and cons" pattern for comparative analysis
        pros_cons_pattern = r'\b(pros and cons|advantages and disadvantages|benefits and drawbacks)\b'
        is_comparative = bool(re.search(pros_cons_pattern, prompt.lower()))

        if is_conceptual or is_comparative:
            return RoutingDecision(
                lane=Lane.REASONING,
                complexity_score=score,
                signals=signals,
                reason=f"Conceptual/analytical query; routing to strong model [fallback]",
            )

        # Short factual queries shouldn't route to REASONING even with common logic words
        is_short_factual = word_count <= 10 and signals.question_depth <= 1

        # Raised threshold to 0.20 for short queries, 0.15 for longer
        logic_threshold = 0.20 if is_short_factual else 0.15
        is_logic_heavy = signals.logic_density > logic_threshold and not is_short_factual

        is_complex = score >= self.config.reasoning_threshold
        needs_deep_reasoning = signals.estimated_steps > 2

        # Don't route short factual queries to REASONING even if they seem complex
        if (is_logic_heavy or (is_complex and needs_deep_reasoning)) and not is_short_factual:
            return RoutingDecision(
                lane=Lane.REASONING,
                complexity_score=score,
                signals=signals,
                reason=f"Complex reasoning (logic={signals.logic_density:.2f}, steps={signals.estimated_steps}); routing to strong model [fallback]",
            )

        # FAST: Simple queries for weak model
        return RoutingDecision(
            lane=Lane.FAST,
            complexity_score=score,
            signals=signals,
            reason=f"Simple query ({score:.2f}); routing to fast model [fallback]",
        )

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
