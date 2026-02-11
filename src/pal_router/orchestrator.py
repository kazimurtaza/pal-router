"""Orchestrator-based router using Nemotron-Orchestrator-8B."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from openai import OpenAI

from pal_router.conversation import (
    ConversationContext,
    ConversationTurn,
    OrchestratorConfig,
    OrchestratorDecision,
    ToolCall,
    ToolResult,
)
from pal_router.router import RouterResult, RoutingDecision
from pal_router.tools import ToolRegistry, parse_orchestrator_response
from pal_router.types import Lane

if TYPE_CHECKING:
    pass


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
                if context.has_retried_no_tool:
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
                        result=ToolResult(success=False, output="", error=str(e)),
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
            answer = (
                "Unable to complete the task. The orchestrator encountered "
                "errors or exceeded budget limits."
            )

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
            final_answer=(
                "I was unable to complete this task after multiple attempts. "
                "The task may require additional information or capabilities."
            ),
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

    def _build_routing_decision(
        self,
        context: ConversationContext,
        orchestrator_decision: OrchestratorDecision,
    ) -> RoutingDecision:
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
            reason=(
                orchestrator_decision.reasoning
                if orchestrator_decision
                else "Orchestrator routing"
            ),
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
