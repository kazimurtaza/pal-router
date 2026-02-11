"""LLM-based orchestrator router for PAL-Router.

This module implements the main OrchestratorRouter class that uses an 8B LLM
to coordinate multiple tools (fast_model, strong_model, code_executor, web_search)
inspired by NVIDIA's ToolOrchestra paper.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from pal_router.conversation import (
    ConversationContext,
    ConversationTurn,
    OrchestratorConfig,
    OrchestratorDecision,
    ToolCall,
    ToolResult,
)
from pal_router.tools import ToolRegistry, parse_orchestrator_response


class OrchestratorRouter:
    """LLM-based orchestrator router.

    This router uses an 8B LLM to intelligently route queries to appropriate
    tools based on the query characteristics and accumulated context.

    Attributes:
        config: Orchestrator configuration
        infra: PAL-Router's existing infrastructure
        tool_registry: Tool registry and executor
    """

    def __init__(self, config: OrchestratorConfig, infrastructure: Any):
        """Initialize the orchestrator router.

        Args:
            config: Orchestrator configuration
            infrastructure: PAL-Router's existing infrastructure
        """
        self.config = config
        self.infra = infrastructure
        self.tool_registry = ToolRegistry(config, infrastructure)

        # Lazy loaded model for transformers backend
        self._model = None
        self._tokenizer = None

    def query(self, user_query: str) -> ConversationContext:
        """Process a user query through orchestration.

        Args:
            user_query: The user's query string

        Returns:
            ConversationContext with all turns and final answer
        """
        # Create new context
        context = ConversationContext(original_query=user_query)

        # Run orchestration loop
        self._run_orchestration_loop(context)

        return context

    def _run_orchestration_loop(self, context: ConversationContext) -> None:
        """Run the main orchestration loop.

        Loop:
        1. Check if we should stop (budget, stuck, max rounds)
        2. Orchestrate next round (query LLM, parse response)
        3. Execute tool(s) and record results
        4. Repeat until stop condition met

        Args:
            context: Current conversation context (modified in-place)
        """
        while not self._should_stop(context):
            # Get next decision from orchestrator
            decision = self._orchestrate_round(context)

            # Check if final answer provided
            if decision.is_final and decision.final_answer:
                self._record_final_answer(context, decision)
                break

            # Execute tool calls
            for tool_call in decision.tool_calls:
                turn = self._execute_tool(tool_call)
                context.turns.append(turn)

                # Record error if tool failed
                if not turn.result.success:
                    context.errors.append(
                        f"{tool_call.name} failed: {turn.result.error}"
                    )

        # If loop exited without final answer, provide fallback
        if not any(t.tool_call.name == "final_answer" for t in context.turns):
            self._provide_fallback_answer(context)

    def _should_stop(self, context: ConversationContext) -> bool:
        """Check if orchestration should stop.

        Stop conditions:
        - Max rounds reached
        - Budget exceeded
        - Stuck in loop

        Args:
            context: Current conversation context

        Returns:
            True if should stop, False otherwise
        """
        # Max rounds check
        if len(context.turns) >= self.config.max_rounds:
            return True

        # Budget check
        if context.budget_exceeded(self.config):
            return True

        # Stuck detection
        if context.is_stuck():
            return True

        return False

    def _orchestrate_round(self, context: ConversationContext) -> OrchestratorDecision:
        """Orchestrate a single round.

        Args:
            context: Current conversation context

        Returns:
            OrchestratorDecision with tool calls and reasoning
        """
        # Build messages
        system_prompt = self._build_system_prompt()
        user_context = context.build_prompt_context(self.config)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

        # Query orchestrator model
        response = self._query_orchestrator_model(messages)

        # Parse response
        return self._parse_orchestrator_response(response)

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool definitions.

        Returns:
            System prompt string
        """
        tools_info = []
        for tool in self.config.tools:
            func = tool["function"]
            name = func["name"]
            desc = func["description"]
            tools_info.append(f"- {name}: {desc}")

        prompt = f"""You are an intelligent orchestrator that routes queries to appropriate tools.

Available tools:
{chr(10).join(tools_info)}

Your job:
1. Analyze the user's query and accumulated context
2. Decide which tool to use next (or provide final_answer)
3. Consider budget constraints and avoid repeating failed approaches
4. Always use final_answer to deliver results to the user

Budget tracking:
- Max rounds: {self.config.max_rounds}
- Max cost: ${self.config.max_cost_usd:.4f}
- Max latency: {self.config.max_latency_ms}ms

Respond with:
- Reasoning about your decision
- Tool call(s) to execute (with parameters)
- final_answer when task is complete
"""
        return prompt

    def _query_orchestrator_model(self, messages: list[dict]) -> Any:
        """Query the orchestrator model via configured backend.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Model response (backend-specific format)
        """
        backend = self.config.backend

        if backend == "llamacpp":
            return self._query_llamacpp(messages)
        elif backend == "vllm":
            return self._query_vllm(messages)
        elif backend == "transformers":
            return self._query_transformers(messages)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _query_llamacpp(self, messages: list[dict]) -> Any:
        """Query llama.cpp server.

        Args:
            messages: List of message dicts

        Returns:
            Mock response object compatible with parse_orchestrator_response
        """
        if requests is None:
            raise ImportError("requests library required for llamacpp backend")

        url = self.config.model_url or "http://localhost:8080/v1/chat/completions"

        payload = {
            "model": self.config.model_path,
            "messages": messages,
            "tools": self.config.tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 512,
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        # Return mock object compatible with parse_orchestrator_response
        class LlamaResponse:
            def __init__(self, data: dict):
                self.data = data
                self.choices = [self._make_choice(data)]

            def _make_choice(self, data: dict):
                class Choice:
                    def __init__(self, msg: dict):
                        self.message = self._make_message(msg)

                    def _make_message(self, msg: dict):
                        class Message:
                            def __init__(self, msg: dict):
                                self.content = msg.get("content")
                                self.tool_calls = [
                                    self._make_tc(tc)
                                    for tc in msg.get("tool_calls", [])
                                ]

                            def _make_tc(self, tc: dict):
                                class ToolCall:
                                    def __init__(self, tc: dict):
                                        self.function = self._make_function(tc)

                                    def _make_function(self, tc: dict):
                                        class Function:
                                            def __init__(self, f: dict):
                                                self.name = f.get("name")
                                                self.arguments = f.get("arguments", "{}")

                                        return Function(tc.get("function", {}))

                                return ToolCall(tc)

                        return Message(msg)

                return Choice(data["choices"][0])

        return LlamaResponse(response.json())

    def _query_vllm(self, messages: list[dict]) -> Any:
        """Query vLLM server.

        Args:
            messages: List of message dicts

        Returns:
            Mock response object compatible with parse_orchestrator_response
        """
        if requests is None:
            raise ImportError("requests library required for vllm backend")

        url = self.config.model_url or "http://localhost:8000/v1/chat/completions"

        payload = {
            "model": self.config.model_path,
            "messages": messages,
            "tools": self.config.tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 512,
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        # Use same response format as llamacpp
        class VLLMResponse:
            def __init__(self, data: dict):
                self.data = data
                self.choices = [self._make_choice(data)]

            def _make_choice(self, data: dict):
                class Choice:
                    def __init__(self, msg: dict):
                        self.message = self._make_message(msg)

                    def _make_message(self, msg: dict):
                        class Message:
                            def __init__(self, msg: dict):
                                self.content = msg.get("content")
                                self.tool_calls = [
                                    self._make_tc(tc)
                                    for tc in msg.get("tool_calls", [])
                                ]

                            def _make_tc(self, tc: dict):
                                class ToolCall:
                                    def __init__(self, tc: dict):
                                        self.function = self._make_function(tc)

                                    def _make_function(self, tc: dict):
                                        class Function:
                                            def __init__(self, f: dict):
                                                self.name = f.get("name")
                                                self.arguments = f.get("arguments", "{}")

                                        return Function(tc.get("function", {}))

                                return ToolCall(tc)

                        return Message(msg)

                return Choice(data["choices"][0])

        return VLLMResponse(response.json())

    def _query_transformers(self, messages: list[dict]) -> Any:
        """Query local model via transformers.

        Args:
            messages: List of message dicts

        Returns:
            Mock response object compatible with parse_orchestrator_response
        """
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers library required for transformers backend")

        # Lazy load model
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_path)

        # Build prompt
        prompt = self._messages_to_prompt(messages)

        # Generate
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

        response_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return mock response
        class TransformersResponse:
            def __init__(self, text: str):
                self.text = text
                self.choices = [self._make_choice(text)]

            def _make_choice(self, text: str):
                class Choice:
                    def __init__(self, text: str):
                        self.message = self._make_message(text)

                    def _make_message(self, text: str):
                        class Message:
                            def __init__(self, text: str):
                                self.content = text
                                self.tool_calls = None

                        return Message(text)

                return Choice(text)

        return TransformersResponse(response_text)

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert messages list to prompt string for transformers.

        Args:
            messages: List of message dicts

        Returns:
            Prompt string
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant:"

    def _parse_orchestrator_response(self, response: Any) -> OrchestratorDecision:
        """Parse orchestrator response into OrchestratorDecision.

        Args:
            response: Raw response from LLM backend

        Returns:
            Parsed OrchestratorDecision
        """
        return parse_orchestrator_response(response, self.config.tools)

    def _execute_tool(self, tool_call: ToolCall) -> ConversationTurn:
        """Execute a tool call and record the result.

        Args:
            tool_call: The tool call to execute

        Returns:
            ConversationTurn with result
        """
        start = time.perf_counter()

        try:
            result = self.tool_registry.execute(tool_call)
        except Exception as e:
            result = ToolResult(
                success=False,
                output="",
                error=str(e),
            )

        latency_ms = (time.perf_counter() - start) * 1000
        cost_usd = result.metadata.get("cost_usd", 0.0)

        return ConversationTurn(
            tool_call=tool_call,
            result=result,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    def _record_final_answer(self, context: ConversationContext, decision: OrchestratorDecision) -> None:
        """Record final answer in context.

        Args:
            context: Conversation context to update
            decision: Decision with final answer
        """
        tool_call = ToolCall(
            name="final_answer",
            parameters={
                "answer": decision.final_answer or "",
                "sources": decision.sources or [],
            },
        )

        result = ToolResult(
            success=True,
            output=decision.final_answer or "",
            metadata={"sources": decision.sources or []},
        )

        turn = ConversationTurn(
            tool_call=tool_call,
            result=result,
            cost_usd=0.0,
            latency_ms=0.0,
        )

        context.turns.append(turn)

    def _provide_fallback_answer(self, context: ConversationContext) -> None:
        """Provide fallback answer when orchestration stops early.

        Args:
            context: Conversation context to update
        """
        # Find best successful result
        successful_results = [t for t in context.turns if t.result.success]

        if not successful_results:
            # No successful results, provide error message
            answer = (
                f"Unable to complete the query. "
                f"Encountered {len(context.errors)} errors. "
                f"Please try a different approach."
            )
            sources = []
        else:
            # Use the most recent successful result
            best = successful_results[-1]
            answer = best.result.output
            sources = [best.tool_call.name]

        tool_call = ToolCall(
            name="final_answer",
            parameters={"answer": answer, "sources": sources},
        )

        result = ToolResult(
            success=True,
            output=answer,
            metadata={"sources": sources, "fallback": True},
        )

        turn = ConversationTurn(
            tool_call=tool_call,
            result=result,
            cost_usd=0.0,
            latency_ms=0.0,
        )

        context.turns.append(turn)
