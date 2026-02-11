"""Tool definitions and registry for orchestrator router."""

from __future__ import annotations

import json
import time
from typing import Any

from pal_router.conversation import (
    OrchestratorConfig,
    ToolCall,
    ToolResult,
    OrchestratorDecision,
)
from pal_router.config import Config as PalRouterConfig


# Tool definitions matching NVIDIA's approach
DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fast_model",
            "description": (
                "Simple factual queries. Cost: ~$0.0001. Latency: <500ms. "
                "Use for: definitions, facts, simple Q&A, translations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to answer"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gpt-4o-mini", "claude-haiku", "llama-8b"],
                        "default": "llama-8b"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "strong_model",
            "description": (
                "Complex reasoning tasks. Cost: ~$0.01. Latency: 1-3s. "
                "Use for: analysis, comparisons, nuanced questions, creative writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to answer"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gpt-4o", "claude-sonnet", "llama-70b"],
                        "default": "llama-70b"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": (
                "Math and computation via Python code. Cost: ~$0.0001 + compute. "
                "Use for: arithmetic, formulas, data processing, anything with numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The problem to solve with code"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gpt-4o-mini", "claude-haiku", "qwen-coder"],
                        "default": "gpt-4o-mini"
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 30,
                        "description": "Max execution time in seconds"
                    }
                },
                "required": ["problem"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search web for current/missing information. Cost: ~$0.001. "
                "Use for: recent events, real-time data, facts you're unsure about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Provide the final answer when task is complete. "
                "Always use this to deliver results to user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer to the user's query"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools/sources used to derive answer"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


class ToolRegistry:
    """Registry and executor for available tools."""

    def __init__(self, config: OrchestratorConfig, existing_infra: Any):
        """Initialize tool registry.

        Args:
            config: Orchestrator configuration
            existing_infra: PAL-Router's existing infrastructure. Expected to provide
                get_client(model_name, tier) -> ModelClient and optionally
                get_agentic_workflow(client, timeout) -> AgenticWorkflow
        """
        self.config = config
        self.infra = existing_infra
        self._tools = {t["function"]["name"]: t for t in config.tools or DEFAULT_TOOLS}
        self.config.tools = self.config.tools or DEFAULT_TOOLS

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return result.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with success status and output

        Raises:
            ValueError: If tool name is unknown
        """
        tool_name = tool_call.name
        params = tool_call.parameters

        if tool_name == "fast_model":
            return self._execute_fast_model(params)
        elif tool_name == "strong_model":
            return self._execute_strong_model(params)
        elif tool_name == "code_executor":
            return self._execute_code(params)
        elif tool_name == "web_search":
            return self._execute_search(params)
        elif tool_name == "final_answer":
            return self._execute_final_answer(params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_fast_model(self, params: dict) -> ToolResult:
        """Execute fast_model tool."""
        model_key = params.get("model", "llama-8b")
        model_name = self.config.fast_models.get(model_key, model_key)

        if not hasattr(self.infra, "get_client"):
            raise ValueError("Infrastructure must provide get_client method")
        client = self.infra.get_client(model_name, tier="fast")

        query = params.get("query")
        if query is None:
            raise ValueError("Missing required parameter: query")

        start = time.perf_counter()
        response = client.complete(query)
        latency_ms = (time.perf_counter() - start) * 1000

        return ToolResult(
            success=True,
            output=response.content,
            metadata={"cost_usd": response.cost_usd, "latency_ms": latency_ms}
        )

    def _execute_strong_model(self, params: dict) -> ToolResult:
        """Execute strong_model tool."""
        model_key = params.get("model", "llama-70b")
        model_name = self.config.strong_models.get(model_key, model_key)
        client = self.infra.get_client(model_name, tier="strong")

        query = params.get("query")
        if query is None:
            raise ValueError("Missing required parameter: query")

        start = time.perf_counter()
        response = client.complete(query)
        latency_ms = (time.perf_counter() - start) * 1000

        return ToolResult(
            success=True,
            output=response.content,
            metadata={"cost_usd": response.cost_usd, "latency_ms": latency_ms}
        )

    def _execute_code(self, params: dict) -> ToolResult:
        """Execute code_executor tool using existing AgenticWorkflow."""
        model_key = params.get("model", "gpt-4o-mini")
        model_name = self.config.code_models.get(model_key, model_key)
        client = self.infra.get_client(model_name, tier="fast")

        problem = params.get("problem")
        if problem is None:
            raise ValueError("Missing required parameter: problem")

        timeout = params.get("timeout", 30)

        # Use infrastructure's workflow if available, otherwise create new one
        if hasattr(self.infra, 'get_agentic_workflow'):
            workflow = self.infra.get_agentic_workflow(client, timeout)
        else:
            from pal_router.agentic import AgenticWorkflow
            # Reuse existing AgenticWorkflow
            workflow = AgenticWorkflow(
                weak_model=client,
                config=PalRouterConfig(agentic_timeout_seconds=timeout)
            )

        result = workflow.execute(problem)

        return ToolResult(
            success=result.success,
            output=result.answer,
            metadata={
                "cost_usd": result.total_cost_usd,
                "latency_ms": result.total_latency_ms,
                "code": result.code,
                "attempts": result.attempts
            }
        )

    def _execute_search(self, params: dict) -> ToolResult:
        """Execute web_search tool.

        Note: This is an intentional placeholder pending search provider integration.
        TODO: Implement actual search provider (Tavily, SerpAPI, Brave, etc.)
        """
        query = params.get("query")
        if query is None:
            raise ValueError("Missing required parameter: query")

        max_results = params.get("max_results", 5)

        # Placeholder implementation - search provider to be integrated later
        return ToolResult(
            success=True,
            output=f"Search results for: {query} (TODO: implement search provider)",
            metadata={"sources": [], "query": query}
        )

    def _execute_final_answer(self, params: dict) -> ToolResult:
        """Final answer is just returned, no execution needed."""
        answer = params.get("answer")
        if answer is None:
            raise ValueError("Missing required parameter: answer")

        return ToolResult(
            success=True,
            output=answer,
            metadata={"sources": params.get("sources", [])}
        )


def parse_orchestrator_response(
    response,
    tools: list[dict]
) -> OrchestratorDecision:
    """Parse OpenAI-style response into structured decision.

    Args:
        response: OpenAI chat completion response
        tools: List of tool definitions

    Returns:
        OrchestratorDecision with parsed data
    """
    message = response.choices[0].message
    tool_calls = []
    is_final = False
    final_answer = None
    sources = None

    # Parse tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                parameters = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                parameters = {"error": f"Invalid JSON: {e}"}

            tc_obj = ToolCall(
                name=tc.function.name,
                parameters=parameters
            )
            tool_calls.append(tc_obj)

            # Check if this is final_answer
            if tc.function.name == "final_answer":
                is_final = True
                final_answer = parameters.get("answer")
                sources = parameters.get("sources")

    return OrchestratorDecision(
        reasoning=message.content or "",
        tool_calls=tool_calls,
        is_final=is_final,
        final_answer=final_answer,
        sources=sources,
        raw_response=str(message)
    )
