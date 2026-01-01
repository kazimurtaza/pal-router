"""PAL-style agentic workflow with code execution."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from pal_router.config import Config
from pal_router.models import CompletionResult, ModelClient


PAL_SYSTEM_PROMPT = """You are a reasoning assistant that solves problems by writing Python code.

For any problem requiring calculation, logic, or multi-step reasoning:
1. Analyze the problem and identify what needs to be computed
2. Write Python code that solves it (use only standard library + math)
3. The code MUST print() the final answer
4. Include comments explaining your reasoning

Respond ONLY with Python code inside ```python``` blocks. No other text."""

VERIFICATION_PROMPT = """The code produced this error:
{stderr}

Please fix the code and try again. Respond only with corrected Python code inside ```python``` blocks."""

SYNTHESIS_PROMPT = """The computation produced this result: {result}

Original question: {question}

Provide a clear, natural language answer based on this result. Be concise."""


@dataclass
class ExecutionResult:
    """Result from executing Python code."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool


@dataclass
class AgenticResult:
    """Result from the agentic workflow."""

    answer: str
    success: bool
    code: str | None = None
    attempts: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    fallback_used: bool = False
    completions: list[CompletionResult] = field(default_factory=list)


def _set_resource_limits(memory_limit_mb: int) -> None:
    """Set resource limits for the subprocess (Unix only)."""
    try:
        import resource

        # Set memory limit
        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ImportError, ValueError):
        # resource module not available on Windows, or limit setting failed
        pass


def execute_python(
    code: str,
    timeout_seconds: int = 30,
    memory_limit_mb: int = 512,
) -> ExecutionResult:
    """Safely execute Python code in a subprocess.

    Args:
        code: Python code to execute.
        timeout_seconds: Maximum execution time.
        memory_limit_mb: Maximum memory usage in MB.

    Returns:
        ExecutionResult with stdout, stderr, return_code, and timeout status.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        temp_path = f.name

    try:
        # Build subprocess command
        # On Unix, we can set resource limits via preexec_fn
        preexec = None
        if sys.platform != "win32":
            preexec = lambda: _set_resource_limits(memory_limit_mb)

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=preexec,
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr="Execution timed out",
            return_code=-1,
            timed_out=True,
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=str(e),
            return_code=-1,
            timed_out=False,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def extract_python_code(text: str) -> str | None:
    """Extract code from ```python``` blocks.

    Args:
        text: Model response potentially containing code blocks.

    Returns:
        Extracted Python code, or None if no valid block found.
    """
    # Try to find ```python blocks
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try ``` blocks without language specifier
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


class AgenticWorkflow:
    """PAL-style agentic workflow with code execution and optional fallback."""

    def __init__(
        self,
        weak_model: ModelClient,
        strong_model: ModelClient | None = None,
        config: Config | None = None,
    ):
        """Initialize the agentic workflow.

        Args:
            weak_model: The weak model client for code generation.
            strong_model: Optional strong model for fallback.
            config: Configuration with retry limits, timeouts, etc.
        """
        self.weak_model = weak_model
        self.strong_model = strong_model
        self.config = config or Config()

    def execute(self, prompt: str) -> AgenticResult:
        """Execute the agentic workflow.

        1. Ask weak model to generate Python code
        2. Execute code
        3. If error, feed error back and retry
        4. If all retries fail and fallback is enabled, use strong model
        5. Return final answer

        Args:
            prompt: The user's query.

        Returns:
            AgenticResult with answer, code, cost, and execution details.
        """
        completions: list[CompletionResult] = []
        total_cost = 0.0
        total_latency = 0.0
        last_code: str | None = None
        last_error = ""

        # Try PAL approach
        for attempt in range(self.config.agentic_max_retries):
            # Generate code
            if attempt == 0:
                response = self.weak_model.complete(prompt, system=PAL_SYSTEM_PROMPT)
            else:
                error_prompt = VERIFICATION_PROMPT.format(stderr=last_error)
                response = self.weak_model.complete(error_prompt, system=PAL_SYSTEM_PROMPT)

            completions.append(response)
            total_cost += response.cost_usd
            total_latency += response.latency_ms

            # Extract code
            code = extract_python_code(response.content)
            if not code:
                last_error = "No valid Python code block found in response"
                continue

            last_code = code

            # Execute code
            result = execute_python(
                code,
                timeout_seconds=self.config.agentic_timeout_seconds,
                memory_limit_mb=self.config.agentic_memory_limit_mb,
            )

            if result.return_code == 0 and result.stdout.strip():
                # Success! Optionally synthesize natural language answer
                answer = result.stdout.strip()

                return AgenticResult(
                    answer=answer,
                    success=True,
                    code=code,
                    attempts=attempt + 1,
                    total_cost_usd=total_cost,
                    total_latency_ms=total_latency,
                    fallback_used=False,
                    completions=completions,
                )

            # Save error for next retry
            last_error = result.stderr or "Code execution produced no output"

        # All retries exhausted - try fallback if enabled
        if self.config.agentic_fallback_to_strong and self.strong_model:
            return self._fallback_to_strong(
                prompt,
                completions,
                total_cost,
                total_latency,
            )

        # No fallback - return failure
        return AgenticResult(
            answer=f"Failed to solve after {self.config.agentic_max_retries} attempts. Last error: {last_error}",
            success=False,
            code=last_code,
            attempts=self.config.agentic_max_retries,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            fallback_used=False,
            completions=completions,
        )

    def _fallback_to_strong(
        self,
        prompt: str,
        prior_completions: list[CompletionResult],
        prior_cost: float,
        prior_latency: float,
    ) -> AgenticResult:
        """Fallback to strong model when PAL fails.

        Args:
            prompt: Original user prompt.
            prior_completions: Completions from failed PAL attempts.
            prior_cost: Cost accumulated from PAL attempts.
            prior_latency: Latency accumulated from PAL attempts.

        Returns:
            AgenticResult from strong model.
        """
        if not self.strong_model:
            return AgenticResult(
                answer="Fallback requested but no strong model available",
                success=False,
                attempts=self.config.agentic_max_retries,
                total_cost_usd=prior_cost,
                total_latency_ms=prior_latency,
                fallback_used=False,
                completions=prior_completions,
            )

        # Simple direct prompt to strong model
        response = self.strong_model.complete(prompt)

        return AgenticResult(
            answer=response.content,
            success=True,
            code=None,  # No code execution in fallback
            attempts=self.config.agentic_max_retries + 1,
            total_cost_usd=prior_cost + response.cost_usd,
            total_latency_ms=prior_latency + response.latency_ms,
            fallback_used=True,
            completions=prior_completions + [response],
        )

    def synthesize_answer(
        self,
        prompt: str,
        raw_result: str,
    ) -> tuple[str, CompletionResult]:
        """Synthesize a natural language answer from code output.

        Args:
            prompt: Original user question.
            raw_result: Raw output from code execution.

        Returns:
            Tuple of (synthesized answer, completion result).
        """
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            result=raw_result,
            question=prompt,
        )
        response = self.weak_model.complete(synthesis_prompt)
        return response.content, response
