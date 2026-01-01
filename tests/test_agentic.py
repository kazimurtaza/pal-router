"""Tests for the agentic workflow module."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pal_router.agentic import (
    AgenticResult,
    AgenticWorkflow,
    ExecutionResult,
    execute_python,
    extract_python_code,
)
from pal_router.models import CompletionResult


class TestExtractPythonCode:
    """Tests for code extraction from model responses."""

    def test_extract_python_block(self):
        """Should extract code from ```python``` blocks."""
        text = """Here's the solution:

```python
x = 5
print(x * 2)
```

This will print 10."""
        code = extract_python_code(text)
        assert code == "x = 5\nprint(x * 2)"

    def test_extract_generic_block(self):
        """Should fallback to ``` blocks without language specifier."""
        text = """Solution:

```
result = 15 * 0.2
print(result)
```
"""
        code = extract_python_code(text)
        assert code == "result = 15 * 0.2\nprint(result)"

    def test_no_code_block(self):
        """Should return None if no code block found."""
        text = "The answer is 42."
        code = extract_python_code(text)
        assert code is None

    def test_multiple_blocks_takes_first(self):
        """Should take the first code block."""
        text = """First:
```python
print("first")
```

Second:
```python
print("second")
```
"""
        code = extract_python_code(text)
        assert code == 'print("first")'


class TestExecutePython:
    """Tests for Python code execution."""

    def test_simple_execution(self):
        """Should execute simple Python code."""
        code = "print(2 + 2)"
        result = execute_python(code)
        assert result.return_code == 0
        assert result.stdout.strip() == "4"
        assert not result.timed_out

    def test_execution_with_error(self):
        """Should capture errors from code execution."""
        code = "print(undefined_variable)"
        result = execute_python(code)
        assert result.return_code != 0
        assert "NameError" in result.stderr

    def test_multiline_output(self):
        """Should capture multiline output."""
        code = """
for i in range(3):
    print(i)
"""
        result = execute_python(code)
        assert result.return_code == 0
        assert result.stdout.strip() == "0\n1\n2"

    def test_math_calculation(self):
        """Should handle math operations."""
        code = """
import math
result = math.sqrt(16) + math.pi
print(f"{result:.2f}")
"""
        result = execute_python(code)
        assert result.return_code == 0
        # sqrt(16) = 4, pi ≈ 3.14, total ≈ 7.14
        assert "7.14" in result.stdout

    def test_timeout(self):
        """Should timeout on long-running code."""
        code = """
import time
time.sleep(10)
print("done")
"""
        result = execute_python(code, timeout_seconds=1)
        assert result.timed_out
        assert result.return_code == -1

    def test_syntax_error(self):
        """Should capture syntax errors."""
        code = "print('unclosed string"
        result = execute_python(code)
        assert result.return_code != 0
        assert "SyntaxError" in result.stderr or "EOL" in result.stderr


class TestAgenticResult:
    """Tests for AgenticResult dataclass."""

    def test_successful_result(self):
        """Should store successful result data."""
        result = AgenticResult(
            answer="30",
            success=True,
            code="print(15 * 2)",
            attempts=1,
            total_cost_usd=0.001,
            total_latency_ms=500,
            fallback_used=False,
        )
        assert result.answer == "30"
        assert result.success is True
        assert result.code == "print(15 * 2)"
        assert result.attempts == 1
        assert not result.fallback_used

    def test_failed_result(self):
        """Should store failed result data."""
        result = AgenticResult(
            answer="Error: could not solve",
            success=False,
            code=None,
            attempts=3,
            total_cost_usd=0.003,
            total_latency_ms=1500,
            fallback_used=False,
        )
        assert result.success is False
        assert result.attempts == 3


class TestAgenticWorkflow:
    """Tests for AgenticWorkflow with mocked models."""

    def test_successful_execution(self):
        """Should succeed when code executes correctly."""
        # Mock weak model that returns valid Python code
        weak_model = Mock()
        weak_model.complete.return_value = CompletionResult(
            content='```python\nprint(15 * 0.15 * 100)\n```',
            model="mock",
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.001,
            latency_ms=100,
        )

        workflow = AgenticWorkflow(weak_model=weak_model)
        result = workflow.execute("Calculate 15% of $100")

        assert result.success is True
        assert "225" in result.answer  # 15 * 0.15 * 100 = 225
        assert result.attempts == 1
        assert not result.fallback_used

    def test_retry_on_error(self):
        """Should retry when code has errors."""
        weak_model = Mock()
        # First call returns buggy code, second returns fixed code
        weak_model.complete.side_effect = [
            CompletionResult(
                content='```python\nprint(undefined_var)\n```',
                model="mock",
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.001,
                latency_ms=100,
            ),
            CompletionResult(
                content='```python\nprint(42)\n```',
                model="mock",
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.001,
                latency_ms=100,
            ),
        ]

        workflow = AgenticWorkflow(weak_model=weak_model)
        result = workflow.execute("What is the answer?")

        assert result.success is True
        assert "42" in result.answer
        assert result.attempts == 2

    def test_fallback_to_strong_model(self):
        """Should fallback to strong model after max retries."""
        from pal_router.config import Config

        weak_model = Mock()
        strong_model = Mock()

        # Weak model always returns bad code
        weak_model.complete.return_value = CompletionResult(
            content='No code here, just text.',
            model="mock-weak",
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.001,
            latency_ms=100,
        )

        # Strong model gives direct answer
        strong_model.complete.return_value = CompletionResult(
            content='The answer is 42.',
            model="mock-strong",
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.01,
            latency_ms=500,
        )

        config = Config(agentic_max_retries=2, agentic_fallback_to_strong=True)
        workflow = AgenticWorkflow(
            weak_model=weak_model,
            strong_model=strong_model,
            config=config,
        )
        result = workflow.execute("What is the meaning of life?")

        assert result.success is True
        assert result.fallback_used is True
        assert "42" in result.answer
        # 2 weak attempts + 1 strong fallback
        assert result.attempts == 3

    def test_no_fallback_when_disabled(self):
        """Should not fallback when fallback is disabled."""
        from pal_router.config import Config

        weak_model = Mock()
        weak_model.complete.return_value = CompletionResult(
            content='No code here.',
            model="mock-weak",
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.001,
            latency_ms=100,
        )

        config = Config(agentic_max_retries=2, agentic_fallback_to_strong=False)
        workflow = AgenticWorkflow(weak_model=weak_model, config=config)
        result = workflow.execute("Calculate something")

        assert result.success is False
        assert result.fallback_used is False
        assert result.attempts == 2

    def test_cost_accumulation(self):
        """Should accumulate costs across attempts."""
        weak_model = Mock()
        weak_model.complete.side_effect = [
            CompletionResult(
                content='```python\nprint(bad)\n```',
                model="mock",
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.001,
                latency_ms=100,
            ),
            CompletionResult(
                content='```python\nprint(42)\n```',
                model="mock",
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.002,
                latency_ms=200,
            ),
        ]

        workflow = AgenticWorkflow(weak_model=weak_model)
        result = workflow.execute("Answer?")

        assert result.success is True
        assert result.total_cost_usd == pytest.approx(0.003)
        assert result.total_latency_ms == pytest.approx(300)
