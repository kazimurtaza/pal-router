"""Model client abstractions for unified LLM access."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from pal_router.config import ModelConfig


@dataclass
class CompletionResult:
    """Result from a model completion."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class ModelClient(Protocol):
    """Protocol for model clients."""

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        """Generate a completion for the given prompt."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class OpenAIClient:
    """OpenAI model client."""

    def __init__(self, config: ModelConfig, api_key: str | None = None):
        """Initialize the OpenAI client.

        Args:
            config: Model configuration with name and pricing.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @property
    def model_name(self) -> str:
        return self.config.name

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        """Generate a completion using OpenAI.

        Args:
            prompt: The user prompt.
            system: Optional system message.

        Returns:
            CompletionResult with content, tokens, cost, and latency.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        cost = (
            (input_tokens / 1000) * self.config.cost_per_1k_input
            + (output_tokens / 1000) * self.config.cost_per_1k_output
        )

        return CompletionResult(
            content=response.choices[0].message.content or "",
            model=self.config.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )


class AnthropicClient:
    """Anthropic model client."""

    def __init__(self, config: ModelConfig, api_key: str | None = None):
        """Initialize the Anthropic client.

        Args:
            config: Model configuration with name and pricing.
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        from anthropic import Anthropic

        self.config = config
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    @property
    def model_name(self) -> str:
        return self.config.name

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        """Generate a completion using Anthropic.

        Args:
            prompt: The user prompt.
            system: Optional system message.

        Returns:
            CompletionResult with content, tokens, cost, and latency.
        """
        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.config.name,
            max_tokens=4096,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost = (
            (input_tokens / 1000) * self.config.cost_per_1k_input
            + (output_tokens / 1000) * self.config.cost_per_1k_output
        )

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return CompletionResult(
            content=content,
            model=self.config.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )


class LlamaCppClient:
    """Client for llama.cpp server (OpenAI-compatible API)."""

    def __init__(
        self,
        config: ModelConfig,
        base_url: str = "http://10.3.0.163:8080/v1",
    ):
        """Initialize the llama.cpp client.

        Args:
            config: Model configuration.
            base_url: URL of the llama.cpp server.
        """
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=120.0,  # Local models can be slow
        )

    @property
    def model_name(self) -> str:
        return self.config.name

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        """Generate a completion using llama.cpp server."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return CompletionResult(
            content=response.choices[0].message.content or "",
            model=self.config.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # Free - local
            latency_ms=latency_ms,
        )


class GroqClient:
    """Groq client (OpenAI-compatible API)."""

    def __init__(self, config: ModelConfig, api_key: str | None = None):
        """Initialize the Groq client.

        Args:
            config: Model configuration.
            api_key: Groq API key. If None, uses GROQ_API_KEY env var.
        """
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.getenv("GROQ_API_KEY"),
        )

    @property
    def model_name(self) -> str:
        return self.config.name

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        """Generate a completion using Groq."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        cost = (
            (input_tokens / 1000) * self.config.cost_per_1k_input
            + (output_tokens / 1000) * self.config.cost_per_1k_output
        )

        return CompletionResult(
            content=response.choices[0].message.content or "",
            model=self.config.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )


class GeminiClient:
    """Google Gemini client with rate limit handling."""

    def __init__(self, config: ModelConfig, api_key: str | None = None):
        """Initialize the Gemini client.

        Args:
            config: Model configuration.
            api_key: Google AI API key. If None, uses GOOGLE_API_KEY env var.
        """
        import google.generativeai as genai

        self.config = config
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(config.name)

    @property
    def model_name(self) -> str:
        return self.config.name

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        """Generate a completion using Gemini."""
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        start = time.perf_counter()

        # Retry with backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(full_prompt)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    import time as t
                    t.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise

        latency_ms = (time.perf_counter() - start) * 1000

        # Gemini doesn't always provide token counts
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        cost = (
            (input_tokens / 1000) * self.config.cost_per_1k_input
            + (output_tokens / 1000) * self.config.cost_per_1k_output
        )

        return CompletionResult(
            content=response.text,
            model=self.config.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )


class FallbackClient:
    """Client that falls back through a chain of providers on rate limits.

    Chain: Gemini → Groq → Local llama.cpp
    """

    def __init__(self, *clients: ModelClient):
        """Initialize with a chain of clients (first = primary, last = final fallback).

        Args:
            *clients: ModelClient instances in priority order.
        """
        if len(clients) < 2:
            raise ValueError("Need at least 2 clients for fallback")
        self.clients = list(clients)
        self._current_index = 0

    @property
    def model_name(self) -> str:
        client = self.clients[self._current_index]
        if self._current_index > 0:
            return f"{client.model_name} (fallback #{self._current_index})"
        return client.model_name

    def complete(self, prompt: str, system: str | None = None) -> CompletionResult:
        last_error = None

        for i, client in enumerate(self.clients):
            try:
                result = client.complete(prompt, system)
                self._current_index = i
                return result
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Continue to next fallback on rate limit or connection errors
                if "429" in str(e) or "rate" in error_str or "connection" in error_str:
                    continue
                # Re-raise non-rate-limit errors
                raise

        # All fallbacks exhausted
        self._current_index = len(self.clients) - 1
        raise last_error or RuntimeError("All providers failed")


def get_client(
    config: ModelConfig,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
) -> ModelClient:
    """Factory function to get the appropriate model client.

    Args:
        config: Model configuration.
        provider: One of "openai", "anthropic", "llamacpp", "groq", "gemini".
        base_url: Optional base URL for llamacpp.
        api_key: Optional API key override.

    Returns:
        A ModelClient instance.
    """
    if provider == "anthropic":
        return AnthropicClient(config, api_key)
    elif provider == "llamacpp":
        return LlamaCppClient(config, base_url or "http://10.3.0.163:8080/v1")
    elif provider == "groq":
        return GroqClient(config, api_key)
    elif provider == "gemini":
        return GeminiClient(config, api_key)
    return OpenAIClient(config, api_key)
