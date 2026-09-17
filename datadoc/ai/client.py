"""Multi-provider LLM client integration using LiteLLM."""

from __future__ import annotations

import os
from typing import Any, Optional


def detect_provider_and_model(preferred_model: Optional[str] = None) -> tuple[str, str]:
    """Detects available LLM credentials and returns (provider, model_name).

    Prioritizes explicit `preferred_model` if provided. Otherwise inspects environment
    variables for Gemini, OpenAI, Anthropic, Groq, or local Ollama.
    """
    if preferred_model:
        if "gemini" in preferred_model.lower():
            return "google", preferred_model
        if "claude" in preferred_model.lower() or "anthropic" in preferred_model.lower():
            return "anthropic", preferred_model
        if "groq" in preferred_model.lower():
            return "groq", preferred_model
        if "ollama" in preferred_model.lower():
            return "ollama", preferred_model
        return "openai", preferred_model

    # Auto-detection from environment
    if os.environ.get("GEMINI_API_KEY"):
        return "google", "gemini/gemini-3.6-flash"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-3-5-haiku-20241022"
    if os.environ.get("GROQ_API_KEY"):
        return "groq", "groq/llama-3.3-70b-versatile"
    if os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_HOST"):
        return "ollama", "ollama/llama3"

    # Default fallback placeholder
    return "none", "gpt-4o-mini"


def is_ai_configured() -> bool:
    """Check whether any LLM API key or local provider is detected in the environment."""
    keys = [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "OLLAMA_API_BASE",
        "OLLAMA_HOST",
    ]
    return any(bool(os.environ.get(k)) for k in keys)


class LLMClient:
    """Thin wrapper around LiteLLM providing resilient timeout and error handling."""

    def __init__(self, model: Optional[str] = None, timeout: int = 45):
        self.provider, self.model = detect_provider_and_model(model)
        self.timeout = timeout

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invokes the configured LLM with prompt and optional system message."""
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "LiteLLM is required for AI features. Install it with: pip install 'datadoc-cli[ai]'"
            ) from e

        # Suppress verbose LiteLLM telemetry/logs
        litellm.telemetry = False
        litellm.suppress_debug_info = True

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response: Any = litellm.completion(
                model=self.model,
                messages=messages,
                timeout=self.timeout,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned an empty response.")
            return str(content).strip()
        except Exception as err:
            err_str = str(err)
            if "AuthenticationError" in err_str or "API key" in err_str or "401" in err_str:
                raise RuntimeError(
                    f"Authentication failed for {self.model}. Please check that your API key is valid."
                ) from err
            if "Timeout" in err_str or "timeout" in err_str:
                raise TimeoutError(f"LLM request timed out after {self.timeout}s.") from err
            raise RuntimeError(f"LLM call to {self.model} failed: {err_str}") from err


def get_llm_client(model: Optional[str] = None, timeout: int = 45) -> LLMClient:
    """Factory helper to obtain an initialized LLMClient."""
    return LLMClient(model=model, timeout=timeout)
