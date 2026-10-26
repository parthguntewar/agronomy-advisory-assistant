"""
LLM Provider Abstraction Layer for the Agronomy Advisory Assistant.

This module provides a unified interface for language model generation:
  1. Base LLMProvider interface (ABC).
  2. MockProvider: Deterministic, offline generator for tests and CI.
  3. OllamaProvider: Local, private inference via Ollama (e.g., Llama 3.2).
  4. APIProvider: Cloud inference via OpenAI-compatible endpoints.
  5. Factory function (get_llm_provider) for dynamic configuration via environment variables.

Safety & Security Guarantees:
  - Secrets (API keys) are read strictly from environment variables and never logged.
  - Missing configurations or provider failures raise structured errors without exposing secrets.
  - Automated tests can run completely offline without an API key or network access using MockProvider.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import requests

from src import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for all LLM provider errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is misconfigured (e.g., missing API key, invalid URL)."""
    pass


class LLMGenerationError(LLMError):
    """Raised when an LLM provider fails during text generation (network, timeout, API error)."""
    pass


class LLMProvider(ABC):
    """Abstract base class for all LLM generation providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate completion text for the given prompt.

        Args:
            prompt: User prompt containing farmer query and retrieved evidence context.
            system_prompt: System prompt with evidence-grounding and safety rules.

        Returns:
            Generated text string.

        Raises:
            LLMConfigurationError: If provider configuration is invalid.
            LLMGenerationError: If provider generation fails.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return human-readable provider identifier."""
        pass


class MockProvider(LLMProvider):
    """
    Deterministic offline provider for automated testing and CI.
    Requires no API keys, no network access, and no local GPU runtime.
    """

    def __init__(
        self,
        default_response: Optional[str] = None,
        canned_responses: Optional[Dict[str, str]] = None,
    ) -> None:
        self.default_response = default_response
        self.canned_responses = canned_responses or {}

    def get_provider_name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Check if an exact canned response matches the prompt
        for key, resp in self.canned_responses.items():
            if key in prompt:
                return resp

        if self.default_response:
            return self.default_response

        # Default behavior: construct an evidence-grounded response reflecting the prompt
        return (
            "Based on historical Kisan Call Centre (KCC) records, here is the verified advisory evidence:\n\n"
            f"{prompt}\n\n"
            "NOTE: This information is derived from historical KCC transcripts. "
            "Chemical recommendations and dosages must be independently confirmed with local "
            "agricultural extension officers (KVK / SAU) before application."
        )


class OllamaProvider(LLMProvider):
    """
    Local model provider connecting to an Ollama server (e.g. Llama 3.2).
    Provides 100% private, zero-cost on-device inference without external network calls.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else config.OLLAMA_TIMEOUT

    def get_provider_name(self) -> str:
        return f"ollama ({self.model})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for strict factual grounding
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.info(f"[OllamaProvider] Sending request to {self.base_url} (model={self.model})")

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "").strip()
            if not response_text:
                raise LLMGenerationError(
                    f"Ollama returned empty response for model '{self.model}'."
                )
            return response_text
        except requests.exceptions.ConnectionError as e:
            raise LLMGenerationError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Ensure Ollama is running (`ollama serve` or start Ollama app)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMGenerationError(
                f"Ollama request timed out after {self.timeout}s (model={self.model})."
            ) from e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            raise LLMGenerationError(
                f"Ollama HTTP error {status}: {e}"
            ) from e
        except Exception as e:
            raise LLMGenerationError(f"Unexpected Ollama generation failure: {e}") from e


class APIProvider(LLMProvider):
    """
    Cloud model provider connecting to OpenAI-compatible Chat Completions endpoints
    (OpenAI, Groq, Together, DeepSeek, Local vLLM, etc.).
    Uses strictly environment-variable-configured credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = (base_url or config.LLM_API_BASE).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT

        # Sanitize check: Ensure API key is configured and not a placeholder
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        placeholder_keys = {
            "",
            "your_api_key_here",
            "sk-placeholder",
            "your_key",
            "<api_key>",
        }
        if not self.api_key or self.api_key.strip().lower() in placeholder_keys:
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured or contains placeholder value. "
                "Set a valid LLM_API_KEY in your environment or .env file."
            )

    def get_provider_name(self) -> str:
        return f"api ({self.model})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Re-check configuration before each request
        self._validate_configuration()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Deterministic, grounded extraction
        }

        # Safe logging: Never log the Authorization header or secret key
        logger.info(f"[APIProvider] Sending request to {self.base_url} (model={self.model})")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMGenerationError("API returned response with no choices.")
            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise LLMGenerationError("API returned empty message content.")
            return content
        except requests.exceptions.ConnectionError as e:
            raise LLMGenerationError(
                f"Cannot connect to API endpoint at {self.base_url}: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMGenerationError(
                f"API request timed out after {self.timeout}s (model={self.model})."
            ) from e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            raise LLMGenerationError(
                f"API provider HTTP error {status}."
            ) from e
        except Exception as e:
            raise LLMGenerationError(f"Unexpected API provider failure: {e}") from e


def get_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """
    Factory function to instantiate the configured LLM provider.

    Resolution order:
      1. Explicit provider_type argument.
      2. LLM_PROVIDER environment variable / config.py (default: 'mock').

    Supported values:
      - 'mock': Deterministic, offline generator (default).
      - 'ollama': Local Ollama runtime.
      - 'api': OpenAI-compatible REST API.
    """
    chosen = (provider_type or config.LLM_PROVIDER).strip().lower()

    if chosen == "mock":
        return MockProvider()
    elif chosen == "ollama":
        return OllamaProvider()
    elif chosen in ("api", "openai"):
        return APIProvider()
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider '{chosen}'. Supported options: 'mock', 'ollama', 'api'."
        )
