"""
Generation package: Evidence-grounded answer generation, LLM providers, and post-generation verification.
"""
from src.generation.answer_gen import AnswerGenerator, GeneratedAnswer, GenerationStatus, SYSTEM_PROMPT
from src.generation.verifier import PostGenerationVerifier, VerificationResult, VerificationStatus
from src.generation.providers import (
    LLMProvider,
    MockProvider,
    OllamaProvider,
    APIProvider,
    LLMError,
    LLMConfigurationError,
    LLMGenerationError,
    get_llm_provider,
)

__all__ = [
    "AnswerGenerator",
    "GeneratedAnswer",
    "GenerationStatus",
    "SYSTEM_PROMPT",
    "PostGenerationVerifier",
    "VerificationResult",
    "VerificationStatus",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "APIProvider",
    "LLMError",
    "LLMConfigurationError",
    "LLMGenerationError",
    "get_llm_provider",
]
