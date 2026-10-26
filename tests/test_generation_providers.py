"""
Unit tests for Phase 4: LLM Provider Integration and Grounded Generation Safety.

Covers:
  1. Provider interface abstraction (MockProvider, OllamaProvider, APIProvider).
  2. Factory configuration and provider switching via environment variables.
  3. Missing configuration handling (e.g., missing or placeholder API key).
  4. Model errors, connection failures, and timeouts.
  5. Graceful fallback when model generation fails.
  6. Evidence-grounded generation with source IDs and uncertainty handling.
  7. Post-generation verifier rejection of fabricated chemical claims.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from typing import List

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
from src.generation.answer_gen import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationStatus,
    SYSTEM_PROMPT,
)
from src.generation.verifier import (
    PostGenerationVerifier,
    VerificationStatus,
    VerificationResult,
)
from src.guard.relevance_gate import GateResult, GateVerdict
from src.retrieval.vector_store import RetrievalResult
from src.config import AGRONOMIC_SAFETY_NOTICE, SOURCE_ATTRIBUTION


def make_test_chunk(
    chunk_id: str = "kcc_chunk_wheat_001",
    crop: str = "Wheat",
    crop_clean: str = "Wheat",
    similarity_score: float = 0.68,
    expert_advisory: str = "Spray tebuconazole 1 ml/litre of water for yellow rust control.",
    farmer_query: str = "How to treat yellow rust in wheat?",
    state: str = "Punjab",
    district: str = "Ludhiana",
    original_year: int = 2020,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        similarity_score=similarity_score,
        crop=crop,
        crop_clean=crop_clean,
        state=state,
        district=district,
        advisory_domain="plant_protection",
        query_type="disease_control",
        original_year=original_year,
        farmer_query=farmer_query,
        expert_advisory=expert_advisory,
        qc_severity="none",
        qc_confidence="high",
        qc_flags=[],
        rag_use_mode="use_as_evidence",
        source_attribution=SOURCE_ATTRIBUTION,
        safety_notice=AGRONOMIC_SAFETY_NOTICE,
        retrieval_text=f"Crop: {crop_clean}. Query: {farmer_query}. Advisory: {expert_advisory}",
    )


class TestLLMProviderInterface(unittest.TestCase):
    """Tests for the LLMProvider abstraction and factory."""

    def test_mock_provider_defaults(self):
        provider = MockProvider()
        self.assertEqual(provider.get_provider_name(), "mock")
        output = provider.generate("Test prompt")
        self.assertIn("historical Kisan Call Centre (KCC) records", output)
        self.assertIn("Test prompt", output)

    def test_mock_provider_custom_responses(self):
        canned = {"rust": "Rust treatment advice grounded in record kcc_001."}
        provider = MockProvider(canned_responses=canned)
        output = provider.generate("How to treat rust in wheat?")
        self.assertEqual(output, "Rust treatment advice grounded in record kcc_001.")

    def test_factory_returns_mock(self):
        provider = get_llm_provider("mock")
        self.assertIsInstance(provider, MockProvider)

    def test_factory_returns_ollama(self):
        provider = get_llm_provider("ollama")
        self.assertIsInstance(provider, OllamaProvider)

    def test_factory_raises_for_invalid_provider(self):
        with self.assertRaises(LLMConfigurationError):
            get_llm_provider("unsupported_provider_xyz")


class TestMissingConfigurationAndSecurity(unittest.TestCase):
    """Tests for missing configuration and secret protection."""

    def test_api_provider_rejects_missing_key(self):
        with self.assertRaises(LLMConfigurationError) as ctx:
            APIProvider(api_key="", base_url="https://api.openai.com/v1")
        self.assertIn("LLM_API_KEY is not configured", str(ctx.exception))

    def test_api_provider_rejects_placeholder_key(self):
        with self.assertRaises(LLMConfigurationError) as ctx:
            APIProvider(api_key="your_api_key_here", base_url="https://api.openai.com/v1")
        self.assertIn("placeholder", str(ctx.exception).lower())

    @patch("src.generation.providers.requests.post")
    def test_api_provider_does_not_log_secret_key(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Sample advisory answer."}}]
        }
        with self.assertLogs("src.generation.providers", level="INFO") as log_capture:
            provider = APIProvider(api_key="sk-real-secret-key-123456789")
            provider.generate("Test question")
            for record in log_capture.output:
                self.assertNotIn("sk-real-secret-key-123456789", record)


class TestModelErrorsAndFallbacks(unittest.TestCase):
    """Tests for network failures, timeouts, and graceful fallbacks."""

    @patch("src.generation.providers.requests.post")
    def test_ollama_connection_error_raises_llm_generation_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        provider = OllamaProvider(base_url="http://127.0.0.1:11434")
        with self.assertRaises(LLMGenerationError) as ctx:
            provider.generate("Farmer query")
        self.assertIn("Cannot connect to Ollama server", str(ctx.exception))

    @patch("src.generation.providers.requests.post")
    def test_ollama_timeout_raises_llm_generation_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", timeout=5.0)
        with self.assertRaises(LLMGenerationError) as ctx:
            provider.generate("Farmer query")
        self.assertIn("timed out", str(ctx.exception).lower())

    @patch("src.generation.providers.requests.post")
    def test_api_http_error_raises_llm_generation_error(self, mock_post):
        import requests
        resp = MagicMock()
        resp.status_code = 401
        mock_post.side_effect = requests.exceptions.HTTPError("401 Client Error: Unauthorized", response=resp)
        provider = APIProvider(api_key="sk-valid-format-key-abc")
        with self.assertRaises(LLMGenerationError) as ctx:
            provider.generate("Farmer query")
        self.assertIn("401", str(ctx.exception))

    def test_answer_generator_falls_back_gracefully_on_provider_error(self):
        """When the LLM provider fails, AnswerGenerator must not crash; it must fall back to the safe template."""
        failing_provider = MagicMock(spec=LLMProvider)
        failing_provider.get_provider_name.return_value = "failing_mock"
        failing_provider.generate.side_effect = LLMGenerationError("Simulated LLM outage")

        generator = AnswerGenerator(provider=failing_provider)
        chunk = make_test_chunk()
        gate_result = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            reason="Good match",
            best_score=0.68,
            best_chunk=chunk,
            sufficient_chunks=[chunk],
        )

        answer = generator.generate("How to treat yellow rust in wheat?", gate_result)
        self.assertEqual(answer.status, GenerationStatus.SUCCESS)
        self.assertIn("kcc_chunk_wheat_001", answer.source_ids)
        # Verify fallback produced structured output containing the advisory
        self.assertIn("tebuconazole", answer.answer_text)
        self.assertIn("historical", answer.answer_text.lower())


class TestGroundedAnswerAndSafetyPipeline(unittest.TestCase):
    """Tests that generated answers preserve source citations, uncertainty notes, and verifier checks."""

    def setUp(self):
        self.chunk = make_test_chunk(
            chunk_id="kcc_chunk_wheat_001",
            crop="Wheat",
            expert_advisory="Spray tebuconazole 1 ml/litre of water for yellow rust control.",
        )
        self.gate_sufficient = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            reason="Sufficient similarity",
            best_score=0.68,
            best_chunk=self.chunk,
            sufficient_chunks=[self.chunk],
        )
        self.gate_weak = GateResult(
            verdict=GateVerdict.EVIDENCE_WEAK,
            reason="Weak score",
            best_score=0.35,
            best_chunk=self.chunk,
            all_chunks=[self.chunk],
        )

    def test_grounded_answer_includes_source_ids_and_notices(self):
        generator = AnswerGenerator(provider=MockProvider())
        answer = generator.generate("How to treat yellow rust in wheat?", self.gate_sufficient)

        self.assertEqual(answer.status, GenerationStatus.SUCCESS)
        self.assertIn("kcc_chunk_wheat_001", answer.source_ids)
        self.assertEqual(answer.crops_cited, ["Wheat"])
        self.assertIn("Kisan Call Centre", answer.provenance_note)
        self.assertIn("NOTICE:", answer.safety_notice)

    def test_weak_evidence_distinguishes_uncertainty(self):
        generator = AnswerGenerator(provider=MockProvider())
        answer = generator.generate("How to treat rust in wheat?", self.gate_weak)

        self.assertEqual(answer.status, GenerationStatus.WEAK_EVIDENCE)
        self.assertIsNotNone(answer.uncertainty_note)
        self.assertIn("low relevance score", answer.uncertainty_note)
        self.assertIn("Krishi Vigyan Kendra", answer.uncertainty_note)

    def test_verifier_passes_grounded_answer(self):
        verifier = PostGenerationVerifier()
        answer = GeneratedAnswer(
            status=GenerationStatus.SUCCESS,
            answer_text="Based on [kcc_chunk_wheat_001], spray tebuconazole 1 ml/litre for yellow rust.",
            source_ids=["kcc_chunk_wheat_001"],
            crops_cited=["Wheat"],
            safety_notice=AGRONOMIC_SAFETY_NOTICE,
        )
        result = verifier.verify(answer, [self.chunk])
        self.assertEqual(result.status, VerificationStatus.PASSED)
        self.assertTrue(result.is_safe_to_present)

    def test_verifier_blocks_hallucinated_chemical_dosage(self):
        """If the LLM invents a chemical or ungrounded dosage, verifier must block it."""
        verifier = PostGenerationVerifier()
        # Evidence only has "tebuconazole 1 ml/litre"
        # The LLM generated an ungrounded recommendation for "chlorpyrifos 5 ml/litre"
        hallucinated_answer = GeneratedAnswer(
            status=GenerationStatus.SUCCESS,
            answer_text="You should apply chlorpyrifos 5 ml/litre for rust control.",
            source_ids=["kcc_chunk_wheat_001"],
            crops_cited=["Wheat"],
            safety_notice=AGRONOMIC_SAFETY_NOTICE,
        )
        result = verifier.verify(hallucinated_answer, [self.chunk])
        self.assertEqual(result.status, VerificationStatus.BLOCKED_HALLUCINATION)
        self.assertTrue(result.is_blocked)
        self.assertFalse(result.is_safe_to_present)
        self.assertEqual(result.safe_answer.status, GenerationStatus.ABSTAINED_NO_EVIDENCE)
        self.assertEqual(result.safe_answer.answer_text, "")
        self.assertIn("HALLUCINATION_RISK", result.issues[0])


if __name__ == "__main__":
    unittest.main()
