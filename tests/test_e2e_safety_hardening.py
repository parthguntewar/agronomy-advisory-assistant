"""
End-to-End Safety and Reliability Hardening Tests (Phase 6C).

Comprehensive scenario verification:
- Evidence and hallucination safety (grounding, weak evidence caveat, cross-crop exclusion, contradictory evidence).
- Unsupported, mismatch, ambiguous, and dynamic query handling.
- Provider failures, timeout simulation, transparent fallback disclosure, and secret sanitization.
- Pipeline metadata integrity and stage consistency.
"""

import unittest
from unittest.mock import MagicMock

from src.generation.answer_gen import AnswerGenerator, GenerationStatus
from src.generation.providers import LLMProvider, LLMGenerationError
from src.generation.verifier import PostGenerationVerifier, VerificationStatus
from src.guard.query_guard import QueryGuard, GuardAction
from src.guard.relevance_gate import RelevanceGate, GateVerdict, GateResult
from src.pipeline import GuardedRAGPipeline
from src.retrieval.retriever import AgronomyRetriever
from src.retrieval.vector_store import RetrievalResult


def make_test_chunk(chunk_id: str, crop: str, score: float, advisory: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        similarity_score=score,
        crop=crop,
        crop_clean=crop,
        state="Maharashtra",
        district="Solapur",
        advisory_domain="plant_protection",
        query_type="Pest Management",
        original_year=2021,
        farmer_query="Sample test query",
        expert_advisory=advisory,
        qc_severity="low",
        qc_confidence="high",
        qc_flags=[],
        rag_use_mode="use_as_evidence",
        source_attribution="KCC Sample Test",
        safety_notice="NOTICE: Historical test record.",
        retrieval_text=advisory,
    )


class FailingProvider(LLMProvider):
    """Simulates an LLM provider that times out or throws an exception containing sensitive data."""
    def __init__(self, error_msg: str = "Request timed out after 30s with Authorization Bearer sk-secret-token-1234567890"):
        self.error_msg = error_msg

    def get_provider_name(self) -> str:
        return "failing_simulated_llm"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        raise LLMGenerationError(self.error_msg)


class EmptyProvider(LLMProvider):
    """Simulates an LLM provider that returns empty or whitespace-only content."""
    def get_provider_name(self) -> str:
        return "empty_simulated_llm"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        return "   \n\t  "


class TestE2ESafetyHardening(unittest.TestCase):
    """Phase 6C End-to-End Safety & Reliability Hardening Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = AgronomyRetriever()
        cls.pipeline = GuardedRAGPipeline(retriever=cls.retriever)
        cls.guard = QueryGuard()
        cls.gate = RelevanceGate()
        cls.verifier = PostGenerationVerifier()

    # ── A. Evidence & Hallucination Safety ────────────────────────────────────

    def test_strong_evidence_produces_grounded_verified_answer(self):
        """Strong, crop-compatible evidence (>0.50) produces SUCCESS status with source IDs."""
        res = self.pipeline.answer("Insect Control in Satawar crop?")
        self.assertEqual(res.gate_result.verdict, GateVerdict.EVIDENCE_SUFFICIENT)
        self.assertEqual(res.final_answer.status, GenerationStatus.SUCCESS)
        self.assertGreaterEqual(len(res.final_answer.source_ids), 1)
        self.assertIn("kcc_chunk_kcc_00563633", res.final_answer.source_ids)
        self.assertEqual(res.verification_result.status, VerificationStatus.PASSED)

    def test_weak_evidence_preserves_explicit_uncertainty_warning(self):
        """Weak evidence (0.30 - 0.50) preserves uncertainty_note and is verified with caveats."""
        res = self.pipeline.answer("What medicine to spray for worms in satawar roots and leaves?")
        self.assertEqual(res.gate_result.verdict, GateVerdict.EVIDENCE_WEAK)
        self.assertEqual(res.final_answer.status, GenerationStatus.WEAK_EVIDENCE)
        self.assertIsNotNone(res.final_answer.uncertainty_note)
        self.assertIn("KVK", res.final_answer.uncertainty_note)
        self.assertEqual(res.verification_result.status, VerificationStatus.PASSED_WITH_WARNINGS)
        self.assertTrue(any("WEAK_EVIDENCE_CAVEAT" in w for w in res.verification_result.warnings))

    def test_contradictory_evidence_generates_warning_preventing_unjustified_certainty(self):
        """When multiple chunks recommend differing chemical active ingredients, verifier warns."""
        c1 = make_test_chunk("chunk_1", "Paddy (Dhan)", 0.65, "Kisan bhai Quinalphos 25% EC 500 ml/acre spray kare.")
        c2 = make_test_chunk("chunk_2", "Paddy (Dhan)", 0.62, "Apply Carbofuran 3G granules 160 gram/cent in paddy field.")

        gen = AnswerGenerator()
        fake_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.65,
            best_chunk=c1,
            sufficient_chunks=[c1, c2],
            compatible_chunks=[c1, c2],
            all_chunks=[c1, c2],
        )
        answer = gen.generate("Paddy pest advice", fake_gate)
        verif = self.verifier.verify(answer, [c1, c2])

        self.assertEqual(verif.status, VerificationStatus.PASSED_WITH_WARNINGS)
        self.assertTrue(any("CONTRADICTORY_EVIDENCE" in w for w in verif.warnings))

    def test_verifier_blocks_fake_source_record_ids(self):
        """If an answer cites a record ID that was not in the approved evidence chunks, verifier blocks it."""
        c1 = make_test_chunk("chunk_real", "Tomato", 0.60, "Spray Copper Oxychloride 600 gm/acre.")
        gen = AnswerGenerator()
        fake_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.60,
            best_chunk=c1,
            sufficient_chunks=[c1],
            compatible_chunks=[c1],
        )
        answer = gen.generate("Tomato blight", fake_gate)
        # Inject hallucinated citation
        answer.source_ids = ["chunk_fake_hallucinated_id"]

        verif = self.verifier.verify(answer, [c1])
        self.assertEqual(verif.status, VerificationStatus.BLOCKED_HALLUCINATION)
        self.assertTrue(any("INVALID_CITATION" in issue for issue in verif.issues))

    def test_verifier_blocks_empty_answer_text(self):
        """An empty or blank answer text must be blocked as invalid."""
        c1 = make_test_chunk("chunk_real", "Tomato", 0.60, "Spray Copper Oxychloride 600 gm/acre.")
        gen = AnswerGenerator()
        fake_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.60,
            best_chunk=c1,
            sufficient_chunks=[c1],
            compatible_chunks=[c1],
        )
        answer = gen.generate("Tomato blight", fake_gate)
        answer.answer_text = "   "

        verif = self.verifier.verify(answer, [c1])
        self.assertEqual(verif.status, VerificationStatus.BLOCKED_HALLUCINATION)
        self.assertTrue(any("EMPTY_ANSWER" in issue for issue in verif.issues))

    # ── B. Unsupported & Ambiguous Queries ────────────────────────────────────

    def test_unsupported_crop_never_borrows_cross_crop_advice(self):
        """Unsupported crops (Quinoa, Avocado, Dragon Fruit) must abstain and never borrow advice."""
        for q, crop in [
            ("Quinoa downy mildew management practices", "Quinoa"),
            ("How to fertilize Hass avocado trees in sandy soil?", "Avocado"),
            ("What is the recommended grafting method for Brazilian Dragon Fruit in winter?", "Dragon Fruit"),
        ]:
            res = self.pipeline.answer(q)
            self.assertEqual(res.guard_decision.detected_crop, crop)
            self.assertEqual(res.gate_result.verdict, GateVerdict.CROP_MISMATCH)
            self.assertEqual(res.final_answer.status, GenerationStatus.ABSTAINED_CROP_MISMATCH)
            self.assertFalse(res.was_answered)

    def test_ambiguous_crop_disease_requests_clarification_with_zero_retrieval(self):
        """Ambiguous query missing crop requests clarification; retriever is never called."""
        res = self.pipeline.answer("My leaves have yellow spots and are curling. What spray should I use?")
        self.assertEqual(res.guard_decision.action, GuardAction.CLARIFY_CROP)
        self.assertEqual(len(res.retrieval_results), 0)
        self.assertFalse(res.was_answered)
        self.assertIn("crop", res.final_answer.abstention_reason.lower())

    def test_regional_crop_names_resolve_correctly(self):
        """Transliterated names (dhan, sarson, kapas, tamatar) resolve correctly in QueryGuard."""
        for query, expected_crop in [
            ("Dhan me blast lag gaya hai kya kare?", "Dhan"),
            ("Sarson me mahu lag gaya hai kya spray kare?", "Sarson"),
            ("Kapas me kit niyantran", "Kapas"),
            ("Tamatar me jhulsa rog ki dawai", "Tamatar"),
        ]:
            decision = self.guard.classify(query)
            self.assertEqual(decision.action, GuardAction.PROCEED)
            self.assertEqual(decision.detected_crop, expected_crop)

    # ── C. Provider Failures & Fallback Transparency ──────────────────────────

    def test_provider_timeout_falls_back_transparently_with_notice_banner(self):
        """When LLM provider times out, AnswerGenerator falls back and exposes metadata & notice banner."""
        failing_provider = FailingProvider(error_msg="Ollama request timed out after 30.0s")
        gen = AnswerGenerator(provider=failing_provider)

        c1 = make_test_chunk("chunk_satawar", "Satawar", 0.65, "Quinalphos 25% EC 500 ml/acre spray kare.")
        fake_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.65,
            best_chunk=c1,
            sufficient_chunks=[c1],
            compatible_chunks=[c1],
        )

        answer = gen.generate("Satawar pest", fake_gate)
        # Must be transparent fallback
        self.assertTrue(answer.is_fallback)
        self.assertEqual(answer.provider_used, "deterministic_template_fallback")
        self.assertIn("Ollama request timed out", answer.fallback_reason)
        # Display text must carry prominent notice banner
        display_text = answer.to_display_text()
        self.assertIn("[FALLBACK NOTICE]", display_text)
        self.assertIn("deterministic fallback template", display_text)

    def test_provider_error_redacts_api_keys_and_bearer_tokens(self):
        """Error messages containing secret keys or Bearer tokens must be sanitized before exposure."""
        sensitive_error = "Connection failed: Authorization: Bearer secret_session_token_12345 and key sk-proj-1234567890abcdef1234567890"
        failing_provider = FailingProvider(error_msg=sensitive_error)
        gen = AnswerGenerator(provider=failing_provider)

        c1 = make_test_chunk("chunk_test", "Tomato", 0.60, "Copper Oxychloride spray.")
        fake_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.60,
            best_chunk=c1,
            sufficient_chunks=[c1],
            compatible_chunks=[c1],
        )

        answer = gen.generate("Tomato spray", fake_gate)
        self.assertTrue(answer.is_fallback)
        # Secrets must be sanitized
        self.assertNotIn("sk-proj-1234567890abcdef1234567890", answer.fallback_reason)
        self.assertIn("[REDACTED_API_KEY]", answer.fallback_reason)
        self.assertIn("[REDACTED_TOKEN]", answer.fallback_reason)

    def test_empty_provider_output_triggers_fallback_or_safe_abstention(self):
        """If an LLM returns whitespace or empty text, system falls back to structured template."""
        empty_provider = EmptyProvider()
        gen = AnswerGenerator(provider=empty_provider)

        c1 = make_test_chunk("chunk_pomegranate", "Pomegranate", 0.60, "Messaile 15ml/15 Lit spray.")
        fake_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.60,
            best_chunk=c1,
            sufficient_chunks=[c1],
            compatible_chunks=[c1],
        )

        answer = gen.generate("Pomegranate fruit borer", fake_gate)
        self.assertTrue(answer.is_fallback)
        self.assertIn("Messaile", answer.answer_text)
        self.assertIn("[FALLBACK NOTICE]", answer.to_display_text())

    def test_empty_evidence_chunks_abstains_safely_without_generating(self):
        """If no crop-compatible evidence chunks are available, generator abstains safely."""
        gen = AnswerGenerator()
        empty_gate = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.0,
            best_chunk=None,
            sufficient_chunks=[],
            compatible_chunks=[],
            all_chunks=[],
        )
        answer = gen.generate("Any query", empty_gate)
        self.assertEqual(answer.status, GenerationStatus.ABSTAINED_NO_EVIDENCE)
        self.assertEqual(answer.answer_text, "")

    # ── D. Pipeline Integrity ─────────────────────────────────────────────────

    def test_pipeline_evidence_consistency_across_stages(self):
        """Gate approved compatible_chunks are exactly the chunks used in generation and verification."""
        res = self.pipeline.answer("Fertilizer recommendation for pomegranate trees")
        gate_res = res.gate_result
        self.assertIsNotNone(gate_res.compatible_chunks)
        # Compatible chunks must contain only Pomegranate
        for c in gate_res.compatible_chunks:
            self.assertIn("Pomegranate", c.crop_clean)

        # Chunks cited in final answer must match compatible_chunks
        compat_ids = {c.chunk_id for c in gate_res.compatible_chunks}
        for sid in res.final_answer.source_ids:
            self.assertIn(sid, compat_ids)


if __name__ == "__main__":
    unittest.main()
