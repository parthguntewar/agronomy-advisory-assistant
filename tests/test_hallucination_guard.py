"""
Phase 3 test suite: Hallucination Guard and Safe Abstention Pipeline.

10 automated test scenarios:
  1.  Supported agricultural question → PROCEED → sufficient evidence → answer cited.
  2.  Non-agricultural question → ABSTAIN_OFFTOPIC.
  3.  Agricultural query with no matching evidence → EVIDENCE_ABSENT → abstain.
  4.  Weather forecast request → ABSTAIN_DYNAMIC.
  5.  Current government scheme question → ABSTAIN_DYNAMIC.
  6.  Disease question without crop name → CLARIFY_CROP.
  7.  Crop mismatch between query and retrieved evidence → CROP_MISMATCH → blocked.
  8.  Unsupported chemical dosage in mock generated answer → BLOCKED_HALLUCINATION.
  9.  Generated answer that correctly cites evidence → PASSED verification.
  10. Historical advice must not be treated as verified current guidance → uncertainty note.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guard.query_guard import QueryGuard, GuardAction, GuardDecision
from src.guard.relevance_gate import RelevanceGate, GateVerdict
from src.generation.answer_gen import AnswerGenerator, GenerationStatus
from src.generation.verifier import PostGenerationVerifier, VerificationStatus
from src.retrieval.vector_store import RetrievalResult
from src.pipeline import GuardedRAGPipeline, PipelineResult
from src.config import AGRONOMIC_SAFETY_NOTICE, SOURCE_ATTRIBUTION


# ---------------------------------------------------------------------------
# Helper: Build a fake RetrievalResult for testing
# ---------------------------------------------------------------------------
def make_fake_result(
    chunk_id: str = "kcc_chunk_test_001",
    crop: str = "Paddy (Dhan)",
    crop_clean: str = "Paddy (Dhan)",
    similarity_score: float = 0.65,
    expert_advisory: str = "Apply chlorpyrifos 2 ml/litre for stem borer control.",
    farmer_query: str = "How to control stem borer in paddy?",
    state: str = "Odisha",
    district: str = "Puri",
    advisory_domain: str = "plant_protection",
    query_type: str = "pest_control",
    original_year: int = 2021,
    qc_severity: str = "none",
    qc_confidence: str = "high",
    qc_flags: List[str] = None,
    rag_use_mode: str = "use_as_evidence",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        similarity_score=similarity_score,
        crop=crop,
        crop_clean=crop_clean,
        state=state,
        district=district,
        advisory_domain=advisory_domain,
        query_type=query_type,
        original_year=original_year,
        farmer_query=farmer_query,
        expert_advisory=expert_advisory,
        qc_severity=qc_severity,
        qc_confidence=qc_confidence,
        qc_flags=qc_flags or [],
        rag_use_mode=rag_use_mode,
        source_attribution=SOURCE_ATTRIBUTION,
        safety_notice=AGRONOMIC_SAFETY_NOTICE,
        retrieval_text=(
            f"Crop: {crop_clean}. Topic: {query_type}. "
            f"Location: {state}, {district}. "
            f"Question: {farmer_query}\nAdvisory Evidence: {expert_advisory}"
        ),
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------
class TestHallucinationGuardPipeline(unittest.TestCase):
    """
    10-scenario automated test suite for the Phase 3 hallucination guard pipeline.
    Tests are designed to be run WITHOUT a live ChromaDB index (using mocks)
    and also to verify correct behavior when the real index is available.
    """

    @classmethod
    def setUpClass(cls):
        cls.guard = QueryGuard()
        cls.gate = RelevanceGate()
        cls.generator = AnswerGenerator()
        cls.verifier = PostGenerationVerifier()

    # ──────────────────────────────────────────────────────────────────────
    # Test 1: Supported agricultural question with sufficient evidence
    # ──────────────────────────────────────────────────────────────────────
    def test_01_supported_agricultural_question(self):
        """
        SCENARIO 1: A well-formed agricultural question with crop context.
        Guard should PROCEED; if evidence is found (score ≥ 0.50), generate answer.
        """
        query = "How to control stem borer in paddy crop?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 01: Supported Agricultural Question ---")
        print(f"Guard action: {decision.action.value} | Crop: {decision.detected_crop}")

        self.assertEqual(decision.action, GuardAction.PROCEED)
        self.assertTrue(decision.is_agricultural)

        # Simulate retrieval with a high-confidence match
        fake_result = make_fake_result(similarity_score=0.62)
        gate_result = self.gate.evaluate(query, [fake_result], query_crop="Paddy")
        print(f"Gate verdict: {gate_result.verdict.value} | Score: {gate_result.best_score:.4f}")

        # Score 0.62 > 0.50 (sufficient threshold) → EVIDENCE_SUFFICIENT
        self.assertEqual(gate_result.verdict, GateVerdict.EVIDENCE_SUFFICIENT)

        # Generate answer
        answer = self.generator.generate(query, gate_result)
        print(f"Answer status: {answer.status.value}")
        self.assertEqual(answer.status, GenerationStatus.SUCCESS)
        self.assertTrue(len(answer.answer_text) > 0)
        self.assertIn("kcc_chunk_test_001", answer.source_ids)

    # ──────────────────────────────────────────────────────────────────────
    # Test 2: Non-agricultural query → ABSTAIN_OFFTOPIC
    # ──────────────────────────────────────────────────────────────────────
    def test_02_non_agricultural_query_abstain(self):
        """
        SCENARIO 2: A clearly non-agricultural question.
        Guard must classify as ABSTAIN_OFFTOPIC without any retrieval.
        """
        query = "How do I configure a React frontend router in Node.js?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 02: Non-Agricultural Query ---")
        print(f"Guard action: {decision.action.value} | Agricultural: {decision.is_agricultural}")

        self.assertEqual(decision.action, GuardAction.ABSTAIN_OFFTOPIC)
        self.assertFalse(decision.is_agricultural)

    # ──────────────────────────────────────────────────────────────────────
    # Test 3: Agricultural query with no matching evidence → abstain
    # ──────────────────────────────────────────────────────────────────────
    def test_03_no_matching_evidence_abstain(self):
        """
        SCENARIO 3: Agricultural query where vector search returns only very
        low similarity results (unindexed crop).
        Gate must return EVIDENCE_ABSENT and generation must abstain.
        """
        query = "What is the recommended grafting method for Brazilian Dragon Fruit?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 03: Agricultural Query, No KB Evidence ---")
        print(f"Guard action: {decision.action.value}")

        self.assertEqual(decision.action, GuardAction.PROCEED)

        # Simulate retrieval with very low similarity (forced nearest neighbor)
        fake_result = make_fake_result(similarity_score=0.20)  # below min viable 0.30
        gate_result = self.gate.evaluate(query, [fake_result], query_crop=None)
        print(f"Gate verdict: {gate_result.verdict.value} | Score: {gate_result.best_score:.4f}")

        self.assertEqual(gate_result.verdict, GateVerdict.EVIDENCE_ABSENT)

        answer = self.generator.generate(query, gate_result)
        print(f"Answer status: {answer.status.value}")
        self.assertEqual(answer.status, GenerationStatus.ABSTAINED_NO_EVIDENCE)
        self.assertEqual(answer.answer_text, "")

    # ──────────────────────────────────────────────────────────────────────
    # Test 4: Weather forecast request → ABSTAIN_DYNAMIC
    # ──────────────────────────────────────────────────────────────────────
    def test_04_weather_forecast_abstain_dynamic(self):
        """
        SCENARIO 4: Farmer asks about tomorrow's weather.
        Guard must classify as ABSTAIN_DYNAMIC — no retrieval should occur.
        """
        query = "Will it rain tomorrow in Mehsana district, Gujarat?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 04: Weather Forecast Request ---")
        print(f"Guard action: {decision.action.value} | Dynamic: {decision.has_dynamic_request}")

        self.assertEqual(decision.action, GuardAction.ABSTAIN_DYNAMIC)
        self.assertTrue(decision.has_dynamic_request)

    # ──────────────────────────────────────────────────────────────────────
    # Test 5: Government scheme deadline → ABSTAIN_DYNAMIC
    # ──────────────────────────────────────────────────────────────────────
    def test_05_government_scheme_abstain_dynamic(self):
        """
        SCENARIO 5: Farmer asks about PM Kisan scheme installment date.
        Guard must classify as ABSTAIN_DYNAMIC (scheme deadlines are live data).
        """
        query = "When is the 10th PM Kisan installment date and how do I apply for the scheme?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 05: Government Scheme Deadline ---")
        print(f"Guard action: {decision.action.value} | Dynamic: {decision.has_dynamic_request}")

        self.assertEqual(decision.action, GuardAction.ABSTAIN_DYNAMIC)
        self.assertTrue(decision.has_dynamic_request)

    # ──────────────────────────────────────────────────────────────────────
    # Test 6: Disease question without crop name → CLARIFY_CROP
    # ──────────────────────────────────────────────────────────────────────
    def test_06_disease_without_crop_clarify(self):
        """
        SCENARIO 6: Farmer asks about disease symptoms without naming the crop.
        Guard must request clarification — no retrieval should occur.
        """
        query = "My leaves have yellow spots and are curling. What spray should I use?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 06: Disease Query Without Crop Name ---")
        print(f"Guard action: {decision.action.value} | Detected crop: {decision.detected_crop}")
        print(f"Clarification: {decision.clarification_prompt[:80] if decision.clarification_prompt else 'None'}...")

        self.assertEqual(decision.action, GuardAction.CLARIFY_CROP)
        self.assertTrue(decision.has_disease_pest_context)
        self.assertIsNone(decision.detected_crop)
        self.assertIsNotNone(decision.clarification_prompt)
        self.assertIn("crop", decision.clarification_prompt.lower())

    # ──────────────────────────────────────────────────────────────────────
    # Test 7: Crop mismatch between query and retrieved evidence
    # ──────────────────────────────────────────────────────────────────────
    def test_07_crop_mismatch_blocked(self):
        """
        SCENARIO 7: Query is about wheat, but vector search returns a paddy record.
        Gate must detect the crop mismatch and return CROP_MISMATCH.
        """
        query = "How to control aphids in wheat field?"
        decision = self.guard.classify(query)
        print(f"\n--- Test 07: Crop Mismatch ---")
        print(f"Guard detected crop: {decision.detected_crop}")

        self.assertEqual(decision.action, GuardAction.PROCEED)

        # Retrieved chunk is for Paddy — different from query crop Wheat
        paddy_result = make_fake_result(
            crop="Paddy (Dhan)", crop_clean="Paddy (Dhan)", similarity_score=0.45
        )
        gate_result = self.gate.evaluate(query, [paddy_result], query_crop="Wheat")
        print(f"Gate verdict: {gate_result.verdict.value}")
        print(f"Mismatch detail: {gate_result.crop_mismatch_detail}")

        self.assertEqual(gate_result.verdict, GateVerdict.CROP_MISMATCH)
        self.assertIsNotNone(gate_result.crop_mismatch_detail)

        answer = self.generator.generate(query, gate_result)
        print(f"Answer status: {answer.status.value}")
        self.assertEqual(answer.status, GenerationStatus.ABSTAINED_CROP_MISMATCH)
        self.assertEqual(answer.answer_text, "")

    # ──────────────────────────────────────────────────────────────────────
    # Test 8: Unsupported chemical dosage blocked by verifier
    # ──────────────────────────────────────────────────────────────────────
    def test_08_unsupported_chemical_dosage_blocked(self):
        """
        SCENARIO 8: A mock generated answer contains a chemical dosage (e.g., 5 ml/litre)
        that is NOT present in the supporting evidence. Verifier must block it.
        """
        print(f"\n--- Test 08: Unsupported Chemical Dosage in Answer ---")
        from src.generation.answer_gen import GeneratedAnswer, GenerationStatus

        # Evidence: mentions only "Apply urea" — no specific ml dosage
        evidence = make_fake_result(
            expert_advisory="Apply urea at recommended rate for better yield.",
            farmer_query="How much urea to apply for wheat?",
        )

        # Mock answer: contains a specific dosage NOT in evidence
        fabricated_answer = GeneratedAnswer(
            status=GenerationStatus.SUCCESS,
            answer_text=(
                "Based on KCC records, apply chlorpyrifos 5 ml/litre of water "
                "for best results in your wheat crop."
            ),
            source_ids=["kcc_chunk_test_001"],
            evidence_scores=[0.65],
            crops_cited=["Paddy (Dhan)"],
            safety_notice=AGRONOMIC_SAFETY_NOTICE,
        )

        result = self.verifier.verify(fabricated_answer, [evidence])
        print(f"Verification status: {result.status.value}")
        print(f"Issues: {result.issues}")

        # The dosage "5 ml/litre" is not in evidence text → HALLUCINATION_RISK → BLOCKED
        self.assertEqual(result.status, VerificationStatus.BLOCKED_HALLUCINATION)
        self.assertTrue(result.is_blocked)
        self.assertTrue(len(result.issues) > 0)

    # ──────────────────────────────────────────────────────────────────────
    # Test 9: Correctly cited answer passes verification
    # ──────────────────────────────────────────────────────────────────────
    def test_09_cited_answer_passes_verification(self):
        """
        SCENARIO 9: A generated answer that uses only text directly from evidence
        (no fabricated chemicals or dosages) must PASS the verifier.
        """
        print(f"\n--- Test 09: Correctly Cited Answer Passes Verification ---")
        from src.generation.answer_gen import GeneratedAnswer, GenerationStatus

        # Evidence with specific advisory content
        evidence = make_fake_result(
            expert_advisory=(
                "For stem borer in paddy, apply chlorpyrifos 2 ml/litre of water. "
                "Repeat after 15 days if infestation persists."
            ),
        )

        # The gate_result with this evidence
        gate_result = self.gate.evaluate(
            "How to control stem borer in paddy?",
            [evidence],
            query_crop="Paddy"
        )

        # Generate answer using the generator (builds answer from evidence)
        answer = self.generator.generate("How to control stem borer in paddy?", gate_result)
        print(f"Generated status: {answer.status.value}")

        # Verify it
        result = self.verifier.verify(answer, [evidence])
        print(f"Verification status: {result.status.value}")

        # Should pass (template-based generation only uses evidence text)
        self.assertIn(result.status, (VerificationStatus.PASSED, VerificationStatus.PASSED_WITH_WARNINGS))
        self.assertFalse(result.is_blocked)
        self.assertIsNotNone(result.safe_answer)

    # ──────────────────────────────────────────────────────────────────────
    # Test 10: Historical advice carries uncertainty note
    # ──────────────────────────────────────────────────────────────────────
    def test_10_historical_advice_uncertainty_note(self):
        """
        SCENARIO 10: Historical KCC records (e.g., from 2016) must NOT be presented
        as verified current guidance. The answer must carry an explicit uncertainty note
        when evidence is weak (below sufficient threshold).
        """
        query = "What fertilizer schedule should I use for paddy during kharif season?"
        print(f"\n--- Test 10: Historical Advice Uncertainty ---")

        # Simulate a 2016 record with a moderate (weak) similarity score
        old_result = make_fake_result(
            similarity_score=0.38,  # above min viable (0.30) but below sufficient (0.50)
            expert_advisory="Apply 100 kg DAP per hectare at the time of sowing. [2016 KCC record]",
            original_year=2016,
        )

        gate_result = self.gate.evaluate(query, [old_result], query_crop=None)
        print(f"Gate verdict: {gate_result.verdict.value} | Score: {gate_result.best_score:.4f}")

        # Score 0.38 → EVIDENCE_WEAK
        self.assertEqual(gate_result.verdict, GateVerdict.EVIDENCE_WEAK)

        answer = self.generator.generate(query, gate_result)
        print(f"Answer status: {answer.status.value}")
        print(f"Uncertainty note present: {answer.uncertainty_note is not None}")

        # Must be WEAK_EVIDENCE with an explicit uncertainty note
        self.assertEqual(answer.status, GenerationStatus.WEAK_EVIDENCE)
        self.assertIsNotNone(answer.uncertainty_note)
        # The uncertainty note must mention consulting an extension officer
        self.assertTrue(
            "KVK" in answer.uncertainty_note or "extension" in answer.uncertainty_note.lower(),
            "Uncertainty note must recommend consulting an agricultural extension officer."
        )


# ---------------------------------------------------------------------------
# Additional integration test: full pipeline with mocked retriever
# ---------------------------------------------------------------------------
class TestGuardedPipelineIntegration(unittest.TestCase):
    """Integration tests using the full GuardedRAGPipeline with a mocked retriever."""

    def _make_pipeline_with_mock(self, mock_results: list) -> GuardedRAGPipeline:
        """Creates a pipeline with a mocked AgronomyRetriever."""
        pipeline = GuardedRAGPipeline.__new__(GuardedRAGPipeline)
        pipeline.query_guard = QueryGuard()
        pipeline.gate = RelevanceGate()
        pipeline.relevance_gate = RelevanceGate()
        pipeline.answer_gen = AnswerGenerator()
        pipeline.verifier = PostGenerationVerifier()
        pipeline.top_k = 3

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_results
        pipeline.retriever = mock_retriever
        return pipeline

    def test_pipeline_offtopic_returns_abstention(self):
        """Full pipeline: off-topic query must abstain without calling retriever."""
        pipeline = self._make_pipeline_with_mock([])
        result = pipeline.answer("How do I configure a React frontend router in Node.js?")
        print(f"\n[Integration] Off-topic: {result.guard_decision.action.value}")
        self.assertEqual(result.guard_decision.action, GuardAction.ABSTAIN_OFFTOPIC)
        self.assertFalse(result.was_answered)
        # Retriever must NOT be called
        pipeline.retriever.retrieve.assert_not_called()

    def test_pipeline_dynamic_returns_abstention(self):
        """Full pipeline: weather query must abstain without calling retriever."""
        pipeline = self._make_pipeline_with_mock([])
        result = pipeline.answer("Will it rain tomorrow in Pune?")
        print(f"[Integration] Dynamic: {result.guard_decision.action.value}")
        self.assertEqual(result.guard_decision.action, GuardAction.ABSTAIN_DYNAMIC)
        pipeline.retriever.retrieve.assert_not_called()

    def test_pipeline_clarify_crop_no_retrieval(self):
        """Full pipeline: ambiguous disease query must trigger clarification, no retrieval."""
        pipeline = self._make_pipeline_with_mock([])
        result = pipeline.answer("My leaves have yellow spots and are curling. What spray should I use?")
        print(f"[Integration] Clarify: {result.guard_decision.action.value}")
        self.assertEqual(result.guard_decision.action, GuardAction.CLARIFY_CROP)
        pipeline.retriever.retrieve.assert_not_called()

    def test_pipeline_sufficient_evidence_answered(self):
        """Full pipeline: in-domain query with sufficient evidence produces an answer."""
        high_conf_result = make_fake_result(similarity_score=0.65)
        pipeline = self._make_pipeline_with_mock([high_conf_result])
        result = pipeline.answer("How to control stem borer in paddy crop?")
        print(
            f"[Integration] Full pipeline: "
            f"Guard={result.guard_decision.action.value} | "
            f"Gate={result.gate_result.verdict.value if result.gate_result else 'N/A'} | "
            f"Answered={result.was_answered}"
        )
        self.assertEqual(result.guard_decision.action, GuardAction.PROCEED)
        self.assertTrue(result.was_answered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
