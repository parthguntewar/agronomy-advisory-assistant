"""
Regression test suite for Phase 6A: Safety Guard and Verifier Hardening.

Covers:
  1. Weak-evidence answers are verified and NOT skipped (removal of SKIPPED_WEAK).
  2. Weak-evidence answers with hallucinated chemicals are strictly blocked.
  3. Unindexed crops (Quinoa, Avocado, Dragon Fruit) do not leak unrelated crop advice.
  4. Chemical claim verification hardening for blind spots (monosultap, acephate, flubendiamide, paraquat).
  5. Explicit fallback metadata tracking (provider_used, is_fallback, fallback_reason).
  6. Abstentions and clarifications cleanly preserve non-advisory state.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import GuardedRAGPipeline, PipelineResult
from src.guard.query_guard import QueryGuard, GuardAction, GuardDecision
from src.guard.relevance_gate import RelevanceGate, GateVerdict, GateResult
from src.generation.answer_gen import AnswerGenerator, GeneratedAnswer, GenerationStatus
from src.generation.verifier import PostGenerationVerifier, VerificationStatus
from src.generation.providers import LLMProvider, MockProvider, LLMGenerationError
from src.retrieval.vector_store import RetrievalResult
from src.ui.helpers import extract_pipeline_diagnostics
from src.config import AGRONOMIC_SAFETY_NOTICE, SOURCE_ATTRIBUTION


def make_test_result(
    chunk_id: str = "kcc_chunk_test_001",
    crop: str = "Paddy (Dhan)",
    similarity_score: float = 0.65,
    farmer_query: str = "How to control blast in paddy?",
    expert_advisory: str = "Spray tricyclazole 18% + mancozeb 62% @ 2 gm/lit of water.",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        similarity_score=similarity_score,
        crop=crop,
        crop_clean=crop,
        state="Odisha",
        district="Cuttack",
        advisory_domain="plant_protection",
        query_type="disease_control",
        original_year=2021,
        farmer_query=farmer_query,
        expert_advisory=expert_advisory,
        qc_severity="none",
        qc_confidence="high",
        qc_flags=[],
        rag_use_mode="use_as_evidence",
        source_attribution=SOURCE_ATTRIBUTION,
        safety_notice=AGRONOMIC_SAFETY_NOTICE,
        retrieval_text=f"Crop: {crop}. Question: {farmer_query}\nAdvisory Evidence: {expert_advisory}",
    )


class TestSafetyHardeningRegression(unittest.TestCase):
    """Regression test cases verifying Phase 6A safety improvements."""

    def setUp(self):
        self.verifier = PostGenerationVerifier()
        self.generator = AnswerGenerator(provider=MockProvider())
        self.gate = RelevanceGate()
        self.guard = QueryGuard()

    # -----------------------------------------------------------------------
    # 1. Weak-Evidence Answers Must Be Verified (Not Skipped)
    # -----------------------------------------------------------------------
    def test_weak_evidence_answer_is_verified_and_preserves_caveat(self):
        """Weak-evidence answers must undergo structural verification and not return SKIPPED_WEAK."""
        chunk = make_test_result(
            chunk_id="kcc_chunk_weak_paddy",
            crop="Paddy (Dhan)",
            similarity_score=0.38,  # weak evidence (0.30 - 0.50)
            expert_advisory="Apply 50 kg urea per hectare at tillering stage.",
        )
        gate_res = GateResult(
            verdict=GateVerdict.EVIDENCE_WEAK,
            best_score=0.38,
            best_chunk=chunk,
            sufficient_chunks=[],
            all_chunks=[chunk],
            detected_query_crop="Paddy",
            reason="Score is weak.",
        )
        answer = self.generator.generate("What fertilizer for paddy?", gate_res)
        self.assertEqual(answer.status, GenerationStatus.WEAK_EVIDENCE)
        self.assertIsNotNone(answer.uncertainty_note)

        # Verify weak evidence answer
        verif = self.verifier.verify(answer, [chunk])
        self.assertNotEqual(verif.status, VerificationStatus.SKIPPED_WEAK, "Must NOT bypass verification!")
        self.assertTrue(verif.is_safe_to_present)
        self.assertEqual(verif.status, VerificationStatus.PASSED_WITH_WARNINGS)
        self.assertTrue(any("WEAK_EVIDENCE_CAVEAT" in w for w in verif.warnings))
        self.assertEqual(verif.safe_answer.uncertainty_note, answer.uncertainty_note)

    def test_weak_evidence_answer_with_hallucinated_chemical_is_blocked(self):
        """If a weak-evidence answer contains an ungrounded chemical dosage, verifier must block it."""
        chunk = make_test_result(
            chunk_id="kcc_chunk_weak_002",
            crop="Paddy (Dhan)",
            similarity_score=0.36,
            expert_advisory="Apply 25 kg nitrogen per acre.",
        )
        # Answer contains ungrounded chemical 'paraquat' not present in chunk
        bad_answer = GeneratedAnswer(
            status=GenerationStatus.WEAK_EVIDENCE,
            answer_text="Based on KCC records, apply paraquat 5 ml/litre for weed control [kcc_chunk_weak_002].",
            source_ids=["kcc_chunk_weak_002"],
            evidence_scores=[0.36],
            crops_cited=["Paddy (Dhan)"],
            uncertainty_note="Weak evidence note.",
        )
        verif = self.verifier.verify(bad_answer, [chunk])
        self.assertEqual(verif.status, VerificationStatus.BLOCKED_HALLUCINATION)
        self.assertTrue(verif.is_blocked)
        self.assertFalse(verif.is_safe_to_present)
        self.assertEqual(verif.safe_answer.answer_text, "")
        self.assertTrue(any("paraquat" in issue for issue in verif.issues))

    # -----------------------------------------------------------------------
    # 2. Close Unknown-Crop Gap (Quinoa, Avocado, Dragon Fruit)
    # -----------------------------------------------------------------------
    def test_unindexed_crops_trigger_crop_mismatch_and_refuse_cross_crop_advice(self):
        """Unindexed crops must not generate advice from arbitrary nearest-neighbor crops."""
        pipeline = GuardedRAGPipeline()

        unindexed_queries = [
            ("Quinoa downy mildew management practices", "Quinoa"),
            ("How to fertilize Hass avocado trees in sandy soil?", "Avocado"),
            ("What is the recommended grafting method for Brazilian Dragon Fruit in winter?", "Dragon Fruit"),
        ]

        for query, expected_crop_token in unindexed_queries:
            result = pipeline.answer(query)
            # The system must NOT generate advice from unrelated crops (e.g. Groundnut or Turmeric or Ber)
            self.assertFalse(
                result.was_answered,
                f"Query '{query}' must NOT be answered with cross-crop advice!"
            )
            # Gate verdict must declare CROP_MISMATCH
            self.assertIsNotNone(result.gate_result)
            self.assertEqual(
                result.gate_result.verdict,
                GateVerdict.CROP_MISMATCH,
                f"Expected CROP_MISMATCH for query '{query}'"
            )
            self.assertEqual(
                result.final_answer.status,
                GenerationStatus.ABSTAINED_CROP_MISMATCH,
            )

    # -----------------------------------------------------------------------
    # 3. Chemical-Claim Blind Spots Hardening
    # -----------------------------------------------------------------------
    def test_chemical_blind_spots_monosultap_acephate_flubendiamide_paraquat_blocked_when_ungrounded(self):
        """Known chemical blind spots must be detected and blocked when ungrounded."""
        chunk = make_test_result(
            chunk_id="kcc_chunk_safe_001",
            crop="Tomato",
            expert_advisory="Recommended to spray copper oxychloride 3 gm per litre of water.",
        )

        blind_spot_chemicals = [
            "monosultap",
            "acephate",
            "flubendiamide",
            "paraquat",
        ]

        for chem in blind_spot_chemicals:
            fabricated_answer = GeneratedAnswer(
                status=GenerationStatus.SUCCESS,
                answer_text=f"For pest control, spray {chem} @ 2 ml/litre [kcc_chunk_safe_001].",
                source_ids=["kcc_chunk_safe_001"],
                evidence_scores=[0.65],
                crops_cited=["Tomato"],
            )
            verif = self.verifier.verify(fabricated_answer, [chunk])
            self.assertEqual(
                verif.status,
                VerificationStatus.BLOCKED_HALLUCINATION,
                f"Verifier failed to block ungrounded chemical: {chem}",
            )
            self.assertTrue(any(chem in issue for issue in verif.issues))

    def test_chemical_blind_spots_pass_when_grounded_in_evidence(self):
        """When blind spot chemicals actually appear in the evidence chunk, they must pass verification."""
        chunk = make_test_result(
            chunk_id="kcc_chunk_chem_grounded",
            crop="Paddy",
            expert_advisory="For stem borer in paddy, spray monosultap 50% SP @ 2 gm/litre of water.",
        )
        grounded_answer = GeneratedAnswer(
            status=GenerationStatus.SUCCESS,
            answer_text="For stem borer in paddy, spray monosultap 50% SP @ 2 gm/litre of water [kcc_chunk_chem_grounded].",
            source_ids=["kcc_chunk_chem_grounded"],
            evidence_scores=[0.65],
            crops_cited=["Paddy"],
        )
        verif = self.verifier.verify(grounded_answer, [chunk])
        self.assertEqual(verif.status, VerificationStatus.PASSED)
        self.assertTrue(verif.is_safe_to_present)

    # -----------------------------------------------------------------------
    # 4. Explicit Fallback Metadata Tracking
    # -----------------------------------------------------------------------
    def test_explicit_fallback_metadata_on_provider_failure(self):
        """When an LLM provider fails, answer must explicitly expose fallback metadata."""
        failing_provider = MagicMock(spec=LLMProvider)
        failing_provider.get_provider_name.return_value = "ollama (llama3.2)"
        failing_provider.generate.side_effect = LLMGenerationError("Ollama daemon timeout")

        generator = AnswerGenerator(provider=failing_provider)
        chunk = make_test_result()
        gate_res = GateResult(
            verdict=GateVerdict.EVIDENCE_SUFFICIENT,
            best_score=0.65,
            best_chunk=chunk,
            sufficient_chunks=[chunk],
        )

        answer = generator.generate("How to control blast in paddy?", gate_res)
        self.assertTrue(answer.is_fallback, "is_fallback must be True on provider failure")
        self.assertEqual(answer.provider_used, "deterministic_template_fallback")
        self.assertIn("Ollama daemon timeout", answer.fallback_reason)
        # to_display_text must state fallback notice explicitly
        display_text = answer.to_display_text()
        self.assertIn("[FALLBACK NOTICE]", display_text)
        self.assertIn("deterministic fallback template", display_text)

        # UI diagnostics must surface fallback metadata
        dummy_pipe_result = PipelineResult(
            query="test",
            guard_decision=GuardDecision(action=GuardAction.PROCEED),
            generated_answer=answer,
        )
        diag = extract_pipeline_diagnostics(dummy_pipe_result)
        self.assertTrue(diag["generation"]["is_fallback"])
        self.assertEqual(diag["generation"]["provider_used"], "deterministic_template_fallback")
        self.assertIsNotNone(diag["generation"]["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
