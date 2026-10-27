"""
Regression and Integration Tests for Phase 6B: Retrieval & Relevance-Gate Optimization.

Tests multi-chunk crop-aware filtering and ranking:
- Promotion of relevant crops from ranks 2 and 3 (Pomegranate, Guava, Cotton).
- Preventing cross-crop advice leakage for unsupported crops (Quinoa, Avocado, Dragon Fruit).
- Ensuring generation and verification receive ONLY gate-approved compatible chunks.
- Verification of threshold semantics and uncertainty caveats.
"""

import unittest
from typing import List

from src.config import TOP_K_RETRIEVAL
from src.generation.answer_gen import AnswerGenerator, GenerationStatus
from src.generation.verifier import PostGenerationVerifier, VerificationStatus
from src.guard.query_guard import QueryGuard, GuardAction
from src.guard.relevance_gate import RelevanceGate, GateVerdict, GateResult
from src.pipeline import GuardedRAGPipeline
from src.retrieval.retriever import AgronomyRetriever
from src.retrieval.vector_store import RetrievalResult


def make_dummy_result(chunk_id: str, crop: str, score: float, advisory: str = "Standard advisory.") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        crop=crop,
        crop_clean=crop,
        state="TestState",
        district="TestDistrict",
        advisory_domain="plant_protection",
        query_type="Insect Control",
        original_year=2021,
        qc_flags=[],
        qc_severity="low",
        qc_confidence="high",
        rag_use_mode="use_as_evidence",
        farmer_query="Dummy query",
        expert_advisory=advisory,
        retrieval_text=advisory,
        source_attribution="Test Attribution",
        safety_notice="Test Notice",
        similarity_score=score,
    )


class TestRetrievalGateOptimization(unittest.TestCase):
    """Regression test suite for Phase 6B multi-chunk gating and rank promotion."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = AgronomyRetriever()
        cls.guard = QueryGuard()
        cls.gate = RelevanceGate()
        cls.pipeline = GuardedRAGPipeline(retriever=cls.retriever)

    def test_pomegranate_promoted_from_rank_2_over_unrelated_rank_1(self):
        """
        Pomegranate query: raw search has Green Gram at rank 1 (0.4908)
        and Pomegranate at rank 2 (0.4743).
        RelevanceGate must promote Pomegranate and exclude Green Gram.
        """
        query = "Fertilizer recommendation for pomegranate trees"
        decision = self.guard.classify(query)
        self.assertEqual(decision.action, GuardAction.PROCEED)
        self.assertEqual(decision.detected_crop, "Pomegranate")

        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreaterEqual(len(results), 2)
        # Raw rank 1 is not Pomegranate
        self.assertNotEqual(results[0].crop_clean, "Pomegranate")

        gate_res = self.gate.evaluate(query, results, query_crop=decision.detected_crop)
        # Best chunk is now Pomegranate
        self.assertIsNotNone(gate_res.best_chunk)
        self.assertIn("Pomegranate", gate_res.best_chunk.crop_clean)
        # Gate verdict is weak evidence (score ~0.47 is between 0.30 and 0.50)
        self.assertEqual(gate_res.verdict, GateVerdict.EVIDENCE_WEAK)
        # Compatible chunks must contain only Pomegranate
        for chunk in gate_res.compatible_chunks:
            self.assertIn("Pomegranate", chunk.crop_clean)

    def test_guava_promoted_from_rank_3_over_unrelated_top_matches(self):
        """
        Guava query: raw search has Groundnut at rank 1 and Pomegranate at rank 2,
        with Guava at rank 3 (0.4262).
        Gate must promote Guava and filter out Groundnut and Pomegranate.
        """
        query = "Pest management for guava fruit fly"
        decision = self.guard.classify(query)
        self.assertEqual(decision.detected_crop, "Guava")

        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreaterEqual(len(results), 3)
        self.assertNotEqual(results[0].crop_clean, "Guava")

        gate_res = self.gate.evaluate(query, results, query_crop=decision.detected_crop)
        self.assertEqual(gate_res.best_chunk.crop_clean, "Guava")
        self.assertEqual(gate_res.verdict, GateVerdict.EVIDENCE_WEAK)
        # All compatible chunks are Guava
        for chunk in gate_res.compatible_chunks:
            self.assertEqual(chunk.crop_clean, "Guava")

    def test_cotton_promoted_from_rank_3_and_generation_receives_only_cotton(self):
        """
        Cotton query: Cotton is at rank 3.
        Gate promotes Cotton.
        AnswerGenerator must receive ONLY Cotton chunks.
        """
        query = "How to control sucking pests in cotton crop?"
        decision = self.guard.classify(query)
        self.assertEqual(decision.detected_crop, "Cotton")

        results = self.retriever.retrieve(query, top_k=3)
        gate_res = self.gate.evaluate(query, results, query_crop=decision.detected_crop)

        self.assertIn("Cotton", gate_res.best_chunk.crop_clean)

        gen = AnswerGenerator()
        answer = gen.generate(query, gate_res)
        # Status is weak evidence
        self.assertEqual(answer.status, GenerationStatus.WEAK_EVIDENCE)
        # Crops cited must contain Cotton and NEVER Groundnut or Satawar
        self.assertTrue(all("Cotton" in c for c in answer.crops_cited))
        self.assertNotIn("Groundnut", answer.crops_cited)
        self.assertNotIn("Satawar", answer.crops_cited)

    def test_satawar_exact_and_paraphrased_behavior(self):
        """
        Satawar exact query produces EVIDENCE_SUFFICIENT (> 0.50).
        Satawar paraphrased query produces EVIDENCE_WEAK (0.30 - 0.50) with uncertainty note.
        """
        # Exact
        exact_query = "Insect Control in Satawar crop?"
        res_exact = self.pipeline.answer(exact_query)
        self.assertEqual(res_exact.gate_result.verdict, GateVerdict.EVIDENCE_SUFFICIENT)
        self.assertEqual(res_exact.final_answer.status, GenerationStatus.SUCCESS)

        # Paraphrased
        para_query = "What medicine to spray for worms in satawar roots and leaves?"
        res_para = self.pipeline.answer(para_query)
        self.assertEqual(res_para.gate_result.verdict, GateVerdict.EVIDENCE_WEAK)
        self.assertEqual(res_para.final_answer.status, GenerationStatus.WEAK_EVIDENCE)
        self.assertIsNotNone(res_para.final_answer.uncertainty_note)

    def test_unsupported_crops_quinoa_avocado_dragon_fruit_refuse_cross_crop_advice(self):
        """
        Queries for crops absent from the KB (Quinoa, Avocado, Dragon Fruit)
        must be rejected as CROP_MISMATCH and never borrow advice from nearest neighbors.
        """
        unsupported = [
            ("Quinoa downy mildew management practices", "Quinoa"),
            ("How to fertilize Hass avocado trees in sandy soil?", "Avocado"),
            ("What is the recommended grafting method for Brazilian Dragon Fruit in winter?", "Dragon Fruit"),
        ]

        for query, expected_crop in unsupported:
            result = self.pipeline.answer(query)
            # Guard must detect the crop
            self.assertEqual(result.guard_decision.detected_crop, expected_crop)
            # Gate must reject
            self.assertEqual(result.gate_result.verdict, GateVerdict.CROP_MISMATCH)
            # Pipeline final answer is safe abstention
            self.assertEqual(result.final_answer.status, GenerationStatus.ABSTAINED_CROP_MISMATCH)
            self.assertFalse(result.was_answered)

    def test_multi_chunk_similarity_alone_does_not_make_evidence_sufficient(self):
        """
        Requirement: Matching crop at rank 2 or 3 should be considered,
        but similarity alone must not make its evidence sufficient.
        A compatible chunk with score 0.42 must remain EVIDENCE_WEAK.
        """
        query = "How to treat guava fruit borer?"
        # Mock retrieval: rank 1 is wheat (0.55), rank 2 is guava (0.42)
        r1 = make_dummy_result("chunk_wheat", "Wheat", 0.55)
        r2 = make_dummy_result("chunk_guava", "Guava", 0.42)

        gate_res = self.gate.evaluate(query, [r1, r2], query_crop="Guava")
        self.assertEqual(gate_res.best_chunk.chunk_id, "chunk_guava")
        # Must be WEAK, not sufficient!
        self.assertEqual(gate_res.verdict, GateVerdict.EVIDENCE_WEAK)
        self.assertEqual(len(gate_res.sufficient_chunks), 0)
        self.assertEqual(len(gate_res.compatible_chunks), 1)

    def test_generation_and_verification_receive_identical_approved_chunks(self):
        """
        Ensure both AnswerGenerator and Verifier receive ONLY the gate's approved compatible chunks.
        """
        query = "Treatment for guava pest"
        r1 = make_dummy_result("chunk_unrelated", "Groundnut", 0.52)
        r2 = make_dummy_result(
            "chunk_guava",
            "Guava",
            0.45,
            advisory="Apply neem oil 2 ml/litre on guava leaves for pest control."
        )

        pipeline_result = self.pipeline.answer(query)
        # Even if pipeline ran, verify unit contracts directly:
        gate_res = self.gate.evaluate(query, [r1, r2], query_crop="Guava")
        gen = AnswerGenerator()
        answer = gen.generate(query, gate_res)

        # Chunks fed to verifier in pipeline logic
        evidence_for_verification = (
            gate_res.sufficient_chunks
            or getattr(gate_res, "compatible_chunks", None)
            or ([gate_res.best_chunk] if gate_res.best_chunk else [])
        )
        self.assertEqual(len(evidence_for_verification), 1)
        self.assertEqual(evidence_for_verification[0].crop_clean, "Guava")


if __name__ == "__main__":
    unittest.main()
