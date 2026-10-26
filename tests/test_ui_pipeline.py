"""
Unit and integration tests for Phase 5: User-Facing Application and UI Helpers.

Tests:
  1. Ollama service health check helper (online vs offline).
  2. Extraction of genuine pipeline diagnostics without fabricating unreached stages.
  3. UI formatting of in-domain answered queries.
  4. UI formatting of pre-retrieval off-topic abstentions (bypassed stages handled cleanly).
  5. UI formatting of dynamic real-time request abstentions.
  6. UI formatting of ambiguous disease queries requiring crop clarification.
  7. UI formatting of verifier-blocked hallucination alerts.
  8. Unexposed field handling ("Not exposed by pipeline" / "Stage not executed").
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.helpers import (
    check_ollama_service,
    extract_pipeline_diagnostics,
    format_stage_badge,
)
from src.pipeline import GuardedRAGPipeline, PipelineResult
from src.guard.query_guard import GuardDecision, GuardAction
from src.guard.relevance_gate import GateResult, GateVerdict
from src.generation.answer_gen import GeneratedAnswer, GenerationStatus, AnswerGenerator
from src.generation.verifier import VerificationResult, VerificationStatus
from src.generation.providers import MockProvider
from src.retrieval.vector_store import RetrievalResult
from src.config import AGRONOMIC_SAFETY_NOTICE, SOURCE_ATTRIBUTION


def make_dummy_result(
    query: str = "How do I control stem borer in paddy?",
    action: GuardAction = GuardAction.PROCEED,
    verdict: GateVerdict = GateVerdict.EVIDENCE_SUFFICIENT,
    gen_status: GenerationStatus = GenerationStatus.SUCCESS,
    verif_status: VerificationStatus = VerificationStatus.PASSED,
) -> PipelineResult:
    chunk = RetrievalResult(
        chunk_id="kcc_chunk_test_paddy_1",
        similarity_score=0.6420,
        crop="Paddy (Dhan)",
        crop_clean="Paddy (Dhan)",
        state="Odisha",
        district="Puri",
        advisory_domain="plant_protection",
        query_type="pest_control",
        original_year=2022,
        farmer_query="Stem borer control in paddy?",
        expert_advisory="Spray chlorpyrifos 2 ml/litre.",
        qc_severity="none",
        qc_confidence="high",
        qc_flags=[],
        rag_use_mode="use_as_evidence",
        source_attribution=SOURCE_ATTRIBUTION,
        safety_notice=AGRONOMIC_SAFETY_NOTICE,
        retrieval_text="Stem borer advisory",
    )

    guard_dec = GuardDecision(
        action=action,
        reason="Query context validated.",
        detected_crop="Paddy" if action == GuardAction.PROCEED else None,
        is_agricultural=True if action != GuardAction.ABSTAIN_OFFTOPIC else False,
        has_dynamic_request=True if action == GuardAction.ABSTAIN_DYNAMIC else False,
        clarification_prompt="Please specify crop" if action == GuardAction.CLARIFY_CROP else None,
    )

    gate_res = GateResult(
        verdict=verdict,
        reason="Score exceeds threshold.",
        best_score=0.6420,
        best_chunk=chunk,
        sufficient_chunks=[chunk],
    )

    gen_ans = GeneratedAnswer(
        status=gen_status,
        answer_text="Apply chlorpyrifos 2 ml/litre for stem borer in paddy.",
        source_ids=["kcc_chunk_test_paddy_1"],
        evidence_scores=[0.6420],
        crops_cited=["Paddy (Dhan)"],
    )

    verif_res = VerificationResult(
        status=verif_status,
        original_answer=gen_ans,
        reason="Passed all structural checks.",
        safe_answer=gen_ans,
    )

    return PipelineResult(
        query=query,
        guard_decision=guard_dec,
        retrieval_results=[chunk],
        gate_result=gate_res,
        generated_answer=gen_ans,
        verification_result=verif_res,
        final_answer=gen_ans,
        pipeline_stages=["QueryGuard", "Retrieval", "RelevanceGate", "Generation", "Verification"],
    )


class TestUIHelpers(unittest.TestCase):
    """Tests for UI helper and diagnostic extraction logic."""

    @patch("src.ui.helpers.requests.get")
    def test_check_ollama_service_online(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "mistral:latest"}]
        }
        res = check_ollama_service("http://127.0.0.1:11434")
        self.assertTrue(res["available"])
        self.assertIn("llama3.2:latest", res["models"])
        self.assertIsNone(res["error"])

    @patch("src.ui.helpers.requests.get")
    def test_check_ollama_service_offline(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        res = check_ollama_service("http://127.0.0.1:11434")
        self.assertFalse(res["available"])
        self.assertEqual(res["models"], [])
        self.assertIsNotNone(res["error"])

    def test_format_stage_badge(self):
        self.assertEqual(format_stage_badge("QueryGuard"), "Stage 1: QueryGuard")
        self.assertEqual(format_stage_badge("Verification"), "Stage 5: Post-Verifier")
        self.assertEqual(format_stage_badge("UnknownStage"), "UnknownStage")

    def test_extract_pipeline_diagnostics_success_case(self):
        result = make_dummy_result()
        diag = extract_pipeline_diagnostics(result)

        self.assertEqual(diag["query"], "How do I control stem borer in paddy?")
        self.assertTrue(diag["was_answered"])
        self.assertFalse(diag["was_blocked"])

        # Guard
        self.assertEqual(diag["guard"]["action"], "proceed")
        self.assertEqual(diag["guard"]["detected_crop"], "Paddy")

        # Gate
        self.assertEqual(diag["gate"]["verdict"], "evidence_sufficient")
        self.assertEqual(diag["gate"]["best_score"], "0.6420")
        self.assertEqual(diag["gate"]["retrieved_chunk_count"], 1)

        # Generation
        self.assertEqual(diag["generation"]["status"], "success")
        self.assertEqual(diag["generation"]["source_ids_cited"], ["kcc_chunk_test_paddy_1"])

        # Verification
        self.assertEqual(diag["verification"]["status"], "passed")
        self.assertTrue(diag["verification"]["is_safe_to_present"])

    def test_extract_pipeline_diagnostics_bypassed_stages(self):
        """When pre-retrieval guard blocks a query, stages 2-5 must show safely bypassed, not fake values."""
        guard_dec = GuardDecision(
            action=GuardAction.ABSTAIN_OFFTOPIC,
            reason="Non-agricultural question.",
            is_agricultural=False,
            has_dynamic_request=False,
        )
        abstained_ans = GeneratedAnswer(
            status=GenerationStatus.ABSTAINED_OFFTOPIC,
            answer_text="",
            abstention_reason="Not related to agriculture.",
        )
        result = PipelineResult(
            query="How do I configure Node.js router?",
            guard_decision=guard_dec,
            retrieval_results=[],
            gate_result=None,
            generated_answer=None,
            verification_result=None,
            final_answer=abstained_ans,
            pipeline_stages=["QueryGuard", "Abstained(OffTopic)"],
        )

        diag = extract_pipeline_diagnostics(result)
        self.assertFalse(diag["was_answered"])
        self.assertEqual(diag["guard"]["action"], "abstain_offtopic")
        # Bypassed stages must report safely bypassed
        self.assertIn("safely bypassed", diag["gate"]["verdict"])
        self.assertIn("safely bypassed", diag["generation"]["status"])
        self.assertIn("safely bypassed", diag["verification"]["status"])

    def test_extract_pipeline_diagnostics_verifier_blocked(self):
        """When an answer is blocked for hallucinations, diagnostics must report blocked status and issues."""
        base_result = make_dummy_result()
        blocked_verif = VerificationResult(
            status=VerificationStatus.BLOCKED_HALLUCINATION,
            original_answer=base_result.generated_answer,
            issues=["HALLUCINATION_RISK: Fabricated chemical detected: ['chlorpyrifos 50 ml/litre']"],
            reason="Blocked due to ungrounded claim.",
            safe_answer=GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                abstention_reason="Blocked due to ungrounded claims.",
            ),
        )
        base_result.verification_result = blocked_verif
        base_result.final_answer = blocked_verif.safe_answer

        diag = extract_pipeline_diagnostics(base_result)
        self.assertTrue(diag["was_blocked"])
        self.assertEqual(diag["verification"]["status"], "blocked_hallucination")
        self.assertIn("Fabricated chemical", diag["verification"]["issues"][0])


class TestUIPipelineIntegration(unittest.TestCase):
    """Integration test verifying end-to-end execution with MockProvider in the UI pipeline."""

    def setUp(self):
        # Create pipeline with deterministic MockProvider
        mock_provider = MockProvider()
        answer_gen = AnswerGenerator(provider=mock_provider)
        self.pipeline = GuardedRAGPipeline(answer_generator=answer_gen)

    def test_ui_pipeline_supported_query_end_to_end(self):
        query = "How to control stem borer in paddy?"
        result = self.pipeline.answer(query)
        self.assertIsNotNone(result)
        self.assertIn("QueryGuard", result.pipeline_stages)
        diag = extract_pipeline_diagnostics(result)
        self.assertIn("stages", diag)

    def test_ui_pipeline_offtopic_query_abstains(self):
        query = "How to configure React frontend router in Node.js?"
        result = self.pipeline.answer(query)
        self.assertEqual(result.guard_decision.action, GuardAction.ABSTAIN_OFFTOPIC)
        self.assertFalse(result.was_answered)

    def test_ui_pipeline_dynamic_query_abstains(self):
        query = "Will it rain tomorrow in Mehsana, Gujarat?"
        result = self.pipeline.answer(query)
        self.assertEqual(result.guard_decision.action, GuardAction.ABSTAIN_DYNAMIC)
        self.assertFalse(result.was_answered)

    def test_ui_pipeline_ambiguous_crop_clarifies(self):
        query = "My plants have yellow spots and are curling. What spray should I use?"
        result = self.pipeline.answer(query)
        self.assertIn(result.guard_decision.action, (GuardAction.CLARIFY_CROP, GuardAction.CLARIFY_SYMPTOMS))
        self.assertFalse(result.was_answered)


if __name__ == "__main__":
    unittest.main()
