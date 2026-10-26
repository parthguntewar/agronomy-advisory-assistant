"""
Hallucination-Guarded RAG Pipeline for the Agronomy Advisory Assistant.

Orchestrates the complete safety-layered pipeline:
  1. QueryGuard  → Pre-retrieval classification (off-topic / dynamic / clarify / proceed)
  2. Retrieval   → Semantic vector search (only if QueryGuard says PROCEED)
  3. RelevanceGate → Post-retrieval threshold gate (sufficient / weak / absent / mismatch)
  4. AnswerGenerator → Evidence-grounded answer construction
  5. PostGenerationVerifier → Structural safety check on generated answer

This module is the single entry point for the entire safe advisory pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

from src.guard.query_guard import QueryGuard, GuardDecision, GuardAction
from src.guard.relevance_gate import RelevanceGate, GateResult
from src.generation.answer_gen import AnswerGenerator, GeneratedAnswer, GenerationStatus
from src.generation.verifier import PostGenerationVerifier, VerificationResult
from src.retrieval.retriever import AgronomyRetriever
from src.retrieval.vector_store import RetrievalResult
from src.config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Complete structured result from the guarded RAG pipeline.
    
    Attributes:
        query: Original input query.
        guard_decision: Pre-retrieval guard output.
        retrieval_results: Raw vector search results (empty if guarded).
        gate_result: Post-retrieval relevance gate output (None if not reached).
        generated_answer: Answer before verification.
        verification_result: Post-generation safety verification.
        final_answer: The verified, safe-to-present GeneratedAnswer.
        pipeline_stages: Ordered list of stages executed.
    """
    query: str
    guard_decision: GuardDecision
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    gate_result: Optional[GateResult] = None
    generated_answer: Optional[GeneratedAnswer] = None
    verification_result: Optional[VerificationResult] = None
    final_answer: Optional[GeneratedAnswer] = None
    pipeline_stages: List[str] = field(default_factory=list)

    @property
    def was_answered(self) -> bool:
        return self.final_answer is not None and self.final_answer.is_answered

    @property
    def was_blocked(self) -> bool:
        return (
            self.verification_result is not None
            and self.verification_result.is_blocked
        )

    def to_display_text(self) -> str:
        """Full human-readable pipeline trace for CLI output."""
        lines = [
            "=" * 72,
            "   AGRONOMY ADVISORY ASSISTANT -- HALLUCINATION-GUARDED RAG",
            "=" * 72,
            f"Query: {self.query}",
            f"Pipeline stages: {' -> '.join(self.pipeline_stages)}",
            "",
            "── GUARD DECISION ───────────────────────────────────────────────────",
            f"Action:     {self.guard_decision.action.value}",
            f"Reason:     {self.guard_decision.reason}",
        ]

        if self.guard_decision.detected_crop:
            lines.append(f"Detected crop: {self.guard_decision.detected_crop}")

        if self.guard_decision.needs_clarification and self.guard_decision.clarification_prompt:
            lines.append("")
            lines.append("── CLARIFICATION REQUIRED ───────────────────────────────────────────")
            lines.append(self.guard_decision.clarification_prompt)

        if self.retrieval_results:
            lines.append("")
            lines.append("── RETRIEVAL RESULTS ─────────────────────────────────────────────────")
            for i, r in enumerate(self.retrieval_results[:3], 1):
                lines.append(
                    f"  [{i}] {r.chunk_id} | Crop: {r.crop_clean} | "
                    f"Score: {r.similarity_score:.4f}"
                )

        if self.gate_result:
            lines.append("")
            lines.append("-- RELEVANCE GATE " + "-" * 52)
            lines.append(f"Verdict:    {self.gate_result.verdict.value}")
            lines.append(f"Best score: {self.gate_result.best_score:.4f}")
            lines.append(f"Reason:     {self.gate_result.reason}")

        if self.verification_result:
            lines.append("")
            lines.append("-- VERIFICATION " + "-" * 54)
            lines.append(f"Status:     {self.verification_result.status.value}")
            if self.verification_result.issues:
                for issue in self.verification_result.issues:
                    lines.append(f"  ISSUE: {issue}")
            if self.verification_result.warnings:
                for warn in self.verification_result.warnings:
                    lines.append(f"  WARN:  {warn}")

        lines.append("")
        if self.final_answer:
            lines.append(self.final_answer.to_display_text())
        else:
            lines.append("No answer generated.")

        return "\n".join(lines)


class GuardedRAGPipeline:
    """
    Full hallucination-guarded RAG pipeline.
    
    Composes QueryGuard → AgronomyRetriever → RelevanceGate →
    AnswerGenerator → PostGenerationVerifier into a single safe() call.
    """

    def __init__(
        self,
        retriever: Optional[AgronomyRetriever] = None,
        top_k: int = TOP_K_RETRIEVAL,
        answer_generator: Optional[AnswerGenerator] = None,
    ) -> None:
        self.query_guard = QueryGuard()
        self.retriever = retriever or AgronomyRetriever()
        self.relevance_gate = RelevanceGate()
        self.answer_gen = answer_generator or AnswerGenerator()
        self.verifier = PostGenerationVerifier()
        self.top_k = top_k

    def answer(self, query: str) -> PipelineResult:
        """
        Runs the complete guarded RAG pipeline for a single query.
        
        Args:
            query: Raw natural language question from the farmer.
            
        Returns:
            PipelineResult with all intermediate decisions and final answer.
        """
        stages: List[str] = []
        result = PipelineResult(query=query, guard_decision=None)  # type: ignore

        # ── Stage 1: Pre-retrieval Query Guard ────────────────────────────
        stages.append("QueryGuard")
        guard_decision = self.query_guard.classify(query)
        result.guard_decision = guard_decision
        logger.info(f"[QueryGuard] Action={guard_decision.action.value} | {guard_decision.reason}")

        # Handle guard blocks (dynamic, off-topic, clarify)
        if guard_decision.action == GuardAction.ABSTAIN_DYNAMIC:
            stages.append("Abstained(Dynamic)")
            final = self.answer_gen.generate_abstention(
                GenerationStatus.ABSTAINED_DYNAMIC,
                (
                    "This query requires real-time information (weather forecasts, "
                    "current market prices, or government scheme deadlines) that this "
                    "system does not have access to. Please check:\n"
                    "• Weather: India Meteorological Department (imd.gov.in)\n"
                    "• Market prices: Agmarknet (agmarknet.gov.in)\n"
                    "• PM Kisan: pmkisan.gov.in"
                ),
            )
            result.pipeline_stages = stages
            result.final_answer = final
            return result

        if guard_decision.action == GuardAction.ABSTAIN_OFFTOPIC:
            stages.append("Abstained(OffTopic)")
            final = self.answer_gen.generate_abstention(
                GenerationStatus.ABSTAINED_OFFTOPIC,
                (
                    "I am an agronomy advisory assistant specializing in crop management, "
                    "pest control, nutrient management, and farming practices. "
                    "Your question does not appear to be related to agriculture. "
                    "Please ask a farming-related question."
                ),
            )
            result.pipeline_stages = stages
            result.final_answer = final
            return result

        if guard_decision.action in (GuardAction.CLARIFY_CROP, GuardAction.CLARIFY_SYMPTOMS):
            stages.append("Clarification(Requested)")
            # Return immediately with the clarification prompt; no retrieval
            final = GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                abstention_reason=guard_decision.clarification_prompt,
            )
            result.pipeline_stages = stages
            result.final_answer = final
            return result

        # ── Stage 2: Semantic Retrieval ────────────────────────────────────
        stages.append("Retrieval")
        retrieval_results = self.retriever.retrieve(
            query,
            top_k=self.top_k,
            crop_filter=None,  # Guard-detected crop used for gate, not as a hard filter
        )
        result.retrieval_results = retrieval_results
        logger.info(f"[Retrieval] Retrieved {len(retrieval_results)} chunks.")

        # ── Stage 3: Relevance Gate ────────────────────────────────────────
        stages.append("RelevanceGate")
        gate_result = self.relevance_gate.evaluate(
            query=query,
            results=retrieval_results,
            query_crop=guard_decision.detected_crop,
        )
        result.gate_result = gate_result
        logger.info(f"[RelevanceGate] Verdict={gate_result.verdict.value} | {gate_result.reason}")

        # ── Stage 4: Answer Generation ─────────────────────────────────────
        stages.append("Generation")
        generated = self.answer_gen.generate(query, gate_result)
        result.generated_answer = generated
        logger.info(f"[Generation] Status={generated.status.value}")

        # ── Stage 5: Post-Generation Verification ─────────────────────────
        stages.append("Verification")
        evidence_for_verification = (
            gate_result.sufficient_chunks or gate_result.all_chunks or []
        )
        verification = self.verifier.verify(generated, evidence_for_verification)
        result.verification_result = verification
        logger.info(f"[Verification] Status={verification.status.value}")

        # Final safe answer is the verifier's output
        result.final_answer = verification.safe_answer or generated
        result.pipeline_stages = stages

        return result
