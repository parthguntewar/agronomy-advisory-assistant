"""
Evidence-Grounded Answer Generator for the Agronomy Advisory Assistant.

This module provides a structured generation interface that:
1. Accepts ONLY pre-approved retrieved evidence as input.
2. Produces answers grounded exclusively in supplied evidence text.
3. Attaches source record IDs and provenance to every substantive answer.
4. Returns a structured GeneratedAnswer instead of a raw string.
5. Explicitly marks answers as UNCERTAIN or ABSTAINED when evidence is insufficient.
6. Delegates language model generation to an LLMProvider abstraction (Mock, Ollama, API).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.retrieval.vector_store import RetrievalResult
from src.guard.relevance_gate import GateResult, GateVerdict
from src.config import AGRONOMIC_SAFETY_NOTICE, SOURCE_ATTRIBUTION
from src.generation.providers import (
    LLMProvider,
    MockProvider,
    LLMError,
    LLMGenerationError,
    get_llm_provider,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an evidence-grounded Agronomy Advisory Assistant synthesizing historical agricultural records.\n"
    "CRITICAL GROUNDING INSTRUCTIONS:\n"
    "1. Base your response EXCLUSIVELY on the provided historical KCC advisory evidence records.\n"
    "2. For every factual advisory statement or recommendation, you MUST explicitly cite the corresponding "
    "record ID (e.g., [kcc_chunk_...]).\n"
    "3. NEVER invent, hallucinate, or extrapolate chemical names, active ingredients, or dosages that are not in the records.\n"
    "4. Clearly distinguish retrieved historical facts from uncertainty. If the evidence does not directly or fully answer "
    "the query, state that clearly.\n"
    "5. Always remind the farmer that this advisory is based on historical records and must be independently verified "
    "with their local Krishi Vigyan Kendra (KVK) or state agricultural extension officer before applying chemicals."
)


class GenerationStatus(str, Enum):
    """Status of the answer generation attempt."""
    SUCCESS = "success"                    # Answer generated from sufficient evidence
    WEAK_EVIDENCE = "weak_evidence"        # Answer generated but evidence is uncertain
    FALLBACK = "fallback"                  # Requested LLM failed; deterministic template fallback used
    ABSTAINED_NO_EVIDENCE = "abstained_no_evidence"  # No evidence; no answer generated
    ABSTAINED_CROP_MISMATCH = "abstained_crop_mismatch"  # Crop mismatch; blocked
    ABSTAINED_DYNAMIC = "abstained_dynamic"             # Live data requested
    ABSTAINED_OFFTOPIC = "abstained_offtopic"           # Off-topic query


@dataclass
class GeneratedAnswer:
    """
    Structured answer produced by the AnswerGenerator.
    
    Attributes:
        status: Generation outcome.
        answer_text: The formatted advisory text (empty string if abstained).
        source_ids: List of KCC chunk IDs used as evidence.
        evidence_scores: Cosine similarity scores of the supporting chunks.
        crops_cited: Crop names from cited evidence.
        provenance_note: Attribution string for the evidence sources.
        safety_notice: Mandatory agronomic safety disclaimer.
        uncertainty_note: If weak evidence, explains the limitation.
        abstention_reason: If abstained, explains why no answer was given.
        provider_used: Identifier of the provider or fallback that synthesized the response.
        is_fallback: True if the configured provider failed and fallback template was used.
        fallback_reason: Detailed error message explaining why fallback occurred.
    """
    status: GenerationStatus
    answer_text: str
    source_ids: List[str] = field(default_factory=list)
    evidence_scores: List[float] = field(default_factory=list)
    crops_cited: List[str] = field(default_factory=list)
    provenance_note: str = SOURCE_ATTRIBUTION
    safety_notice: str = AGRONOMIC_SAFETY_NOTICE
    uncertainty_note: Optional[str] = None
    abstention_reason: Optional[str] = None
    provider_used: str = "mock"
    is_fallback: bool = False
    fallback_reason: Optional[str] = None

    @property
    def is_answered(self) -> bool:
        return self.status in (
            GenerationStatus.SUCCESS,
            GenerationStatus.WEAK_EVIDENCE,
            GenerationStatus.FALLBACK,
        )

    def to_display_text(self) -> str:
        """Formats the answer for CLI/UI display with all caveats."""
        lines = ["=" * 70]

        if self.status == GenerationStatus.SUCCESS:
            lines.append("ADVISORY RESPONSE (Evidence-Grounded)")
        elif self.status == GenerationStatus.WEAK_EVIDENCE:
            lines.append("ADVISORY RESPONSE (Low-Confidence Evidence)")
        elif self.status == GenerationStatus.FALLBACK:
            lines.append("ADVISORY RESPONSE (Deterministic Template Fallback)")
        elif self.status == GenerationStatus.ABSTAINED_NO_EVIDENCE:
            lines.append("UNABLE TO ANSWER: No Relevant Evidence Found")
        elif self.status == GenerationStatus.ABSTAINED_CROP_MISMATCH:
            lines.append("BLOCKED: Crop Mismatch -- Evidence Not Safe to Use")
        elif self.status == GenerationStatus.ABSTAINED_DYNAMIC:
            lines.append("UNABLE TO ANSWER: Real-Time Data Required")
        elif self.status == GenerationStatus.ABSTAINED_OFFTOPIC:
            lines.append("UNABLE TO ANSWER: Query Not Related to Agriculture")
        lines.append("=" * 70)

        if self.is_fallback:
            lines.append(
                f"[FALLBACK NOTICE] Configured provider failed: {self.fallback_reason or 'unknown error'}. "
                "Output generated from verified deterministic fallback template, not direct LLM generation."
            )
            lines.append("")

        if self.answer_text:
            lines.append(self.answer_text)
            lines.append("")

        if self.uncertainty_note:
            lines.append(f"UNCERTAINTY NOTE: {self.uncertainty_note}")
            lines.append("")

        if self.abstention_reason:
            lines.append(f"Reason: {self.abstention_reason}")
            lines.append("")

        if self.source_ids:
            lines.append("EVIDENCE SOURCES:")
            for i, (sid, score) in enumerate(zip(self.source_ids, self.evidence_scores), 1):
                lines.append(f"  [{i}] Record ID: {sid}  |  Similarity: {score:.4f}")
            if self.crops_cited:
                lines.append(f"  Crops in evidence: {', '.join(set(self.crops_cited))}")
            lines.append(f"  Attribution: {self.provenance_note}")
            lines.append(f"  Provider Used: {self.provider_used} (Fallback: {self.is_fallback})")
            lines.append("")

        lines.append(f"NOTICE: {self.safety_notice}")
        lines.append("=" * 70)
        return "\n".join(lines)


class AnswerGenerator:
    """
    Evidence-grounded advisory answer generator.
    
    Delegates response synthesis to an LLMProvider while ensuring:
      - Only approved evidence chunks are forwarded to the model.
      - Source IDs and agronomic safety disclaimers are attached to every substantive answer.
      - Safe abstentions are generated when evidence is absent or crops mismatch.
      - Graceful fallback occurs if the model runtime fails.
    """

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider or get_llm_provider()

    def generate(
        self,
        query: str,
        gate_result: GateResult,
    ) -> GeneratedAnswer:
        """
        Generates a structured answer based on the gate evaluation result.
        
        Args:
            query: Original farmer query.
            gate_result: Output from the RelevanceGate evaluation.
            
        Returns:
            GeneratedAnswer with status, text, and source provenance.
        """
        verdict = gate_result.verdict

        # -- Stage 3 Gate Checks: No evidence or crop mismatch: abstain immediately --
        if verdict == GateVerdict.EVIDENCE_ABSENT:
            return GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                abstention_reason=gate_result.reason,
            )

        if verdict == GateVerdict.CROP_MISMATCH:
            return GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_CROP_MISMATCH,
                answer_text="",
                abstention_reason=(
                    f"{gate_result.reason}\n\n"
                    "To protect you from incorrect advice, the system will not "
                    "present evidence from a different crop as though it applies "
                    "to your crop. Please provide the correct crop name to retry."
                ),
            )

        # -- Approved evidence chunks for generation --
        # Strictly use approved crop-compatible evidence; never include unrelated cross-crop chunks
        evidence_chunks = (
            gate_result.sufficient_chunks
            or getattr(gate_result, "compatible_chunks", None)
            or ([gate_result.best_chunk] if gate_result.best_chunk else [])
        )

        if not evidence_chunks:
            return GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                abstention_reason=(
                    gate_result.reason
                    or "No approved crop-compatible evidence found to generate safe advice."
                ),
            )

        source_ids = [c.chunk_id for c in evidence_chunks]
        scores = [c.similarity_score for c in evidence_chunks]
        crops = [c.crop_clean for c in evidence_chunks]

        # Synthesize answer text using the configured provider
        answer_text, is_fallback, provider_used, fallback_reason = self._synthesize_answer(
            query=query,
            chunks=evidence_chunks,
            is_weak=(verdict == GateVerdict.EVIDENCE_WEAK),
        )

        # Fail safely if neither generation nor fallback produced substantive text
        if not answer_text or not answer_text.strip():
            return GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                source_ids=source_ids,
                evidence_scores=scores,
                crops_cited=crops,
                provider_used=provider_used,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
                abstention_reason=(
                    f"Advisory generation failed safely. Provider: {provider_used}. "
                    f"Details: {fallback_reason or 'empty generation output'}."
                ),
            )

        if verdict == GateVerdict.EVIDENCE_SUFFICIENT:
            return GeneratedAnswer(
                status=GenerationStatus.SUCCESS,
                answer_text=answer_text,
                source_ids=source_ids,
                evidence_scores=scores,
                crops_cited=crops,
                provider_used=provider_used,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
            )
        else:  # EVIDENCE_WEAK
            return GeneratedAnswer(
                status=GenerationStatus.WEAK_EVIDENCE,
                answer_text=answer_text,
                source_ids=source_ids,
                evidence_scores=scores,
                crops_cited=crops,
                provider_used=provider_used,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
                uncertainty_note=(
                    "The retrieved evidence has a low relevance score and may not "
                    "directly address your question. The information below is the "
                    "closest available historical advisory from the KCC knowledge base. "
                    "Please consult your local Krishi Vigyan Kendra (KVK) or "
                    "agricultural extension officer for verified guidance."
                ),
            )

    def _synthesize_answer(
        self,
        query: str,
        chunks: List[RetrievalResult],
        is_weak: bool = False,
    ) -> Tuple[str, bool, str, Optional[str]]:
        """
        Synthesizes the answer text using the active LLM provider.
        If using default MockProvider without custom canned responses, produces
        the standard structured format. If an LLM error occurs, falls back gracefully.
        
        Returns:
            Tuple of (answer_text, is_fallback, provider_used, fallback_reason)
        """
        provider_name = self.provider.get_provider_name()

        # If using standard MockProvider without custom responses, use deterministic template
        if (
            isinstance(self.provider, MockProvider)
            and not self.provider.canned_responses
            and not self.provider.default_response
        ):
            return self._format_evidence_answer(query, chunks), False, "mock", None

        prompt = self._build_llm_prompt(query, chunks, is_weak=is_weak)

        try:
            text = self.provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            if not text or not text.strip():
                raise LLMGenerationError(f"Provider '{provider_name}' returned empty response text.")
            return text, False, provider_name, None
        except LLMError as e:
            clean_err = self._sanitize_error(str(e))
            logger.warning(
                f"[AnswerGenerator] LLM provider '{provider_name}' "
                f"failed: {clean_err}. Falling back to structured evidence template."
            )
            try:
                fallback_text = self._format_evidence_answer(query, chunks)
                return (
                    fallback_text,
                    True,
                    "deterministic_template_fallback",
                    f"LLMError in provider '{provider_name}': {clean_err}",
                )
            except Exception as fe:
                return (
                    "",
                    True,
                    "failed_fallback",
                    f"Both provider and fallback failed: {clean_err} | {self._sanitize_error(str(fe))}",
                )
        except Exception as e:
            clean_err = self._sanitize_error(str(e))
            logger.error(
                f"[AnswerGenerator] Unexpected failure in provider '{provider_name}': {clean_err}. "
                "Falling back to structured evidence template."
            )
            try:
                fallback_text = self._format_evidence_answer(query, chunks)
                return (
                    fallback_text,
                    True,
                    "deterministic_template_fallback",
                    f"Unexpected error in provider '{provider_name}': {clean_err}",
                )
            except Exception as fe:
                return (
                    "",
                    True,
                    "failed_fallback",
                    f"Both provider and fallback failed: {clean_err} | {self._sanitize_error(str(fe))}",
                )

    @staticmethod
    def _sanitize_error(err_str: str) -> str:
        """Sanitizes potential API keys or authorization headers from error messages."""
        redacted = re.sub(r"([sS][kK]-[A-Za-z0-9_\-\.]{8,})", "[REDACTED_API_KEY]", err_str)
        redacted = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED_TOKEN]", redacted, flags=re.I)
        return redacted

    def _build_llm_prompt(
        self,
        query: str,
        chunks: List[RetrievalResult],
        is_weak: bool = False,
    ) -> str:
        """Constructs the prompt containing query and evidence records for the LLM."""
        lines = [
            f"Farmer Question: {query}\n",
            "Available Historical KCC Advisory Records:",
        ]
        for i, c in enumerate(chunks, 1):
            year_str = str(c.original_year) if c.original_year else "Historical"
            lines.append(f"\n[Record ID: {c.chunk_id}]")
            lines.append(f"Crop: {c.crop_clean}")
            lines.append(f"Location: {c.district}, {c.state} (Year: {year_str})")
            lines.append(f"Domain: {c.advisory_domain} | Topic: {c.query_type}")
            lines.append(f"Historical Query: {c.farmer_query}")
            lines.append(f"Historical Advisory: {c.expert_advisory}")

        if is_weak:
            lines.append(
                "\nNOTE: The relevance similarity score for this evidence is weak. "
                "Explicitly communicate uncertainty and urge local validation."
            )

        lines.append(
            "\nProvide an answer summarizing the historical advisory. Cite the source [Record ID] for all facts. "
            "Never recommend chemicals not found in the records above."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_evidence_answer(query: str, chunks: List[RetrievalResult]) -> str:
        """
        Formats retrieved evidence chunks into a structured advisory response.
        Used by the mock generator and as an offline fallback.
        """
        lines = [
            "Based on historical Kisan Call Centre (KCC) records, here is the "
            "available advisory evidence for your question:",
            "",
        ]

        for i, chunk in enumerate(chunks, 1):
            lines.append(f"--- Advisory Record {i} ---")
            lines.append(f"Record ID: {chunk.chunk_id}")
            lines.append(f"Crop: {chunk.crop_clean}")
            lines.append(f"Location: {chunk.district}, {chunk.state}")
            year_str = str(chunk.original_year) if chunk.original_year else "Historical (year unspecified)"
            lines.append(f"Record Year: {year_str}")
            lines.append(f"Domain: {chunk.advisory_domain}")
            lines.append(f"Similarity Score: {chunk.similarity_score:.4f} (retrieval signal only)")
            lines.append("")
            lines.append("KCC Expert Advisory:")
            lines.append(chunk.expert_advisory)
            lines.append("")

        lines.append(
            "NOTE: The above is a verbatim excerpt from historical KCC records. "
            "It reflects advice given at a specific time and location. "
            "Chemical names, dosages, and product registrations may have changed "
            "since the record date. Always verify with your local agricultural officer."
        )
        return "\n".join(lines)

    def generate_abstention(
        self,
        status: GenerationStatus,
        reason: str,
    ) -> GeneratedAnswer:
        """
        Generates a safe abstention response for pre-retrieval guard decisions.
        Used when the QueryGuard blocks the query before any retrieval occurs.
        """
        return GeneratedAnswer(
            status=status,
            answer_text="",
            abstention_reason=reason,
        )
