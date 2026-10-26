"""
Post-Generation Verifier for the Agronomy Advisory Assistant.

Performs a lightweight structural and safety check on generated answers
AFTER the AnswerGenerator produces output, before the answer is presented
to the farmer.

What the verifier checks:
  1. SOURCE CITATION COMPLETENESS: Every substantive answer must cite at
     least one source record ID.
  2. ANSWER GROUNDING: The answer text must not contain fabricated chemical
     names or dosages that are not present in the supporting evidence.
  3. UNSUPPORTED CHEMICAL CLAIM DETECTION: If the answer contains chemical
     dosages/product names that do not appear in any evidence chunk text,
     flag as HALLUCINATION_RISK.
  4. HISTORICAL DISCLAIMER PRESENCE: Generated answers must retain the
     historical provenance disclaimer.

IMPORTANT LIMITATIONS:
  - This verifier does NOT guarantee factual correctness of the evidence.
  - It does NOT verify that the cited KCC advisory is agronomically valid today.
  - It does NOT eliminate hallucinations — it reduces ungrounded fabrication risk.
  - Chemical name matching is approximate (substring / token level), not semantic.
  - A proper verifier requires an entailment model (NLI) for production use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.generation.answer_gen import GeneratedAnswer, GenerationStatus
from src.retrieval.vector_store import RetrievalResult


class VerificationStatus(str, Enum):
    """Outcome of the post-generation verification step."""
    PASSED = "passed"              # Answer passed all checks
    PASSED_WITH_WARNINGS = "passed_with_warnings"  # Minor issues noted
    BLOCKED_HALLUCINATION = "blocked_hallucination"  # Unsupported claims detected
    SKIPPED_ABSTENTION = "skipped_abstention"        # No answer to verify (abstained)
    SKIPPED_WEAK = "skipped_weak"                    # Weak evidence — not verified


@dataclass
class VerificationResult:
    """Result of the post-generation verification."""
    status: VerificationStatus
    original_answer: GeneratedAnswer
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reason: str = ""
    # The safe answer to present — may be modified or blocked
    safe_answer: Optional[GeneratedAnswer] = None

    @property
    def is_safe_to_present(self) -> bool:
        return self.status in (VerificationStatus.PASSED, VerificationStatus.PASSED_WITH_WARNINGS)

    @property
    def is_blocked(self) -> bool:
        return self.status == VerificationStatus.BLOCKED_HALLUCINATION


# ---------------------------------------------------------------------------
# Chemical / agrochemical token patterns
# Used to detect specific claims in generated text that must be grounded.
# ---------------------------------------------------------------------------
_CHEMICAL_PATTERNS: List[re.Pattern] = [
    # Dosage patterns: "X ml/litre", "X g/ha", "X kg/acre" etc.
    re.compile(r"\b\d+\.?\d*\s*(?:ml|g|kg|gm|liter|litre|L)\s*/\s*(?:litre|liter|ha|hectare|acre|plant|tree|l)\b", re.I),
    # Active ingredient / product name followed by dose
    re.compile(r"\b(?:spray|apply|use|mix)\s+\d+\.?\d*\s*(?:ml|g|kg|gm)\b", re.I),
    # Specific chemical name indicators (common in KCC data)
    re.compile(r"\b(?:chlorpyrifos|imidacloprid|thiamethoxam|lambda-cyhalothrin|profenofos|carbendazim|mancozeb|copper sulphate|tricel|urea|dap|npk|potash|zinc sulphate|borax|sulphur|emamectin|spinosad|azoxystrobin|propiconazole|tebuconazole|monocrotophos)\b", re.I),
]


class PostGenerationVerifier:
    """
    Structural safety verifier applied to GeneratedAnswer objects.
    
    Checks that every substantive claim in the generated answer is traceable
    to the provided evidence chunks and that source citations are present.
    
    This is an APPROXIMATE verifier — production use requires an NLI model.
    """

    def verify(
        self,
        answer: GeneratedAnswer,
        evidence_chunks: List[RetrievalResult],
    ) -> VerificationResult:
        """
        Verifies a generated answer against its evidence sources.
        
        Args:
            answer: The GeneratedAnswer to verify.
            evidence_chunks: The RetrievalResult chunks used to generate the answer.
            
        Returns:
            VerificationResult indicating whether the answer is safe to present.
        """
        # ── Skip verification for abstentions ────────────────────────────
        if not answer.is_answered:
            return VerificationResult(
                status=VerificationStatus.SKIPPED_ABSTENTION,
                original_answer=answer,
                reason="Answer was an abstention — no generated text to verify.",
                safe_answer=answer,
            )

        # ── Skip detailed verification for weak evidence ──────────────────
        if answer.status == GenerationStatus.WEAK_EVIDENCE:
            return VerificationResult(
                status=VerificationStatus.SKIPPED_WEAK,
                original_answer=answer,
                reason=(
                    "Answer generated under weak evidence conditions. "
                    "Structural verification skipped; uncertainty note is already present."
                ),
                safe_answer=answer,
            )

        issues: List[str] = []
        warnings: List[str] = []

        # ── Check 1: Source citation completeness ─────────────────────────
        if not answer.source_ids:
            issues.append(
                "CITATION_MISSING: Answer has no source record IDs. "
                "Every substantive advisory answer must cite at least one KCC record."
            )

        # ── Check 2: Safety notice presence ───────────────────────────────
        if answer.safety_notice and len(answer.safety_notice) < 20:
            warnings.append(
                "DISCLAIMER_SHORT: Safety notice appears truncated or missing. "
                "Agronomic advisory answers must carry the full safety disclaimer."
            )

        # ── Check 3: Chemical claim grounding ─────────────────────────────
        answer_lower = answer.answer_text.lower()
        evidence_text_combined = " ".join(
            (c.expert_advisory + " " + c.farmer_query).lower()
            for c in evidence_chunks
        )

        ungrounded_claims: List[str] = []
        for pat in _CHEMICAL_PATTERNS:
            for m in pat.finditer(answer_lower):
                claim = m.group(0).lower()
                # Check if this exact claim appears in any evidence text
                if claim not in evidence_text_combined:
                    ungrounded_claims.append(claim)

        if ungrounded_claims:
            issues.append(
                f"HALLUCINATION_RISK: The following specific chemical claims in the "
                f"answer were NOT found in any supporting evidence chunk: "
                f"{ungrounded_claims}. These may be fabricated or incorrectly generalized."
            )

        # ── Check 4: Evidence crop consistency ────────────────────────────
        cited_crops = set(answer.crops_cited) if answer.crops_cited else set()
        evidence_crops = {c.crop_clean for c in evidence_chunks}
        extra_crops = cited_crops - evidence_crops
        if extra_crops:
            warnings.append(
                f"CROP_CONSISTENCY: Answer cites crops {extra_crops} "
                f"not present in retrieved evidence. Possible template error."
            )

        # ── Final verdict ─────────────────────────────────────────────────
        if issues:
            # If there are any issues (not just warnings), block the answer
            blocked_answer = GeneratedAnswer(
                status=answer.status,
                answer_text="",  # Clear answer text — do not present it
                source_ids=answer.source_ids,
                evidence_scores=answer.evidence_scores,
                crops_cited=answer.crops_cited,
                safety_notice=answer.safety_notice,
                abstention_reason=(
                    "This answer was blocked by the post-generation safety verifier "
                    "because it contained ungrounded claims not traceable to the "
                    "retrieved evidence. Issues: " + " | ".join(issues)
                ),
            )
            # Override status to abstained for display
            blocked_answer = GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                abstention_reason=(
                    "Post-generation verifier blocked this answer due to "
                    "ungrounded chemical claims or missing citations. "
                    "Details: " + " | ".join(issues)
                ),
                safety_notice=answer.safety_notice,
            )
            return VerificationResult(
                status=VerificationStatus.BLOCKED_HALLUCINATION,
                original_answer=answer,
                issues=issues,
                warnings=warnings,
                reason="Answer blocked: ungrounded claims detected.",
                safe_answer=blocked_answer,
            )

        if warnings:
            return VerificationResult(
                status=VerificationStatus.PASSED_WITH_WARNINGS,
                original_answer=answer,
                warnings=warnings,
                reason="Answer passed verification with minor warnings.",
                safe_answer=answer,
            )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            original_answer=answer,
            reason="Answer passed all structural safety checks.",
            safe_answer=answer,
        )
