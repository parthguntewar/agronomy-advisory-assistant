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
# Expanded catalog of common agricultural active ingredients and formulations.
# Note: This is a structural heuristic and does NOT provide 100% chemical safety coverage.
# ---------------------------------------------------------------------------
_CHEMICAL_PATTERNS: List[re.Pattern] = [
    # Explicit active ingredients and commercial formulations (including known blind spots: monosultap, acephate, flubendiamide, paraquat)
    re.compile(
        r"\b(?:monosultap|acephate|flubendiamide|paraquat|glyphosate|cartap|cartap hydrochloride|"
        r"carbofuran|quinalphos|dimethoate|malathion|fipronil|tricyclazole|hexaconazole|"
        r"metalaxyl|streptocycline|chlorantraniliprole|pendimethalin|atrazine|2,4-d|"
        r"cypermethrin|deltamethrin|chlorpyrifos|imidacloprid|thiamethoxam|lambda-cyhalothrin|"
        r"profenofos|carbendazim|mancozeb|copper sulphate|copper oxychloride|tricel|rogor|"
        r"confidor|bavistin|coragen|urea|dap|npk|potash|zinc sulphate|borax|sulphur|"
        r"emamectin|emamectin benzoate|spinosad|azoxystrobin|propiconazole|tebuconazole|monocrotophos|"
        r"endosulfan|phorate)\b",
        re.I,
    ),
    # Agrochemical class suffixes
    re.compile(r"\b[a-z]{3,}(?:sultap|quat|oxystrobin|fenozide)\b", re.I),
    # Formulation codes: e.g. 25% EC, 80% WP, 50% SC, 3G
    re.compile(r"\b\d+\.?\d*\s*%\s*(?:ec|wp|sc|sl|sp|gr|dp|wg|fs|cs)\b", re.I),
    re.compile(r"\b\d+\.?\d*\s*(?:g|gr)\s+granul(?:es?|ar)\b", re.I),
    # Dosage patterns: "X ml/litre", "X g/ha", "X kg/acre", "X ml per liter", etc.
    re.compile(r"\b\d+\.?\d*\s*(?:ml|g|kg|gm|liter|litre|l)\s*(?:/|\bper\b)\s*(?:litre|liter|ha|hectare|acre|plant|tree|l|pump|tank)\b", re.I),
    # Explicit application rates: e.g., "spray 2 ml", "apply 500 ml/acre"
    re.compile(r"\b(?:spray|apply|use|mix)\s+\d+\.?\d*\s*(?:ml|g|kg|gm|liter|litre|l)\b", re.I),
    re.compile(r"\b\d+\.?\d*\s*(?:ml|g|kg|gm)\s+(?:in|per)\s+\d+\.?\d*\s*(?:l|liter|litres?|pani)\b", re.I),
]


def _normalize_tokens(text: str) -> str:
    """Normalizes whitespace and common separators for token-level grounding comparison."""
    t = text.lower()
    t = re.sub(r"[/\-_,;:()\[\]]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class PostGenerationVerifier:
    """
    Structural safety verifier applied to GeneratedAnswer objects.
    
    Checks that substantive claims in the generated answer are traceable
    to the provided evidence chunks and that source citations are present.
    
    IMPORTANT LIMITATION:
    This is an automated structural/token-level verifier. It reduces ungrounded
    hallucination risk, but does NOT guarantee 100% agronomic or chemical safety.
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
        # ── Skip verification strictly for non-advisory abstentions ──────
        if not answer.is_answered:
            return VerificationResult(
                status=VerificationStatus.SKIPPED_ABSTENTION,
                original_answer=answer,
                reason="Answer was an abstention — no generated text to verify.",
                safe_answer=answer,
            )

        issues: List[str] = []
        warnings: List[str] = []

        # ── Note weak evidence caveat if applicable (do NOT skip verification!) ─
        if answer.status == GenerationStatus.WEAK_EVIDENCE:
            warnings.append(
                "WEAK_EVIDENCE_CAVEAT: Advisory was generated under low-confidence evidence conditions. "
                "Local expert validation (KVK/SAU) is required before implementing any advice."
            )

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
        norm_evidence = _normalize_tokens(evidence_text_combined)

        ungrounded_claims: List[str] = []
        for pat in _CHEMICAL_PATTERNS:
            for m in pat.finditer(answer_lower):
                claim = m.group(0).lower().strip()
                norm_claim = _normalize_tokens(claim)
                # Check both exact substring and normalized token match
                if claim not in evidence_text_combined and norm_claim not in norm_evidence:
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
            blocked_answer = GeneratedAnswer(
                status=GenerationStatus.ABSTAINED_NO_EVIDENCE,
                answer_text="",
                source_ids=answer.source_ids,
                evidence_scores=answer.evidence_scores,
                crops_cited=answer.crops_cited,
                safety_notice=answer.safety_notice,
                abstention_reason=(
                    "Post-generation verifier blocked this answer due to "
                    "ungrounded chemical claims or missing citations. "
                    "Details: " + " | ".join(issues)
                ),
                provider_used=getattr(answer, "provider_used", "mock"),
                is_fallback=getattr(answer, "is_fallback", False),
                fallback_reason=getattr(answer, "fallback_reason", None),
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
