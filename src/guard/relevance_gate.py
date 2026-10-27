"""
Retrieval Relevance Gate for the Agronomy Advisory Assistant.

Evaluates retrieved evidence AFTER vector search to decide whether the
retrieved chunks are sufficiently relevant and crop-compatible to support
answer generation.

Key design principle:
  Cosine similarity is a DISTANCE METRIC, not a calibrated probability of
  correctness. Thresholds here are heuristic gates, validated against the
  5-scenario baseline. They are NOT percentages of factual accuracy.

Gate decisions:
  EVIDENCE_SUFFICIENT   - At least one chunk exceeds relevance threshold and 
                          crop is compatible with the query.
  EVIDENCE_WEAK         - Chunks found but below confidence threshold; answer 
                          with explicit uncertainty warning.
  EVIDENCE_ABSENT       - No chunks above the minimum viable threshold; abstain.
  CROP_MISMATCH         - Retrieved crop is incompatible with the queried crop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.retrieval.vector_store import RetrievalResult


class GateVerdict(str, Enum):
    """Verdict from the relevance gate evaluation."""
    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    EVIDENCE_WEAK = "evidence_weak"
    EVIDENCE_ABSENT = "evidence_absent"
    CROP_MISMATCH = "crop_mismatch"


# ---------------------------------------------------------------------------
# Threshold configuration
# These were calibrated against 5 baseline scenarios:
#   Scenario 1 (in-domain Satawar):              0.6206 → should PASS
#   Scenario 2 (non-agri React/Node.js):         0.0815 → should ABSENT
#   Scenario 3 (unindexed Dragon Fruit):         0.4247 → WEAK (no crop match)
#   Scenario 4 (weather/PM Kisan time-sensitive):0.3835 → WEAK (pre-filtered by guard)
#   Scenario 5 (ambiguous disease no-crop):      0.2737 → ABSENT/MISMATCH
#
# SUFFICIENT_THRESHOLD: chunk must exceed this for confident evidence use.
# MINIMUM_VIABLE_THRESHOLD: below this, we will not attempt to generate an answer.
# ---------------------------------------------------------------------------
SUFFICIENT_THRESHOLD = 0.50   # Strong evidence for answer generation
MINIMUM_VIABLE_THRESHOLD = 0.30  # Below this: abstain entirely


@dataclass
class GateResult:
    """Result produced by the RelevanceGate for a given retrieval set."""
    verdict: GateVerdict
    best_score: float
    best_chunk: Optional[RetrievalResult]
    sufficient_chunks: List[RetrievalResult] = field(default_factory=list)
    compatible_chunks: List[RetrievalResult] = field(default_factory=list)
    all_chunks: List[RetrievalResult] = field(default_factory=list)
    detected_query_crop: Optional[str] = None
    crop_mismatch_detail: Optional[str] = None
    reason: str = ""

    @property
    def has_sufficient_evidence(self) -> bool:
        return self.verdict == GateVerdict.EVIDENCE_SUFFICIENT

    @property
    def should_abstain(self) -> bool:
        return self.verdict in (GateVerdict.EVIDENCE_ABSENT, GateVerdict.CROP_MISMATCH)


class RelevanceGate:
    """
    Post-retrieval relevance gate that checks whether retrieved evidence
    meets minimum quality and crop-compatibility requirements.
    
    Args:
        sufficient_threshold: Cosine similarity above which evidence is "sufficient".
        minimum_viable_threshold: Below this score, abstain entirely.
    """

    def __init__(
        self,
        sufficient_threshold: float = SUFFICIENT_THRESHOLD,
        minimum_viable_threshold: float = MINIMUM_VIABLE_THRESHOLD,
    ) -> None:
        self.sufficient_threshold = sufficient_threshold
        self.minimum_viable_threshold = minimum_viable_threshold

    def evaluate(
        self,
        query: str,
        results: List[RetrievalResult],
        query_crop: Optional[str] = None,
    ) -> GateResult:
        """
        Evaluates retrieved results against relevance thresholds.
        
        Args:
            query: The original farmer query (used for context logging).
            results: Retrieved RetrievalResult objects from the vector store.
            query_crop: Crop name detected from the query by QueryGuard, if any.
            
        Returns:
            GateResult with verdict and filtered evidence chunks.
        """
        if not results:
            return GateResult(
                verdict=GateVerdict.EVIDENCE_ABSENT,
                best_score=0.0,
                best_chunk=None,
                detected_query_crop=query_crop,
                reason="Vector store returned zero results.",
            )

        best_raw = results[0]
        best_raw_score = best_raw.similarity_score

        # ── Gate 1: Minimum viable threshold on best raw score ───────────
        if best_raw_score < self.minimum_viable_threshold:
            return GateResult(
                verdict=GateVerdict.EVIDENCE_ABSENT,
                best_score=best_raw_score,
                best_chunk=best_raw,
                sufficient_chunks=[],
                compatible_chunks=[],
                all_chunks=results,
                detected_query_crop=query_crop,
                reason=(
                    f"Best similarity score ({best_raw_score:.4f}) is below minimum viable "
                    f"threshold ({self.minimum_viable_threshold}). "
                    "No sufficiently relevant evidence found for this query."
                ),
            )

        # ── Gate 2: Multi-chunk Crop compatibility check ─────────────────
        promoted_rank_note = ""
        if query_crop is not None:
            # Look for any chunk that is crop-compatible with query_crop
            crop_compatible = [
                r for r in results
                if self._check_crop_mismatch(query_crop, r) is None
            ]
            if not crop_compatible:
                mismatch_desc = self._check_crop_mismatch(query_crop, best_raw) or (
                    f"Query crop '{query_crop}' does not match any retrieved chunk "
                    f"(top match was '{best_raw.crop_clean}')."
                )
                return GateResult(
                    verdict=GateVerdict.CROP_MISMATCH,
                    best_score=best_raw_score,
                    best_chunk=best_raw,
                    sufficient_chunks=[],
                    compatible_chunks=[],
                    all_chunks=results,
                    detected_query_crop=query_crop,
                    crop_mismatch_detail=mismatch_desc,
                    reason=(
                        f"Query crop '{query_crop}' does not match any retrieved chunk "
                        f"(top match was '{best_raw.crop_clean}' with score {best_raw_score:.4f}). "
                        "Providing cross-crop evidence could result in dangerous chemical misapplication."
                    ),
                )
            # Prioritize the best crop-compatible chunk
            best = crop_compatible[0]
            candidate_chunks = crop_compatible
            if best.chunk_id != best_raw.chunk_id:
                orig_rank = results.index(best) + 1
                promoted_rank_note = (
                    f" [Promoted compatible crop '{best.crop_clean}' from rank {orig_rank} "
                    f"(score {best.similarity_score:.4f}) over incompatible rank 1 '{best_raw.crop_clean}' "
                    f"(score {best_raw_score:.4f})]"
                )
        else:
            # QueryGuard did not detect a crop name: check if retrieved chunks are mentioned
            crop_mentioned = [
                r for r in results
                if self._is_chunk_crop_mentioned(query, r)
            ]
            if crop_mentioned:
                best = crop_mentioned[0]
                candidate_chunks = crop_mentioned
                if best.chunk_id != best_raw.chunk_id:
                    orig_rank = results.index(best) + 1
                    promoted_rank_note = (
                        f" [Promoted mentioned crop '{best.crop_clean}' from rank {orig_rank} "
                        f"(score {best.similarity_score:.4f}) over unmentioned rank 1 '{best_raw.crop_clean}']"
                    )
            elif self._has_advisory_intent(query):
                # Query has specific agronomic intent but unindexed/unmentioned crop evidence was retrieved
                return GateResult(
                    verdict=GateVerdict.CROP_MISMATCH,
                    best_score=best_raw_score,
                    best_chunk=best_raw,
                    sufficient_chunks=[],
                    compatible_chunks=[],
                    all_chunks=results,
                    detected_query_crop=None,
                    crop_mismatch_detail=(
                        f"Retrieved crop '{best_raw.crop_clean}' is not mentioned in your query. "
                        "Cannot safely apply crop-specific advice without establishing crop compatibility."
                    ),
                    reason=(
                        f"The query asks for agricultural advisory without specifying '{best_raw.crop_clean}', "
                        f"but nearest-neighbor search returned '{best_raw.crop_clean}' evidence. "
                        "To prevent dangerous chemical misapplication across different crops, the system will not guess."
                    ),
                )
            else:
                best = best_raw
                candidate_chunks = results

        best_score = best.similarity_score

        # ── Gate 3: Check minimum viable threshold on compatible chunk ───
        if best_score < self.minimum_viable_threshold:
            return GateResult(
                verdict=GateVerdict.EVIDENCE_ABSENT,
                best_score=best_score,
                best_chunk=best,
                sufficient_chunks=[],
                compatible_chunks=[],
                all_chunks=results,
                detected_query_crop=query_crop,
                reason=(
                    f"Best compatible similarity score ({best_score:.4f}) is below minimum viable "
                    f"threshold ({self.minimum_viable_threshold}). "
                    f"No sufficiently relevant evidence found for this query.{promoted_rank_note}"
                ),
            )

        # ── Gate 4: Sufficient vs weak evidence ───────────────────────────
        sufficient = [r for r in candidate_chunks if r.similarity_score >= self.sufficient_threshold]

        if sufficient:
            return GateResult(
                verdict=GateVerdict.EVIDENCE_SUFFICIENT,
                best_score=best_score,
                best_chunk=best,
                sufficient_chunks=sufficient,
                compatible_chunks=candidate_chunks,
                all_chunks=results,
                detected_query_crop=query_crop,
                reason=(
                    f"Found {len(sufficient)} chunk(s) exceeding sufficient threshold "
                    f"({self.sufficient_threshold}). Best score: {best_score:.4f}.{promoted_rank_note}"
                ),
            )
        else:
            # Score is above minimum viable but below sufficient → weak evidence
            return GateResult(
                verdict=GateVerdict.EVIDENCE_WEAK,
                best_score=best_score,
                best_chunk=best,
                sufficient_chunks=[],
                compatible_chunks=candidate_chunks,
                all_chunks=results,
                detected_query_crop=query_crop,
                reason=(
                    f"Best similarity score ({best_score:.4f}) is above minimum viable "
                    f"({self.minimum_viable_threshold}) but below sufficient threshold "
                    f"({self.sufficient_threshold}). Evidence will be presented with "
                    f"explicit uncertainty caveat.{promoted_rank_note}"
                ),
            )

    ALIASES: dict = {
        "paddy": {"rice", "dhan"},
        "rice": {"paddy", "dhan"},
        "maize": {"corn", "makka"},
        "corn": {"maize", "makka"},
        "arhar": {"tur", "pigeon pea"},
        "tur": {"arhar", "pigeon pea"},
        "moong": {"green gram", "mung"},
        "urad": {"black gram"},
        "bhindi": {"okra", "ladysfinger", "ladyfinger"},
        "okra": {"bhindi", "ladysfinger", "ladyfinger"},
        "groundnut": {"peanut", "pea nut", "mung phalli", "mungphalli"},
        "chilli": {"chillies", "chili", "mirch"},
        "chillies": {"chilli", "chili", "mirch"},
        "satawar": {"shatavari", "satavar"},
        "dragon fruit": {"pitaya", "pitahaya"},
        "cotton": {"kapas"},
        "kapas": {"cotton"},
        "pomegranate": {"anaar", "anar", "dalimb"},
        "guava": {"amrood", "amrud", "peru"},
        "mustard": {"sarson", "rai", "toria"},
        "tomato": {"tamatar"},
        "brinjal": {"eggplant", "baingan", "aubergine"},
        "wheat": {"gehun", "gehu"},
    }

    @classmethod
    def _check_crop_mismatch(cls, query_crop: str, result: RetrievalResult) -> Optional[str]:
        """
        Checks whether the retrieved chunk's crop is incompatible with the query crop.
        Returns a mismatch description string if there is a mismatch, else None.
        """
        q_norm = query_crop.lower().strip()
        r_norm = result.crop_clean.lower().strip()

        # Direct or alias match → no mismatch
        if q_norm == r_norm:
            return None
        if q_norm in r_norm or r_norm in q_norm:
            return None
        for alias_set in cls.ALIASES.values():
            if (q_norm in alias_set or any(a in q_norm for a in alias_set)) and (
                r_norm in alias_set or any(a in r_norm for a in alias_set)
            ):
                return None

        return (
            f"Query crop '{query_crop}' vs retrieved chunk crop '{result.crop_clean}'. "
            "These are different crops; advisory evidence may not be applicable."
        )

    @classmethod
    def _is_chunk_crop_mentioned(cls, query: str, result: RetrievalResult) -> bool:
        """
        Checks whether the retrieved chunk's crop (or any of its known aliases/tokens)
        is actually mentioned in the user's query text.
        """
        q_lower = query.lower()
        r_crop = result.crop_clean.lower()
        tokens = [t for t in re.findall(r"\b[a-z]{3,}\b", r_crop) if t not in {"crop", "bean", "peas"}]
        for token in tokens:
            if re.search(r"\b" + re.escape(token) + r"\b", q_lower):
                return True
        for alias_set in cls.ALIASES.values():
            if any(t in alias_set for t in tokens):
                if any(re.search(r"\b" + re.escape(alias) + r"\b", q_lower) for alias in alias_set):
                    return True
        return False

    @staticmethod
    def _has_advisory_intent(query: str) -> bool:
        """Checks if the query has specific agronomic treatment, disease, or cultivation intent."""
        keywords = [
            "disease", "pest", "insect", "spray", "pesticide", "fertilizer", "dose", "dosage",
            "treatment", "control", "fungus", "blight", "rot", "mildew", "rust", "grafting",
            "pruning", "planting", "cultivation", "seed", "variety", "weed", "chemical", "worm"
        ]
        q_lower = query.lower()
        return any(k in q_lower for k in keywords)
