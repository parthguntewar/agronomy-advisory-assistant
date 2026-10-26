"""
Pre-Retrieval Query Guard for the Agronomy Advisory Assistant.

Performs lightweight rule-based + keyword classification of incoming farmer queries
BEFORE any retrieval happens. The guard is designed to be modular so that a stronger
ML-based classifier can replace the heuristic layer later without changing the interface.

GuardAction categories:
  PROCEED          - Query is agricultural and has enough context; proceed to retrieval.
  ABSTAIN_DYNAMIC  - Query requires live/real-time data (weather, market prices, scheme deadlines).
  ABSTAIN_OFFTOPIC - Query is not related to agriculture.
  CLARIFY_CROP     - Disease/pest/spray query is missing the crop name.
  CLARIFY_SYMPTOMS - Treatment query lacks symptoms; generic "what spray" style.

Design decision: We intentionally keep this as a SIGNAL, not a hard filter.
The caller (pipeline) decides what to do with GuardAction; this module only classifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Set

# ---------------------------------------------------------------------------
# Taxonomy of known agricultural crop terms (lower-cased substrings)
# This is intentionally not exhaustive; it is a bootstrap list expanded
# from the KCC-Krishi KB plus common Indian agricultural terminology.
# ---------------------------------------------------------------------------
_AGRI_CROP_TERMS: Set[str] = {
    # Cereals
    "paddy", "rice", "wheat", "maize", "corn", "jowar", "bajra", "ragi",
    "sorghum", "barley", "millet",
    # Pulses
    "moong", "urad", "arhar", "tur", "pigeon pea", "chickpea", "chana",
    "lentil", "masur", "cowpea", "moth bean", "green gram", "black gram",
    # Oilseeds
    "groundnut", "soybean", "mustard", "sunflower", "sesame", "linseed",
    "safflower", "castor", "rapeseed",
    # Vegetables & Fruits
    "tomato", "potato", "onion", "brinjal", "eggplant", "cabbage",
    "cauliflower", "okra", "bhindi", "pea", "beans", "cucumber",
    "gourd", "pumpkin", "mango", "banana", "papaya", "guava", "pomegranate",
    "grapes", "orange", "lemon", "lime", "apple", "pear", "jackfruit",
    "strawberry", "watermelon", "melon",
    # Cash Crops
    "cotton", "sugarcane", "jute", "tobacco", "hemp", "rubber",
    # Spices & Plantation
    "turmeric", "ginger", "cardamom", "pepper", "chilli", "coriander",
    "cumin", "fenugreek", "garlic", "tea", "coffee", "coconut",
    # Misc Indian crops from KB sample
    "satawar", "ber", "amla", "drumstick", "moringa", "cluster bean",
    "bottle gourd", "ridge gourd", "bitter gourd", "aonla",
}

# ---------------------------------------------------------------------------
# Indicator phrases for dynamic/live-data domains that we cannot answer
# ---------------------------------------------------------------------------
_DYNAMIC_INDICATORS: List[str] = [
    # Weather
    r"\b(weather|rain(fall)?|temperature|forecast|humidity|wind|storm|cyclone)\b",
    r"\b(will it rain|will there be rain|mausam|barish)\b",
    r"\b(tomorrow|today|next week|this week)\b.*\b(rain|weather|temp)\b",
    r"\b(rain|weather|forecast).*\b(tomorrow|today|next week)\b",
    # Market prices
    r"\b(market price|mandi price|msp|minimum support price|market rate|current price)\b",
    r"\bprice of\b.*\b(today|yesterday|this week|now|current)\b",
    r"\b(onion price|wheat price|cotton price|paddy price).*\b(today|current|mandi)\b",
    # Government scheme deadlines / dates
    r"\b(pm kisan|pm-kisan|pmksy|pradhan mantri|government scheme|yojana)\b.*\b(date|deadline|last date|when|installment|kist)\b",
    r"\b(installment|kist|subsidy|apply online|application deadline)\b.*\b(scheme|yojana|kisan|govt)\b",
    r"\b(registration deadline|last date to apply|scheme closed|scheme open)\b",
]

# ---------------------------------------------------------------------------
# Indicator phrases for disease/pest/spray queries
# ---------------------------------------------------------------------------
_DISEASE_PEST_INDICATORS: List[str] = [
    r"\b(disease|pest|insect|fungus|fungi|virus|bacteria|nematode|worm|borer|aphid|mite|thrips|whitefly|caterpillar|larva)\b",
    r"\b(spray|pesticide|insecticide|fungicide|herbicide|weedicide|chemical|dawa|davai)\b",
    r"\b(yellow(ing)?|spot(s)?|curl(ing)?|wilt(ing)?|rot|blight|rust|mosaic|lesion|necrosis)\b",
    r"\b(attack|infestation|damage|infected|affected|sick|dying)\b",
    r"\b(control|treat|prevent|manage|application|dose|dosage|recommendation)\b.*\b(disease|pest|insect|spray|chemical)\b",
]

# ---------------------------------------------------------------------------
# Crop name extraction: simple heuristic regex patterns
# ---------------------------------------------------------------------------
_CROP_PATTERNS: List[re.Pattern] = [
    # "in my wheat crop" / "in my paddy field"
    re.compile(r"\bin\s+(?:my\s+)?(\w[\w\s]{1,20}?)\s+(?:crop|field|plant|farm|garden|orchard)\b", re.I),
    # "for wheat" / "for paddy crop" — only for known crop names
    re.compile(r"\bfor\s+(wheat|rice|paddy|cotton|tomato|potato|onion|maize|soybean|groundnut|sugarcane|brinjal|okra|mustard|tea|coffee|mango|banana|turmeric|ginger|chilli|satawar|ber|arhar|moong|urad|jowar|bajra|ragi|sorghum|barley|lentil|cowpea|sunflower|sesame|castor|guava|pomegranate|grapes|orange|lemon|apple|watermelon|cumin|coriander|fenugreek|garlic|cardamom|pepper|rubber|jute|tobacco|hemp|coconut|moringa|drumstick|amla|aonla)\b", re.I),
    # Stand-alone crop names at word boundaries — only known crop vocabulary
    re.compile(r"\b(wheat|rice|paddy|cotton|tomato|potato|onion|maize|soybean|groundnut|sugarcane|brinjal|okra|mustard|tea|coffee|mango|banana|turmeric|ginger|chilli|satawar|ber|arhar|moong|urad|jowar|bajra|ragi|sorghum|barley|lentil|cowpea|sunflower|sesame|castor|guava|pomegranate|grapes|orange|lemon|apple|watermelon|cumin|coriander|fenugreek|garlic|cardamom|pepper|rubber|jute|tobacco|hemp|coconut|moringa|drumstick|amla|aonla)\b", re.I),
]

# ---------------------------------------------------------------------------
# Non-agricultural topic keywords (heuristic — not exhaustive)
# ---------------------------------------------------------------------------
_NON_AGRI_INDICATORS: List[str] = [
    r"\b(javascript|python script|react|node\.js|html|css|sql|database|api|frontend|backend|server|docker|kubernetes|cloud|aws|azure|gcp|machine learning|deep learning|neural network|chatgpt|llm|openai)\b",
    r"\b(physics|chemistry|mathematics|algebra|calculus|history|geography|economics|politics|election|cricket|football|movie|recipe|cooking|music|travel|hotel|flight|ticket|visa|passport)\b",
    r"\b(stock market|share market|nse|bse|sensex|nifty|cryptocurrency|bitcoin|forex)\b",
    r"\b(medical|doctor|hospital|surgery|medicine|drug|tablet|injection|disease|treatment)\b(?!.*\b(crop|plant|field|farm|agricultural|pest|insect|spray)\b)",
]


class GuardAction(str, Enum):
    """Action directive returned by the QueryGuard."""
    PROCEED = "proceed"
    ABSTAIN_DYNAMIC = "abstain_dynamic"
    ABSTAIN_OFFTOPIC = "abstain_offtopic"
    CLARIFY_CROP = "clarify_crop"
    CLARIFY_SYMPTOMS = "clarify_symptoms"


@dataclass
class GuardDecision:
    """
    Structured decision produced by the QueryGuard for a given query.
    
    Attributes:
        action: The directive indicating what the pipeline should do next.
        detected_crop: Crop name extracted from the query, if any.
        has_disease_pest_context: Whether the query mentions disease/pest indicators.
        has_dynamic_request: Whether the query requests live data.
        is_agricultural: Whether the query is classified as agricultural.
        clarification_prompt: If action is CLARIFY_*, the suggested prompt to show the farmer.
        reason: Human-readable explanation of the guard decision.
        confidence: 'high', 'medium', or 'low' — qualitative, not calibrated probability.
    """
    action: GuardAction
    detected_crop: Optional[str] = None
    has_disease_pest_context: bool = False
    has_dynamic_request: bool = False
    is_agricultural: bool = True
    clarification_prompt: Optional[str] = None
    reason: str = ""
    confidence: str = "medium"
    triggered_patterns: List[str] = field(default_factory=list)

    @property
    def should_proceed(self) -> bool:
        return self.action == GuardAction.PROCEED

    @property
    def should_abstain(self) -> bool:
        return self.action in (GuardAction.ABSTAIN_DYNAMIC, GuardAction.ABSTAIN_OFFTOPIC)

    @property
    def needs_clarification(self) -> bool:
        return self.action in (GuardAction.CLARIFY_CROP, GuardAction.CLARIFY_SYMPTOMS)


class QueryGuard:
    """
    Modular pre-retrieval query classification guard.
    
    Applies a tiered heuristic pipeline to classify incoming queries into one
    of five GuardAction categories before any vector search is invoked.
    The heuristics are intentionally simple and conservative: when uncertain,
    the guard defaults to PROCEED and lets downstream evidence-gating decide.
    
    This class is designed for easy replacement: swap _classify() with an
    ML classifier by subclassing and overriding classify().
    """

    def __init__(self) -> None:
        # Pre-compile all regex patterns for performance
        self._dynamic_patterns = [re.compile(p, re.I) for p in _DYNAMIC_INDICATORS]
        self._disease_patterns = [re.compile(p, re.I) for p in _DISEASE_PEST_INDICATORS]
        self._non_agri_patterns = [re.compile(p, re.I) for p in _NON_AGRI_INDICATORS]

    def classify(self, query: str) -> GuardDecision:
        """
        Classifies the input query and returns a GuardDecision.
        
        Args:
            query: The raw natural language question from the farmer.
            
        Returns:
            GuardDecision with action, detected context, and optional clarification.
        """
        q = query.strip()
        if not q:
            return GuardDecision(
                action=GuardAction.ABSTAIN_OFFTOPIC,
                is_agricultural=False,
                reason="Empty query.",
                confidence="high",
            )
        q_lower = q.lower()

        # ── Step 1: Check for dynamic/live-data requests ──────────────────
        triggered_dynamic = self._match_patterns(q_lower, self._dynamic_patterns)
        if triggered_dynamic:
            return GuardDecision(
                action=GuardAction.ABSTAIN_DYNAMIC,
                has_dynamic_request=True,
                reason=(
                    "Query requests real-time or time-sensitive data (weather, "
                    "market prices, or government scheme deadlines). "
                    "This assistant does not have access to live data feeds."
                ),
                confidence="high",
                triggered_patterns=triggered_dynamic,
            )

        # ── Step 2: Check for non-agricultural off-topic content ──────────
        triggered_offtopic = self._match_patterns(q_lower, self._non_agri_patterns)
        if triggered_offtopic:
            # Secondary check: does the query also contain clear agricultural terms?
            if not self._has_agricultural_context(q_lower):
                return GuardDecision(
                    action=GuardAction.ABSTAIN_OFFTOPIC,
                    is_agricultural=False,
                    reason=(
                        "Query does not appear to be related to agriculture, "
                        "crops, farming, or agronomic advisory topics."
                    ),
                    confidence="high",
                    triggered_patterns=triggered_offtopic,
                )

        # ── Step 3: Detect disease/pest context ───────────────────────────
        triggered_disease = self._match_patterns(q_lower, self._disease_patterns)
        has_disease_context = len(triggered_disease) > 0

        # ── Step 4: Attempt crop extraction ──────────────────────────────
        detected_crop = self._extract_crop(q)

        # ── Step 5: Apply clarification rules for disease/pest queries ────
        if has_disease_context and detected_crop is None:
            # Ambiguous chemical/spray/disease query with no crop mentioned
            clarification = (
                "To give you accurate advice, could you please tell me:\n"
                "1. Which crop is affected? (e.g., paddy, wheat, tomato)\n"
                "2. What symptoms do you see? (e.g., yellow spots, wilting, holes in leaves)\n\n"
                "Different crops require different treatments, and prescribing chemicals "
                "without knowing the crop could cause harm."
            )
            return GuardDecision(
                action=GuardAction.CLARIFY_CROP,
                detected_crop=None,
                has_disease_pest_context=True,
                reason=(
                    "Query mentions disease/pest/spray context but does not "
                    "identify the crop. Cannot safely retrieve crop-specific advice."
                ),
                clarification_prompt=clarification,
                confidence="high",
                triggered_patterns=triggered_disease,
            )

        # ── Step 6: Default — agricultural query with sufficient context ──
        is_agri = self._has_agricultural_context(q_lower) or has_disease_context or bool(detected_crop)
        return GuardDecision(
            action=GuardAction.PROCEED,
            detected_crop=detected_crop,
            has_disease_pest_context=has_disease_context,
            is_agricultural=is_agri,
            reason=(
                "Query is agricultural with sufficient context for retrieval. "
                f"Detected crop: {detected_crop or 'not specified'}."
            ),
            confidence="medium",
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _match_patterns(text: str, patterns: List[re.Pattern]) -> List[str]:
        """Returns list of matched pattern strings for any patterns that fire."""
        matched = []
        for pat in patterns:
            m = pat.search(text)
            if m:
                matched.append(m.group(0))
        return matched

    @staticmethod
    def _has_agricultural_context(q_lower: str) -> bool:
        """Broad check: does the query contain any agricultural vocabulary?"""
        agri_broad = [
            "crop", "farm", "field", "seed", "fertilizer", "pesticide",
            "irrigation", "harvest", "sow", "yield", "soil", "manure",
            "compost", "kisan", "krishi", "kheti", "fasal",
        ]
        for term in agri_broad:
            if term in q_lower:
                return True
        # Check against crop name vocabulary
        for crop_term in _AGRI_CROP_TERMS:
            if crop_term in q_lower:
                return True
        return False

    @staticmethod
    def _extract_crop(query: str) -> Optional[str]:
        """
        Attempts to extract a crop name from the query using ordered regex patterns.
        Returns the first successful match, or None.
        """
        for pat in _CROP_PATTERNS:
            m = pat.search(query)
            if m:
                # The matched crop name is in the last non-None group
                groups = [g for g in m.groups() if g is not None]
                if groups:
                    return groups[-1].strip().title()
        return None
