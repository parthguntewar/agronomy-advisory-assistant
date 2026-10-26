"""
CLI Demonstration: Hallucination-Guarded RAG Pipeline with Modular LLM Providers.

Demonstrates 4 scenarios across safety layers:
  1. A supported crop-management question that retrieves and cites evidence.
  2. A non-agricultural question that safely abstains.
  3. An ambiguous disease question without a crop name (requests clarification).
  4. A time-sensitive weather/scheme query that abstains (no live data).

Usage:
  .venv\\Scripts\\python.exe scripts/demo_guard.py
  .venv\\Scripts\\python.exe scripts/demo_guard.py --provider ollama
  .venv\\Scripts\\python.exe scripts/demo_guard.py --provider api
  .venv\\Scripts\\python.exe scripts/demo_guard.py --live --provider mock
"""

import sys
import argparse
import logging
import io
from pathlib import Path
from unittest.mock import MagicMock

# Force UTF-8 output on Windows to handle Unicode characters in banners
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import GuardedRAGPipeline
from src.generation.answer_gen import AnswerGenerator
from src.generation.providers import get_llm_provider
from src.retrieval.vector_store import RetrievalResult
from src.config import SOURCE_ATTRIBUTION, AGRONOMIC_SAFETY_NOTICE

logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs for clean demo output

DEMO_BANNER = """
================================================================================
     AGRONOMY ADVISORY ASSISTANT -- Phase 4: Guarded RAG with Modular LLM
     DEMONSTRATION: Safety Pipeline with Pluggable Generation Providers
================================================================================
"""

SEPARATOR = "\n" + "-" * 80 + "\n"


def make_demo_result(
    chunk_id: str = "kcc_chunk_demo_001",
    crop_clean: str = "Paddy (Dhan)",
    similarity_score: float = 0.65,
    expert_advisory: str = (
        "For stem borer (Scirpophaga incertulas) in paddy, apply chlorpyrifos 2 ml/litre "
        "of water by foliar spray. Drain field water before spraying and refill after 48 hours. "
        "For severe infestation, apply granular carbofuran 3G at 25 kg/hectare."
    ),
    farmer_query: str = "How to control stem borer in paddy crop?",
    state: str = "Odisha",
    district: str = "Puri",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        similarity_score=similarity_score,
        crop=crop_clean,
        crop_clean=crop_clean,
        state=state,
        district=district,
        advisory_domain="plant_protection",
        query_type="pest_control",
        original_year=2022,
        farmer_query=farmer_query,
        expert_advisory=expert_advisory,
        qc_severity="none",
        qc_confidence="high",
        qc_flags=[],
        rag_use_mode="use_as_evidence",
        source_attribution=SOURCE_ATTRIBUTION,
        safety_notice=AGRONOMIC_SAFETY_NOTICE,
        retrieval_text=(
            f"Crop: {crop_clean}. Topic: pest_control. "
            f"Location: {state}, {district}. "
            f"Question: {farmer_query}\nAdvisory Evidence: {expert_advisory}"
        ),
    )


def build_pipeline(use_live_index: bool = False, provider_name: str = "mock") -> GuardedRAGPipeline:
    """Builds a GuardedRAGPipeline with the chosen provider."""
    provider = get_llm_provider(provider_name)
    answer_gen = AnswerGenerator(provider=provider)

    if use_live_index:
        try:
            return GuardedRAGPipeline(answer_generator=answer_gen)
        except Exception as e:
            print(f"[NOTE] Live index unavailable ({e}). Falling back to mock retrieval.")

    # Mock retrieval setup (always available, no ChromaDB required)
    pipeline = GuardedRAGPipeline.__new__(GuardedRAGPipeline)
    from src.guard.query_guard import QueryGuard
    from src.guard.relevance_gate import RelevanceGate
    from src.generation.verifier import PostGenerationVerifier

    pipeline.query_guard = QueryGuard()
    pipeline.relevance_gate = RelevanceGate()
    pipeline.answer_gen = answer_gen
    pipeline.verifier = PostGenerationVerifier()
    pipeline.top_k = 3

    mock_retriever = MagicMock()
    # Scenario 1 returns an approved high-similarity Paddy result
    mock_retriever.retrieve.return_value = [make_demo_result()]
    pipeline.retriever = mock_retriever
    return pipeline


def run_scenario(pipeline: GuardedRAGPipeline, scenario_num: int, label: str, query: str):
    print(f"\n{'=' * 80}")
    print(f"SCENARIO {scenario_num}: {label}")
    print(f"{'=' * 80}")
    print(f"Farmer Query: \"{query}\"")
    print()
    result = pipeline.answer(query)
    print(result.to_display_text())


def main():
    parser = argparse.ArgumentParser(
        description="Agronomy Advisory Assistant -- Guarded RAG Demo"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real ChromaDB index instead of mock retrieval",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "ollama", "api"],
        help="LLM generation provider: mock (default), ollama (local), or api (cloud)",
    )
    args = parser.parse_args()

    print(DEMO_BANNER)
    print(f"Active Provider: {args.provider.upper()} | Index Mode: {'LIVE ChromaDB' if args.live else 'MOCK Retrieval'}")

    pipeline = build_pipeline(use_live_index=args.live, provider_name=args.provider)

    # -- Scenario 1: Supported question with retrieved evidence -------------
    run_scenario(
        pipeline, 1,
        "Supported Agricultural Question (Evidence Grounded)",
        "How do I control stem borer in my paddy crop?",
    )

    # -- Scenario 2: Non-agricultural -> safe abstention --------------------
    run_scenario(
        pipeline, 2,
        "Non-Agricultural Question (Safe Abstention)",
        "How do I configure a React frontend router in Node.js?",
    )

    # -- Scenario 3: Ambiguous disease without crop -> clarification ---------
    run_scenario(
        pipeline, 3,
        "Ambiguous Disease Query Without Crop (Clarification Requested)",
        "My plants have yellow spots and the leaves are curling. What spray should I use?",
    )

    # -- Scenario 4: Weather / time-sensitive -> abstention -----------------
    run_scenario(
        pipeline, 4,
        "Time-Sensitive / Dynamic Data Request (Abstention)",
        "Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?",
    )

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(
        "\nKey safety and provider properties demonstrated:\n"
        "  - Scenario 1: Advisory cited with source record ID, crop, location, and year.\n"
        "  - Scenario 2: Off-topic queries blocked before any retrieval.\n"
        "  - Scenario 3: Disease queries without crop name ask for clarification.\n"
        "  - Scenario 4: Real-time requests abstain and refer to authoritative sources.\n"
        "\nProvider commands:\n"
        "  .venv\\Scripts\\python.exe scripts/demo_guard.py --provider mock\n"
        "  .venv\\Scripts\\python.exe scripts/demo_guard.py --provider ollama\n"
        "  .venv\\Scripts\\python.exe scripts/demo_guard.py --provider api\n"
    )


if __name__ == "__main__":
    main()
