"""
UI Helper functions and diagnostics extraction for the Agronomy Advisory Assistant.

Provides data transformation and diagnostic inspection for the Streamlit UI,
ensuring genuine pipeline results are surfaced without fabricating values.
Strictly free of emojis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

from src.pipeline import GuardedRAGPipeline, PipelineResult
from src.generation.answer_gen import AnswerGenerator, GenerationStatus
from src.generation.providers import get_llm_provider, LLMProvider
from src.guard.query_guard import GuardAction
from src.guard.relevance_gate import GateVerdict
from src.generation.verifier import VerificationStatus
from src import config


def check_ollama_service(base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Checks whether the local Ollama service is reachable and lists available models.
    Returns status dict without throwing unhandled exceptions.
    """
    url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
    try:
        r = requests.get(f"{url}/api/tags", timeout=2.0)
        if r.status_code == 200:
            models_data = r.json().get("models", [])
            model_names = [m.get("name", "") for m in models_data]
            return {
                "available": True,
                "url": url,
                "models": model_names,
                "error": None,
            }
        return {
            "available": False,
            "url": url,
            "models": [],
            "error": f"HTTP {r.status_code}",
        }
    except Exception as e:
        return {
            "available": False,
            "url": url,
            "models": [],
            "error": str(e),
        }


def format_stage_badge(stage_name: str) -> str:
    """Formats a pipeline stage into a clean display label without emojis."""
    mapping = {
        "QueryGuard": "Stage 1: QueryGuard",
        "Retrieval": "Stage 2: Vector Retrieval",
        "RelevanceGate": "Stage 3: Relevance Gate",
        "Generation": "Stage 4: Grounded Generation",
        "Verification": "Stage 5: Post-Verifier",
        "Abstained(OffTopic)": "Abstained (Off-Topic)",
        "Abstained(Dynamic)": "Abstained (Real-Time)",
        "Clarification(Requested)": "Clarification Required",
    }
    return mapping.get(stage_name, stage_name)


def extract_pipeline_diagnostics(result: PipelineResult) -> Dict[str, Any]:
    """
    Safely extracts diagnostic fields from a PipelineResult.
    Any unexposed or unreached stage returns 'Not exposed by pipeline'
    or 'Not executed' rather than inventing values.
    """
    NOT_EXPOSED = "Not exposed by pipeline"
    NOT_REACHED = "Not executed"

    # Stage 1: QueryGuard
    guard_info = {
        "executed": "QueryGuard" in result.pipeline_stages,
        "action": result.guard_decision.action.value if result.guard_decision else NOT_EXPOSED,
        "reason": result.guard_decision.reason if result.guard_decision else NOT_EXPOSED,
        "detected_crop": result.guard_decision.detected_crop if result.guard_decision and result.guard_decision.detected_crop else "None detected",
        "is_agricultural": str(result.guard_decision.is_agricultural) if result.guard_decision else NOT_EXPOSED,
        "is_dynamic": str(result.guard_decision.has_dynamic_request) if result.guard_decision else NOT_EXPOSED,
        "needs_clarification": str(result.guard_decision.needs_clarification) if result.guard_decision else NOT_EXPOSED,
        "clarification_prompt": result.guard_decision.clarification_prompt if result.guard_decision and result.guard_decision.clarification_prompt else None,
    }

    # Stage 2 & 3: Retrieval & RelevanceGate
    if result.gate_result:
        is_mismatch = (
            result.gate_result.verdict == GateVerdict.CROP_MISMATCH
            or bool(result.gate_result.crop_mismatch_detail)
        )
        gate_info = {
            "executed": True,
            "verdict": result.gate_result.verdict.value,
            "best_score": f"{result.gate_result.best_score:.4f}" if result.gate_result.best_score is not None else NOT_EXPOSED,
            "reason": result.gate_result.reason or NOT_EXPOSED,
            "retrieved_chunk_count": len(result.retrieval_results),
            "sufficient_chunk_count": len(result.gate_result.sufficient_chunks) if result.gate_result.sufficient_chunks else 0,
            "crop_mismatch_detected": is_mismatch,
            "mismatch_detail": result.gate_result.crop_mismatch_detail or "None",
        }
    else:
        gate_info = {
            "executed": False,
            "verdict": NOT_REACHED,
            "best_score": NOT_REACHED,
            "reason": NOT_REACHED,
            "retrieved_chunk_count": len(result.retrieval_results),
            "sufficient_chunk_count": 0,
            "crop_mismatch_detected": False,
            "mismatch_detail": NOT_REACHED,
        }

    # Stage 4: Generation
    if result.generated_answer:
        gen_info = {
            "executed": True,
            "status": result.generated_answer.status.value,
            "source_ids_cited": result.generated_answer.source_ids or [],
            "crops_cited": result.generated_answer.crops_cited or [],
            "has_uncertainty_note": result.generated_answer.uncertainty_note is not None,
            "uncertainty_note": result.generated_answer.uncertainty_note or None,
            "abstention_reason": result.generated_answer.abstention_reason or None,
            "provider_used": getattr(result.generated_answer, "provider_used", "unknown"),
            "is_fallback": getattr(result.generated_answer, "is_fallback", False),
            "fallback_reason": getattr(result.generated_answer, "fallback_reason", None),
        }
    else:
        gen_info = {
            "executed": False,
            "status": NOT_REACHED,
            "source_ids_cited": [],
            "crops_cited": [],
            "has_uncertainty_note": False,
            "uncertainty_note": None,
            "abstention_reason": None,
            "provider_used": NOT_REACHED,
            "is_fallback": False,
            "fallback_reason": None,
        }

    # Stage 5: Verification
    if result.verification_result:
        verif_info = {
            "executed": True,
            "status": result.verification_result.status.value,
            "reason": result.verification_result.reason or NOT_EXPOSED,
            "issues": result.verification_result.issues or [],
            "warnings": result.verification_result.warnings or [],
            "is_blocked": result.verification_result.is_blocked,
            "is_safe_to_present": result.verification_result.is_safe_to_present,
        }
    else:
        verif_info = {
            "executed": False,
            "status": NOT_REACHED,
            "reason": NOT_REACHED,
            "issues": [],
            "warnings": [],
            "is_blocked": False,
            "is_safe_to_present": result.was_answered,
        }

    return {
        "query": result.query,
        "stages": result.pipeline_stages,
        "was_answered": result.was_answered,
        "was_blocked": result.was_blocked,
        "guard": guard_info,
        "gate": gate_info,
        "generation": gen_info,
        "verification": verif_info,
    }
