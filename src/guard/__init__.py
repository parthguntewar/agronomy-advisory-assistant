"""
Guard package: Pre-retrieval query classification and post-retrieval relevance gating.
Provides the hallucination-mitigation safety layer for the Agronomy Advisory Assistant.
"""
from src.guard.query_guard import QueryGuard, GuardDecision, GuardAction

__all__ = ["QueryGuard", "GuardDecision", "GuardAction"]
