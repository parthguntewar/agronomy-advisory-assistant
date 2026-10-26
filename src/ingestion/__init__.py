"""
Ingestion module for validating, filtering, and preparing KCC agronomy chunks.
"""
from src.ingestion.schema import KnowledgeBaseChunk, validate_and_filter_record

__all__ = ["KnowledgeBaseChunk", "validate_and_filter_record"]
