"""
Data models and validation logic for KCC knowledge base ingestion.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import re

from src.config import (
    ALLOWED_ADVISORY_DOMAINS,
    EXCLUDED_ADVISORY_DOMAINS,
    ALLOWED_RAG_MODES,
    DISALLOWED_RAG_MODES,
    DISALLOWED_QC_SEVERITIES,
    SOURCE_ATTRIBUTION,
    CURATED_CORPUS_ATTRIBUTION,
    AGRONOMIC_SAFETY_NOTICE
)

@dataclass
class KnowledgeBaseChunk:
    """Standardized schema for a retrieved agronomy knowledge chunk."""
    chunk_id: str
    source_example_id: str
    crop: str
    crop_clean: str
    state: str
    district: str
    block: str
    advisory_domain: str
    query_type: str
    farmer_query: str
    expert_advisory: str
    retrieval_text: str
    rag_use_mode: str
    qc_severity: str
    qc_confidence: str
    qc_flags: List[str]
    original_year: Optional[int]
    provenance: Dict[str, str] = field(default_factory=lambda: {
        "authoritative_source": SOURCE_ATTRIBUTION,
        "curated_corpus": CURATED_CORPUS_ATTRIBUTION,
    })
    safety_notice: str = AGRONOMIC_SAFETY_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseChunk":
        return cls(**data)


def validate_and_filter_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[KnowledgeBaseChunk]]:
    """
    Validates a raw or sampled KCC-Krishi record against agronomic RAG quality standards.
    
    Returns:
        (is_valid: bool, rejection_reason: Optional[str], chunk: Optional[KnowledgeBaseChunk])
    """
    # Check top-level format: supports both nested format and flat metadata format
    meta = record.get("metadata", record)
    example_id = meta.get("example_id") or record.get("example_id", "unknown_id")
    
    # 1. Check Crop validity
    crop = str(meta.get("crop", "")).strip()
    if not crop:
        return False, "missing_crop", None
    if crop.isdigit():
        return False, f"numeric_crop_id_{crop}", None
    if crop.lower() in {"others", "other", "null", "none"}:
        return False, f"unspecified_generic_crop_{crop}", None

    # 2. Check Advisory Domain validity
    domain = str(meta.get("advisory_domain", "")).strip().lower()
    if not domain:
        return False, "missing_advisory_domain", None
    if domain in EXCLUDED_ADVISORY_DOMAINS:
        return False, f"unsuitable_domain_{domain}", None
    if domain not in ALLOWED_ADVISORY_DOMAINS:
        return False, f"unknown_disallowed_domain_{domain}", None

    # 3. Check RAG Use Mode
    rag_mode = str(meta.get("rag_use_mode", "")).strip().lower()
    if rag_mode in DISALLOWED_RAG_MODES:
        return False, f"disallowed_rag_mode_{rag_mode}", None
    if rag_mode not in ALLOWED_RAG_MODES:
        return False, f"unsupported_rag_mode_{rag_mode}", None

    # 4. Check QC Severity
    qc_severity = str(meta.get("qc_severity", "none")).strip().lower()
    if qc_severity in DISALLOWED_QC_SEVERITIES:
        return False, f"high_qc_severity_corruption", None

    # 5. Extract Farmer Query and Expert Advisory text
    farmer_query = ""
    expert_advisory = ""

    # Check for direct fields
    if "farmer_query" in record and "expert_advisory" in record:
        farmer_query = str(record["farmer_query"]).strip()
        expert_advisory = str(record["expert_advisory"]).strip()
    elif "user_prompt" in record and "expert_response" in record:
        farmer_query = str(record["user_prompt"]).strip()
        expert_advisory = str(record["expert_response"]).strip()
    elif "messages" in record and isinstance(record["messages"], list):
        for msg in record["messages"]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                farmer_query = content.strip()
            elif role == "assistant":
                expert_advisory = content.strip()
    elif "question_en" in record and "answer_en" in record:
        farmer_query = str(record["question_en"]).strip()
        expert_advisory = str(record["answer_en"]).strip()

    if not farmer_query:
        return False, "missing_farmer_query", None
    if not expert_advisory:
        return False, "missing_expert_advisory", None

    # Clean crop text
    crop_clean = meta.get("clean_crop_for_prompt") or crop
    
    # Process QC flags
    qc_flags_raw = meta.get("qc_flags", [])
    if isinstance(qc_flags_raw, str):
        qc_flags = [f.strip() for f in qc_flags_raw.split("|") if f.strip()]
    elif isinstance(qc_flags_raw, list):
        qc_flags = [str(f).strip() for f in qc_flags_raw]
    else:
        qc_flags = []

    # Parse year if present
    year = meta.get("year")
    if year is None:
        created_at = str(meta.get("created_at", ""))
        match = re.search(r"20\d\d", created_at)
        if match:
            year = int(match.group(0))

    # Clean query text for retrieval embedding
    # Format: Crop: <crop>. Query: <query>. Advisory: <advisory>
    retrieval_text = (
        f"Crop: {crop_clean}. "
        f"Topic: {meta.get('query_type', domain)}. "
        f"Location: {meta.get('state', 'India')}, {meta.get('district', '')}. "
        f"Question: {farmer_query}\n"
        f"Advisory Evidence: {expert_advisory}"
    )

    chunk = KnowledgeBaseChunk(
        chunk_id=f"kcc_chunk_{example_id}",
        source_example_id=str(example_id),
        crop=crop,
        crop_clean=str(crop_clean),
        state=str(meta.get("state", "UNKNOWN")),
        district=str(meta.get("district", "UNKNOWN")),
        block=str(meta.get("block", "")),
        advisory_domain=domain,
        query_type=str(meta.get("query_type", "general")),
        farmer_query=farmer_query,
        expert_advisory=expert_advisory,
        retrieval_text=retrieval_text,
        rag_use_mode=rag_mode,
        qc_severity=qc_severity,
        qc_confidence=str(meta.get("qc_confidence", "medium")),
        qc_flags=qc_flags,
        original_year=year,
    )

    return True, None, chunk
