import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Sample File Paths
SAMPLE_RAG_KB = SAMPLE_DIR / "rag_knowledge_sample_30.jsonl"
SAMPLE_EVAL_ABSTENTION = SAMPLE_DIR / "evaluation_abstention_sample_20.jsonl"
SAMPLE_METADATA = SAMPLE_DIR / "sample_metadata.json"

# Processed File Paths
PROCESSED_KB = PROCESSED_DIR / "knowledge_base.jsonl"
INGESTION_REPORT = PROCESSED_DIR / "ingestion_report.json"

# Ingestion Filter Rules
ALLOWED_ADVISORY_DOMAINS = {
    "plant_protection",
    "nutrient_management",
    "weed_management",
    "cultural_practices",
    "varieties",
}

EXCLUDED_ADVISORY_DOMAINS = {
    "weather",              # Time-sensitive, requires dynamic weather API
    "market_information",  # Volatile prices, requires real-time mandi API
    "government_scheme",   # Expired application dates, requires official portal
    "training_or_contact", # Historical call center contacts
    "other",
}

ALLOWED_RAG_MODES = {
    "use_as_evidence",
    "use_with_caution",
}

DISALLOWED_RAG_MODES = {
    "historical_only",
    "exclude",
}

DISALLOWED_QC_SEVERITIES = {
    "high",  # Exclude severe translation corruption, extreme unit mismatches
}

# Embedding & Retrieval Settings
# Lightweight English embedding model (~80 MB, 384 dimensions)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3
SIMILARITY_THRESHOLD_BASELINE = 0.40

# Attribution & Safety Notices
SOURCE_ATTRIBUTION = (
    "Kisan Call Centre (KCC) Transcripts of Farmers Queries and Answers, "
    "Ministry of Agriculture and Farmers Welfare, Government of India (Data.gov.in / AIKosh)."
)

CURATED_CORPUS_ATTRIBUTION = (
    "KCC-Krishi RAG/SFT Advisory Corpus, "
    "URAV Advanced Learning Systems Private Limited."
)

AGRONOMIC_SAFETY_NOTICE = (
    "NOTICE: This document contains historical Kisan Call Centre advisory evidence. "
    "It is NOT verified current legal or agronomic advice. "
    "Chemical dosages, restricted pesticides, and local practices must be independently "
    "verified with local agricultural extension officers (KVK / State Agricultural University) "
    "and authorized product labels."
)

# LLM Provider Configuration
# Options: 'mock' (default, offline), 'ollama' (local inference), 'api' (OpenAI-compatible)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()

# Ollama Settings (Local inference)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest").strip()
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))

# Cloud API Settings (OpenAI-compatible endpoints)
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))
