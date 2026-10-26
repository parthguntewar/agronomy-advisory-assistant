# Agronomy Advisory Assistant: Hallucination-Guarded RAG Pipeline

**Project Title:** Agronomy Advisory Assistant with Hallucination-Guarded RAG & Local Dialect Support  
**Current Phase:** 4 — Language Model Integration (Modular LLM Providers & Grounded Synthesis)  
**Target OS:** Windows 11 (PowerShell) / Python 3.12+

---

## 1. Overview

This project builds a modular, safety-layered Retrieval-Augmented Generation (RAG) pipeline for agricultural advisory evidence, grounded in historical Kisan Call Centre (KCC) records.

### Phase Summary

| Phase | Description | Status |
| :---: | :--- | :---: |
| **1** | Dataset investigation, selection, and sample preparation | ✅ Done |
| **2** | Data ingestion, quality filtering, and semantic retrieval pipeline | ✅ Done |
| **3** | Hallucination guard, relevance gating, and safe abstention pipeline | ✅ Done |
| **4** | Clean LLM provider interface (Mock, Ollama, API) & grounded answer generation | ✅ Done |
| **5** | Interactive Chatbot UI with Local Dialect Support | 🔜 Next |

> **CRITICAL AGRONOMIC DISCLAIMER:**  
> This project indexes historical Kisan Call Centre (KCC) records for research and evaluation. It **does NOT prescribe or automate agricultural chemical advice**. Historical recommendations (dating back to 2014–2024) may contain restricted chemicals, outdated formulations, or regional inaccuracies. All agrochemical decisions require field validation by qualified extension specialists (KVK / SAU) and inspection of active product labels.

---

## 2. Guarded RAG Architecture with Modular Generation

```
Farmer Query
     │
     ▼
[Stage 1] QueryGuard (pre-retrieval)
     │  ├─ ABSTAIN_OFFTOPIC  → Polite rejection (no retrieval, no LLM call)
     │  ├─ ABSTAIN_DYNAMIC   → Abstain + refer to live data sources (IMD, Agmarknet)
     │  ├─ CLARIFY_CROP      → Ask farmer for crop name & symptoms (no retrieval)
     │  └─ PROCEED           → Continue to retrieval
     ▼
[Stage 2] AgronomyRetriever (semantic vector search)
     │   ChromaDB + sentence-transformers/all-MiniLM-L6-v2
     ▼
[Stage 3] RelevanceGate (post-retrieval threshold gate)
     │  ├─ EVIDENCE_ABSENT   → Abstain (score < 0.30; no LLM call)
     │  ├─ CROP_MISMATCH     → Block (retrieved crop ≠ query crop; no LLM call)
     │  ├─ EVIDENCE_WEAK     → Forward with low-confidence flag
     │  └─ EVIDENCE_SUFFICIENT → Forward with high-confidence flag
     ▼
[Stage 4] AnswerGenerator & LLM Provider Layer
     │   Strict grounding prompt (cite record IDs, no fabricated chemicals)
     │   Delegates to pluggable LLMProvider:
     │     ├─ MockProvider (default offline, deterministic, zero-network)
     │     ├─ OllamaProvider (local private inference, e.g. Llama 3.2 on Windows)
     │     └─ APIProvider (cloud OpenAI-compatible endpoint with env API key)
     │   Graceful fallback to structured evidence template if runtime fails.
     ▼
[Stage 5] PostGenerationVerifier (structural safety check)
     │  ├─ BLOCKED_HALLUCINATION → Block answer if ungrounded chemical claims detected
     │  ├─ PASSED_WITH_WARNINGS → Present with caution warnings
     │  └─ PASSED              → Safe to present to farmer
     ▼
Structured Advisory Response (GeneratedAnswer with source IDs & disclaimers)
```

**Safety limitations (not eliminated by this system):**
- The system **reduces** hallucinations; it **does NOT eliminate** them.
- Small local models (1B–3B parameters) may occasionally fail strict negative constraints, which is why the Stage 5 post-generation verifier is mandatory.
- The verifier performs structural and token-level grounding checks, not full natural language entailment (NLI).
- Historical KCC records may reflect outdated pest control practices or chemical registrations.

---

## 3. Environment & LLM Integration Comparison

### System Hardware & Environment Profile
- **OS:** Windows 11 Home (x64)
- **Python:** 3.12.6
- **Installed Runtimes:** Ollama (`ollama.exe`) is installed locally at `C:\Users\parth\AppData\Local\Programs\Ollama\ollama.exe` with `llama3.2:latest` (2.0 GB) present.
- **Hardware:** Intel UHD Graphics 620 + NVIDIA GeForce MX250 (2 GB VRAM), 16 GB Total RAM.

### Options Evaluated

| Dimension | Option A: Local Ollama (`llama3.2:latest`) | Option B: Cloud API (`gpt-4o-mini` / Groq) | Option C: Mock Provider |
| :--- | :--- | :--- | :--- |
| **Cost** | **$0 (100% Free)** | Pay-per-token; requires credit card / API credits | **$0** (Offline) |
| **Privacy** | **100% Local**: No farmer query leaves laptop | Prompts transmitted to external cloud servers | **100% Local** |
| **Windows Compatibility** | Native Windows executable running on `127.0.0.1:11434` | Full HTTPS compatibility via standard HTTP client | Native Python |
| **Hardware Fit** | 2.0 GB quantized model runs on CPU / hybrid offload | Zero local compute/VRAM overhead | Minimal CPU/RAM |
| **Grounding Adherence** | Good on basic extraction; occasional drift under complex rules | High instruction adherence and strict citation accuracy | Deterministic template |
| **Setup Complexity** | Zero extra download needed (already installed) | Requires signing up and generating API key | Zero setup |

### Architectural Recommendation
We implement a **Clean Provider Interface** (`LLMProvider`) supporting all three options:
1. **MockProvider (Default):** Used for automated tests, CI, and zero-network environments. Guarantees 100% offline, reproducible testing with no external dependencies.
2. **OllamaProvider (Recommended for Local Privacy):** Uses the existing local Ollama installation (`http://127.0.0.1:11434`) with `llama3.2:latest` without downloading any new models or incurring costs.
3. **APIProvider (Recommended for Highest Instruction Quality):** Supports any OpenAI-compatible cloud endpoint via `LLM_API_KEY` without hardcoding credentials or provider-specific SDKs.

---

## 4. Project Directory Structure

```text
ai project/
├── .env.example                        # Template for environment configuration
├── .gitignore                          # Excludes .env, .venv, and local caches
├── data/
│   ├── sample/                         # Validated source samples from Phase 1
│   │   ├── rag_knowledge_sample_30.jsonl
│   │   ├── evaluation_abstention_sample_20.jsonl
│   │   └── sample_metadata.json
│   ├── processed/                      # Output from reproducible ingestion
│   │   ├── knowledge_base.jsonl        # Cleaned, standardized knowledge chunks
│   │   └── ingestion_report.json       # Ingestion audit and rejection metrics
│   └── vector_store/                   # Persistent local ChromaDB storage
├── src/
│   ├── __init__.py
│   ├── config.py                       # Central paths, provider configs, and safety rules
│   ├── pipeline.py                     # GuardedRAGPipeline orchestrator
│   ├── ingestion/
│   │   ├── schema.py                   # Chunk schema and validation rules
│   │   └── ingest.py                   # Ingestion pipeline & rejection logger
│   ├── retrieval/
│   │   ├── embedder.py                 # SentenceTransformer wrapper
│   │   ├── vector_store.py             # ChromaDB client & RetrievalResult model
│   │   └── retriever.py                # High-level semantic retriever & CLI formatter
│   ├── guard/
│   │   ├── query_guard.py              # Pre-retrieval query classifier (GuardAction)
│   │   └── relevance_gate.py           # Post-retrieval threshold gate (GateVerdict)
│   └── generation/
│       ├── __init__.py
│       ├── providers.py                # LLMProvider abstraction (Mock, Ollama, API)
│       ├── answer_gen.py               # Grounded answer generator with LLM prompt
│       └── verifier.py                 # Post-generation structural safety verifier
├── scripts/
│   ├── build_index.py                  # CLI script to embed and index chunks
│   ├── query_retrieval.py              # CLI tester for farmer questions
│   └── demo_guard.py                   # CLI demo showing safety scenarios & providers
├── tests/
│   ├── __init__.py
│   ├── test_retrieval_baseline.py      # 5 Phase 2 diagnostic retrieval baseline tests
│   ├── test_hallucination_guard.py     # 14 Phase 3 guard & pipeline tests
│   └── test_generation_providers.py    # 16 Phase 4 provider & grounding safety tests
├── DATASET_REPORT.md                   # Completed Phase 1 dataset report
├── requirements.txt                    # Pinned package dependencies
└── README.md                           # This documentation
```

---

## 5. Setup & Configuration (Windows PowerShell)

Open **Windows PowerShell** in the project directory: `c:\Users\parth\Desktop\ai project`.

### Step 1: Environment Setup
```powershell
# Create virtual environment if not already present
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 2: Configure LLM Provider
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Edit `.env` to select your provider:

#### Option A: Offline Mock (Default — No setup, no network)
```env
LLM_PROVIDER=mock
```

#### Option B: Local Ollama (100% Private, Zero Cost)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_TIMEOUT=30.0
```
*Note: Verify that Ollama is running by opening the Ollama desktop app or running `ollama list`.*

#### Option C: Cloud API (OpenAI-compatible)
```env
LLM_PROVIDER=api
LLM_API_KEY=sk-your-actual-api-key-here
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT=30.0
```
*Security notice: Never commit `.env` to version control. Keys are never logged.*

---

## 6. Execution Workflow

### Step 3: Run Ingestion and Vector Indexing (One-Time Setup)
```powershell
# 1. Ingest raw samples and filter out corrupted records
.\.venv\Scripts\python.exe -m src.ingestion.ingest

# 2. Build local ChromaDB vector store
.\.venv\Scripts\python.exe scripts/build_index.py --reset
```

### Step 4: Run the Interactive Demo
Run the 4 safety scenarios using the default mock provider:
```powershell
.\.venv\Scripts\python.exe scripts/demo_guard.py
```

Run with local Ollama:
```powershell
.\.venv\Scripts\python.exe scripts/demo_guard.py --provider ollama
```

Run with live ChromaDB retrieval:
```powershell
.\.venv\Scripts\python.exe scripts/demo_guard.py --live --provider mock
```

---

## 7. Automated Test Suite Results

The project includes 35 comprehensive automated unit tests covering all safety and generation stages. All tests run offline using mocks without external network calls.

### Run All Tests:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Test Suite Breakdown:

| Test Module | Tests | Scenarios Tested | Status |
| :--- | :---: | :--- | :---: |
| `tests.test_retrieval_baseline` | 5 | Normal in-domain, out-of-domain nearest neighbor, unindexed crop, time-sensitive queries, ambiguous queries | ✅ 5/5 PASS |
| `tests.test_hallucination_guard` | 14 | Pre-retrieval QueryGuard, RelevanceGate thresholds, crop mismatch detection, dynamic abstention, structural verifier | ✅ 14/14 PASS |
| `tests.test_generation_providers` | 16 | MockProvider, OllamaProvider, APIProvider, missing API keys, timeout handling, graceful fallback, grounded citations, ungrounded chemical rejection | ✅ 16/16 PASS |
| **Total** | **35** | **Complete end-to-end safety & generation verification** | **✅ 35/35 PASS** |

---

## 8. Examples of Pipeline Decisions

### Example 1: Grounded Advisory Response (Supported Query)
- **Input Query:** *"How do I control stem borer in my paddy crop?"*
- **Pipeline Stage:** `QueryGuard -> Retrieval -> RelevanceGate -> Generation -> Verification`
- **Gate Verdict:** `EVIDENCE_SUFFICIENT` (Score: 0.6500)
- **Generation Output:**
  ```text
  ADVISORY RESPONSE (Evidence-Grounded)
  ----------------------------------------------------------------------
  Based on historical Kisan Call Centre (KCC) records, here is the available advisory evidence:
  For stem borer (Scirpophaga incertulas) in paddy, apply chlorpyrifos 2 ml/litre of water by foliar spray...
  
  EVIDENCE SOURCES:
    [1] Record ID: kcc_chunk_demo_001 | Similarity: 0.6500
    Crops in evidence: Paddy (Dhan)
    Attribution: Kisan Call Centre (KCC) Transcripts, Ministry of Agriculture & Farmers Welfare, GoI.
  
  NOTICE: This document contains historical Kisan Call Centre advisory evidence.
  It is NOT verified current legal or agronomic advice. Chemical dosages must be independently
  verified with local agricultural extension officers (KVK / SAU).
  ```

### Example 2: Non-Agricultural Query (Pre-Retrieval Safe Abstention)
- **Input Query:** *"How do I configure a React frontend router in Node.js?"*
- **Pipeline Stage:** `QueryGuard -> Abstained(OffTopic)`
- **Retrieval & LLM Calls:** **0 (Intercepted before any vector search)**
- **Response:**
  ```text
  UNABLE TO ANSWER: Query Not Related to Agriculture
  Reason: I am an agronomy advisory assistant specializing in crop management, pest control,
  nutrient management, and farming practices. Your question does not appear to be related to agriculture.
  ```

### Example 3: Ambiguous Query Without Crop Name (Clarification Requested)
- **Input Query:** *"My plants have yellow spots and the leaves are curling. What spray should I use?"*
- **Pipeline Stage:** `QueryGuard -> Clarification(Requested)`
- **Response:**
  ```text
  To give you accurate advice, could you please tell me:
  1. Which crop is affected? (e.g., paddy, wheat, tomato)
  2. What symptoms do you see? (e.g., yellow spots, wilting, holes in leaves)
  Different crops require different treatments, and prescribing chemicals without knowing the crop could cause harm.
  ```

### Example 4: Real-Time / Weather Query (Safe Abstention with Referral)
- **Input Query:** *"Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?"*
- **Pipeline Stage:** `QueryGuard -> Abstained(Dynamic)`
- **Response:**
  ```text
  UNABLE TO ANSWER: Real-Time Data Required
  Reason: This query requires real-time information (weather forecasts, current market prices,
  or government scheme deadlines) that this system does not have access to. Please check:
  • Weather: India Meteorological Department (imd.gov.in)
  • Market prices: Agmarknet (agmarknet.gov.in)
  • PM Kisan: pmkisan.gov.in
  ```

### Example 5: Crop Mismatch (Post-Retrieval Relevance Gate Block)
- **Input Query:** *"How to treat rust disease in wheat?"* (Retrieved chunk: *Paddy stem borer*)
- **Gate Verdict:** `CROP_MISMATCH`
- **Action:** Generation blocked; no cross-crop chemical advice is presented.

### Example 6: Fabricated Chemical Claim (Post-Generation Verifier Block)
- **Generated Text Claim:** *"Apply chlorpyrifos 5 ml/litre for rust control."*
- **Evidence Chunks:** Contain only *tebuconazole 1 ml/litre*.
- **Verifier Action:** `BLOCKED_HALLUCINATION`
- **Output:** The answer text is stripped, and the response is safely converted to an abstention citing `HALLUCINATION_RISK`.

---

## 9. Remaining Risks & Phase 5 Roadmap

1. **Agronomic Validity of Historical Records:** Historical KCC advice (2014–2024) may contain chemicals that are now banned or restricted by the Central Insecticides Board & Registration Committee (CIB&RC).
2. **Local Dialect Translation Nuances:** Hindi and regional language queries often use colloquial terms for pests and diseases (e.g., *sundi*, *mahla*, *geedan*). Multilingual embeddings and local dialect normalization are required.
3. **Natural Language Entailment (NLI):** The current verifier uses regex-based chemical and dosage matching. Integrating a formal NLI cross-encoder will provide deeper semantic entailment checks.
4. **Interactive Chatbot Interface:** Phase 5 will implement a Streamlit chatbot UI supporting multi-turn dialogues, clarification workflows, and evidence inspection panels.
