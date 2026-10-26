# Agronomy Advisory Assistant: Hallucination-Guarded RAG Pipeline

**Project Title:** Agronomy Advisory Assistant with Hallucination-Guarded RAG & Local Dialect Support  
**Current Phase:** 5 — User-Facing Application (Streamlit Academic Demonstration Interface)  
**Target OS:** Windows 11 (PowerShell) / Python 3.12+  
**Repository:** [https://github.com/parthguntewar/agronomy-advisory-assistant](https://github.com/parthguntewar/agronomy-advisory-assistant)

---

## 1. Overview

This project builds a modular, safety-layered Retrieval-Augmented Generation (RAG) pipeline and academic demonstration interface for agricultural advisory evidence, grounded in historical Kisan Call Centre (KCC) records.

### Phase Summary

| Phase | Description | Status |
| :---: | :--- | :---: |
| **1** | Dataset investigation, selection, and sample preparation | ✅ Done |
| **2** | Data ingestion, quality filtering, and semantic retrieval pipeline | ✅ Done |
| **3** | Hallucination guard, relevance gating, and safe abstention pipeline | ✅ Done |
| **4** | Clean LLM provider interface (Mock, Ollama, API) & grounded answer generation | ✅ Done |
| **5** | User-facing Streamlit application with live pipeline diagnostics & evidence inspector | ✅ Done |

> **CRITICAL AGRONOMIC DISCLAIMER:**  
> This project indexes historical Kisan Call Centre (KCC) records for research and evaluation. It **does NOT prescribe or automate agricultural chemical advice**. Historical recommendations (dating back to 2014–2024) may contain restricted chemicals, outdated formulations, or regional inaccuracies. All agrochemical decisions require field validation by qualified extension specialists (KVK / SAU) and inspection of active product labels.

---

## 2. Five-Stage Guarded RAG Architecture

```
Farmer Query (Entered via Streamlit UI or CLI)
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
     │     ├─ OllamaProvider (local private inference on Windows, e.g. Llama 3.2)
     │     ├─ MockProvider (default offline, deterministic, zero-network)
     │     └─ APIProvider (cloud OpenAI-compatible endpoint with env API key)
     │   Graceful fallback to structured evidence template if runtime fails.
     ▼
[Stage 5] PostGenerationVerifier (structural safety check)
     │  ├─ BLOCKED_HALLUCINATION → Block answer if ungrounded chemical claims detected
     │  ├─ PASSED_WITH_WARNINGS → Present with caution warnings
     │  └─ PASSED              → Safe to present to farmer
     ▼
Streamlit UI Display:
  • Structured Advisory Response (with Cited Record IDs & Disclaimers)
  • Real-Time Pipeline Diagnostic Trace
  • Expandable Supporting Evidence Transcripts
```

**Scientific Safety Limitations (Not Eliminated by this System):**
- The system **reduces** hallucinations; it **does NOT eliminate** them.
- Small local models (1B–3B parameters) may occasionally fail strict negative constraints, which is why the Stage 5 post-generation verifier is mandatory.
- The verifier performs structural and token-level grounding checks, not full natural language entailment (NLI).
- Historical KCC records may reflect outdated pest control practices or chemical registrations.

---

## 3. Project Directory Structure

```text
ai project/
├── .env.example                        # Template for environment configuration
├── .gitignore                          # Excludes .env, .venv, local vector store, and caches
├── app.py                              # Phase 5: Streamlit user-facing application
├── DATASET_REPORT.md                   # Completed Phase 1 dataset report
├── README.md                           # This documentation
├── requirements.txt                    # Pinned package dependencies (including streamlit)
├── data/
│   ├── sample/                         # Validated source samples from Phase 1
│   │   ├── rag_knowledge_sample_30.jsonl
│   │   ├── evaluation_abstention_sample_20.jsonl
│   │   └── sample_metadata.json
│   ├── processed/                      # Output from reproducible ingestion
│   │   ├── knowledge_base.jsonl        # Cleaned, standardized knowledge chunks
│   │   └── ingestion_report.json       # Ingestion audit and rejection metrics
│   └── vector_store/                   # Persistent local ChromaDB storage (git-ignored)
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
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── providers.py                # LLMProvider abstraction (Mock, Ollama, API)
│   │   ├── answer_gen.py               # Grounded answer generator with LLM prompt
│   │   └── verifier.py                 # Post-generation structural safety verifier
│   └── ui/
│       ├── __init__.py
│       └── helpers.py                  # Diagnostic extraction & Ollama health check helpers
├── scripts/
│   ├── build_index.py                  # CLI script to embed and index chunks
│   ├── query_retrieval.py              # CLI tester for farmer questions
│   └── demo_guard.py                   # CLI demo showing safety scenarios & providers
└── tests/
    ├── __init__.py
    ├── test_retrieval_baseline.py      # 5 Phase 2 diagnostic retrieval baseline tests
    ├── test_hallucination_guard.py     # 14 Phase 3 guard & pipeline tests
    ├── test_generation_providers.py    # 16 Phase 4 provider & grounding safety tests
    └── test_ui_pipeline.py             # 10 Phase 5 UI helpers & failure-case integration tests
```

---

## 4. Setup & Launch Instructions (Windows PowerShell)

Open **Windows PowerShell** in the project directory: `c:\Users\parth\Desktop\ai project`.

### Step 1: Environment Setup
```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install all required dependencies (including Streamlit)
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 2: Knowledge Base Ingestion & Vector Index Preparation
Prepare the clean knowledge chunks and persistent ChromaDB index (one-time setup):
```powershell
# 1. Run reproducible ingestion & quality filtering
.\.venv\Scripts\python.exe -m src.ingestion.ingest

# 2. Build local ChromaDB vector store
.\.venv\Scripts\python.exe scripts/build_index.py --reset
```

### Step 3: Local Ollama Requirements
The application integrates with the local Ollama runtime on Windows (`http://127.0.0.1:11434`):
- Ensure Ollama is running in your taskbar, or launch it via Windows Start Menu.
- Model: `llama3.2:latest` (already present locally).
- If Ollama is offline or times out, the application automatically falls back to the deterministic safe template without throwing unhandled exceptions.

### Step 4: Launch the User-Facing Streamlit Application
Execute the following command in PowerShell:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Streamlit will launch locally in your browser at `http://localhost:8501`.

---

## 5. User Interface Features & Academic Evaluation Guide

The redesigned Streamlit interface (`app.py`) provides a professional, research-grade agricultural intelligence dashboard:

1. **Brand Navigation & System Status Header:**
   - Top banner displaying "Agronomy Advisory Assistant: Evidence-grounded agricultural intelligence".
   - Compact live status indicators for Ollama runtime connectivity and active ChromaDB vector index.
   - Restrained tab navigation across three core operational views:
     - **Advisory Workspace:** Primary query interface, guided benchmark scenarios, structured advice cards, citations, and safety notices.
     - **Evidence Explorer:** Detailed inspection of retrieved historical KCC records with metadata, similarity scores, and long-form transcripts.
     - **System Diagnostics:** Transparent audit trail of QueryGuard, AgronomyRetriever, RelevanceGate, AnswerGenerator, and PostGenerationVerifier. Stages that did not run display as `Not executed`.

2. **Main Advisory Workspace:**
   - Clean, high-contrast query input with actionable placeholder guidance.
   - Primary "Generate Advisory" action button.
   - Five standardized academic benchmark cards for reproducible evaluation:
     - **Paddy Stem Borer Control** (Normal in-domain query -> answered with grounded evidence).
     - **Ambiguous Symptoms (No Crop)** (Disease question missing crop name -> triggers pre-retrieval clarification).
     - **Weather & PM Kisan Inquiry** (Dynamic time-sensitive request -> triggers safe abstention with official portals).
     - **React/Node.js Technical Query** (Non-agricultural inquiry -> triggers off-topic safe abstention).
     - **Dragon Fruit Winter Grafting** (Unindexed crop -> rejected by post-retrieval Relevance Gate).
   - Execution progress indicator while the guarded RAG pipeline runs.

3. **Advisory Results & Evidence Attribution:**
   - Visual execution breadcrumbs highlighting the path traversed through the pipeline.
   - Structured result panel with clear status badges, verified advisory text, uncertainty notifications, and source citations.
   - High-contrast regulatory disclaimer detailing limitations of historical KCC records and the necessity of independent expert verification.

4. **Evidence Explorer:**
   - Expandable evidence cards displaying Crop, Location, Year, Record ID, and Cosine Similarity Score.
   - Clear distinction between historical call-centre transcripts and modern generated advice.

5. **System Diagnostics & Configuration:**
   - Live stage-by-stage audit showing inputs, verdicts, and outputs. Stages that did not execute are cleanly marked `Not executed`.
   - Compact configuration panel in the sidebar for provider selection (`ollama`, `mock`, `api`) and active safety threshold inspection.

---

## 6. Automated Test Suite Results

The project includes **45 comprehensive automated unit and integration tests** covering all phases. All tests run offline without network access or live API keys.

### Run All Tests:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Complete Test Results:
```text
Ran 45 tests in 3.554s

OK
```

### Test Breakdown by Phase:

| Test Module | Tests | Focus Areas | Status |
| :--- | :---: | :--- | :---: |
| `tests.test_retrieval_baseline` | 5 | In-domain retrieval, forced nearest-neighbor baseline, unindexed crops, dynamic query hazards, ambiguity hazards | ✅ 5/5 PASS |
| `tests.test_hallucination_guard` | 14 | Pre-retrieval QueryGuard, calibrated relevance gate, crop mismatch detection, dynamic abstention, structural verifier | ✅ 14/14 PASS |
| `tests.test_generation_providers` | 16 | MockProvider, OllamaProvider, APIProvider, configuration security, timeout handling, graceful fallback, ungrounded chemical rejection | ✅ 16/16 PASS |
| `tests.test_ui_pipeline` | 10 | Ollama service health check, genuine diagnostic extraction, bypassed stage handling, UI integration across all 5 evaluation scenarios | ✅ 10/10 PASS |
| **Total** | **45** | **Complete end-to-end safety, generation, and UI verification** | **✅ 45/45 PASS** |

---

## 7. Examples of Pipeline Decisions in the UI

### 1. In-Domain Supported Question
- **Query:** *"How do I control stem borer in my paddy crop?"*
- **UI Output:**
  - Status: `SUCCESS (Evidence-Grounded)`
  - Stages: `Stage 1: QueryGuard ➔ Stage 2: Retrieval ➔ Stage 3: Relevance Gate ➔ Stage 4: Grounded Generation ➔ Stage 5: Post-Verifier`
  - Cited Evidence: `kcc_chunk_kcc_00563633` (Score: `0.6500`)
  - Grounded text with regulatory disclaimer and attribution.

### 2. Missing Crop Name (Clarification Requested)
- **Query:** *"My plants have yellow spots and the leaves are curling. What spray should I use?"*
- **UI Output:**
  - Status: `CROP / SYMPTOM CLARIFICATION REQUIRED`
  - Stages: `Stage 1: QueryGuard ➔ Clarification Needed`
  - Prompt shown: *"Which crop is affected? (e.g. paddy, wheat, tomato)..."*
  - Vector retrieval and LLM generation: **0 calls (Safely bypassed)**.

### 3. Dynamic Weather Query (Safe Abstention)
- **Query:** *"Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?"*
- **UI Output:**
  - Status: `REAL-TIME DATA REQUIRED (Safe Abstention)`
  - Stages: `Stage 1: QueryGuard ➔ Abstained (Real-Time)`
  - Directs farmer to official live portals: `imd.gov.in`, `agmarknet.gov.in`, `pmkisan.gov.in`.

### 4. Off-Topic Technical Query
- **Query:** *"How do I configure a React frontend router in Node.js?"*
- **UI Output:**
  - Status: `QUERY NOT RELATED TO AGRICULTURE (Safe Abstention)`
  - Stages: `Stage 1: QueryGuard ➔ Abstained (Off-Topic)`
  - Explains assistant's agricultural scope.

### 5. Fabricated Chemical Claim Blocked
- **Trigger:** Generated text recommends an active ingredient or dosage not found in retrieved chunks.
- **UI Output:**
  - Status: `BLOCKED BY SAFETY VERIFIER: Ungrounded Claims Detected`
  - Verifier Verdict: `BLOCKED_HALLUCINATION`
  - Response text is stripped to prevent farmer from acting on ungrounded chemical advice.
