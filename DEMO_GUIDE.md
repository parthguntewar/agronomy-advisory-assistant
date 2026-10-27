# Agronomy Advisory Assistant: Demo Guide & Runtime Verification

**Project Title:** Agronomy Advisory Assistant with Hallucination-Guarded RAG & Local Dialect Support  
**Target Environment:** Windows 11 (PowerShell) / Python 3.12+  
**Repository:** `parthguntewar/agronomy-advisory-assistant` (Branch: `main`)  
**Status:** Phase 7 Demo-Ready (Academic Verification & Evaluation Dashboard)

---

## 1. Project Architecture & 5-Stage Execution Flow

The system employs a strict, 5-stage defensive pipeline designed to prioritize grounding, transparency, and safe abstention over ungrounded generation:

```text
Farmer Query (Entered in Streamlit UI or CLI)
     │
     ▼
[Stage 1] QueryGuard (Pre-Retrieval Gate)
     ├─ ABSTAIN_OFFTOPIC   ➔ Out-of-scope non-agri query (0 retrieval calls, polite refusal)
     ├─ ABSTAIN_DYNAMIC    ➔ Real-time/weather/scheme query (0 retrieval calls, official portals)
     ├─ CLARIFY_CROP       ➔ Symptoms without crop name (0 retrieval calls, clarification prompt)
     └─ PROCEED            ➔ Valid agri query with identified crop -> continue
     │
     ▼
[Stage 2] AgronomyRetriever (Semantic Vector Search)
     Dense semantic search via ChromaDB using sentence-transformers/all-MiniLM-L6-v2 (top_k=3).
     │
     ▼
[Stage 3] RelevanceGate (Relevance, Similarity & Multi-Chunk Crop Compatibility)
     ├─ EVIDENCE_ABSENT    ➔ Best similarity score < 0.30 (abstain, no generation)
     ├─ CROP_MISMATCH      ➔ None of top-k chunks match query crop (abstain, prevent cross-crop advice)
     ├─ EVIDENCE_WEAK      ➔ 0.30 <= Score < 0.50 (forward with uncertainty flag)
     └─ EVIDENCE_SUFFICIENT➔ Score >= 0.50 (forward with high-confidence flag)
     │
     ▼
[Stage 4] AnswerGenerator (Evidence-Grounded Synthesis & Pluggable Providers)
     Formats ONLY approved compatible chunks into system prompt.
     Providers: MockProvider (default offline), OllamaProvider (local Llama 3.2), APIProvider.
     Failure handling: Automatic fallback to deterministic template with [FALLBACK NOTICE].
     │
     ▼
[Stage 5] PostGenerationVerifier (Structural Safety & Grounding Verification)
     Checks: Citation validity (Check 1), Regulatory disclaimer (Check 2),
             Ungrounded chemical active ingredients & dosages (Check 3),
             Crop consistency (Check 4), Contradictory evidence across chunks (Check 5).
     Verdicts: PASSED, PASSED_WITH_WARNINGS, or BLOCKED_HALLUCINATION.
     │
     ▼
Final User Presentation (Streamlit Dashboard)
     Grounded Advisory Text | Execution Breadcrumbs | Supporting Records | Diagnostics
```

---

## 2. Prerequisites & Environment Setup

### System Prerequisites
- **Operating System:** Windows 10/11 with PowerShell.
- **Python:** Python 3.12 (in virtual environment `.venv`).
- **RAM:** Minimum 8 GB recommended for local vector store and embeddings.
- **Ollama (Optional for local LLM inference):** Ollama daemon running on `http://127.0.0.1:11434` with model `llama3.2:latest`. If Ollama is offline or not installed, the application automatically runs using the deterministic `mock` provider or fallback template with zero network dependencies.

### Clean Setup Steps
```powershell
# 1. Open PowerShell and navigate to project root
cd "C:\Users\parth\Desktop\ai project"

# 2. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify installed dependencies
.\.venv\Scripts\pip.exe list
```

### Knowledge Base Ingestion & Index Verification (One-Time Setup)
```powershell
# Ingest raw historical sample data and filter for quality
.\.venv\Scripts\python.exe -m src.ingestion.ingest

# Build/verify ChromaDB local vector store (indexes 26 quality-filtered chunks)
.\.venv\Scripts\python.exe scripts/build_index.py --reset
```

---

## 3. Launch Commands

### A. Launch User-Facing Streamlit Application
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
- Open browser at `http://localhost:8501`.
- The dashboard automatically detects whether local Ollama is active or offline and displays operational status in the top-right header.

### B. Run Complete Automated Test Suite (71 Tests)
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### C. Run Golden Retrieval Benchmark Evaluation
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py
```

---

## 4. Safe Demonstration Scenarios & Expected Outcomes

The Streamlit interface provides one-click benchmark scenario buttons under the query box for live demonstrations:

| # | Demo Query / Scenario | Pipeline Path & Stages Executed | Expected UI Presentation & Status |
| :-: | :--- | :--- | :--- |
| **1** | **Paddy Stem Borer Control**<br>`"How do I control stem borer in my paddy crop?"` | `QueryGuard (PROCEED)` ➔ `Retrieval` ➔ `RelevanceGate (SUFFICIENT)` ➔ `Generation (Grounded)` ➔ `Verifier (PASSED)` | **Green Badge:** `Verified Evidence-Grounded Advisory`<br>Cites record `kcc_chunk_kcc_00563633` (Paddy, Score > 0.60). Grounded pesticide recommendations with regulatory disclaimer. |
| **2** | **Ambiguous Disease (No Crop)**<br>`"My plants have yellow spots and the leaves are curling. What spray should I use?"` | `QueryGuard (CLARIFY_CROP)`<br>*(Retrieval, Gating, and Generation are safely BYPASSED)* | **Blue Badge:** `Clarification Required`<br>Displays polite clarification card asking farmer to identify the crop and affected plant parts. 0 vector chunks retrieved. |
| **3** | **Dynamic Weather & PM Kisan Inquiry**<br>`"Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?"` | `QueryGuard (ABSTAIN_DYNAMIC)`<br>*(Retrieval, Gating, and Generation are safely BYPASSED)* | **Blue Badge:** `Real-Time Data Required (Abstained)`<br>Refuses static advice and routes farmer directly to live official portals (`imd.gov.in`, `pmkisan.gov.in`). |
| **4** | **Non-Agricultural Query**<br>`"How do I configure a React frontend router in Node.js?"` | `QueryGuard (ABSTAIN_OFFTOPIC)`<br>*(Retrieval, Gating, and Generation are safely BYPASSED)* | **Blue Badge:** `Out-of-Scope Query (Abstained)`<br>Politely explains assistant's agricultural scope. 0 chunks retrieved. |
| **5** | **Unsupported Crop Query (Quinoa / Dragon Fruit)**<br>`"What is the recommended grafting method for Brazilian Dragon Fruit in winter?"` | `QueryGuard (PROCEED, Dragon Fruit)` ➔ `Retrieval (Nearest: Ber, 0.4247)` ➔ `RelevanceGate (CROP_MISMATCH)` | **Yellow Badge:** `No Sufficient Evidence Found`<br>Gate detects mismatch between query crop (Dragon Fruit) and retrieved chunks (Ber). Safely refuses advice rather than borrowing treatment from unrelated fruit. |
| **6** | **Supported Crop with Weak Evidence**<br>`"What medicine to spray for worms in satawar roots and leaves?"` | `QueryGuard (PROCEED, Satawar)` ➔ `Retrieval (0.4787)` ➔ `RelevanceGate (EVIDENCE_WEAK)` ➔ `Verifier (PASSED_WITH_WARNINGS)` | **Amber Badge:** `Low-Confidence Advisory (Uncertain Evidence)`<br>Displays prominent uncertainty notice urging independent SAU/KVK extension officer validation. |
| **7** | **Simulated Provider Timeout / Outage**<br>*(Select `ollama` when service is stopped, or set invalid API key)* | `QueryGuard` ➔ `Retrieval` ➔ `RelevanceGate` ➔ `Generation (Fallback)` ➔ `Verifier` | **Fallback Notice:** Clear `[FALLBACK NOTICE]` banner indicating deterministic template fallback was used. Pipeline does not crash. |

---

## 5. How Safety, Verification & Fallback Mechanisms Work

1. **Pre-Retrieval Guarding (`QueryGuard`):**
   - Heuristically inspects vocabulary for non-agricultural tokens, dynamic time-sensitive markers (weather, market prices, subsidies), and crop entity names.
   - Saves computational cost and prevents forced nearest-neighbor matching for queries that cannot be answered from a static historical corpus.

2. **Relevance Gating & Multi-Chunk Crop Promotion (`RelevanceGate`):**
   - Examines top-3 retrieved chunks. If chunk 1 is an unrelated crop with a slightly higher embedding score (e.g. Green Gram at 0.4908), but chunk 2 is an exact crop match (e.g. Pomegranate at 0.4901), the gate promotes chunk 2.
   - Enforces a dual-threshold band: `>= 0.50` is sufficient evidence; `0.30 to 0.49` is weak evidence requiring explicit caveats; `< 0.30` triggers automatic abstention.

3. **Grounded Generation & Citation Enforcement (`AnswerGenerator`):**
   - Only `compatible_chunks` approved by the RelevanceGate are supplied to the generation prompt. Unrelated crop chunks are permanently filtered out.
   - If the provider fails or times out, the generator synthesizes a deterministic response directly from the approved evidence text and flags `is_fallback = True`.

4. **Structural Post-Generation Verification (`PostGenerationVerifier`):**
   - Extracts all chemical names, active ingredients, formulations (EC, WP, SC), and dosages from the generated answer.
   - Cross-checks each extracted entity against the approved evidence chunks. Any ungrounded active ingredient triggers `BLOCKED_HALLUCINATION`, stripping the advisory text to prevent hazardous farmer action.
   - Checks that all cited chunk IDs exist in the retrieved evidence, preventing citation hallucination.
   - Detects disjoint chemical recommendations across multiple chunks and emits a `CONTRADICTORY_EVIDENCE` warning.

---

## 6. Known Limitations & Research Boundaries

To maintain rigorous scientific accuracy, the following system boundaries and limitations must be clearly understood:

1. **Small Sample Knowledge Base:**
   - The active local vector store contains **26 indexed chunks** derived from a 30-record historical Kisan Call Centre sample.
   - It is intended for academic proof-of-concept and safety verification, not production advisory across all Indian crops or agro-climatic zones.
2. **Historical Advice Disclaimer:**
   - Source KCC records date from 2014–2024. Certain recommended chemical formulations (e.g. Carbofuran, Monocrotophos, or Chlorpyrifos formulations) may have been banned, restricted, or superseded by modern ICAR/CIBRC regulations.
   - The system is an evidence retrieval demonstration, **not an active regulatory authority**.
3. **Lexical & Regex Chemical Verification vs Full Registry:**
   - The Stage 5 verifier utilizes regex pattern matching and an agronomic active ingredient lexicon.
   - While effective for common insecticides, fungicides, and herbicides, it does not replace a comprehensive, live CIBRC (Central Insecticides Board & Registration Committee) database lookup.
4. **Structural Grounding vs Full Cross-Encoder NLI:**
   - The verifier validates entity presence, token containment, chunk ID integrity, and multi-chunk contradiction.
   - Full semantic entailment / Natural Language Inference (NLI) for complex natural language reasoning requires an external cross-encoder model.
5. **No Production Readiness Claim:**
   - Automated testing verifies that known error paths abstain safely. It does **not** prove universal agronomic safety across arbitrary natural language inputs. Live deployment requires formal field trials and agricultural university certification.
