# Agronomy Advisory Assistant: Academic Viva Preparation & Demo Guide

**Project Title:** Agronomy Advisory Assistant with Hallucination-Guarded RAG & Local Dialect Support  
**Student Reference Document:** Technical Viva Voce, Architecture Deep-Dive, & Presentation Checklist  
**Repository:** `parthguntewar/agronomy-advisory-assistant`  
**Target Environment:** Windows 11 / Python 3.12 / Streamlit / Ollama (`llama3.2:latest`)

---

## 1. Quick Pitch & Problem Statement

### 1.1 The Core Problem
General-purpose Large Language Models (LLMs) hallucinate plausible-sounding but factually lethal advice when answering agricultural questions. In agronomy, hallucinating a pesticide active ingredient, mixing incompatible chemicals, recommending a dosage that is ten times too high, or prescribing a treatment intended for cotton to a vegetable farmer can destroy an entire harvest, poison groundwater, or cause acute human toxicity.

### 1.2 The Failure of "Naive RAG"
Standard Retrieval-Augmented Generation (RAG) fails in agricultural domains because:
1. **Forced Nearest Neighbors:** Dense vector search always returns the mathematically closest vector, even when the query is completely outside the knowledge base (e.g., asking about Quinoa when only Groundnut is indexed).
2. **Top-1 Crop Blindness:** A high-similarity match for an unrelated crop will drown out a lower-ranked chunk for the actual crop.
3. **Weak Evidence Amplification:** Naive RAG treats a low-similarity match (e.g., cosine similarity 0.35) as factual gospel and instructs the LLM to answer with full confidence.
4. **No Post-Generation Chemical Verification:** Standard RAG pipelines pass LLM output straight to the user without checking whether mentioned active ingredients or dosages actually existed in the retrieved source text.

### 1.3 Our Solution: The 5-Stage Hallucination-Guarded RAG Pipeline
We built a safety-first, 5-stage defensive RAG pipeline that enforces pre-retrieval scope gating, multi-chunk crop compatibility ranking, calibrated dual-threshold relevance gating, evidence-exclusive synthesis, and post-generation structural chemical and citation verification.

---

## 2. Technical System Deep-Dive

### 2.1 Technology Stack & Architecture
- **User Interface:** Streamlit (`app.py`), designed as an academic evaluation dashboard with real-time stage execution breadcrumbs, pipeline diagnostics, and an evidence explorer.
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, lightweight ~80 MB CPU footprint).
- **Vector Database:** ChromaDB (`ChromaAgronomyStore`) using persistent on-disk storage with cosine distance metric (`hnsw:space: cosine`).
- **Generation Layer:** Pluggable `LLMProvider` abstraction supporting:
  - `OllamaProvider`: Local private inference running Llama 3.2 on Windows (`http://127.0.0.1:11434`).
  - `MockProvider`: Deterministic offline template generator for testing without network dependencies.
  - `APIProvider`: Cloud-compatible OpenAI format for external endpoints.
- **Fallback Architecture:** Automatic fallback to deterministic evidence templates (`[FALLBACK NOTICE]`) if the LLM times out or crashes.

### 2.2 End-to-End Pipeline Workflow
```text
Farmer Query
     │
     ▼
[Stage 1: QueryGuard] (src/guard/query_guard.py)
  ├─ Off-topic check (non-agri keywords) ──➔ ABSTAIN_OFFTOPIC (0 vector calls)
  ├─ Dynamic check (weather, mandi prices)──➔ ABSTAIN_DYNAMIC (0 vector calls)
  ├─ Crop identification & symptoms check ─➔ CLARIFY_CROP (0 vector calls)
  └─ Identified crop + agronomic question ──➔ PROCEED
     │
     ▼
[Stage 2: AgronomyRetriever] (src/retrieval/retriever.py)
  Embed query via all-MiniLM-L6-v2 ➔ Query ChromaDB for top_k=3 nearest vectors.
     │
     ▼
[Stage 3: RelevanceGate] (src/guard/relevance_gate.py)
  ├─ Cross-crop check: Prioritize matching crop from top-3 ranks (promotes rank 2/3).
  ├─ Score < 0.30 ────────────────────────➔ EVIDENCE_ABSENT (Abstain, no LLM call)
  ├─ Query crop ≠ Retrieved crop ─────────➔ CROP_MISMATCH (Abstain, prevent cross-crop advice)
  ├─ 0.30 <= Score < 0.50 ────────────────➔ EVIDENCE_WEAK (Forward with uncertainty flag)
  └─ Score >= 0.50 ───────────────────────➔ EVIDENCE_SUFFICIENT (Forward with high confidence)
     │
     ▼
[Stage 4: AnswerGenerator] (src/generation/answer_gen.py)
  Formats ONLY approved compatible chunks into system prompt.
  Synthesizes response via configured provider (Ollama / Mock / API).
  Catches timeouts/errors, sanitizes secrets, and generates fallback template if needed.
     │
     ▼
[Stage 5: PostGenerationVerifier] (src/generation/verifier.py)
  ├─ Check 0: Non-empty answer text.
  ├─ Check 1: Citation existence (validates cited chunk IDs against approved evidence).
  ├─ Check 2: Mandatory regulatory disclaimer presence.
  ├─ Check 3: Chemical & dosage grounding (extracts active ingredients, formulations,
  │           and dosages; flags BLOCKED_HALLUCINATION if ungrounded).
  ├─ Check 4: Crop entity consistency.
  └─ Check 5: Contradictory evidence detection across multi-chunk advice.
     │
     ▼
Streamlit Display: Verified Grounded Advisory / Safe Abstention Card
```

### 2.3 Dataset Origin & Preprocessing
- **Source Data:** Kisan Call Centre (KCC) farmer inquiry and expert advisory transcripts published by the Ministry of Agriculture & Farmers Welfare, Government of India (via Data.gov.in / AIKosh), curated into the KCC-Krishi corpus by URAV Advanced Learning Systems.
- **Sample Selection:** 30 validated knowledge records (`data/sample/rag_knowledge_sample_30.jsonl`) and 20 evaluation abstention records (`data/sample/evaluation_abstention_sample_20.jsonl`).
- **Ingestion Pipeline (`src/ingestion/ingest.py`):**
  - Excluded time-sensitive domains: `weather`, `market_information`, `government_scheme`.
  - Disallowed high-severity corruption flags (`high_qc_severity_corruption`).
  - Retained 26 standardized, high-quality knowledge chunks (`knowledge_base.jsonl`) indexed into ChromaDB.
- **Metadata Preserved:** `crop_clean`, `state`, `district`, `block`, `advisory_domain`, `query_type`, `farmer_query`, `expert_advisory`, `qc_flags`, `rag_use_mode`.

### 2.4 Distance Metric & Threshold Calibration
- ChromaDB calculates cosine distance $d \in [0, 2]$.
- Similarity is computed as:
  $$\text{Similarity Score} = \max(0.0, \min(1.0, 1.0 - d))$$
- **Dual Calibrated Thresholds:**
  - **Sufficient Threshold ($\ge 0.50$):** High-confidence factual alignment. Answer generated with standard grounding.
  - **Minimum Viable Threshold ($0.30 \le \text{Score} < 0.50$):** Weak evidence. Answer generated with an explicit uncertainty caveat and verified under `PASSED_WITH_WARNINGS`.
  - **Absent Threshold ($< 0.30$):** Insufficient evidence. Pipeline safely abstains without invoking the LLM.

---

## 3. Measured Evaluation Metrics (N = 30 Benchmark)

Evaluated using `scripts/evaluate_retrieval.py` on the curated benchmark dataset `data/evaluation/retrieval_benchmark.jsonl`:

### 3.1 Ground-Truth Retrieval Metrics (N = 15 Labeled In-Domain Queries)
| Metric | Raw Vector Search (top_k=3) | Relevance-Gated & Crop-Promoted | Absolute Improvement / Lift |
| :--- | :---: | :---: | :---: |
| **Hit@1 (Top-1 Accuracy)** | 73.33% (11/15) | **93.33% (14/15)** | **+20.00%** |
| **Hit@3 (Recall Ceiling)** | 100.00% (15/15) | **100.00% (15/15)** | Maintained 100% |
| **MRR (Mean Reciprocal Rank)** | 0.8444 | **0.9667** | **+0.1222 Lift** |
| **Selected Chunk Crop Match** | 73.33% (11/15) | **100.00% (15/15)** | **+26.67%** |

*Why does Crop Promotion help?* In queries like Guava or Pomegranate, unrelated crops (e.g. Groundnut or Green Gram) had slightly higher similarity at rank 1 due to shared symptom words. RelevanceGate promoted the matching crop from rank 2 or 3 to rank 1, eliminating cross-crop errors.

### 3.2 Safety Abstention & Refusal Metrics (N = 15 Refusal / Out-of-Domain Queries)
- **Unsupported Crop Rejection Rate:** **100.00% (4/4)** (Quinoa, Avocado, Dragon Fruit rejected; Kiwi refused as absent).
- **Cross-Crop Advice Leakage Rate:** **0.00% (0/4)** (Zero cross-crop contamination).
- **Crop Mismatch Blocking Rate:** **100.00% (3/3)** (Wheat, Maize, Barley blocked from borrowing advice).
- **Ambiguous Query Clarification Rate:** **75.00% (3/4)** (Prompted for missing crop before retrieval).
- **Off-Topic / Dynamic Abstention Rate:** **75.00% (3/4)** (Refused pre-retrieval).

---

## 4. Two-Minute Presentation Script

> "Good morning, respected examiners. My project is the **Agronomy Advisory Assistant with Hallucination-Guarded RAG and Local Dialect Support**.
>
> In agriculture, Large Language Model hallucinations are dangerous. If an LLM hallucinates an unverified pesticide dosage, mixes incompatible chemicals, or recommends cotton pesticide to a vegetable farmer, it can destroy crops, poison soil, or endanger human life.
>
> Standard or 'naive' RAG does not solve this problem. Vector databases will always return a nearest neighbor vector, even if the farmer asks about an unsupported crop like Quinoa or Dragon Fruit. Naive RAG then feeds that unrelated crop data to the LLM, producing dangerous cross-crop advice.
>
> To solve this, we designed an architecture with **five defensive layers**:
> 1. **Pre-Retrieval QueryGuard:** Rejects non-agricultural questions and dynamic queries like weather or live mandi prices, and detects when a disease query is missing a crop name—asking for clarification before searching the vector database.
> 2. **AgronomyRetriever:** Searches local ChromaDB embeddings using `all-MiniLM-L6-v2`.
> 3. **Calibrated RelevanceGate:** Evaluates the top-3 retrieved chunks, promotes matching crop evidence from ranks 2 and 3, and blocks crop mismatches. It enforces calibrated thresholds: scores above 0.50 are sufficient, scores between 0.30 and 0.50 carry explicit uncertainty warnings, and scores below 0.30 safely abstain.
> 4. **Grounded Answer Generator:** Supplies only approved, crop-compatible evidence to a local Ollama Llama 3.2 model, with automatic fallback to a deterministic template if the model times out.
> 5. **Post-Generation Verifier:** Parses the generated answer to verify that all cited chunk IDs actually exist, and checks that every single chemical name, formulation, and dosage exists in the supporting evidence. If an ungrounded chemical is detected, the answer is immediately blocked.
>
> On our 30-query benchmark, our multi-chunk crop promotion increased Hit@1 accuracy from 73.3% to **93.3%**, achieved an MRR of **0.9667**, and delivered **100% rejection** of unsupported crops with **zero cross-crop advice leakage**. The entire test suite consists of **71 passing automated tests**, and the system runs completely offline on a local laptop without external APIs. Thank you."

---

## 5. Five-Minute Live Demo Checklist

### Step 0: Launch Application
Run in Windows PowerShell:
```powershell
cd "C:\Users\parth\Desktop\ai project"
.\.venv\Scripts\streamlit.exe run app.py
```
Open browser at `http://localhost:8501`.

### Step 1: Normal In-Domain Grounded Query (1 minute)
- **Action:** Click the benchmark button **"Paddy Stem Borer Control"** (`How do I control stem borer in my paddy crop?`) and click **"Generate Advisory"**.
- **Point out to examiner:**
  - Pipeline stage badges at the top showing all 5 stages executed: `QueryGuard ➔ Vector Retrieval ➔ Relevance Gate ➔ Grounded Generation ➔ Post-Verifier`.
  - Green status badge: `Verified Evidence-Grounded Advisory`.
  - The cited KCC Record ID (`kcc_chunk_kcc_00563633`).
  - Regulatory safety notice displayed directly below the text.
  - Expand **Supporting Evidence Records** to show the exact historical KCC advisory transcript retrieved from ChromaDB.

### Step 2: Ambiguous Query Clarification (1 minute)
- **Action:** Click the benchmark button **"Ambiguous Symptoms (No Crop)"** (`My plants have yellow spots and the leaves are curling. What spray should I use?`) and click **"Generate Advisory"**.
- **Point out to examiner:**
  - Stage execution shows ONLY `Stage 1: QueryGuard` was executed.
  - Blue status badge: `Clarification Required`.
  - The system politely explains: *"To give you accurate advice, could you please tell me which crop is affected?"*
  - Emphasize that **0 vector retrieval calls** and **0 LLM calls** were made, saving compute and preventing arbitrary nearest-neighbor matching.

### Step 3: Dynamic Weather Query Safe Abstention (30 seconds)
- **Action:** Click **"Weather & Government Scheme"** (`Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?`).
- **Point out to examiner:**
  - Status: `Real-Time Data Required (Abstained)`.
  - Explains that static historical records cannot provide real-time weather forecasts or policy dates, routing the farmer to official portals (`imd.gov.in`, `pmkisan.gov.in`).

### Step 4: Unsupported Crop Rejection (1 minute)
- **Action:** Click **"Dragon Fruit Grafting (Unindexed)"** (`What is the recommended grafting method for Brazilian Dragon Fruit in winter?`).
- **Point out to examiner:**
  - QueryGuard identifies the crop as Dragon Fruit and proceeds to retrieval.
  - Vector search returns nearest neighbor (Ber) with score 0.4247.
  - **RelevanceGate intercepts the mismatch:** Query crop `Dragon Fruit` does not match chunk crop `Ber`.
  - Status: `No Sufficient Evidence Found (Crop Mismatch)`.
  - Crucial takeaway: Naive RAG would have prescribed Ber advice for Dragon Fruit. Our system safely refused.

### Step 5: System Diagnostics & Provider Architecture (1 minute)
- **Action:** Click on the **System Diagnostics** tab.
- **Point out to examiner:**
  - Generation Provider selector (`ollama`, `mock`, `api`).
  - Live status indicator for Ollama daemon.
  - Displayed Calibrated Relevance Gate Thresholds: Sufficient ($\ge 0.50$), Minimum Viable ($\ge 0.30$), Top-K ($3$).
  - Full transparent stage diagnostics table showing whether stages executed or were safely bypassed.

### Fallback & Recovery Procedures During Demo:
- If Ollama service is stopped: The application automatically falls back to the deterministic evidence template (`[FALLBACK NOTICE]`) without crashing.
- If Streamlit disconnects: Refresh browser (`Ctrl + F5`) or restart with `.\.venv\Scripts\streamlit.exe run app.py`.

---

## 6. Technical Cheat Sheet: Modules, Functions & File Paths

| File Path | Primary Class / Functions | Key Responsibility |
| :--- | :--- | :--- |
| [`src/pipeline.py`](file:///c:/Users/parth/Desktop/ai%20project/src/pipeline.py) | `GuardedRAGPipeline.answer(query)` | Single orchestrator coordinating the 5 pipeline stages and returning `PipelineResult`. |
| [`src/guard/query_guard.py`](file:///c:/Users/parth/Desktop/ai%20project/src/guard/query_guard.py) | `QueryGuard.classify(query)` | Pre-retrieval gate; detects off-topic queries, dynamic requests, and extracts crop entities. |
| [`src/retrieval/retriever.py`](file:///c:/Users/parth/Desktop/ai%20project/src/retrieval/retriever.py) | `AgronomyRetriever.retrieve(query)` | Embeds text with `all-MiniLM-L6-v2` and searches ChromaDB for top-3 candidate chunks. |
| [`src/retrieval/vector_store.py`](file:///c:/Users/parth/Desktop/ai%20project/src/retrieval/vector_store.py) | `ChromaAgronomyStore.search()` | Persistent ChromaDB client; computes cosine similarity $1.0 - \text{distance}$. |
| [`src/guard/relevance_gate.py`](file:///c:/Users/parth/Desktop/ai%20project/src/guard/relevance_gate.py) | `RelevanceGate.evaluate()` | Multi-chunk crop compatibility check, rank promotion, and threshold gating (0.50 / 0.30). |
| [`src/generation/answer_gen.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/answer_gen.py) | `AnswerGenerator.generate()` | Constructs grounded prompt using approved evidence; handles Ollama/Mock calls & fallbacks. |
| [`src/generation/verifier.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/verifier.py) | `PostGenerationVerifier.verify()` | Structural verification: citation IDs, chemical active ingredients, dosages, and contradictions. |
| [`src/generation/providers.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/providers.py) | `OllamaProvider`, `MockProvider`, `APIProvider` | Unified LLM provider interface with exception handling and secret sanitization. |
| [`src/ui/helpers.py`](file:///c:/Users/parth/Desktop/ai%20project/src/ui/helpers.py) | `extract_pipeline_diagnostics()` | Transforms `PipelineResult` for UI presentation without fabricating missing data. |
| [`app.py`](file:///c:/Users/parth/Desktop/ai%20project/app.py) | Streamlit Entry Point | Academic evaluation dashboard with 3 tabs: Workspace, Evidence Explorer, Diagnostics. |

---

## 7. 25 Realistic Viva Voce Questions & Answers

### Architecture & Conceptual Questions

#### Q1: What is the main difference between your system and standard (naive) RAG?
**Answer:** Standard RAG embeds a query, retrieves the top-$k$ nearest neighbors regardless of relevance, and prompts an LLM to generate an answer. Our system wraps retrieval in defensive layers: `QueryGuard` blocks off-topic, dynamic, or ambiguous queries before retrieval; `RelevanceGate` evaluates all top-$k$ chunks to promote crop matches and block crop mismatches; and `PostGenerationVerifier` checks that all cited chemicals, dosages, and record IDs exist in the source evidence before displaying the answer.

#### Q2: Why did you choose a 5-stage pipeline instead of letting a large model like GPT-4 do everything in one prompt?
**Answer:** Relying entirely on LLM prompt instructions is non-deterministic and prone to jailbreaks, attention drift, and hallucinations. A modular 5-stage pipeline provides deterministic, testable software gates. Non-agricultural queries are filtered at Stage 1 with zero compute cost, crop mismatches are blocked at Stage 3 using deterministic logic, and Stage 5 provides independent programmatic verification that cannot be bypassed by LLM prompt drift.

#### Q3: What is the role of Streamlit in this project?
**Answer:** Streamlit serves as an academic evaluation dashboard rather than a simple consumer chatbot. It exposes real-time pipeline breadcrumbs, displays whether intermediate stages were executed or bypassed, provides an Evidence Explorer to audit raw historical KCC transcripts, and renders a System Diagnostics panel showing calibrated thresholds and live LLM provider parameters.

#### Q4: Why did you choose ChromaDB as the vector store?
**Answer:** ChromaDB is a lightweight, open-source, embedded vector database that runs in-process with Python on Windows without requiring a Docker container or dedicated cloud cluster. It supports persistent on-disk storage (`chroma.sqlite3`) and provides exact HNSW cosine space indexing suitable for our local evaluation dataset.

#### Q5: What embedding model did you use, and why?
**Answer:** We used `sentence-transformers/all-MiniLM-L6-v2`. It produces 384-dimensional dense embeddings, has a tiny disk footprint of approximately 80 MB, and executes in milliseconds on standard CPU hardware. It strikes an optimal balance between semantic retrieval quality and local execution speed without requiring a dedicated GPU.

---

### Dataset & Preprocessing Questions

#### Q6: Why is your active knowledge base only 26 records? Isn't that too small for a real system?
**Answer:** This repository is an academic proof-of-concept and safety verification prototype focused on testing defensive failure modes, threshold boundaries, and hallucination verifiers. A small, strictly curated dataset allowed us to hand-annotate a ground-truth benchmark, manually verify every single chemical claim and QC corruption flag, and prove that our safety gates prevent cross-crop contamination and forced nearest-neighbor leaks. The code is modular and scales to larger datasets by re-running `build_index.py`.

#### Q7: Where does the dataset come from, and what is its legal/agronomic status?
**Answer:** The data originates from historical Kisan Call Centre (KCC) call transcripts recorded by the Ministry of Agriculture and Farmers Welfare, Government of India, released on Data.gov.in / AIKosh, and curated into the KCC-Krishi corpus by URAV Advanced Learning Systems. Because these records date from 2014 to 2024, they are treated strictly as historical research evidence, not active regulatory prescriptions.

#### Q8: What criteria did you use to filter and reject records during ingestion?
**Answer:** In `src/ingestion/ingest.py`, we filtered records using four rules: we excluded time-sensitive domains (`weather`, `market_information`, and `government_scheme`), removed records flagged with `high_qc_severity_corruption` (such as severe translation errors or extreme unit mismatches), enforced crop validity (rejecting numeric or malformed crop names), and restricted usage modes to `use_as_evidence` and `use_with_caution`.

#### Q9: Why is the `original_year` metadata missing or null for most records?
**Answer:** In the raw KCC call center sample extracts, neither the top-level JSON nor the nested metadata fields included a reliable timestamp or year. Rather than fabricating a fictitious date, our pipeline explicitly records `original_year = None`, displaying it transparently in the UI and CLI as "Historical (year unspecified)" to maintain scientific integrity.

---

### Retrieval & Threshold Calibration Questions

#### Q10: How do you compute similarity scores from ChromaDB distances?
**Answer:** ChromaDB returns cosine distance $d \in [0, 2]$. We convert this to cosine similarity using the formula:
$$\text{Similarity Score} = \max(0.0, \min(1.0, 1.0 - d))$$
This bounds the similarity between 0.0 (completely orthogonal or opposite) and 1.0 (identical vectors).

#### Q11: Why are cosine similarity thresholds inherently imperfect in RAG?
**Answer:** Cosine similarity in dense embedding space measures geometric and topical proximity, not factual truth or agronomic equivalence. An unindexed crop like Dragon Fruit can match a Ber record with a similarity of 0.4247 simply because both discuss fruit pruning in winter. Therefore, similarity scores cannot be used alone; they must be paired with explicit entity-level crop compatibility checks.

#### Q12: Why did you set the thresholds at 0.50 and 0.30? Why not lower 0.50 to 0.40 to answer more queries?
**Answer:** Our empirical score audit showed that unrelated nearest neighbors forced by dense embeddings routinely score between 0.36 and 0.47. Lowering the sufficient threshold below 0.50 would cause unrelated nearest-neighbor crops to be treated as high-confidence evidence. Maintaining 0.50 ensures that lower-scoring chunks (0.30–0.49) are flagged as `EVIDENCE_WEAK`, triggering mandatory uncertainty warnings and strict chemical verification.

#### Q13: What is Multi-Chunk Crop Promotion and why is it important?
**Answer:** In naive RAG, only the rank-1 chunk is inspected. If a farmer asks about Guava, rank 1 might be Groundnut with score 0.4908, while Guava sits at rank 2 with score 0.4901. Naive RAG either aborts or gives Groundnut advice. Our `RelevanceGate` inspects all top-$k$ retrieved chunks; if a lower-ranked chunk matches the query crop, it promotes that chunk to rank 1, increasing our benchmark Hit@1 accuracy from 73.3% to 93.3%.

---

### Hallucination Guarding & Safety Questions

#### Q14: Does your system guarantee zero hallucinations?
**Answer:** No. In computer science and AI, claiming "zero hallucinations" with probabilistic language models is scientifically unsupportable. Instead, our system provides deterministic defense-in-depth: it bounds hallucinations by restricting the model to approved evidence, strips ungrounded chemical claims via regex verification, flags weak evidence with uncertainty caveats, and safely abstains when evidence is absent or contradictory.

#### Q15: How does the Post-Generation Verifier detect ungrounded chemicals?
**Answer:** `PostGenerationVerifier` uses compiled regular expressions to scan the generated answer for pesticide active ingredients (e.g. Chlorpyrifos, Quinalphos, Mancozeb), formulation types (EC, WP, SC, SL, GR), and numerical dosages (e.g. `2 ml/litre`, `500 gm/acre`). It then verifies whether each extracted chemical entity appears in the retrieved evidence text. If an entity is not grounded in the source chunks, it raises `BLOCKED_HALLUCINATION` and strips the text.

#### Q16: How do you handle contradictory evidence between multiple retrieved chunks?
**Answer:** In Check 5 of `PostGenerationVerifier`, we extract the chemical active ingredients from each retrieved supporting chunk. If multiple chunks for the same crop prescribe completely disjoint active ingredients (for instance, one recommends Quinalphos while another recommends Carbofuran), the system emits a `CONTRADICTORY_EVIDENCE` warning and sets the verification status to `PASSED_WITH_WARNINGS`, advising the farmer to consult local extension officers before deciding.

#### Q17: What prevents an LLM from hallucinating fake source citations?
**Answer:** In Check 1 of `PostGenerationVerifier`, we inspect `answer.source_ids`. We cross-check every cited ID against the set of approved chunk IDs in `evidence_chunks`. If the model fabricates a citation (e.g., citing `kcc_chunk_fake_999`), the verifier flags `INVALID_CITATION` and blocks the response.

#### Q18: What happens when an unsupported crop like Quinoa or Avocado is queried?
**Answer:** `QueryGuard` extracts the crop name (e.g., "Quinoa"). Vector search retrieves the nearest available chunk (e.g., Groundnut). Next, `RelevanceGate` compares the query crop ("Quinoa") against the retrieved chunk crop ("Groundnut"). Because they do not match, the gate issues a `CROP_MISMATCH` verdict and the system abstains safely (`ABSTAINED_CROP_MISMATCH`), preventing cross-crop advice leakage.

#### Q19: How do you handle regional crop names or dialect variations?
**Answer:** In `QueryGuard`, we maintain lexical mappings for common regional and transliterated Hindi crop terms (e.g., `Dhan` maps to Paddy, `Sarson` to Mustard, `Kapas` to Cotton, `Tamatar` to Tomato, `Baingan` to Brinjal, and `Makka` to Maize). This allows the guard to extract the crop entity correctly regardless of whether the farmer types English or transliterated Hindi.

---

### Evaluation, Failure Handling & Reliability Questions

#### Q20: What are Hit@1, Hit@3, and MRR, and what do your benchmark results show?
**Answer:** Hit@1 measures whether the correct relevant chunk was returned at rank 1; Hit@3 measures whether it appeared anywhere in the top 3; and MRR (Mean Reciprocal Rank) averages the reciprocal ranks ($1/\text{rank}$) of the first relevant chunk. On our 15 labeled ground-truth benchmark queries, raw vector search achieved Hit@1 of 73.33% and MRR of 0.8444. With our multi-chunk crop promotion, Hit@1 reached **93.33%** and MRR increased to **0.9667** (+0.1222 lift), while maintaining a 100% Hit@3 recall ceiling.

#### Q21: Why don't you compute Hit@k or MRR for the 15 refusal/abstention queries in your benchmark?
**Answer:** Hit@k and MRR mathematically require the existence of a true positive ground-truth relevant document. For queries regarding unsupported crops (like Dragon Fruit), off-topic technical questions, or weather inquiries, there is no relevant document in the knowledge base. Computing retrieval accuracy for them would be scientifically invalid; instead, we evaluate them using refusal metrics: Unsupported Crop Rejection Rate (100%) and Cross-Crop Leakage Rate (0%).

#### Q22: What happens if the local Ollama provider times out or crashes during a live demo?
**Answer:** In `AnswerGenerator`, the LLM call is wrapped in exception handling. If Ollama times out (default 30–60s) or is offline, the exception is caught, secret tokens or keys are sanitized, and the system automatically synthesizes a deterministic response directly from the approved KCC evidence text using a structured template. The answer is flagged with `is_fallback = True` and displays a prominent `[FALLBACK NOTICE]` banner in the UI.

#### Q23: How do you protect API keys and secrets from being leaked in error logs or UI diagnostics?
**Answer:** In `src/generation/providers.py` and `src/generation/answer_gen.py`, error messages pass through a dedicated `_sanitize_error()` method. This uses regular expressions to redact Bearer tokens (`Bearer [REDACTED_TOKEN]`) and API keys matching patterns like `sk-...` (`[REDACTED_API_KEY]`), preventing credentials from being displayed in diagnostics or logged in plain text.

#### Q24: What is your automated testing strategy, and how many tests pass?
**Answer:** We implemented **71 automated unit and integration tests** across 7 test modules executed via Python's standard `unittest` framework. All 71 tests run offline without network access or live API calls, completing in under 4 seconds. Tests cover baseline retrieval, query guard rules, provider timeouts, UI diagnostic extraction, chemical blind spot detection, multi-chunk crop promotion, and end-to-end failure handling.

#### Q25: If you had another semester to work on this project, what would you improve?
**Answer:** I would focus on three major enhancements:
1. **Live CIBRC Registry Integration:** Replace regex chemical verification with an official, live database lookup against the Central Insecticides Board & Registration Committee to check banned/restricted pesticide statuses dynamically.
2. **Cross-Encoder NLI:** Add a lightweight cross-encoder model (such as a MiniLM NLI classifier) to perform full semantic entailment verification rather than token-level grounding checks.
3. **Multilingual ASR:** Integrate a fine-tuned Whisper or Bhashini speech-to-text pipeline to allow farmers to speak queries directly in regional Indian dialects.
