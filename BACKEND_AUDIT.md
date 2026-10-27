# Agronomy Advisory Assistant: Backend Audit & Retrieval Evaluation

**Date:** 2026-10-27  
**Project:** Agronomy Advisory Assistant with Hallucination-Guarded RAG & Local Dialect Support  
**Target Environment:** Windows 11 / Python 3.12+ / PowerShell  
**Repository:** `c:\Users\parth\Desktop\ai project`  
**Current Test Status:** 45 / 45 Tests Passing (Executed offline via `unittest`)  
**Scope:** In-depth backend correctness, retrieval semantics, hallucination guards, verification integrity, and evaluation methodology.

---

## 1. Executive Summary

A comprehensive forensic audit of the ingestion, retrieval, gating, generation, and verification layers was conducted. While the pipeline architecture is modular and all 45 automated unit/integration tests pass, the backend contains **four critical safety vulnerabilities** and several heuristic limitations that could permit unsafe chemical advice or false confidence in real-world deployment:

1. **Unverified Weak-Evidence Generation (Critical Severity):** When evidence similarity falls between 0.30 and 0.50, the generator synthesizes advice, but the `PostGenerationVerifier` **completely bypasses safety checks** (`SKIPPED_WEAK`). Unchecked hallucinations can be served directly to farmers.
2. **Unindexed Crop Mismatch Leakage (Critical Severity):** Crops not present in `_AGRI_CROP_TERMS` (e.g., Quinoa, Avocado, Dragon Fruit) bypass QueryGuard crop extraction (`detected_crop = None`). Consequently, the post-retrieval crop-mismatch gate is skipped, and nearest-neighbor records for completely unrelated crops (e.g. Groundnut or Turmeric) are returned with scores >0.30, triggering unverified advice generation.
3. **Silent Provider Fallback Masking (High Severity):** When an LLM provider (Ollama or API) times out or throws an exception, `AnswerGenerator` falls back to a deterministic string template, but returns `GenerationStatus.SUCCESS`. Upstream consumers cannot distinguish between model-synthesized responses and fallback text.
4. **Heuristic Calibration of Similarity Thresholds (High Severity):** Thresholds `0.50` (sufficient) and `0.30` (minimum viable) were derived from an ad-hoc 5-scenario baseline. In empirical testing, legitimate paraphrased in-domain queries drop below 0.50, while out-of-domain queries routinely score 0.35–0.46, creating a narrow separation window.

---

## 2. Dataset Quality Audit

### 2.1 File Inventory and Counts
- `data/sample/rag_knowledge_sample_30.jsonl`: **30 records**, 0 malformed lines. 100% schema consistency.
- `data/sample/evaluation_abstention_sample_20.jsonl`: **20 records**, 0 malformed lines. 100% schema consistency.
- `data/processed/knowledge_base.jsonl`: **26 records**, 0 malformed lines. Output of `ingest.py` on the 30-sample file.
- `data/sample/sample_metadata.json`: Documents provenance (Kisan Call Centre via Data.gov.in / AIKosh and URAV Advanced Learning Systems / KCC-Krishi).

### 2.2 Rejections and Filtering (`src/ingestion/ingest.py`, `src/ingestion/schema.py`)
- **Retention rate:** 86.67% (26 of 30 retained).
- **Rejected records (4):** Rows 4 (`kcc_00018845`), 6 (`kcc_00094206`), 16 (`kcc_00638911`), and 28 (`kcc_00214059`) were rejected with reason `high_qc_severity_corruption`.
- **Validation logic:** Correctly enforces crop validity, disallows numeric crop IDs, excludes `weather`, `market_information`, and `government_scheme` domains, and rejects `high` QC severity.

### 2.3 Preserved vs. Missing Metadata
- **Preserved:** `crop`, `crop_clean`, `state`, `district`, `block` (extracted from prompt), `advisory_domain`, `query_type`, `farmer_query`, `expert_advisory`, `qc_flags`, `qc_severity`, `qc_confidence`, `rag_use_mode`.
- **Missing / Null:**
  - `original_year`: `Counter({None: 30})` in source samples. Neither the top-level sample JSON nor `raw_metadata` includes a `year` or `created_at` timestamp. All 26 indexed chunks have `original_year = None`, displaying as "Historical (year unspecified)".
  - `block` is missing at top-level; extracted only via regex from the prompt string.

### 2.4 Agronomic Safety and Chemical Risk in Raw Records
Historical records dating back to 2014–2022 contain chemicals that are restricted or banned in various Indian states:
- `kcc_chunk_kcc_00563633` (Satawar): Quinalphos 25% EC (organophosphate).
- `kcc_chunk_kcc_00313852` (Paddy): Carbofuran 3G granular (restricted/banned in multiple states for high mammalian toxicity).
- `kcc_chunk_kcc_00271164` (Tea): Tricel (Chlorpyrifos formulation; restricted).

**Finding:** The knowledge base must treat historical advice strictly as research evidence, never as active regulatory prescriptions.

---

## 3. Retrieval Quality & Embedding Audit

### 3.1 Vector Store & Distance Semantics (`src/retrieval/vector_store.py`)
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional normalized embeddings.
- **Storage:** Persistent ChromaDB collection `agronomic_advisory_kb` with `hnsw:space: cosine`.
- **Score Calculation:**
  ```python
  similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
  ```
- **Semantic Reality:** ChromaDB returns cosine distance $d \in [0, 2]$. $1.0 - d$ computes cosine similarity. However, cosine similarity in embedding space is a geometric alignment measure, **not a calibrated probability of factual accuracy**. Treating 0.50 as "sufficient factual evidence" is heuristic.

### 3.2 Empirical Score Audit Across Query Archetypes
Empirical evaluation against the 26 indexed chunks revealed:

| Category | Query | Top Match Crop | Similarity Score | Gating Outcome |
| :--- | :--- | :--- | :---: | :---: |
| **In-domain exact** | "How to control insects in Satawar crop?" | Satawar | `0.6206` | `EVIDENCE_SUFFICIENT` |
| **In-domain paraphrase 1** | "Pest management for satawar plants" | Satawar | `0.7178` | `EVIDENCE_SUFFICIENT` |
| **In-domain paraphrase 2** | "What medicine to spray for worms in satawar roots and leaves?" | Satawar | `0.4787` | **`EVIDENCE_WEAK` (False Demotion)** |
| **In-domain Paddy blast** | "My paddy field has neck blast fungus, what chemical should I apply?" | Paddy (Dhan) | `0.5514` | `EVIDENCE_SUFFICIENT` |
| **Unindexed crop (Quinoa)** | "Quinoa downy mildew management practices" | Groundnut | `0.4653` | **`EVIDENCE_WEAK` (Dangerous Leak)** |
| **Unindexed crop (Avocado)** | "How to fertilize Hass avocado trees in sandy soil?" | Turmeric | `0.4027` | **`EVIDENCE_WEAK` (Dangerous Leak)** |
| **Unindexed crop (Dragon Fruit)** | "What is the recommended grafting method for Brazilian Dragon Fruit in winter?" | Ber | `0.4247` | **`EVIDENCE_WEAK` (Dangerous Leak)** |
| **Crop Mismatch (Wheat)** | "Control of yellow rust disease in wheat crop" | Groundnut | `0.3660` | `CROP_MISMATCH` (Blocked) |
| **Off-topic technical** | "How do I configure a React frontend router in Node.js?" | Ber | `0.0815` | `EVIDENCE_ABSENT` |
| **Off-topic general** | "What is the capital of France and who is the president?" | Pomegranate | `0.0633` | `EVIDENCE_ABSENT` |
| **Dynamic Weather** | "Will it rain tomorrow in Mehsana district Gujarat?" | Ber | `0.3416` | Nearest neighbor forced (blocked by guard) |

### 3.3 Retrieval Flaws Identified
1. **Forced Nearest Neighbors for Unindexed Queries:** Vector search always returns the nearest vector regardless of relevance. Unindexed crops score between 0.36 and 0.47, easily clearing the `0.30` minimum viable threshold.
2. **Top-1 Gating Bottleneck (`src/guard/relevance_gate.py:121`):** `evaluate()` only evaluates `results[0]` for crop mismatch. If chunk 1 is an unrelated crop with score 0.52 and chunk 2 is an exact crop match with score 0.51, the gate aborts on chunk 1 as `CROP_MISMATCH`.
3. **No Hybrid/Keyword Re-ranking:** Dense embeddings alone struggle with specific agronomic entity matching (e.g. distinguishing leaf curl virus vs yellow vein mosaic).

---

## 4. Generation, Grounding & Verification Audit

### 4.1 Provider Layer (`src/generation/providers.py`)
- Provides `MockProvider`, `OllamaProvider`, and `APIProvider` behind a unified interface.
- Low temperature (`0.1`) configured on Ollama for factual grounding.
- Proper error handling with `LLMConfigurationError` and `LLMGenerationError`.

### 4.2 Prompt Synthesis & Information Flow (`src/generation/answer_gen.py`)
- Evidence is formatted into `_build_llm_prompt()`.
- System prompt strictly instructs the model to cite record IDs and avoid ungrounded chemical claims.
- **Vulnerability — Silent Fallback Masking:**
  ```python
  except LLMError as e:
      logger.warning(f"LLM provider failed: {e}. Falling back...")
      return self._format_evidence_answer(query, chunks)
  ```
  If Ollama crashes or times out, the answer falls back to the deterministic template, but `GeneratedAnswer.status` remains `GenerationStatus.SUCCESS`. The user or downstream system receives no metadata flag indicating the LLM failed.

### 4.3 Structural Verifier Flaws (`src/generation/verifier.py`)
1. **Bypassed Verification for Weak Evidence (Critical):**
   ```python
   if answer.status == GenerationStatus.WEAK_EVIDENCE:
       return VerificationResult(status=VerificationStatus.SKIPPED_WEAK, ...)
   ```
   Weak-evidence answers are the **most likely to suffer from hallucinations**, yet verification is skipped entirely.
2. **False Acceptance via Vocabulary Limitation:**
   `_CHEMICAL_PATTERNS` contains only 21 hardcoded chemical names. If an LLM recommends *cartap hydrochloride, acephate, monocrotophos, paraquat, glyphosate, or flubendiamide* without matching a specific dosage regex, it passes unchecked.
3. **False Rejection via Exact String Containment:**
   `claim not in evidence_text_combined` uses raw substring matching. Legitimate formatting differences (e.g., "2 ml per liter" vs "2 ml/litre" or "0.5 L/acre" vs "500 ml/acre") are flagged as `HALLUCINATION_RISK` and trigger complete answer suppression.

---

## 5. Safety Pipeline Audit (`src/pipeline.py`, `src/guard/`)

### 5.1 Pipeline Stage Bypassing
Execution trace verification:
- **Dynamic Weather / Scheme Query:** Bypasses vector retrieval and LLM generation (0 calls). Returns official portal links (`imd.gov.in`, `agmarknet.gov.in`, `pmkisan.gov.in`).
- **Off-topic Technical Query:** Bypasses retrieval and LLM generation (0 calls). Explains agronomic scope.
- **Ambiguous Disease Query (Missing Crop):** Bypasses retrieval and LLM generation (0 calls). Issues `CLARIFY_CROP` prompt.
- **Crop Mismatch (Wheat -> Groundnut):** Retrieves chunks, but RelevanceGate blocks generation (0 LLM calls).

### 5.2 The Unindexed Crop Escape Hole
When a query names a crop outside `_AGRI_CROP_TERMS` (e.g., "Quinoa downy mildew management"):
1. `QueryGuard` regex finds no crop match -> `detected_crop = None`.
2. Guard decides `PROCEED` because disease keywords ("mildew", "management") are present.
3. `AgronomyRetriever` executes and returns Groundnut chunks (score 0.4653).
4. `RelevanceGate` checks `if query_crop is not None:` -> Skipped because `query_crop` is `None`!
5. `best_score` (0.4653) > 0.30 -> Verdict is `EVIDENCE_WEAK`.
6. Generator synthesizes advice using Groundnut evidence for a Quinoa query.
7. Verifier skips checks due to `WEAK_EVIDENCE`.
8. Result: **A farmer asking about Quinoa receives Groundnut chemical advice with zero verification.**

---

## 6. Testing & Evaluation Audit

### 6.1 Unit Tests vs. Real Retrieval Evaluation
- Current test suite contains 45 tests across 4 files:
  - `test_retrieval_baseline.py` (5 tests)
  - `test_hallucination_guard.py` (14 tests)
  - `test_generation_providers.py` (16 tests)
  - `test_ui_pipeline.py` (10 tests)
- **Coverage Gap:** Tests verify that code runs without crashing and that mocks satisfy contracts, but they do **NOT** evaluate:
  - Retrieval Recall@k or MRR against an annotated benchmark.
  - LLM hallucination rate under adversarial prompting.
  - Dialect / transliteration robustness (Hinglish / Latin Indian phrasing).
  - Out-of-vocabulary crop rejection.

### 6.2 Proposed Reproducible Benchmark Dataset
To evaluate retrieval and generation objectively without fabricating results, create a 50-query golden evaluation benchmark (`data/evaluation/benchmark_50.jsonl`) with the following distribution:

| Evaluation Category | Target Count | Expected Pipeline Outcome | Evaluation Metric |
| :--- | :---: | :--- | :--- |
| **In-Domain Supported** | 15 | `EVIDENCE_SUFFICIENT` + Verified Answer | Hit@1, Crop-Match Recall, Verifier Pass Rate |
| **In-Domain Paraphrased** | 10 | `EVIDENCE_SUFFICIENT` + Verified Answer | Hit@3, Score > 0.50 stability |
| **Unindexed Crop Questions** | 8 | Safe Abstention / Refusal | Rejection Rate (Precision of Refusal) |
| **Ambiguous Disease / Spray** | 7 | `CLARIFY_CROP` (Zero retrieval) | Clarification Recall |
| **Dynamic Live-Data** | 5 | `ABSTAIN_DYNAMIC` (Zero retrieval) | Abstention Precision |
| **Non-Agricultural Off-Topic** | 5 | `ABSTAIN_OFFTOPIC` (Zero retrieval) | Abstention Precision |

---

## 7. Risk Matrix & Remediation Status (Phase 6A Updated)

| Severity | Issue | Status | Remediation Detail | Code Reference |
| :---: | :--- | :---: | :--- | :--- |
| **CRITICAL** | Verifier skips safety checks on weak-evidence generation | **RESOLVED (Phase 6A)** | Removed `SKIPPED_WEAK` bypass. Weak-evidence answers undergo full claim extraction & grounding. Ungrounded chemical claims are blocked (`BLOCKED_HALLUCINATION`). Grounded claims pass with `PASSED_WITH_WARNINGS` and an explicit weak-evidence caveat. | [`src/generation/verifier.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/verifier.py) |
| **CRITICAL** | Unindexed crop names bypass crop-mismatch gate | **RESOLVED (Phase 6A)** | Expanded `_AGRI_CROP_TERMS` and syntactic crop-pattern extractors in `QueryGuard` (e.g. Quinoa, Avocado, Dragon Fruit). Hardened `RelevanceGate` to evaluate crop compatibility across all candidate chunks and refuse cross-crop advisory generation with `CROP_MISMATCH`. | [`src/guard/query_guard.py`](file:///c:/Users/parth/Desktop/ai%20project/src/guard/query_guard.py), [`src/guard/relevance_gate.py`](file:///c:/Users/parth/Desktop/ai%20project/src/guard/relevance_gate.py) |
| **HIGH** | Silent fallback masks LLM provider failure | **RESOLVED (Phase 6A)** | Added `provider_used`, `is_fallback: bool`, and `fallback_reason` metadata to `GeneratedAnswer` and diagnostics. Fallback responses carry a prominent `[FALLBACK NOTICE]` banner in display text without breaking backward compatibility. | [`src/generation/answer_gen.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/answer_gen.py), [`src/ui/helpers.py`](file:///c:/Users/parth/Desktop/ai%20project/src/ui/helpers.py) |
| **HIGH** | Limited chemical vocabulary in verifier (21 chemicals) | **RESOLVED (Phase 6A)** | Added explicit detection for blind spots (`monosultap`, `acephate`, `flubendiamide`, `paraquat`), formulation codes (`EC`, `WP`, `SL`, `SC`, `G`), and dosage rates (`ml/litre`, `kg/acre`, `pani`). Normalized punctuation/whitespace to eliminate false rejections. | [`src/generation/verifier.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/verifier.py) |
| **HIGH** | Uncalibrated relevance thresholds (0.50 / 0.30) | **RESOLVED & DOCUMENTED (Phase 6B)** | Investigated empirical cosine distributions. Maintained 0.50 (sufficient) and 0.30 (minimum viable) as conservative safety gates. Weak evidence (0.30–0.50) is safely caveated and verified rather than demoted or arbitrarily elevated. | [`src/guard/relevance_gate.py`](file:///c:/Users/parth/Desktop/ai%20project/src/guard/relevance_gate.py) |
| **MEDIUM** | Top-1 only crop-mismatch evaluation | **RESOLVED (Phase 6B)** | Multi-chunk candidate evaluation and crop-aware rank promotion implemented. Matching crops at rank 2 or 3 (e.g., Pomegranate, Guava, Cotton) are promoted to best chunk; incompatible cross-crop chunks are stripped out. | [`src/guard/relevance_gate.py`](file:///c:/Users/parth/Desktop/ai%20project/src/guard/relevance_gate.py), [`src/generation/answer_gen.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/answer_gen.py) |
| **MEDIUM** | Missing `original_year` in sample dataset | **DOCUMENTED** | Dataset limitation: KCC sample data lacks timestamp metadata. Treated as research historical baseline. | [`src/ingestion/schema.py`](file:///c:/Users/parth/Desktop/ai%20project/src/ingestion/schema.py) |
| **LOW** | String-exact matching causes false rejections | **RESOLVED (Phase 6A)** | Implemented token-level normalization in verifier to handle minor whitespace/punctuation variance. | [`src/generation/verifier.py`](file:///c:/Users/parth/Desktop/ai%20project/src/generation/verifier.py) |

---

## 8. Empirical Retrieval & Relevance-Gate Evaluation Results (Phase 6B)

A reproducible 30-query evaluation benchmark (`data/evaluation/retrieval_benchmark.jsonl`) was executed via `scripts/evaluate_retrieval.py` against the active 26-chunk ChromaDB knowledge base.

### 8.1 Ground-Truth Retrieval Metrics (N = 15 labeled queries)
Where verified ground-truth relevance labels exist (exact in-domain queries and rank > 1 promoted queries):

| Metric | Raw Vector Search (top_k=3) | Relevance-Gated & Promoted | Absolute Improvement / Lift |
| :--- | :---: | :---: | :---: |
| **Hit@1 (Top-1 Accuracy)** | 73.33% (11/15) | **93.33% (14/15)** | **+20.00%** |
| **Hit@3 (Top-3 Recall)** | 100.00% (15/15) | **100.00% (15/15)** | 0.00% (Maintained ceiling) |
| **MRR (Mean Reciprocal Rank)** | 0.8444 | **0.9667** | **+0.1222** |
| **Selected Chunk Crop Match** | 73.33% (11/15) | **100.00% (15/15)** | **+26.67%** |

### 8.2 Safety Abstention & Refusal Metrics (N = 15 refusal/abstention queries)
Queries lacking relevant records evaluated for safety and refusal precision (Hit@k is NOT computed here as no relevant record exists):
- **Unsupported Crop Rejection Rate:** `100.00%` (4/4 safe refusals; Quinoa, Avocado, Dragon Fruit rejected as `CROP_MISMATCH`, Kiwi rejected as `EVIDENCE_ABSENT` at score 0.2997).
- **Cross-Crop Advice Leakage Rate:** `0.00%` (0/4 leaked cross-crop advice).
- **Crop Mismatch Blocking Rate:** `100.00%` (3/3 blocked; Wheat, Maize, Barley).
- **Ambiguous Query Clarification Rate:** `75.00%` (3/4 prompted for missing crop; 1 non-disease query cleanly refused).
- **Off-Topic / Dynamic Abstention Rate:** `75.00%` (3/4 pre-retrieval refused; general trivia query proceeded to retrieval and was blocked by RelevanceGate at score 0.0633 as `EVIDENCE_ABSENT`).

### 8.3 Empirical Cosine Score Distributions
- **In-Domain Target Chunks (N=15):** Mean = `0.5623` | Min = `0.3542` | Max = `0.7582`
- **Unsupported Crop Forced Nearest Neighbors (N=4):** Mean = `0.3981` | Min = `0.2997` | Max = `0.4653`
- **Crop Mismatch Forced Nearest Neighbors (N=3):** Mean = `0.3825` | Min = `0.3635` | Max = `0.4180`
- **Non-Agricultural / Off-Topic (N=4):** Mean = `0.2061` | Min = `0.0633` | Max = `0.3416`

### 8.4 Findings on Threshold Calibration & Limitations
1. **Narrow Cosine Separation Window:** Unrelated crops forced by dense embeddings score between `0.36` and `0.47`, directly overlapping with valid in-domain paraphrases (which score between `0.35` and `0.52`).
2. **Threshold Preservation:** Thresholds `0.50` (sufficient) and `0.30` (minimum viable) must **not** be lowered. Lowering `0.50` would elevate unverified cross-crop nearest neighbors to sufficient evidence. Chunks scoring between `0.30` and `0.50` are appropriately gated as `EVIDENCE_WEAK`, ensuring uncertainty caveats and strict chemical verification.
3. **Statistical Limitation:** Calibrating continuous scalar thresholds with mathematical confidence bounds requires a larger annotated dataset (e.g., 1,000+ labeled rows), which is outside the scope of this offline sample dataset.

---

## 9. Phase Execution Status

### Phase 6A: Safety Guard & Verifier Hardening (COMPLETED)
- [x] Removed `SKIPPED_WEAK` bypass from `PostGenerationVerifier`.
- [x] Enforced ungrounded chemical detection and blocking on weak-evidence answers (`BLOCKED_HALLUCINATION`).
- [x] Closed unindexed crop escape hole for crops such as Quinoa, Avocado, and Dragon Fruit.
- [x] Added multi-chunk crop compatibility check to `RelevanceGate`.
- [x] Exposed explicit `is_fallback`, `provider_used`, and `fallback_reason` in `GeneratedAnswer`, `PipelineResult`, and UI diagnostics.
- [x] Expanded verifier patterns for blind-spot chemicals (`monosultap`, `acephate`, `flubendiamide`, `paraquat`), formulations, and dosages.
- [x] Test suite: **51 / 51 tests passing**.

### Phase 6B: Retrieval & Relevance-Gate Optimization (COMPLETED)
- [x] Evaluated all retrieved top-$k$ chunks in `RelevanceGate` and promoted matching crops from ranks 2 and 3.
- [x] Prioritized crop-compatible evidence and ensured cross-crop chunks are never forwarded to generator or verifier (`compatible_chunks`).
- [x] Investigated 0.50 and 0.30 thresholds; demonstrated that similarity alone does not make a promoted chunk sufficient.
- [x] Created annotated evaluation benchmark dataset (`data/evaluation/retrieval_benchmark.jsonl`).
- [x] Implemented evaluation harness (`scripts/evaluate_retrieval.py`) measuring Hit@1, Hit@3, MRR, and score distributions.
- [x] Added regression tests in `tests/test_retrieval_gate_optimization.py` for Pomegranate, Guava, Cotton, Satawar, Quinoa, Avocado, and Dragon Fruit.
- [x] Full test suite verified: **58 / 58 tests passing** (0 failures, 0 errors).

### Phase 6C: Systematic Benchmark Evaluation & Final Presentation Report (UPCOMING)
- [ ] Expand golden evaluation benchmark to full 50-query suite.
- [ ] Automate end-to-end evaluation report (`EVALUATION_REPORT.json`) across all 5 pipeline stages.
- [ ] Finalize backend documentation for presentation.


