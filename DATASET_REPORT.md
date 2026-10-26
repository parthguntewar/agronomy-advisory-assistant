# Dataset Investigation & Selection Report: Agronomy Advisory Assistant

**Project:** Agronomy Advisory Assistant with Hallucination-Guarded RAG & Local Dialect Support  
**Environment:** Windows 11 / Antigravity IDE / Python 3.12.6  
**Status:** Phase 1 Completed (Dataset Investigation & Sample Validation)

---

## A. Sources Investigated

1. **KCC-Krishi RAG/SFT Advisory Corpus**
   - **Repository:** [`uralstech/kcc-krishi-rag-sft-advisory-corpus`](https://huggingface.co/datasets/uralstech/kcc-krishi-rag-sft-advisory-corpus)
   - **Curator / Derived Work:** Udayashankar Ravikumar / URAV Advanced Learning Systems Pvt. Ltd. (Preprint: Zenodo DOI 10.5281/zenodo.21040992).
   - **Source Provenance:** Kisan Call Centre Transcripts of Farmers Queries and Answers, Ministry of Agriculture and Farmers Welfare, Government of India (accessed via Data.gov.in / AIKosh).
   - **Documentation Inspected:** `README.md`, `DATA_DICTIONARY.md`, `LICENSE.md`, `SOURCE_RISK_NOTES.md`, `release_summary.json`, `dataset_summary_step12p_locked.json`, `train_filter_report.json`, and `sample_review_step12p_locked.csv`.

2. **Official Government of India Kisan Call Centre Dataset**
   - **Portal:** [Open Government Data (OGD) Platform India (`data.gov.in`)](https://data.gov.in/datasets_webservices/datasets/6622307)
   - **Dataset ID / Resource:** `6622307` (*District-wise and Month-wise Queries of Farmers in Kisan Call Centre*)
   - **Publisher:** Department of Agriculture and Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Government of India.
   - **Documentation Inspected:** OGD Platform India catalog metadata, API specifications, and Government Open Data License (GODL) India documentation.

---

## B. Dataset Comparison

| Comparison Dimension | Official Data.gov.in (`6622307`) | KCC-Krishi Corpus (`uralstech/...`) | Project Impact & Assessment |
| :--- | :--- | :--- | :--- |
| **Accessibility on Windows** | Requires Janparichay SSO account, API key registration, rate-limited paginated REST endpoints. Web pages serve JS shells. | Direct, public HTTPS access via standard REST and Git LFS endpoints (`/resolve/main/`). | **KCC-Krishi wins**: Zero-friction download, no credential exposure or complex auth on Windows. |
| **Language & Script** | Multilingual raw text: Indian scripts (Devanagari, Telugu, Tamil, Bengali, etc.), Latin transliterations, and English. | Clean English: Faithful English translations of regional language queries and answers. | **KCC-Krishi wins**: Directly matches our English initial application requirement without building translation layers. |
| **File Formats & Splits** | Fragmented monthly CSV files across states/years (2014–2024), totaling multiple gigabytes. | Unified `.jsonl` files (`train`, `validation`, `test`, `safety_eval`) + `.csv` review file + metadata JSON. | **KCC-Krishi wins**: Standard JSONL chat and document structures are plug-and-play for RAG chunking and LLM benchmarking. |
| **Dataset Size** | Hundreds of monthly CSVs, >10M raw rows (>25 GB uncompressed). | 407,610 total curated rows (~780 MB total; `train`: 365k, `val`: 20k, `test`: 20k, `safety_eval`: 2k). | **KCC-Krishi wins**: Fits comfortably on local Windows developer storage without crashing RAM or storage. |
| **Data Quality & Hygiene** | Headerless lines, corrupted rows, numbers in place of text labels, missing columns. | Pre-cleaned, bad JSON/rows eliminated, language corruption and malformed queries filtered or salvaged. | **KCC-Krishi wins**: Saves weeks of raw ETL, regex deduplication, and anomaly scrubbing. |
| **Hallucination & Safety Metadata** | None. Pure call-log transcript with no safety annotations. | Contains explicit `qc_flags`, `qc_severity`, `qc_confidence`, and pre-computed `rag_use_mode`. | **KCC-Krishi wins**: Indispensable for hallucination guardrails and safe refusal mechanisms. |
| **Out-of-Scope / Stale Data Handling** | Mixes current agronomy with 10-year-old weather forecasts and expired scheme application deadlines. | Explicitly separates domains and routes weather/scheme data into `safety_refusal` or `historical_only`. | **KCC-Krishi wins**: Provides ready-made ground truth for safe abstention evaluation. |

---

## C. License and Access Findings

### 1. Data.gov.in License Terms
- **License:** Government Open Data License — India (GODL-India).
- **Permissions:** Worldwide, royalty-free, non-exclusive license to use, adapt, translate, distribute, and reproduce the data for both commercial and non-commercial purposes.
- **Obligations:**
  - Mandatory attribution: *"Source: Data.gov.in / Ministry of Agriculture and Farmers Welfare"*.
  - Must not present data in a misleading manner or claim government endorsement.
- **Access Reality:** Direct URL access (`6622307`) renders an empty sandbox HTML container requiring dynamic JavaScript hydration. Access requires programmatic authentication through Janparichay single sign-on to fetch an API token.

### 2. KCC-Krishi Corpus License Terms
- **License Tag:** `other` (Custom permissive research and attribution notice in `LICENSE.md`).
- **Composition:** Combines GODL-India source records with derived processing, translations, and quality-control tagging generated by URAV Advanced Learning Systems Private Limited.
- **Attribution Requirements:**
  - Authoritative source: *Kisan Call Centre (KCC) Transcripts of Farmers Queries and Answers, Ministry of Agriculture and Farmers Welfare, Data.gov.in / AIKosh*.
  - Derived work: *KCC-Krishi Project, URAV Advanced Learning Systems Private Limited*.
- **Crucial Restrictions & Disclaimers:**
  - **Not agronomic ground truth:** Must not be deployed as an unrestricted field agronomic authority.
  - **No PII / Credential usage:** Explicitly warns users not to operationalize source-derived IDs, passwords, PINs, or phone numbers.
  - **No warranty:** Dataset is strictly for scientific research, RAG benchmarking, and controlled assistant prototyping.

---

## D. Sample Data Inspection

Actual records from both raw review CSVs and JSONL splits were downloaded, parsed, and inspected locally.

### 1. Available Fields in Inspected Records
The following fields were verified across actual records:
- **Location Fields:** `state` (e.g., TAMILNADU, TELANGANA, GUJARAT), `district` (e.g., THANJAVUR, MEERUT, SIVASAGAR), and `block` (e.g., JOKIHAT, PUNASA).
- **Agronomic Context:** `crop` (e.g., Paddy, Cotton, Tea, Satawar, Fodder Sorghum), `category` (Cereals, Fiber Crops), `sector` (Agriculture, Horticulture).
- **Query & Advisory:** `query_type` (Nutrient Management, Plant Protection, Weed Management), `user_prompt` (Farmer's question), and `expert_response` (Advisory recommendation).
- **Temporal Metadata:** `created_at` timestamp (dating from 2014 to 2024).
- **Safety & Quality Metadata:** `advisory_domain`, `qc_flags`, `qc_severity` (low, medium, high), `qc_confidence` (low, medium, high), `rag_use_mode` (`use_as_evidence`, `use_with_caution`, `historical_only`), and `example_type` (`direct`, `rag_aware`, `caution`, `safety_refusal`).

### 2. Discovered Data-Quality & Safety Issues
1. **Raw Metadata Corruption:**
   - In 18,055 records, `crop` is a numeric code rather than a text string.
   - In 37,973 records, `query_type` contains a raw numeric database ID (e.g., `query_type: 187`).
   - In several raw records, the `village` column contains season names (e.g., `village: KHARIF`).
2. **Translation Artifacts & Number Mismatches:**
   - In Record `kcc_00140984` (Cotton NPK advisory translated from Telugu), the NPK fertilizer ratio `13:0:45` was split by translation into *"13; 0; Spray 45 potassium nitrate 5 gm + Agromini Max 3 gm per litre"*. The dataset's QC pipeline flagged this as `unit_mismatch_in_dose_or_chemical_row` and routed it to `caution`.
3. **Outdated Agricultural Practices & Stale Agrochemicals:**
   - Historical records from 2014–2018 recommend chemicals that are now severely restricted or banned in various Indian states (e.g., Carbofuran 3G, Monocrotophos).
   - Records mentioning historical PM-Kisan installments (e.g. 10th installment in 2022) are obsolete. KCC-Krishi correctly routed these to `safety_refusal`.
4. **Terse Evidence & Missing Diagnostics:**
   - Many farmer questions lack crucial context (e.g., *“Enquired abotu the control measures of termite in tea”* without noting infestation severity, soil moisture, or harvest interval).
5. **Presence of Audit Anomalies:**
   - The corpus audit log reveals 708 instances of `ml_per_ml`, 776 unit warnings, and 83 instances of `acres_of_water`.

---

## E. Recommended Dataset & Rationale

### **Recommendation: Adopt KCC-Krishi RAG/SFT Advisory Corpus**
We recommend using **`uralstech/kcc-krishi-rag-sft-advisory-corpus`** as our primary project dataset, while referencing the official Data.gov.in KCC portal for authoritative provenance.

### **Core Rationale:**
1. **Direct Alignment with English RAG Scope:** All regional queries and recommendations have been pre-translated into English while preserving the original agricultural concepts.
2. **Built-in Hallucination Guardrails:** Unlike raw government dumps, KCC-Krishi provides verified QC flags (`qc_severity`, `qc_flags`, `rag_use_mode`) that allow us to automatically index only high-confidence evidence into our retrieval index while discarding or isolating risky records.
3. **Turnkey Evaluation Benchmark (`safety_eval.jsonl`):** Contains 2,000 pre-curated test cases specifically designed to evaluate whether a RAG system hallucinates or correctly refuses outdated weather forecasts, stale government schemes, and unverified chemical dosages.
4. **Local Windows Viability:** With structured JSONL files totaling under 800 MB, the dataset can be inspected, filtered, embedded, and managed locally on Windows without enterprise database infrastructure.

### **Limitations to Guard Against:**
- **Historical Nature:** Must never treat 2014–2022 chemical recommendations as 2026 legal guidelines.
- **Translation Noise:** Records tagged with `high` QC severity or `unit_mismatch` must either be normalized or excluded from the knowledge base.

---

## F. Proposed RAG Knowledge-Base Schema

To ensure that the retrieval system only serves safe, grounded agronomic advice, we propose the following document schema for our vector knowledge base:

```json
{
  "doc_id": "kcc_doc_00563633",
  "crop": "Satawar",
  "crop_category": "Medicinal / Commercial",
  "state": "UTTAR PRADESH",
  "district": "BAREILLY",
  "advisory_domain": "plant_protection",
  "query_topic": "Insect Control",
  "problem_description": "Farmer asks for insect control methods in Satawar crop.",
  "expert_advisory_text": "Kisan bhai Quinalphos 25% EC 500 ml/acre 200 liter pani me milakar chidkav kare.",
  "original_year": 2018,
  "rag_use_mode": "use_as_evidence",
  "qc_severity": "low",
  "qc_confidence": "high",
  "qc_flags": [
    "dose_bearing_expert_advice",
    "chemical_or_fertilizer_row"
  ],
  "safety_disclaimer": "Verify pest identity, crop stage, and local state agricultural university recommendations before chemical application.",
  "provenance": {
    "source": "Kisan Call Centre (KCC)",
    "publisher": "Data.gov.in / Ministry of Agriculture and Farmers Welfare",
    "curator": "KCC-Krishi Corpus (URAV Advanced Learning Systems)"
  }
}
```

### Knowledge Base Ingestion Filter Rules:
1. **Include Only:** `advisory_domain` in `['plant_protection', 'nutrient_management', 'weed_management', 'cultural_practices', 'varieties']`.
2. **Exclude Strictly:** `advisory_domain` in `['weather', 'market_information', 'government_scheme']` (these require live dynamic APIs, not static historical RAG).
3. **QC Filter:** Discard records where `crop` is numeric, `qc_severity == 'high'`, or `rag_use_mode == 'exclude'`.

---

## G. Proposed Evaluation Methodology

### 1. Evaluation Dimensions & Metrics
1. **Evidence Grounding (Faithfulness):**
   - *Metric:* RAG Triad Faithfulness score / Token Attribution.
   - *Definition:* Does every claim, chemical name, dilution ratio, and timing mentioned in the answer trace back directly to the retrieved KCC document?
2. **Answer Correctness & Completeness:**
   - *Metric:* Semantic Similarity against target expert advice + Agronomic completeness checklist.
   - *Definition:* Does the response accurately communicate the problem diagnosis and recommended practice?
3. **Safe Abstention & Guardrail Precision:**
   - *Metric:* Abstention Accuracy & Fallback Appropriateness.
   - *Definition:* When presented with an ambiguous query, historical weather request, or outdated scheme question, does the system safely decline and guide the farmer to local agricultural extension offices (KVKs)?

### 2. Evaluation Dataset Structure
We utilize a dedicated evaluation suite derived from `safety_eval.jsonl` and held-out `test.jsonl`:
- **50 Grounded Agronomic Scenarios:** Testing accurate retrieval and exact dosage reproduction without numeric hallucinations.
- **25 Outdated / Time-Sensitive Scenarios:** (Historical weather, outdated PM-Kisan queries, old market rates) testing proper refusal.
- **25 Insufficient Context / High-Risk Scenarios:** (Unspecified pest symptoms, high-toxicity chemical requests) testing clarifying questions and safety warnings.

---

## H. Exact Next Steps for Beginning Implementation

1. **Step 1: Ingestion & Filtering Script**
   - Filter `validation.jsonl` and `train.jsonl` according to our RAG knowledge-base filter rules.
   - Export 1,000–5,000 high-quality, verified agronomic evidence chunks to `data/processed/knowledge_base.jsonl`.
2. **Step 2: Local Vector Store Setup (Windows)**
   - Initialize a local vector database using ChromaDB or FAISS.
   - Use a lightweight, open-source embedding model suitable for local Windows inference (e.g., `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`).
3. **Step 3: Dual-Stage Hallucination Guard Architecture**
   - **Pre-retrieval Guard:** Classifies query domain (routes weather, market prices, and government schemes to live disclaimers or abstention).
   - **Post-generation Guard:** Compares generated chemicals, quantities, and units against the retrieved evidence chunk using an exact token-match verifier.
4. **Step 4: Automated Evaluation Pipeline**
   - Run the benchmark evaluation script against our saved 20 abstention test cases and 30 knowledge-base test cases.
   - Measure baseline retrieval recall, faithfulness, and safe abstention rate.
