"""
Agronomy Advisory Assistant -- User-Facing Streamlit Application (Phase 5).

Provides a clean, academic-grade demonstration of the complete 5-stage
Hallucination-Guarded RAG Pipeline:
  1. Pre-retrieval QueryGuard
  2. Semantic Retrieval (ChromaDB)
  3. Relevance Threshold Gate
  4. Grounded Synthesis (Modular LLM Providers: Ollama / Mock / API)
  5. Post-Generation Structural Safety Verifier
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.pipeline import GuardedRAGPipeline, PipelineResult
from src.generation.answer_gen import AnswerGenerator
from src.generation.providers import get_llm_provider
from src.ui.helpers import check_ollama_service, extract_pipeline_diagnostics, format_stage_badge
from src import config

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agronomy Advisory Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling for Academic Demonstration
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4d3e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 0.85rem 1.1rem;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        color: #92400e;
        line-height: 1.45;
    }
    .stage-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        background-color: #e2e8f0;
        color: #1e293b;
    }
    .stage-pill-active {
        background-color: #0d9488;
        color: white;
    }
    .card-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Pipeline Factory
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(provider_name: str) -> GuardedRAGPipeline:
    """Instantiates the pipeline with the chosen generation provider."""
    provider = get_llm_provider(provider_name)
    answer_gen = AnswerGenerator(provider=provider)
    return GuardedRAGPipeline(answer_generator=answer_gen)


# -----------------------------------------------------------------------------
# Sidebar: System Configuration & Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")

    # LLM Provider Configuration
    st.subheader("Language Model Provider")
    provider_options = ["mock", "ollama", "api"]
    default_provider_index = 0
    if config.LLM_PROVIDER in provider_options:
        default_provider_index = provider_options.index(config.LLM_PROVIDER)

    selected_provider = st.selectbox(
        "Active Generation Provider",
        options=provider_options,
        index=default_provider_index,
        help="Select between deterministic offline mock, local private Ollama runtime, or cloud API.",
    )

    # Local Ollama Status Check
    ollama_status = check_ollama_service()
    if ollama_status["available"]:
        st.success(f"🟢 Ollama Local Service: Connected ({ollama_status['url']})")
        if ollama_status["models"]:
            st.caption(f"Installed models: {', '.join(ollama_status['models'])}")
    else:
        st.warning(f"🟡 Ollama Local Service: Offline ({ollama_status['error']})")
        st.caption("Ollama provider will fall back to deterministic safe template if selected.")

    # Vector Database Storage Status
    st.subheader("Knowledge Base")
    vector_dir = config.VECTOR_STORE_DIR
    index_exists = (vector_dir / "chroma.sqlite3").exists()
    if index_exists:
        st.success(f"🟢 ChromaDB Vector Store: Active")
        st.caption(f"Path: `{vector_dir.name}/`")
    else:
        st.warning("🟡 ChromaDB: Unindexed (Run `build_index.py`)")

    # Gate Threshold Settings
    st.subheader("Calibrated Gate Thresholds")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            f"<div class='metric-label'>Sufficient</div>"
            f"<div class='metric-value'>≥ {config.SIMILARITY_THRESHOLD_BASELINE + 0.10:.2f}</div>",
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            f"<div class='metric-label'>Min Viable</div>"
            f"<div class='metric-value'>≥ 0.30</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "**Academic RAG Architecture:**\n"
        "• Stage 1: Pre-retrieval QueryGuard\n"
        "• Stage 2: ChromaDB Semantic Search\n"
        "• Stage 3: Relevance Gate & Mismatch\n"
        "• Stage 4: Constrained Generation\n"
        "• Stage 5: Structural Safety Verifier\n\n"
        "*Scientific Note: Hallucinations are reduced, NOT eliminated.*"
    )

# -----------------------------------------------------------------------------
# Main Content: Header & Agronomic Disclaimer
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>🌾 Agronomy Advisory Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Hallucination-Guarded Retrieval-Augmented Generation (RAG) with Historical Evidence Provenance</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='disclaimer-box'>
        <b>CRITICAL AGRONOMIC DISCLAIMER:</b><br>
        This research prototype indexes historical Kisan Call Centre (KCC) records (2014–2024) for evaluation.
        It <b>does NOT automate agricultural chemical advice</b>. Recommendations may contain restricted formulations
        or regional inaccuracies. All agrochemical decisions require independent field validation by qualified extension
        specialists (KVK / SAU) and inspection of authorized product labels.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Query Console & Benchmark Scenarios
# -----------------------------------------------------------------------------
st.subheader("💬 Farmer Query Console")

# Preset benchmark query definitions
BENCHMARK_QUERIES = {
    "Paddy Stem Borer Control (Supported In-Domain)": "How do I control stem borer in my paddy crop?",
    "Ambiguous Symptoms (Missing Crop Context)": "My plants have yellow spots and the leaves are curling. What spray should I use?",
    "Weather & PM Kisan Inquiry (Dynamic Request)": "Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?",
    "React/Node.js Question (Off-Topic Technical)": "How do I configure a React frontend router in Node.js?",
    "Dragon Fruit Winter Grafting (Unindexed Crop)": "What is the recommended grafting method for Brazilian Dragon Fruit in winter?",
}

preset_choice = st.selectbox(
    "Load an Academic Evaluation Scenario (or write your own below):",
    options=["Custom Input"] + list(BENCHMARK_QUERIES.keys()),
    index=0,
)

# Determine default query text
initial_query = ""
if preset_choice != "Custom Input":
    initial_query = BENCHMARK_QUERIES[preset_choice]

user_query = st.text_input(
    "Enter agricultural question:",
    value=initial_query,
    placeholder="e.g., How to control stem borer in paddy crop?",
    key="query_input",
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    submit_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Pipeline Execution & Result Display
# -----------------------------------------------------------------------------
if submit_clicked and user_query.strip():
    with st.spinner("Processing through 5-stage hallucination-guarded pipeline..."):
        try:
            pipeline = load_pipeline(selected_provider)
            result = pipeline.answer(user_query.strip())
            diag = extract_pipeline_diagnostics(result)
        except Exception as e:
            st.error(f"Pipeline execution encountered an unexpected error: {e}")
            st.stop()

    # Display Pipeline Progression Breadcrumb
    st.markdown("### 🚦 Pipeline Execution Trace")
    stages_html = "".join(
        f"<span class='stage-pill stage-pill-active'>{format_stage_badge(s)}</span>"
        for s in result.pipeline_stages
    )
    st.markdown(stages_html, unsafe_allow_html=True)
    st.markdown("")

    # Display Response Card
    st.markdown("### 📋 Advisory Response")
    final_ans = result.final_answer

    if result.was_answered and final_ans:
        # Determine status banner
        if final_ans.status.value == "success":
            st.success("✅ **ADVISORY RESPONSE (Evidence-Grounded)**")
        else:
            st.warning("⚠️ **ADVISORY RESPONSE (Low-Confidence Evidence)**")

        # Uncertainty callout
        if final_ans.uncertainty_note:
            st.info(f"**Uncertainty Notice:** {final_ans.uncertainty_note}")

        # Advisory Text
        st.markdown(final_ans.answer_text)

        # Citations & Provenance
        if final_ans.source_ids:
            with st.expander("📌 Source Attribution & Record Provenance", expanded=True):
                st.markdown(f"**Cited KCC Record IDs:** `{', '.join(final_ans.source_ids)}`")
                if final_ans.crops_cited:
                    st.markdown(f"**Evidence Crop(s):** {', '.join(set(final_ans.crops_cited))}")
                st.caption(f"**Source Attribution:** {final_ans.provenance_note}")
                st.caption(f"**Regulatory Notice:** {final_ans.safety_notice}")

    else:
        # Abstention or Clarification Response
        action = result.guard_decision.action.value if result.guard_decision else "abstained"

        if action in ("clarify_crop", "clarify_symptoms"):
            st.info("ℹ️ **CROP / SYMPTOM CLARIFICATION REQUIRED**")
            st.markdown(
                f"**The system cannot safely retrieve advice without knowing the crop:**\n\n"
                f"{result.guard_decision.clarification_prompt}"
            )
        elif action == "abstain_dynamic":
            st.info("ℹ️ **REAL-TIME DATA REQUIRED (Safe Abstention)**")
            st.markdown(final_ans.abstention_reason if final_ans else result.guard_decision.reason)
        elif action == "abstain_offtopic":
            st.info("ℹ️ **QUERY NOT RELATED TO AGRICULTURE (Safe Abstention)**")
            st.markdown(final_ans.abstention_reason if final_ans else result.guard_decision.reason)
        elif result.was_blocked:
            st.error("⛔ **BLOCKED BY SAFETY VERIFIER: Ungrounded Claims Detected**")
            st.markdown(final_ans.abstention_reason if final_ans else "Blocked due to ungrounded claims.")
        else:
            st.warning("⚠️ **UNABLE TO ANSWER: No Sufficient Evidence Found**")
            st.markdown(final_ans.abstention_reason if final_ans else "Evidence score fell below viable threshold.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Diagnostic Tabs for Academic Evaluation
    # -------------------------------------------------------------------------
    tab_evidence, tab_diagnostics, tab_trace = st.tabs([
        "📚 Supporting Evidence Chunks",
        "🛡️ Five-Stage Guard Diagnostics",
        "📜 Raw Trace & JSON",
    ])

    with tab_evidence:
        st.markdown("#### Retrieved KCC Historical Evidence Chunks")
        if result.retrieval_results:
            st.caption(f"Vector search returned {len(result.retrieval_results)} candidate chunks.")
            for i, chunk in enumerate(result.retrieval_results, 1):
                with st.expander(
                    f"Chunk {i}: [{chunk.chunk_id}] -- Crop: {chunk.crop_clean} (Cosine Similarity: {chunk.similarity_score:.4f})",
                    expanded=(i == 1),
                ):
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.markdown(f"**Crop:** {chunk.crop_clean}")
                    col_e2.markdown(f"**Location:** {chunk.district}, {chunk.state}")
                    year_val = chunk.original_year if chunk.original_year else "Historical"
                    col_e3.markdown(f"**Record Year:** {year_val}")

                    st.markdown(f"**Historical Farmer Query:** *\"{chunk.farmer_query}\"*")
                    st.markdown(f"**KCC Advisory Evidence:**\n> {chunk.expert_advisory}")
                    st.caption(f"Domain: {chunk.advisory_domain} | Topic: {chunk.query_type} | QC: {chunk.qc_severity}")
        else:
            st.info("No chunks retrieved. The query was intercepted by pre-retrieval QueryGuard before vector search.")

    with tab_diagnostics:
        st.markdown("#### Real-Time Pipeline Stage Metrics")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### Stage 1: QueryGuard")
            g = diag["guard"]
            st.markdown(f"• **Action:** `{g['action']}`")
            st.markdown(f"• **Reason:** {g['reason']}")
            st.markdown(f"• **Detected Crop:** {g['detected_crop']}")
            st.markdown(f"• **Agricultural Context:** `{g['is_agricultural']}`")
            st.markdown(f"• **Dynamic Request:** `{g['is_dynamic']}`")

            st.markdown("##### Stage 4: Answer Generator")
            gen = diag["generation"]
            st.markdown(f"• **Status:** `{gen['status']}`")
            st.markdown(f"• **Provider Used:** `{selected_provider}`")
            st.markdown(f"• **Records Cited:** {', '.join(gen['source_ids_cited']) if gen['source_ids_cited'] else 'None'}")
            st.markdown(f"• **Uncertainty Flag:** `{gen['has_uncertainty_note']}`")

        with col_d2:
            st.markdown("##### Stage 2 & 3: Retrieval & RelevanceGate")
            gt = diag["gate"]
            st.markdown(f"• **Verdict:** `{gt['verdict']}`")
            st.markdown(f"• **Top Similarity Score:** `{gt['best_score']}`")
            st.markdown(f"• **Gate Rationale:** {gt['reason']}")
            st.markdown(f"• **Candidate Chunks:** {gt['retrieved_chunk_count']}")
            st.markdown(f"• **Crop Mismatch:** `{gt['crop_mismatch_detected']}`")

            st.markdown("##### Stage 5: Post-Generation Verifier")
            v = diag["verification"]
            st.markdown(f"• **Verification Status:** `{v['status']}`")
            st.markdown(f"• **Safe to Present:** `{v['is_safe_to_present']}`")
            st.markdown(f"• **Hallucination Issues:** {v['issues'] if v['issues'] else 'None detected'}")
            st.markdown(f"• **Warnings:** {v['warnings'] if v['warnings'] else 'None'}")

    with tab_trace:
        st.markdown("#### Human-Readable Pipeline Log")
        st.text(result.to_display_text())

        st.markdown("#### Structured Diagnostic Payload")
        st.json(diag)

elif submit_clicked and not user_query.strip():
    st.warning("Please enter a question or select a preset evaluation scenario above.")
