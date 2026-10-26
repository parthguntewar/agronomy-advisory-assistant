"""
Agronomy Advisory Assistant -- Professional Agricultural Intelligence Platform.

Redesigned user-facing application providing an academic-grade demonstration
of the 5-stage Hallucination-Guarded RAG Pipeline:
  1. Pre-retrieval QueryGuard
  2. Semantic Vector Retrieval (ChromaDB)
  3. Calibrated Relevance Gate & Crop Mismatch Detection
  4. Evidence-Grounded Synthesis (Modular Providers: Ollama / Mock / API)
  5. Post-Generation Structural Safety Verifier

Designed in a minimal, professional aesthetic without emojis.
"""

from __future__ import annotations

import sys
import json
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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Design System & Custom CSS Injection
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography and Base Palette */
    :root {
        --color-primary: #1b4332;
        --color-primary-hover: #2d6a4f;
        --color-sage: #52b788;
        --color-sage-light: #e8f5e9;
        --color-bg-subtle: #f8f9fa;
        --color-surface: #ffffff;
        --color-border: #e5e7eb;
        --color-border-dark: #d1d5db;
        --color-text-main: #111827;
        --color-text-muted: #4b5563;
        --color-text-subtle: #6b7280;
    }

    /* Streamlit Container Resets */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top Brand Navigation */
    .brand-header {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 1.1rem;
        margin-bottom: 1.4rem;
    }
    .brand-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1b4332;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.9rem;
        font-weight: 400;
        color: #4b5563;
        margin-top: 0.2rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
    }
    .status-dot-active {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
    }
    .status-dot-offline {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #f59e0b;
    }

    /* Section Cards & Panels */
    .ui-panel {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }
    .panel-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .panel-caption {
        font-size: 0.86rem;
        color: #4b5563;
        line-height: 1.45;
        margin-bottom: 1rem;
    }

    /* Regulatory Notice */
    .regulatory-notice {
        background-color: #fafafa;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #1b4332;
        padding: 0.85rem 1rem;
        border-radius: 4px;
        font-size: 0.83rem;
        line-height: 1.45;
        color: #374151;
        margin-bottom: 1.2rem;
    }
    .notice-heading {
        font-weight: 600;
        color: #1b4332;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }

    /* Stage Progression Badges */
    .stage-track {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        align-items: center;
        margin: 0.8rem 0 1.2rem 0;
    }
    .stage-badge {
        font-size: 0.76rem;
        font-weight: 500;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
    }
    .stage-badge-active {
        background-color: #1b4332;
        color: #ffffff;
        border-color: #1b4332;
    }
    .stage-arrow {
        color: #9ca3af;
        font-size: 0.8rem;
    }

    /* Result Badges */
    .verdict-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .verdict-success {
        background-color: #e8f5e9;
        color: #1b4332;
        border: 1px solid #c8e6c9;
    }
    .verdict-warning {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .verdict-info {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .verdict-blocked {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Metric Grid */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .metric-cell {
        background-color: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 0.7rem 0.85rem;
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .metric-data {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
    }

    /* Form and Button Styling */
    .stButton > button {
        background-color: #1b4332 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: 1px solid #1b4332 !important;
        border-radius: 4px !important;
        padding: 0.45rem 1.4rem !important;
        transition: background-color 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #2d6a4f !important;
        border-color: #2d6a4f !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.2rem;
        font-size: 0.92rem;
        font-weight: 500;
        color: #4b5563;
    }
    .stTabs [aria-selected="true"] {
        color: #1b4332 !important;
        font-weight: 600 !important;
        border-bottom-color: #1b4332 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Pipeline Factory & Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_pipeline(provider_name: str) -> GuardedRAGPipeline:
    """Instantiates and caches the RAG pipeline with the chosen provider."""
    provider = get_llm_provider(provider_name)
    answer_gen = AnswerGenerator(provider=provider)
    return GuardedRAGPipeline(answer_generator=answer_gen)


# -----------------------------------------------------------------------------
# Application State Initialization
# -----------------------------------------------------------------------------
if "active_provider" not in st.session_state:
    st.session_state.active_provider = config.LLM_PROVIDER if config.LLM_PROVIDER in ("mock", "ollama", "api") else "mock"

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# Check local services
ollama_info = check_ollama_service()
vector_store_active = (config.VECTOR_STORE_DIR / "chroma.sqlite3").exists()

# -----------------------------------------------------------------------------
# Header: Top Branding & System Status
# -----------------------------------------------------------------------------
col_brand_l, col_brand_r = st.columns([3, 2])

with col_brand_l:
    st.markdown(
        """
        <div class="brand-header">
            <h1 class="brand-title">Agronomy Advisory Assistant</h1>
            <div class="brand-subtitle">Evidence-grounded agricultural intelligence and advisory verification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_brand_r:
    provider_label = st.session_state.active_provider.upper()
    ollama_dot = "status-dot-active" if ollama_info["available"] else "status-dot-offline"
    ollama_text = "Ollama Local Service: Operational" if ollama_info["available"] else "Ollama Local Service: Offline"
    kb_text = "Knowledge Base: Active (26 Records)" if vector_store_active else "Knowledge Base: Not Indexed"

    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem; padding-top: 0.2rem;">
            <div class="status-badge">
                <span class="{ollama_dot}"></span>
                <span>{ollama_text}</span>
            </div>
            <div class="status-badge">
                <span>{kb_text}</span>
                <span style="color: #9ca3af;">|</span>
                <span>Provider: {provider_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_advisory, tab_evidence, tab_diagnostics = st.tabs([
    "Advisory Workspace",
    "Evidence Explorer",
    "System Diagnostics",
])

# =============================================================================
# TAB 1: ADVISORY WORKSPACE (Main Interface)
# =============================================================================
with tab_advisory:
    # Regulatory Notice
    st.markdown(
        """
        <div class="regulatory-notice">
            <div class="notice-heading">Agronomic Regulatory Notice</div>
            This system indexes historical Kisan Call Centre (KCC) records for research and evaluation.
            It does not prescribe or automate agricultural chemical advice. Historical recommendations
            may reflect restricted formulations or localized conditions. Agrochemical decisions must be
            independently confirmed with local Krishi Vigyan Kendra (KVK) extension officers and official product labels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Input Panel
    st.markdown("<div class='panel-title'>Agricultural Advisory</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-caption'>Submit an inquiry regarding crop management, pest identification, plant disease control, or cultivation practices. The assistant grounds responses strictly in validated Kisan Call Centre historical records.</div>",
        unsafe_allow_html=True,
    )

    # Benchmark Evaluation Scenarios
    BENCHMARKS = [
        ("Paddy Stem Borer Control", "How do I control stem borer in my paddy crop?"),
        ("Ambiguous Symptoms (No Crop)", "My plants have yellow spots and the leaves are curling. What spray should I use?"),
        ("Weather & Government Scheme", "Will it rain tomorrow in Mehsana, Gujarat? Also when is the PM Kisan installment date?"),
        ("Software Configuration (Off-Topic)", "How do I configure a React frontend router in Node.js?"),
        ("Dragon Fruit Grafting (Unindexed)", "What is the recommended grafting method for Brazilian Dragon Fruit in winter?"),
    ]

    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.4rem;'>Benchmark Evaluation Scenarios:</div>", unsafe_allow_html=True)
    cols_btn = st.columns(len(BENCHMARKS))

    for idx, (label, query_sample) in enumerate(BENCHMARKS):
        with cols_btn[idx]:
            if st.button(label, key=f"bench_{idx}", use_container_width=True):
                st.session_state.query_text = query_sample

    # Query Input
    input_text = st.text_input(
        "Enter agricultural question:",
        value=st.session_state.query_text,
        placeholder="e.g., How to control stem borer in paddy crop?",
        label_visibility="collapsed",
    )

    col_sub_l, col_sub_r = st.columns([1, 4])
    with col_sub_l:
        generate_clicked = st.button("Generate Advisory", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Execution & Results Presentation
    # -------------------------------------------------------------------------
    if generate_clicked and input_text.strip():
        with st.spinner("Executing 5-stage verification and grounded retrieval pipeline..."):
            pipeline = get_cached_pipeline(st.session_state.active_provider)
            result = pipeline.answer(input_text.strip())
            diag = extract_pipeline_diagnostics(result)

        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

        # Stage Progression Badges
        stage_items = []
        for s in result.pipeline_stages:
            stage_items.append(f"<span class='stage-badge stage-badge-active'>{format_stage_badge(s)}</span>")
        stage_chain = " <span class='stage-arrow'>&rarr;</span> ".join(stage_items)

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #6b7280; margin-bottom: 0.35rem;">Pipeline Stages Executed:</div>
            <div class="stage-track">{stage_chain}</div>
            """,
            unsafe_allow_html=True,
        )

        # Advisory Result Presentation
        ans = result.final_answer

        if result.was_answered and ans:
            verdict_class = "verdict-success" if ans.status.value == "success" else "verdict-warning"
            verdict_title = "Verified Evidence-Grounded Advisory" if ans.status.value == "success" else "Low-Confidence Advisory (Uncertain Evidence)"

            st.markdown(f"<span class='verdict-badge {verdict_class}'>{verdict_title}</span>", unsafe_allow_html=True)

            if ans.uncertainty_note:
                st.info(f"Uncertainty Notice: {ans.uncertainty_note}")

            st.markdown(ans.answer_text)

            # Attribution section
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            with st.expander("Source Attribution & Evidence Metadata", expanded=True):
                st.markdown(f"**Cited KCC Record IDs:** `{', '.join(ans.source_ids)}`")
                if ans.crops_cited:
                    st.markdown(f"**Crops Cited in Evidence:** {', '.join(set(ans.crops_cited))}")
                st.caption(f"**Provenance:** {ans.provenance_note}")
                st.caption(f"**Agronomic Safety Disclaimer:** {ans.safety_notice}")

        else:
            action = result.guard_decision.action.value if result.guard_decision else "abstained"

            if action in ("clarify_crop", "clarify_symptoms"):
                st.markdown("<span class='verdict-badge verdict-info'>Clarification Required</span>", unsafe_allow_html=True)
                st.markdown(
                    f"**The system cannot retrieve advisory evidence without identifying the affected crop:**\n\n"
                    f"{result.guard_decision.clarification_prompt}"
                )
            elif action == "abstain_dynamic":
                st.markdown("<span class='verdict-badge verdict-info'>Real-Time Data Required (Abstained)</span>", unsafe_allow_html=True)
                st.markdown(ans.abstention_reason if ans else result.guard_decision.reason)
            elif action == "abstain_offtopic":
                st.markdown("<span class='verdict-badge verdict-info'>Out-of-Scope Query (Abstained)</span>", unsafe_allow_html=True)
                st.markdown(ans.abstention_reason if ans else result.guard_decision.reason)
            elif result.was_blocked:
                st.markdown("<span class='verdict-badge verdict-blocked'>Blocked by Post-Generation Verifier</span>", unsafe_allow_html=True)
                st.markdown(ans.abstention_reason if ans else "The generated response was blocked due to ungrounded claims.")
            else:
                st.markdown("<span class='verdict-badge verdict-warning'>No Sufficient Evidence Found</span>", unsafe_allow_html=True)
                st.markdown(ans.abstention_reason if ans else "Retrieved evidence did not meet minimum relevance thresholds.")

        # Supporting Evidence Preview
        if result.retrieval_results:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            with st.expander(f"Supporting Evidence Records ({len(result.retrieval_results)} retrieved candidates)", expanded=False):
                for idx, chunk in enumerate(result.retrieval_results, 1):
                    st.markdown(f"**Record {idx}: `{chunk.chunk_id}` | Crop: {chunk.crop_clean} | Similarity: {chunk.similarity_score:.4f}**")
                    st.markdown(f"Location: {chunk.district}, {chunk.state} | Year: {chunk.original_year or 'Historical'} | Domain: {chunk.advisory_domain}")
                    st.markdown(f"> *Expert Advisory Transcript:* {chunk.expert_advisory}")
                    st.markdown("---")

    elif generate_clicked and not input_text.strip():
        st.warning("Please enter a question or select an evaluation scenario above.")


# =============================================================================
# TAB 2: EVIDENCE EXPLORER
# =============================================================================
with tab_evidence:
    st.markdown("<div class='panel-title'>Evidence Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-caption'>Inspect the underlying Kisan Call Centre (KCC) records used for semantic retrieval. These records represent validated transcripts of agricultural advisory interactions from Government of India call centres.</div>",
        unsafe_allow_html=True,
    )

    kb_file = config.PROCESSED_KB
    if kb_file.exists():
        records = []
        with open(kb_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-cell">
                    <div class="metric-title">Validated Chunks</div>
                    <div class="metric-data">{len(records)}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-title">Indexed Crops</div>
                    <div class="metric-data">{len(set(r.get('crop_clean') for r in records if r.get('crop_clean')))}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-title">Embedding Model</div>
                    <div class="metric-data">MiniLM-L6-v2</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-title">Dimensions</div>
                    <div class="metric-data">384</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        filter_crop = st.selectbox(
            "Filter records by crop:",
            options=["All Crops"] + sorted(list(set(r.get("crop_clean", "") for r in records if r.get("crop_clean")))),
        )

        filtered_records = [
            r for r in records
            if filter_crop == "All Crops" or r.get("crop_clean") == filter_crop
        ]

        st.caption(f"Showing {len(filtered_records)} records")

        for r in filtered_records[:15]:
            with st.expander(f"Record: {r.get('chunk_id')} | Crop: {r.get('crop_clean')} ({r.get('state', 'Unknown')})"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**District:** {r.get('district', 'N/A')}")
                c2.markdown(f"**State:** {r.get('state', 'N/A')}")
                c3.markdown(f"**Year:** {r.get('original_year', 'Historical')}")

                st.markdown(f"**Farmer Query:** {r.get('farmer_query', 'N/A')}")
                st.markdown(f"**Expert Advisory Transcript:**\n> {r.get('expert_advisory', 'N/A')}")
                st.caption(f"Domain: {r.get('advisory_domain')} | Sector: {r.get('sector')} | QC Severity: {r.get('qc_severity')}")
    else:
        st.warning("Processed knowledge base file not found. Run 'python -m src.ingestion.ingest' to generate.")


# =============================================================================
# TAB 3: SYSTEM DIAGNOSTICS & PIPELINE AUDIT
# =============================================================================
with tab_diagnostics:
    st.markdown("<div class='panel-title'>System Diagnostics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-caption'>Audit multi-stage hallucination-guard metrics, threshold calibrations, and language model integration parameters.</div>",
        unsafe_allow_html=True,
    )

    # Provider Configuration
    st.markdown("#### Generation Provider Configuration")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        new_provider = st.selectbox(
            "Select Generation Provider:",
            options=["mock", "ollama", "api"],
            index=["mock", "ollama", "api"].index(st.session_state.active_provider),
            help="Select the backend generation provider.",
        )
        if new_provider != st.session_state.active_provider:
            st.session_state.active_provider = new_provider
            st.rerun()

    with col_p2:
        st.markdown(f"**Ollama Base URL:** `{config.OLLAMA_BASE_URL}`")
        st.markdown(f"**Ollama Model:** `{config.OLLAMA_MODEL}`")

    with col_p3:
        st.markdown(f"**API Base URL:** `{config.LLM_API_BASE}`")
        st.markdown(f"**API Model:** `{config.LLM_MODEL}`")

    st.markdown("---")

    # Threshold Settings
    st.markdown("#### Calibrated Relevance Gate Thresholds")
    st.markdown(
        """
        <div class="metric-container">
            <div class="metric-cell">
                <div class="metric-title">Sufficient Threshold</div>
                <div class="metric-data">&ge; 0.50</div>
            </div>
            <div class="metric-cell">
                <div class="metric-title">Minimum Viable Threshold</div>
                <div class="metric-data">&ge; 0.30</div>
            </div>
            <div class="metric-cell">
                <div class="metric-title">Top-K Retrieval</div>
                <div class="metric-data">3 Chunks</div>
            </div>
            <div class="metric-cell">
                <div class="metric-title">Vector Metric</div>
                <div class="metric-data">Cosine Similarity</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Limitations
    st.markdown("#### Scientific Scope & Limitations")
    st.markdown(
        """
        - **Hallucination Reduction:** Multi-stage gating and post-generation structural verification substantially reduce ungrounded assertions but do not completely eliminate hallucinations.
        - **Agronomic Temporality:** Historical recommendations from 2014 to 2024 may reference chemical formulations that have been revised or restricted by the Central Insecticides Board & Registration Committee (CIB&RC).
        - **Verification Bounds:** The current verifier validates exact token and chemical dosage grounding. Full natural language entailment (NLI) is an ongoing enhancement area.
        """
    )
