"""
AskHR — entry point.
All logic lives in src/. This file only wires things together.
"""

import sys
import traceback as tb

import streamlit as st

st.set_page_config(
    page_title="AskHR",
    page_icon="💬",
    layout="wide",
)

# ── Import guard — show a clear error on screen if any package is missing ─────

_import_errors = []


try:
    from src.rag import build_vectorstore
except Exception as e:
    _import_errors.append(("src.rag (FAISS / fastembed)", str(e), tb.format_exc()))


try:
    from src.ui.sidebar import render_sidebar
    from src.ui.landing import render_landing
    from src.ui.tab_chat import render_chat_tab
    from src.ui.tab_docs import render_docs_tab
    from src.ui.tab_metrics import render_metrics_tab
except Exception as e:
    _import_errors.append(("src.ui", str(e), tb.format_exc()))


if _import_errors:
    st.title("🛑 Startup Error — Missing Dependencies")
    st.error("One or more packages failed to import. See details below.")
    for pkg, msg, trace in _import_errors:
        with st.expander(f"❌  {pkg}  —  {msg}"):
            st.code(trace, language="python")
    st.markdown("**Fix:** `pip install -r requirements.txt`")
    st.info(f"Python: `{sys.version}`")
    st.stop()


# ── Sidebar — returns user inputs ─────────────────────────────────────────────

groq_key, guard_model, chat_model, eval_enabled = render_sidebar()

# ── Landing page when no key provided ────────────────────────────────────────

if not groq_key:
    render_landing()
    st.stop()


# ── Build vector store once (cached across sessions) ─────────────────────────

try:
    vectorstore = build_vectorstore()
except Exception:
    st.error("**Failed to build the HR policy vector store.**")
    with st.expander("Error details"):
        st.code(tb.format_exc(), language="python")
    st.stop()


# ── Page + tabs ───────────────────────────────────────────────────────────────

st.title("AskHR")
st.caption("RAG + NeMo Guardrails + FAISS  + Evaluation + LangSmith")

tab_chat, tab_metrics, tab_docs = st.tabs(["💬 Assistant", "📊 RAG Metrics", "📄 HR Policy Documents"])

with tab_chat:
    render_chat_tab(vectorstore, groq_key, guard_model, chat_model, eval_enabled)

with tab_metrics:
    render_metrics_tab()

with tab_docs:
    render_docs_tab()
