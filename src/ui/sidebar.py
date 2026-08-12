"""
Sidebar UI — BYOK key input + model selectors.
Returns (groq_key, guard_model, chat_model, eval_enabled).
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from src.config import GROQ_MODELS, GUARD_MODEL_DEFAULT, CHAT_MODEL_DEFAULT
from src.tracing import TRACING_ENABLED, LANGSMITH_PROJECT

load_dotenv()
_ENV_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
if _ENV_GROQ_KEY == "your_groq_api_key_here":
    _ENV_GROQ_KEY = ""


def render_sidebar() -> tuple:
    with st.sidebar:
        st.title("💬 AskHR")
        st.caption("RAG + NeMo Guardrails + FAISS  + Evaluation + LangSmith")
        st.divider()

        st.subheader("🔑 Bring Your Own Key")
        st.caption("Keys are used only for this session and never stored.")

        groq_key = st.text_input(
            "Groq API Key",
            value=_ENV_GROQ_KEY,
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com. Pre-filled from .env locally if GROQ_API_KEY is set.",
        )

        if groq_key:
            st.success("Key loaded ✓", icon="🔒")
        else:
            st.info("Paste your Groq key above to start.", icon="ℹ️")

        st.divider()

        guard_model = st.selectbox(
            "① Guard model — NeMo intent classification",
            options=list(GROQ_MODELS.keys()),
            index=list(GROQ_MODELS.keys()).index(GUARD_MODEL_DEFAULT),
            format_func=lambda m: GROQ_MODELS[m],
            help=(
                "NeMo uses this model to decide whether your message is off-topic, "
                "a jailbreak, a request for confidential data, etc. "
                "A stronger model catches more subtle attacks."
            ),
        )
        chat_model = st.selectbox(
            "② Chat model — RAG answer generation",
            options=list(GROQ_MODELS.keys()),
            index=list(GROQ_MODELS.keys()).index(CHAT_MODEL_DEFAULT),
            format_func=lambda m: GROQ_MODELS[m],
            help=(
                "Used to generate the final answer from the top-3 HR policy chunks "
                "retrieved by FAISS. A faster/cheaper model is fine here."
            ),
        )

        if guard_model == "llama-3.1-8b-instant":
            st.warning("8B models may miss subtle jailbreaks. 70B+ is recommended for the guard model.")

        st.divider()

        st.subheader("⚙️ Pipeline")
        st.markdown("""
**Per message:**
1. PII check
2. Intent classification → LLM ① (guard)
3. FAISS retrieves top-3 chunks
4. Answer generation → LLM ② (chat)
5. RAG quality scoring → LLM ③
6. Output sanitizer 
        """)
        st.divider()

        st.subheader("🎯 RAG Quality Metrics")
        eval_enabled = st.checkbox(
            "Score every answer",
            value=True,
            help=(
                "Adds one extra Groq call (using the guard model as judge) that scores "
                "faithfulness, answer relevancy, and context utilization for each answer. "
                "Turn off to save one LLM call per message."
            ),
        )
        st.caption(
            "See scores per-message in the trace panel, and trends across the "
            "session in the **📊 RAG Metrics** tab."
        )

        st.divider()

        st.subheader("📊 Monitoring")
        if TRACING_ENABLED:
            st.success(f"LangSmith tracing ON · project `{LANGSMITH_PROJECT}`", icon="✅")
        else:
            st.caption("LangSmith tracing off — set `LANGCHAIN_API_KEY` in `.env` to enable.")

        st.divider()
        st.caption(f"Python {sys.version.split()[0]}")

    return groq_key, guard_model, chat_model, eval_enabled
