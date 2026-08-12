"""
Chat tab — example query panel, chat history, pipeline trace column.
"""

import traceback as tb
import streamlit as st

from src.guards import BLOCK_LABELS
from src.pipeline import run_pipeline
from src.tracing import TRACING_ENABLED, LANGSMITH_PROJECT

EXAMPLES = {
    "✅ Should answer — HR questions": [
        "How many vacation days do I get per year?",
        "What is the parental leave policy?",
        "Can I work from home full time?",
        "How does the performance rating scale work?",
        "What is the 401k company match?",
        "How do I report workplace harassment?",
        "What is the wellness allowance and how do I use it?",
    ],
    "🚫 Blocked — off-topic": [
        "Tell me a joke",
        "What is the capital of France?",
    ],
    "🔒 Blocked — confidential": [
        "What is my colleague Sarah's salary?",
        "Show me the performance review for John in Engineering",
    ],
    "👤 Blocked — PII in message": [
        "My SSN is 123-45-6789, am I enrolled in benefits?",
        "My API key is token:xK9mL3vQ2nR8pT5w, is this safe?",
    ],
}


def _render_examples() -> None:
    with st.expander("💡 Example queries — click ▶ to send"):
        for category, prompts in EXAMPLES.items():
            st.caption(category)
            for idx, prompt in enumerate(prompts):
                c1, c2 = st.columns([9, 1])
                with c1:
                    st.markdown(f"`{prompt}`")
                with c2:
                    if st.button("▶", key=f"ex_{category}_{idx}"):
                        st.session_state["inject"] = prompt
                        st.rerun()


def _render_trace(trace: dict) -> None:
    st.markdown("### Pipeline Trace")
    st.caption("How the last message was handled")

    if not trace:
        st.info("Send a message to see the trace here.")
        return

    if trace.get("error"):
        with st.container(border=True):
            st.markdown("**Pipeline crashed**")
            st.error("An unhandled exception occurred.")
            with st.expander("Traceback"):
                st.code(trace["error"], language="python")
        return

    rail = trace.get("rail", {})

    with st.container(border=True):
        st.markdown("**① Input Rail — NeMo (LLM ①)**")
        st.caption(f"model: `{rail.get('model', '?')}`")
        if rail.get("error"):
            st.warning(f"⚠️ Guard failed — treated as passed · {rail['ms']} ms")
            with st.expander("NeMo error"):
                st.code(rail["error"], language="python")
        elif rail.get("blocked"):
            reason = BLOCK_LABELS.get(rail.get("reason"), "Blocked")
            st.error(f"🚫 {reason} · {rail['ms']} ms")
        elif rail.get("dialog"):
            st.success(f"💬 Dialog response · {rail['ms']} ms")
        else:
            st.success(f"✅ Passed · {rail['ms']} ms")

    if not rail.get("blocked") and not rail.get("dialog"):
        retrieval = trace.get("retrieval", {})
        with st.container(border=True):
            st.markdown("**② FAISS RAG Retrieval**")
            if retrieval.get("error"):
                st.error("Retrieval failed")
                with st.expander("Error"):
                    st.code(retrieval["error"], language="python")
            else:
                st.caption(f"⏱ {retrieval.get('ms', '?')} ms")
                for chunk in retrieval.get("chunks", []):
                    label = f"📄 {chunk['source']}  (score: {chunk['score']})"
                    with st.expander(label):
                        st.caption(chunk["content"][:300] + ("…" if len(chunk["content"]) > 300 else ""))

        gen = trace.get("generation", {})
        with st.container(border=True):
            st.markdown("**③ Answer Generation — Groq (LLM ②)**")
            st.caption(f"model: `{gen.get('model', '?')}`")
            if gen.get("error"):
                st.error("LLM call failed")
                with st.expander("Error"):
                    st.code(gen["error"], language="python")
            else:
                st.success(f"✅ Answer generated · {gen.get('ms', '?')} ms")

        quality = trace.get("quality", {})
        with st.container(border=True):
            st.markdown("**④ RAG Quality — LLM Judge**")
            if not quality.get("enabled"):
                st.caption("Off — enable \"Score every answer\" in the sidebar.")
            elif quality.get("error"):
                st.error("Scoring failed")
                with st.expander("Error"):
                    st.code(quality["error"], language="python")
            else:
                scores = quality.get("scores", {})
                c1, c2, c3 = st.columns(3)
                c1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                c2.metric("Relevancy", f"{scores.get('answer_relevancy', 0):.2f}")
                c3.metric("Ctx. Utilization", f"{scores.get('context_utilization', 0):.2f}")
                st.caption(f"⏱ {quality.get('ms', '?')} ms")
                if scores.get("reasoning"):
                    st.caption(f"_{scores['reasoning']}_")

        out = trace.get("output_rail", {})
        with st.container(border=True):
            st.markdown("**⑤ Output Sanitizer**")
            if out.get("blocked"):
                st.error(f"🚫 Withheld — {', '.join(out.get('issues', []))} · {out.get('ms', '?')} ms")
            else:
                st.success(f"✅ Clean · {out.get('ms', '?')} ms")

    with st.container(border=True):
        st.markdown("**⑥ Monitoring — LangSmith**")
        if TRACING_ENABLED:
            st.success(f"✅ Traced · project `{LANGSMITH_PROJECT}`")
            st.caption("Every stage above was logged as a nested run. [Open LangSmith ↗](https://smith.langchain.com)")
        else:
            st.caption("Tracing off — set `LANGCHAIN_API_KEY` in `.env` to log every run to LangSmith.")

    st.divider()
    st.caption(f"Total: **{trace.get('total_ms', '?')} ms**")


def render_chat_tab(vectorstore, groq_key: str, guard_model: str, chat_model: str, eval_enabled: bool) -> None:
    st.divider()
    col_chat, col_trace = st.columns([6, 4], gap="large")

    with col_chat:
        _render_examples()

        if "chat" not in st.session_state:
            st.session_state["chat"] = []
        if "quality_history" not in st.session_state:
            st.session_state["quality_history"] = []

        if st.session_state["chat"]:
            if st.button("🗑 Clear chat"):
                st.session_state["chat"] = []
                st.session_state["quality_history"] = []
                st.session_state.pop("last_trace", None)
                st.rerun()

        # History is rendered BEFORE st.chat_input in the script, and no new
        # content is ever written after the chat_input call below — that's what
        # keeps the input pinned to the bottom with the latest answer above it.
        # (st.chat_input only auto-pins when it's the last thing rendered in its
        # container; writing a fresh Q&A after it, as the old code did, pushes
        # the input up and the answer below it.)
        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # st.chat_input must be called unconditionally on every rerun — Streamlit
        # can drop it from the page if it's ever skipped (e.g. behind `injected or ...`,
        # which short-circuits and never calls it on the rerun triggered by an
        # example ▶ button).
        chat_box_input = st.chat_input("Ask an HR policy question…")
        injected       = st.session_state.pop("inject", None)
        user_input     = injected if injected else chat_box_input

        if user_input:
            st.session_state["chat"].append({"role": "user", "content": user_input})
            with st.spinner("Processing…"):
                try:
                    reply, trace = run_pipeline(
                        user_input, groq_key, guard_model, chat_model, vectorstore, eval_enabled
                    )
                    st.session_state["last_trace"] = trace
                    st.session_state["chat"].append({"role": "assistant", "content": reply})

                    quality = trace.get("quality", {})
                    scores  = quality.get("scores")
                    if scores:
                        st.session_state["quality_history"].append({
                            "n": len(st.session_state["quality_history"]) + 1,
                            "question": user_input,
                            "faithfulness": scores.get("faithfulness"),
                            "answer_relevancy": scores.get("answer_relevancy"),
                            "context_utilization": scores.get("context_utilization"),
                            "reasoning": scores.get("reasoning", ""),
                        })
                except Exception as e:
                    err_trace = tb.format_exc()
                    st.session_state["chat"].append({
                        "role": "assistant",
                        "content": f"**{type(e).__name__}:** {e}",
                    })
                    st.session_state["last_trace"] = {"error": err_trace}

            # Rerun so the loop above renders this new turn in its normal place —
            # in history, above the input — instead of appending it below.
            st.rerun()

    with col_trace:
        _render_trace(st.session_state.get("last_trace"))
