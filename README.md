# AskHR — NeMo Guardrails + RAG

A production-style RAG assistant for internal HR policy questions, protected by **NVIDIA NeMo Guardrails** acting as a semantic security gate. Built with Streamlit, FAISS, and Groq.

## Architecture

Two or three LLM calls per message, plus Python regex checks at input and output:

1. **Input Rail (NeMo Guardrails, LLM ①)** — a systematic PII regex check, then semantic intent classification (off-topic / jailbreak / confidential-data / greeting / HR question) written in Colang.
2. **FAISS RAG retrieval** — the question is embedded locally (FastEmbed, `BAAI/bge-small-en-v1.5`) and the top-3 relevant chunks are pulled from an in-memory FAISS index of the HR policy documents.
3. **Answer generation (LLM ②)** — a Groq chat model answers using only the retrieved chunks.
4. **RAG quality scoring (LLM ③, optional)** — an LLM-judge call scores the answer on faithfulness, answer relevancy, and context utilization. Toggle in the sidebar; adds one extra Groq call per message when on.
5. **Output sanitizer** — regex scan for credential leaks, SSNs, and hardcoded salary figures before the answer is shown.

Every stage is also traced to **LangSmith** (optional — see below) as a nested run under one root span per message, and the quality scores from stage 4 are logged as LangSmith feedback for cross-session monitoring.

See `src/guards.py` for the Colang rules, `src/eval.py` for the judge prompt, and `src/pipeline.py` for how the stages are wired together.

## Project layout

```
app.py                  Entry point — page config, import guard, tab wiring
src/
  config.py             Model list, system prompt, output-scan patterns
  hr_docs.py            HR policy documents (the knowledge base)
  rag.py                FAISS vector store builder + retrieval
  guards.py             NeMo Guardrails: Colang rules, YAML config, PII action
  eval.py               LLM-as-judge: faithfulness / answer relevancy / context utilization
  tracing.py             LangSmith setup — reads LANGCHAIN_* env vars, exposes TRACING_ENABLED
  pipeline.py           run_pipeline() — guard -> RAG -> generation -> eval -> sanitizer
  ui/
    sidebar.py           BYOK key input + model selectors + quality-scoring toggle
    landing.py            Landing page shown before a key is entered
    tab_chat.py           Chat interface + live pipeline trace
    tab_metrics.py         RAG Metrics tab — session score trends, stat tiles, table view
    tab_docs.py           Policy document browser
requirements.txt
.env.example             Template for GROQ_API_KEY / LANGCHAIN_* vars
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) and paste it into the sidebar — no `.env` file or Streamlit secrets needed. The key only lives in memory for the session (BYOK pattern).

## LangSmith tracing & RAG quality metrics

Both are optional and off by default until you add a key:

1. Get a free API key at [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys.
2. Add it to `.env`:
   ```
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_PROJECT=hr-policy-rag
   ```
3. Restart the app — the sidebar shows "LangSmith tracing ON" once a valid key is detected (`src/tracing.py` checks this at import time, before any LLM call is made).

With tracing on, every message produces one root trace (`hr_policy_rag_pipeline`) with nested spans for the guard check, retrieval, generation, quality eval, and output sanitizer — viewable at smith.langchain.com under your project.

**RAG quality scoring** works independently of LangSmith — toggle "Score every answer" in the sidebar to have an LLM judge (using the guard model) score each answer 0.0–1.0 on:
- **Faithfulness** — is every claim in the answer supported by the retrieved context?
- **Answer relevancy** — does the answer address the question asked?
- **Context utilization** — was the retrieved context actually relevant and used?

Scores show per-message in the Assistant tab's trace panel, and as session trends in the **📊 RAG Metrics** tab. When LangSmith is also on, the three scores are additionally logged as feedback on that message's trace, so they're trackable across sessions, not just within one.

## Notes

- FAISS is used instead of ChromaDB to avoid a `protobuf`/`opentelemetry` version conflict on Python 3.13+, and because it has no telemetry dependency.
- `nemoguardrails`'s `generate_async()` calls `asyncio.run()` internally, which conflicts with Streamlit's own event loop. `src/pipeline.py` runs the guardrails call in a `ThreadPoolExecutor` worker thread to give it an isolated event loop, and uses `contextvars.copy_context()` when submitting so the LangSmith trace context (which is thread-local) still nests correctly across that thread boundary.
- The quality judge (`src/eval.py`) scores all three dimensions in a single Groq call rather than using the `ragas` library directly — `ragas`'s real algorithms need 4+ LLM/embedding calls per metric set, which is too slow/expensive to run live on every chat message. Swap in `ragas` if you need its more rigorous, peer-reviewed scoring for offline/batch evaluation instead.
- All data in `src/hr_docs.py` is synthetic, for demo purposes.
