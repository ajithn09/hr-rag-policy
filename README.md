# AskHR — RAG + Faiss + NeMo Guardrails + Evaluation + langsmith

A production-style RAG assistant for internal HR policy questions, protected by **NVIDIA NeMo Guardrails** acting as a semantic security gate, with built-in **RAG quality evaluation** and **LangSmith tracing/monitoring**. Built with Streamlit, FAISS, and Groq.

## What it does

- Answers employee HR questions (leave, benefits, remote work, performance reviews, code of conduct) grounded only in a local FAISS-indexed policy knowledge base — never from general model knowledge.
- Blocks off-topic questions, jailbreak attempts, requests for confidential employee data, and messages containing PII (email, phone, SSN, API keys, credit cards) *before* they ever reach the answer-generation model.
- Scores every answer for quality (faithfulness, relevancy, context utilization) using an LLM judge.
- Traces every stage of every request to LangSmith, with quality scores logged as feedback for cross-session monitoring.
- BYOK (bring your own key) — no API keys are stored server-side; each session supplies its own Groq key.

## Architecture

At most 3 LLM calls per message, plus Python regex checks at input and output. Blocked messages and greetings stop after the first call.

| # | Stage | Runs when | What it does |
|---|---|---|---|
| 1 | **Input Rail** — NeMo Guardrails (LLM ①) | Always | Systematic PII regex check, then semantic intent classification (off-topic / jailbreak / confidential-data / greeting / HR question) via Colang rules |
| 2 | **FAISS RAG retrieval** | Guard passes | Question embedded locally (FastEmbed), top-3 relevant chunks pulled from the in-memory FAISS index |
| 3 | **Answer generation** (LLM ②) | Guard passes | Groq chat model answers using only the retrieved chunks |
| 4 | **RAG quality scoring** (LLM ③, optional) | Guard passes + toggle on | LLM judge scores the answer on faithfulness, answer relevancy, context utilization |
| 5 | **Output sanitizer** | Guard passes | Regex scan for credential leaks, SSNs, hardcoded salary figures before the answer is shown |

Call-count summary: **1 call** for anything blocked or a greeting, **2 calls** for a normal answered question, **3 calls** if quality scoring is on (the default). The guard model doubles as the quality judge, so no extra model selection is needed for stage 4.

See `src/guards.py` for the Colang rules, `src/eval.py` for the judge prompt, and `src/pipeline.py` for how the stages are wired together and traced.

## Guardrails (NeMo)

Two kinds of input rail, both run before the answer model ever sees the message:

- **Systematic rail** (`detect_pii` in `src/guards.py`) — a Python regex action that runs on every message, no LLM involved. Catches emails, phone numbers, SSNs, API keys/tokens, and credit card numbers.
- **Semantic rails** (Colang flows) — the guard LLM compares the message against example intents to catch things regex can't: off-topic questions, jailbreak attempts ("ignore all previous instructions", "you are now DAN", etc.), and requests for confidential employee data (salaries, performance reviews, termination records).
- **Greeting flow** — matched entirely by Colang with a scripted response; no LLM call needed for the guard *or* the answer.

Every blocking response is prefixed `[RAIL_BLOCKED:<REASON>]` so the app can reliably detect what happened (`src/guards.py::parse_nemo_response`) and route the UI accordingly.

## RAG (retrieval)

- **Knowledge base**: 6 synthetic HR policy documents (`src/hr_docs.py`) — leave/time off, remote work, code of conduct, performance reviews, benefits/compensation, anti-harassment.
- **Chunking**: `RecursiveCharacterTextSplitter`, 500-char chunks with 100-char overlap.
- **Embedding**: FastEmbed, `BAAI/bge-small-en-v1.5`, runs locally — no API call, no cost.
- **Index**: FAISS (in-memory, built once via `@st.cache_resource`). Chosen over ChromaDB to avoid a `protobuf`/`opentelemetry` version conflict on Python 3.13+, and because it has no telemetry dependency.
- **Retrieval**: top-3 chunks by relevance score per query (`src/rag.py::retrieve`).

## RAG quality evaluation

Toggle **"Score every answer"** in the sidebar (on by default) to have an LLM judge score each answer, reference-free, on three dimensions from 0.0–1.0 (`src/eval.py`):

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Is every claim in the answer directly supported by the retrieved context? |
| **Answer relevancy** | Does the answer directly and completely address the question asked? |
| **Context utilization** | Was the retrieved context actually relevant and used, or mostly noise? |

**Design choice:** these are scored in a **single extra Groq call** that returns structured JSON for all three metrics at once, rather than using the `ragas` library directly. `ragas`'s real algorithms need 4+ LLM/embedding calls just for this metric set (question generation + embedding comparisons for relevancy, statement extraction + verification for faithfulness), which is too slow and expensive to run live on every chat message. The guard model is reused as the judge, so there's no extra model to configure. Swap in `ragas` if you need its more rigorous, peer-reviewed scoring for offline/batch evaluation instead.

**Where the scores show up:**
- Per-message, in the Assistant tab's live pipeline trace panel (block ④).
- As session trends in the **📊 RAG Metrics** tab (`src/ui/tab_metrics.py`) — stat tiles with session averages, a 3-series trend chart (Plotly, using the dataviz-skill's pre-validated colorblind-safe categorical palette), and a raw-scores table with the judge's reasoning for each answer.
- As LangSmith feedback on that message's trace, if LangSmith is also enabled — see below.

## LangSmith tracing & monitoring

Optional, off by default, enabled by adding a LangSmith API key — no code changes needed.

**Setup:**
1. Get a free API key at [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys.
2. Add to `.env` (see `.env.example`):
   ```
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_PROJECT=hr-policy-rag
   ```
3. Restart the app. The sidebar shows "LangSmith tracing ON · project `hr-policy-rag`" once a valid key is detected — `src/tracing.py` checks this at import time, before any LLM call is made, and leaves tracing off (no errors, no network calls) if no key is present.

**What gets traced:** every message produces one root trace, `hr_policy_rag_pipeline`, with nested child spans for each stage:

```
hr_policy_rag_pipeline          (root — one per message)
├─ guardrails_check             (guard LLM call, via a worker thread)
│   └─ ChatGroq                 (the actual guard model call)
├─ faiss_retrieve                (retriever span — no LLM)
├─ answer_generation
│   └─ ChatGroq                 (the answer model call)
├─ rag_quality_eval              (only if scoring is on)
│   └─ ChatGroq                 (the judge model call)
└─ output_sanitizer              (regex only — no LLM)
```

Two implementation details make this nesting work correctly:
- The guard check runs in a `ThreadPoolExecutor` worker thread (to isolate NeMo's internal `asyncio.run()` from Streamlit's own event loop). Thread pools don't propagate `contextvars` by default, which would otherwise orphan that span as a disconnected root trace — `src/pipeline.py` uses `contextvars.copy_context()` when submitting so the LangSmith run-tree context crosses the thread boundary correctly.
- Traced inputs are built manually as plain dicts (never the raw function arguments), so the Groq API key can never end up in a LangSmith trace.

**Feedback logging:** when quality scoring is on, the three scores (faithfulness, answer_relevancy, context_utilization) are also logged as LangSmith feedback on that message's root trace via `create_feedback`, so they're queryable/chartable across sessions in the LangSmith UI — not just visible in-app for the current session.

## Project layout

```
app.py                     Entry point — page config, import guard, tab wiring
src/
  config.py                Model list, system prompt, output-scan patterns
  hr_docs.py                HR policy documents (the knowledge base)
  rag.py                   FAISS vector store builder + retrieval
  guards.py                NeMo Guardrails: Colang rules, YAML config, PII action
  eval.py                  LLM-as-judge: faithfulness / answer relevancy / context utilization
  tracing.py                LangSmith setup — reads LANGCHAIN_* env vars, exposes TRACING_ENABLED
  pipeline.py              run_pipeline() — guard -> RAG -> generation -> eval -> sanitizer
  ui/
    sidebar.py              BYOK key input, model selectors, quality-scoring toggle, tracing status
    landing.py               Landing page shown before a key is entered
    tab_chat.py              Chat interface + live per-message pipeline trace
    tab_metrics.py            RAG Metrics tab — session score trends, stat tiles, table view
    tab_docs.py              Policy document browser
requirements.txt
.env.example                Template for GROQ_API_KEY / LANGCHAIN_* vars
```

## Configuration

All via `.env` (see `.env.example`) or environment variables — nothing is hardcoded or required to run:

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | No | Pre-fills the sidebar's key field locally; visitors can also paste their own key at runtime (BYOK) |
| `LANGCHAIN_API_KEY` | No | Enables LangSmith tracing + feedback logging when set |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default `hr-policy-rag`) |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_ENDPOINT` | No | Set automatically once a valid `LANGCHAIN_API_KEY` is detected |

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) and paste it into the sidebar — no `.env` file needed unless you want it pre-filled or want LangSmith tracing. The key only lives in memory for the session (BYOK pattern).

## Notes

- FAISS is used instead of ChromaDB to avoid a `protobuf`/`opentelemetry` version conflict on Python 3.13+, and because it has no telemetry dependency.
- `nemoguardrails`'s `generate_async()` calls `asyncio.run()` internally, which conflicts with Streamlit's own event loop. `src/pipeline.py` runs the guardrails call in a `ThreadPoolExecutor` worker thread to give it an isolated event loop, and uses `contextvars.copy_context()` when submitting so the LangSmith trace context (thread-local by default) still nests correctly across that thread boundary.
- The quality judge (`src/eval.py`) scores all three dimensions in a single Groq call rather than using the `ragas` library directly — see the RAG quality evaluation section above for why.
- `st.chat_input` is only ever called as the last thing rendered in its column, with the chat history loop rendered before it and nothing written after it — this is what keeps the input pinned at the bottom with the latest answer above it (`src/ui/tab_chat.py`).
- All data in `src/hr_docs.py` is synthetic, for demo purposes.
