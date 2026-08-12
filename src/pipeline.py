"""
Core pipeline logic — no Streamlit imports.

run_pipeline() wires together:
  NeMo input rails (LLM ①) → FAISS retrieval → answer generation (LLM ②) →
  RAG quality scoring (LLM ③, optional) → output sanitizer

LangSmith tracing: the whole call is wrapped in a root "hr_policy_rag_pipeline"
span, with child spans for the guard check, retrieval, generation, quality
scoring, and output sanitizer. Traced inputs are built manually (never pass
groq_key/**kwargs through) so the API key can never end up in a LangSmith trace.
"""

import asyncio
import contextvars
import re
import time
import traceback as tb
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from langsmith.run_helpers import trace as ls_trace

import src.tracing  # noqa: F401 — sets LANGCHAIN_* env vars before any LLM/traced call
from src.tracing import TRACING_ENABLED, ls_client, LANGSMITH_PROJECT_ID
from src.guards import build_rails, parse_nemo_response
from src.rag import retrieve
from src.eval import evaluate_answer
from src.config import HR_SYSTEM_PROMPT, SENSITIVE_OUTPUT_PATTERNS


_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nemo")


def check_output(text: str) -> list:
    return [label for label, pat in SENSITIVE_OUTPUT_PATTERNS.items()
            if re.search(pat, text)]


def run_pipeline(
    message: str,
    groq_key: str,
    guard_model: str,
    chat_model: str,
    vectorstore,
    eval_enabled: bool = False,
) -> tuple:
    """
    Returns (reply: str, trace: dict).
    Makes 2 LLM calls (guard + generation), or 3 when eval_enabled and the
    guard passes (guard + generation + quality judge).
    NeMo runs in a worker thread to isolate asyncio from Streamlit's event loop.
    """
    trace   = {}
    t_total = time.time()

    with ls_trace(
        name="hr_policy_rag_pipeline",
        run_type="chain",
        inputs={"message": message, "guard_model": guard_model, "chat_model": chat_model},
    ) as pipeline_run:

        # ── LLM ① — NeMo input rails ─────────────────────────────────────────
        def _nemo_worker():
            llm   = ChatGroq(api_key=groq_key, model=guard_model, temperature=0)
            rails = build_rails(llm)
            async def _run():
                return await rails.generate_async(
                    messages=[{"role": "user", "content": message}]
                )
            return asyncio.run(_run())

        t0 = time.time()
        nemo_error = None
        with ls_trace(name="guardrails_check", run_type="chain", inputs={"message": message}) as guard_run:
            try:
                # copy_context() carries the LangSmith run-tree contextvar into the
                # worker thread — ThreadPoolExecutor does not propagate it by default.
                ctx = contextvars.copy_context()
                raw = _executor.submit(ctx.run, _nemo_worker).result(timeout=60)
            except Exception:
                raw        = "Guard error"
                nemo_error = tb.format_exc()
            rail_ms = round((time.time() - t0) * 1000)

            text, is_blocked, block_reason, is_dialog, _ = parse_nemo_response(raw)
            guard_run.end(outputs={
                "blocked": is_blocked,
                "reason":  block_reason,
                "dialog":  is_dialog,
                "error":   nemo_error,
            })

        trace["rail"] = {
            "blocked": is_blocked,
            "reason":  block_reason,
            "dialog":  is_dialog,
            "ms":      rail_ms,
            "model":   guard_model,
            "error":   nemo_error,
        }

        if is_blocked or is_dialog:
            trace["total_ms"] = round((time.time() - t_total) * 1000)
            pipeline_run.end(outputs={"reply": text, "blocked": is_blocked, "dialog": is_dialog})
            return text, trace

        # ── FAISS retrieval ───────────────────────────────────────────────────
        t1 = time.time()
        rag_error = None
        try:
            chunks = retrieve(message, vectorstore, k=3)
        except Exception:
            chunks    = []
            rag_error = tb.format_exc()
        rag_ms = round((time.time() - t1) * 1000)
        trace["retrieval"] = {"chunks": chunks, "ms": rag_ms, "error": rag_error}

        # ── LLM ② — answer generation ────────────────────────────────────────
        context_text = "\n\n---\n\n".join(
            f"[{c['source']}]\n{c['content']}" for c in chunks
        ) if chunks else "No relevant policy excerpts were retrieved."

        t2 = time.time()
        gen_error = None
        answer    = ""
        with ls_trace(
            name="answer_generation",
            run_type="chain",
            inputs={"message": message, "chunks_used": len(chunks), "model": chat_model},
        ) as gen_run:
            try:
                llm  = ChatGroq(api_key=groq_key, model=chat_model, temperature=0)
                resp = llm.invoke([
                    {"role": "system", "content": HR_SYSTEM_PROMPT.format(context=context_text)},
                    {"role": "user",   "content": message},
                ])
                answer = resp.content
            except Exception:
                gen_error = tb.format_exc()
                answer    = "LLM call failed — see trace for details."
            gen_run.end(outputs={"answer": answer, "error": gen_error})
        gen_ms = round((time.time() - t2) * 1000)
        trace["generation"] = {"ms": gen_ms, "model": chat_model, "error": gen_error}

        # ── LLM ③ — RAG quality scoring (optional, judged on the raw answer) ──
        t_eval = time.time()
        scores     = {}
        eval_error = None
        if eval_enabled and not gen_error:
            with ls_trace(
                name="rag_quality_eval",
                run_type="chain",
                inputs={"message": message, "chunks_used": len(chunks)},
            ) as eval_run:
                try:
                    scores = evaluate_answer(message, context_text, answer, groq_key, guard_model)
                except Exception:
                    eval_error = tb.format_exc()
                eval_run.end(outputs={"scores": scores, "error": eval_error})

            if TRACING_ENABLED and scores and ls_client is not None:
                for key in ("faithfulness", "answer_relevancy", "context_utilization"):
                    ls_client.create_feedback(
                        pipeline_run.id, key=key, score=scores[key], session_id=LANGSMITH_PROJECT_ID,
                    )
        eval_ms = round((time.time() - t_eval) * 1000)
        trace["quality"] = {"scores": scores, "ms": eval_ms, "error": eval_error, "enabled": eval_enabled}

        # ── Output sanitizer ────────────────────────────────────────────────
        t3     = time.time()
        with ls_trace(name="output_sanitizer", run_type="tool", inputs={"answer_len": len(answer)}) as out_run:
            issues = check_output(answer)
            out_run.end(outputs={"issues": issues})
        out_ms = round((time.time() - t3) * 1000)

        if issues:
            answer = "My response contained potentially sensitive information and has been withheld. Please contact hr@AskHRcorp.com directly."
            trace["output_rail"] = {"blocked": True, "issues": issues, "ms": out_ms}
        else:
            trace["output_rail"] = {"blocked": False, "ms": out_ms}

        trace["total_ms"] = round((time.time() - t_total) * 1000)
        pipeline_run.end(outputs={"reply": answer, "output_blocked": bool(issues)})
        return answer, trace
