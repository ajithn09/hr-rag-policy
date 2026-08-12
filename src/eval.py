"""
RAG quality scoring — LLM-as-judge.

Scores a single (question, context, answer) triple on three reference-free
RAGAS-style dimensions in ONE extra Groq call (not the 4+ calls the real
ragas library needs per metric), since this runs live on every message:

- faithfulness:        is every claim in the answer supported by the context?
- answer_relevancy:    does the answer directly address the question?
- context_utilization: was the retrieved context actually relevant/useful?

No Streamlit imports — pure function, called from src/pipeline.py.
"""

import json
import re

from langchain_groq import ChatGroq

EVAL_SYSTEM_PROMPT = """You are an impartial evaluator scoring a RAG (Retrieval-Augmented Generation) system's answer.

Score the ANSWER against the QUESTION and the retrieved CONTEXT on three dimensions, each a float from 0.0 to 1.0:

- faithfulness: Is every factual claim in the answer directly supported by the context? 1.0 = fully grounded, no unsupported claims. 0.0 = answer contradicts or invents facts not present in the context.
- answer_relevancy: Does the answer directly and completely address the question, without padding or irrelevant content? 1.0 = fully relevant and complete. 0.0 = off-topic or a non-answer.
- context_utilization: How much of the retrieved context was actually relevant and necessary to produce the answer? 1.0 = all retrieved context was useful. 0.0 = the context was irrelevant noise.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"faithfulness": <float>, "answer_relevancy": <float>, "context_utilization": <float>, "reasoning": "<one sentence explaining the scores>"}
"""

EVAL_USER_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}"
)


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def evaluate_answer(question: str, context_text: str, answer: str, groq_key: str, eval_model: str) -> dict:
    """
    Returns {"faithfulness": float, "answer_relevancy": float, "context_utilization": float, "reasoning": str}.
    Raises on LLM/parse failure — caller is expected to catch and record the error, matching
    the error-handling pattern used for the other pipeline stages.
    """
    llm = ChatGroq(api_key=groq_key, model=eval_model, temperature=0)
    resp = llm.invoke([
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": EVAL_USER_TEMPLATE.format(
            question=question, context=context_text, answer=answer,
        )},
    ])
    raw = resp.content

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0) if match else raw)

    return {
        "faithfulness":        _clamp(data.get("faithfulness")),
        "answer_relevancy":    _clamp(data.get("answer_relevancy")),
        "context_utilization": _clamp(data.get("context_utilization")),
        "reasoning":           str(data.get("reasoning", "")).strip(),
    }
