"""
Landing page shown when no Groq API key has been entered.
"""

import streamlit as st


def render_landing() -> None:
    st.title("AskHR")
    st.info("Enter your Groq API key in the sidebar to begin.", icon="🔑")
    st.markdown("""
**What this demo shows:**

NeMo Guardrails acting as a semantic gate protecting a RAG pipeline — a common production pattern.

| Stage | What happens |
|---|---|
| **Input Rail (LLM ①)** | Guard model classifies intent — blocks off-topic, jailbreak, confidential requests, PII |
| **FAISS RAG** | Finds the 3 most relevant HR policy chunks from the vector store |
| **Answer (LLM ②)** | Chat model generates an answer grounded only in retrieved policy text |
| **Output Rail** | Regex scan ensures no sensitive data leaked in the response |
| **LangSmith** | Every stage above is traced end-to-end for monitoring and debugging |

**Ask about:** leave, vacation, sick days, parental leave, remote work, benefits, 401k,
performance reviews, code of conduct, harassment, PIP, promotions…

**Try blocking:** an off-topic question, asking a colleague's salary, a jailbreak attempt
or including your SSN in the message.
    """)
