"""
LangSmith tracing / monitoring setup.

Reads LANGCHAIN_* env vars (via .env) and only turns tracing on when a real
API key is present, so the app runs fine with no LangSmith account at all.
Import this before anything that calls an LLM so the env vars are set first.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "hr-policy-rag")

_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if _api_key and _api_key != "your_langsmith_api_key_here":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    TRACING_ENABLED = True
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    TRACING_ENABLED = False

# Shared client for logging feedback (e.g. RAG quality scores) onto traced runs.
# The project id is resolved once and cached so create_feedback() calls don't
# need a network round-trip per message and don't hit the "no session_id" warning.
if TRACING_ENABLED:
    from langsmith import Client
    ls_client = Client()
    try:
        LANGSMITH_PROJECT_ID = ls_client.read_project(project_name=LANGSMITH_PROJECT).id
    except Exception:
        LANGSMITH_PROJECT_ID = None
else:
    ls_client = None
    LANGSMITH_PROJECT_ID = None
