"""
App-wide constants — models, prompts, and output-scan patterns.
"""

GROQ_MODELS = {
    "openai/gpt-oss-120b": "OpenAI OSS · 120B  ★ recommended for guardrails",
    "openai/gpt-oss-20b":  "OpenAI OSS · 20B  ★ fast & cheap",
}


GUARD_MODEL_DEFAULT = "openai/gpt-oss-120b"
CHAT_MODEL_DEFAULT  = "openai/gpt-oss-20b"

HR_SYSTEM_PROMPT = (
    "You are AskHR, an internal HR policy assistant. "
    "Answer the employee's question using ONLY the policy excerpts provided below. "
    "Be concise, cite the relevant policy section, and include specific numbers or rules where applicable. "
    "If the answer is not covered in the excerpts, say so clearly and direct the employee to hr@AskHRcorp.com.\n\n"
    "Policy Excerpts:\n{context}"
)

SENSITIVE_OUTPUT_PATTERNS = {
    "credential_leak":  r"(?i)(password|passwd|secret|api[_\-]?key|token)\s*[:=]\s*['\"]?\w{6,}",
    "ssn_in_output":    r"\b\d{3}-\d{2}-\d{4}\b",
    "hardcoded_salary": r"(?i)\b(earns?|is paid|salary of)\s+\$[\d,]+\b",
}
