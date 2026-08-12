"""
RAG Metrics tab — session-level trends for the LLM-judge quality scores
(faithfulness, answer relevancy, context utilization) computed in src/eval.py.

Colors are the reference dataviz palette's first three categorical slots —
pre-validated as CVD-safe for exactly a 3-series, all-pairs case in both
light and dark modes (see dataviz skill, references/palette.md).
"""

import streamlit as st
import plotly.graph_objects as go

METRICS = [
    ("faithfulness",        "Faithfulness",        "Is every claim in the answer supported by the retrieved context?"),
    ("answer_relevancy",    "Answer Relevancy",     "Does the answer directly address the question asked?"),
    ("context_utilization", "Context Utilization",  "Was the retrieved context actually relevant and used?"),
]


def _theme():
    """Best-effort light/dark detection; defaults to light if unavailable."""
    try:
        if st.context.theme and st.context.theme.type == "dark":
            return "dark"
    except Exception:
        pass
    return "light"


_COLORS = {
    "light": {
        "faithfulness":        "#2a78d6",
        "answer_relevancy":    "#eb6834",
        "context_utilization": "#1baf7a",
        "ink":                 "#0b0b0b",
        "muted":               "#898781",
        "grid":                "#e1e0d9",
    },
    "dark": {
        "faithfulness":        "#3987e5",
        "answer_relevancy":    "#d95926",
        "context_utilization": "#199e70",
        "ink":                 "#ffffff",
        "muted":               "#c3c2b7",
        "grid":                "#2c2c2a",
    },
}


def _build_chart(history: list) -> go.Figure:
    theme  = _theme()
    colors = _COLORS[theme]
    x = [row["n"] for row in history]

    fig = go.Figure()
    for key, label, _ in METRICS:
        fig.add_trace(go.Scatter(
            x=x,
            y=[row[key] for row in history],
            name=label,
            mode="lines+markers",
            line=dict(color=colors[key], width=2),
            marker=dict(size=8, color=colors[key]),
            hovertemplate=f"{label}: " + "%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["muted"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=colors["ink"])),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        xaxis=dict(
            title="Message #",
            tickmode="linear",
            dtick=1,
            gridcolor=colors["grid"],
            zeroline=False,
            color=colors["muted"],
        ),
        yaxis=dict(
            title="Score",
            range=[0, 1.05],
            gridcolor=colors["grid"],
            zeroline=False,
            color=colors["muted"],
        ),
        height=340,
    )
    return fig


def render_metrics_tab() -> None:
    st.divider()
    st.subheader("RAG Quality Metrics")
    st.caption(
        "Faithfulness, answer relevancy, and context utilization — scored by an LLM judge "
        "on every answered message this session. Enable \"Score every answer\" "
        "in the sidebar to populate this tab."
    )
    st.divider()

    history = st.session_state.get("quality_history", [])

    if not history:
        st.info("No scored answers yet. Ask an HR question in the Assistant tab with scoring enabled.")
        return

    avg = {key: sum(row[key] for row in history) / len(history) for key, _, _ in METRICS}
    latest = history[-1]

    cols = st.columns(3)
    for col, (key, label, help_text) in zip(cols, METRICS):
        delta = latest[key] - avg[key]
        col.metric(
            f"Session avg · {label}",
            f"{avg[key]:.2f}",
            delta=f"{delta:+.2f} vs latest" if len(history) > 1 else None,
            help=help_text,
        )

    st.plotly_chart(_build_chart(history), use_container_width=True, config={"displayModeBar": False})

    with st.expander(f"📋 Raw scores — table view ({len(history)} messages)"):
        st.dataframe(
            [
                {
                    "#": row["n"],
                    "Question": row["question"],
                    "Faithfulness": round(row["faithfulness"], 2),
                    "Relevancy": round(row["answer_relevancy"], 2),
                    "Ctx. Utilization": round(row["context_utilization"], 2),
                    "Reasoning": row["reasoning"],
                }
                for row in history
            ],
            use_container_width=True,
            hide_index=True,
        )
