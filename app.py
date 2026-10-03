"""
app.py — Streamlit UI for Movie Intelligence & Follow-up Assistant
==================================================================

Run with:
    streamlit run app.py
"""

import asyncio
import sys
import re
import streamlit as st

from app.agent.agent import MovieAgent
from app.rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Intelligence & Follow-up Assistant",
    page_icon="🎬",
    layout="centered",
)


# ---------------------------------------------------------------------------
# CSS — minimal polish as requested ("no polish needed")
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .citation-block {
        background: #1e2130;
        border-left: 3px solid #4f8ef7;
        padding: 8px 14px;
        border-radius: 4px;
        margin: 4px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .intent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .badge-informational { background:#1a4a1a; color:#6ddb6d; }
    .badge-email         { background:#1a2a4a; color:#6db0ff; }
    .badge-clarification { background:#4a3a1a; color:#ffd06d; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine synchronously (safe for Streamlit)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@st.cache_resource(show_spinner="Loading vector store…")
def _get_movie_titles() -> list[str]:
    """Fetch the distinct movie titles from ChromaDB."""
    try:
        vs = VectorStore()
        count = vs.count()
        if count == 0:
            return []
        records = vs.collection.get(
            limit=min(count, 5000),
            include=["metadatas"],
        )
        titles = sorted(
            set(m["movie_title"] for m in records["metadatas"])
        )
        return titles
    except Exception:
        return []


@st.cache_resource(show_spinner="Initialising agent…")
def _get_agent() -> MovieAgent:
    return MovieAgent()


def _badge(intent: str) -> str:
    cls = f"badge-{intent}"
    label = intent.upper()
    return f'<span class="intent-badge {cls}">{label}</span>'


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        st.markdown(
            f'<div class="citation-block">'
            f"<b>{s['movie_title']}</b><br>"
            f"{s['start_time']} &rarr; {s['end_time']}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_result(result: dict) -> None:
    intent = result.get("intent", "")
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    st.markdown(_badge(intent), unsafe_allow_html=True)

    if intent == "clarification":
        st.info(answer)

    elif intent == "informational":
        st.markdown("**Answer**")
        st.write(answer)
        _render_sources(sources)

    elif intent == "email":
        email_res = result.get("email", {})
        if email_res.get("success"):
            st.success("✅ Email sent successfully (mock transport)")
            st.markdown(f"**Recipient:** `{result.get('recipient', 'unknown')}`")
            st.markdown("**Subject:** Movie Information")
        else:
            err = email_res.get("error", "Unknown error")
            st.error(f"Email failed: {err}")

        st.markdown("**Generated Content**")
        st.write(answer)
        _render_sources(sources)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🎬 Settings")

movie_titles = _get_movie_titles()
sidebar_options = ["All Movies"] + movie_titles

selected_movie = st.sidebar.selectbox(
    "Movie filter",
    sidebar_options,
    help="Restrict retrieval to a specific movie, or search all.",
)

movie_filter: str | None = (
    None if selected_movie == "All Movies" else selected_movie
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Uses local Ollama (`qwen3.5:9b`) + ChromaDB.\n"
    "Emails are sent via a **mock** transport only."
)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Movie Intelligence & Follow-up Assistant")
st.caption(
    "Ask questions about movie subtitles, or request movie information "
    "by email."
)

st.markdown("---")

# Example prompts
with st.expander("💡 Example questions"):
    st.markdown(
        """
- *What does the artist say about art?*
- *Why does Walter become important to the artists?*
- *What happens when Walter shows his sculpture?*
- *Email me a breakdown of what the artist says about art to you@example.com*
- *Email me the scene to you@example.com*  ← triggers clarification
        """
    )

user_input = st.text_area(
    "Your question or request",
    placeholder="e.g. What does the artist say about art?",
    height=90,
    key="user_input",
)

run_btn = st.button("Ask", type="primary", use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

if run_btn:
    if not user_input or not user_input.strip():
        st.warning("Please enter a question or request.")
    else:
        with st.spinner("Thinking…"):
            try:
                agent = _get_agent()
                result = _run_async(
                    agent.run(user_input, movie_title=movie_filter)
                )
                _render_result(result)

            except RuntimeError as exc:
                msg = str(exc)
                if "Ollama" in msg or "connect" in msg.lower():
                    st.error(
                        "⚠️ Cannot reach Ollama. "
                        "Make sure `ollama serve` is running and "
                        "the `qwen3.5:9b` model is available."
                    )
                else:
                    st.error(f"An error occurred: {msg}")
            except Exception as exc:
                st.error(
                    f"An unexpected error occurred. "
                    f"Please check the terminal for details.\n\n`{exc}`"
                )
