"""
app.py — Streamlit UI for Movie Intelligence & Follow-up Assistant
==================================================================

Run with:
    streamlit run app.py
"""

import asyncio
import streamlit as st

from app.agent.agent import MovieAgent
from pathlib import Path

from app.rag.parser import movie_title_from_filename


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

    .badge-informational {
        background: #1a4a1a;
        color: #6ddb6d;
    }

    .badge-email {
        background: #1a2a4a;
        color: #6db0ff;
    }

    .badge-clarification {
        background: #4a3a1a;
        color: #ffd06d;
    }
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
@st.cache_data
def _get_movie_titles() -> list[str]:
    """Discover all available movie titles from the subtitle directory."""
    subtitle_dir = Path(__file__).resolve().parent / "subtitles"

    if not subtitle_dir.exists():
        return []

    titles = []

    for subtitle_file in subtitle_dir.glob("*.srt"):
        try:
            title = movie_title_from_filename(subtitle_file)

            if title:
                titles.append(title)

        except Exception:
            continue

    return sorted(set(titles))

@st.cache_resource(show_spinner="Initialising agent…")
def _get_agent() -> MovieAgent:
    return MovieAgent()


def _badge(intent: str) -> str:
    cls = f"badge-{intent}"
    label = intent.upper()

    return (
        f'<span class="intent-badge {cls}">'
        f"{label}"
        f"</span>"
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    st.markdown("**Sources**")

    for source in sources:
        st.markdown(
            f'<div class="citation-block">'
            f"<b>{source['movie_title']}</b><br>"
            f"{source['start_time']} &rarr; {source['end_time']}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_result(result: dict) -> None:
    intent = result.get("intent", "")
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    st.markdown(
        _badge(intent),
        unsafe_allow_html=True,
    )

    if intent == "clarification":

        st.info(answer)

    elif intent == "informational":

        st.markdown("**Answer**")
        st.write(answer)

        _render_sources(sources)

    elif intent == "email":

        email_res = result.get("email", {})

        if email_res.get("success"):
            st.success(
                "✅ Email sent successfully (mock transport)"
            )

            st.markdown(
                f"**Recipient:** "
                f"`{result.get('recipient', 'unknown')}`"
            )

            st.markdown(
                "**Subject:** Movie Information"
            )

        else:
            err = email_res.get(
                "error",
                "Unknown error",
            )

            st.error(
                f"Email failed: {err}"
            )

        st.markdown("**Generated Content**")
        st.write(answer)

        _render_sources(sources)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🎬 Settings")

movie_titles = _get_movie_titles()


# ---------------------------------------------------------------------------
# Movie selection state
# ---------------------------------------------------------------------------

if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []


# ---------------------------------------------------------------------------
# Select All / Clear All
# ---------------------------------------------------------------------------

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button(
        "Select All",
        use_container_width=True,
    ):
        st.session_state.selected_movies = movie_titles.copy()
        st.rerun()


with col2:
    if st.button(
        "Clear All",
        use_container_width=True,
    ):
        st.session_state.selected_movies = []
        st.rerun()


# ---------------------------------------------------------------------------
# Multi-movie selector
# ---------------------------------------------------------------------------

selected_movies = st.sidebar.multiselect(
    "Movie subtitles",
    options=movie_titles,
    default=st.session_state.selected_movies,
    help=(
        "Select one or more movies. "
        "Retrieval will be restricted to the selected subtitles."
    ),
)


# Keep session state synchronized with manual selections
st.session_state.selected_movies = selected_movies


# ---------------------------------------------------------------------------
# Selection counter
# ---------------------------------------------------------------------------

st.sidebar.caption(
    f"Selected: {len(selected_movies)} / {len(movie_titles)} movies"
)


# ---------------------------------------------------------------------------
# Backend filter
# ---------------------------------------------------------------------------

# NOTE:
# The backend currently accepts a single movie_title.
# This will be updated in the next step to support movie lists.
movie_filter = selected_movies


# ---------------------------------------------------------------------------
# Sidebar information
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.caption(
    "Uses local Ollama (`qwen3.5:9b`) + ChromaDB.\n"
    "Emails are sent via a **mock** transport only."
)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title(
    "Movie Intelligence & Follow-up Assistant"
)

st.caption(
    "Ask questions about movie subtitles, or request "
    "movie information by email."
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

with st.expander("💡 Example questions"):

    st.markdown(
        """
- *What does the artist say about art?*
- *Why does Walter become important to the artists?*
- *What happens when Walter shows his sculpture?*
- *Email me a breakdown of what the artist says about art to you@example.com*
- *Email me the scene to you@example.com* ← triggers clarification
        """
    )


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

user_input = st.text_area(
    "Your question or request",
    placeholder=(
        "e.g. What does the artist say about art?"
    ),
    height=90,
    key="user_input",
)


# ---------------------------------------------------------------------------
# Ask button
# ---------------------------------------------------------------------------

run_btn = st.button(
    "Ask",
    type="primary",
    use_container_width=True,
)


st.markdown("---")


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

if run_btn:

    # ---------------------------------------------------------------
    # Validate user input
    # ---------------------------------------------------------------

    if not user_input or not user_input.strip():

        st.warning(
            "Please enter a question or request."
        )

    # ---------------------------------------------------------------
    # Validate movie selection
    # ---------------------------------------------------------------

    elif not selected_movies:

        st.warning(
            "Please select at least one movie subtitle."
        )

    # ---------------------------------------------------------------
    # Run agent
    # ---------------------------------------------------------------

    else:

        with st.spinner("Thinking…"):

            try:

                agent = _get_agent()

                result = _run_async(
                    agent.run(
                        user_input,
                        movie_title=movie_filter,
                    )
                )

                _render_result(result)

            except RuntimeError as exc:

                msg = str(exc)

                if (
                    "Ollama" in msg
                    or "connect" in msg.lower()
                ):

                    st.error(
                        "⚠️ Cannot reach Ollama. "
                        "Make sure `ollama serve` is running "
                        "and the `qwen3.5:9b` model is available."
                    )

                else:

                    st.error(
                        f"An error occurred: {msg}"
                    )

            except Exception as exc:

                st.error(
                    "An unexpected error occurred. "
                    "Please check the terminal for details.\n\n"
                    f"`{exc}`"
                )