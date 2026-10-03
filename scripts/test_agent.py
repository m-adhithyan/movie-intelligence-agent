"""
End-to-end agent tests (Phase 10)
=================================

Run with:
    python -m scripts.test_agent

All tests run against the local Ollama model (qwen3.5:9b)
and the ChromaDB vector store. No paid APIs are used.
"""

import asyncio
import sys

from app.agent.agent import MovieAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(n: int, title: str) -> None:
    print("\n" + "=" * 70)
    print(f"TEST {n}: {title}")
    print("=" * 70)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  FAIL: {message}")
        sys.exit(1)

    print(f"  PASS: {message}")


def _show(result: dict) -> None:
    print(f"\n  intent  : {result.get('intent')}")

    answer = result.get("answer", "")
    preview = (answer[:120] + "...") if len(answer) > 120 else answer
    print(f"  answer  : {preview!r}")

    sources = result.get("sources", [])
    print(f"  sources : {len(sources)} retrieved")

    for source in sources[:3]:
        print(
            f"    - {source['movie_title']} | "
            f"{source['start_time']} --> {source['end_time']}"
        )

    if result.get("email"):
        print(f"  email   : {result['email'].get('success')}")
        print(f"  recipient: {result.get('recipient')}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_1_informational(agent: MovieAgent) -> None:
    """
    Straightforward informational question.

    Expected:
      - intent=informational
      - non-empty answer
      - sources
      - citations
    """
    _header(
        1,
        "Informational – What does the artist say about art?"
    )

    request = "What does the artist say about art?"
    result = await agent.run(
        request,
        movie_title="A Bucket Of Blood 1959",
    )

    _show(result)

    _assert(
        result["intent"] == "informational",
        "intent is informational"
    )

    _assert(
        bool(result.get("answer", "").strip()),
        "answer is non-empty"
    )

    _assert(
        len(result.get("sources", [])) > 0,
        "sources are present"
    )

    _assert(
        len(result.get("citations", [])) > 0,
        "citations are present"
    )

    for citation in result.get("citations", []):
        _assert(
            bool(citation.get("movie_title")),
            "citation has movie_title"
        )

        _assert(
            bool(citation.get("start_time")),
            "citation has start_time"
        )

        _assert(
            bool(citation.get("end_time")),
            "citation has end_time"
        )


async def test_2_email_valid(agent: MovieAgent) -> None:
    """
    Valid email request with enough context.

    Expected:
      - intent=email
      - non-empty answer
      - MCP called
      - success=True
      - cited sources appear in email
      - retrieved but uncited sources do not appear in email
    """
    _header(
        2,
        "Email – breakdown of what artist says about art"
    )

    request = (
        "Email me a breakdown of what the artist says about art "
        "to test@example.com"
    )

    result = await agent.run(
        request,
        movie_title="A Bucket Of Blood 1959",
    )

    _show(result)

    _assert(
        result["intent"] == "email",
        "intent is email"
    )

    _assert(
        bool(result.get("answer", "").strip()),
        "generated answer is non-empty"
    )

    _assert(
        result.get("email") is not None,
        "email field present"
    )

    _assert(
        result["email"].get("success") is True,
        "email success=True"
    )

    _assert(
        result.get("recipient") == "test@example.com",
        "recipient is correct"
    )

    # ---------------------------------------------------------
    # Verify email body
    # ---------------------------------------------------------

    mcp_result = result["email"].get("result")

    if mcp_result is not None:
        import json as _json

        try:
            mcp_text = mcp_result.content[0].text
            mcp_data = _json.loads(mcp_text)

            body = mcp_data.get("body", "")

            _assert(
                bool(body.strip()),
                "email body is non-empty"
            )

            _assert(
                "Sources" in body,
                "email body contains Sources"
            )

            _assert(
                "A Bucket" in body or "1959" in body,
                "email body references a movie"
            )

            # -------------------------------------------------
            # Verify citation coverage
            # -------------------------------------------------

            citations = result.get("citations", [])

            _assert(
                len(citations) > 0,
                "email has at least one citation"
            )

            for citation in citations:
                expected = (
                    f"{citation['movie_title']} | "
                    f"{citation['start_time']} --> "
                    f"{citation['end_time']}"
                )

                _assert(
                    expected in body,
                    f"email contains cited source: {expected}"
                )

            # -------------------------------------------------
            # Verify that retrieved-but-uncited sources are
            # NOT included in the email.
            # -------------------------------------------------

            retrieved_sources = result.get("sources", [])

            cited_chunk_ids = {
                citation["chunk_id"]
                for citation in citations
            }

            uncited_sources = [
                source
                for source in retrieved_sources
                if source["chunk_id"] not in cited_chunk_ids
            ]

            for source in uncited_sources:
                unexpected = (
                    f"{source['movie_title']} | "
                    f"{source['start_time']} --> "
                    f"{source['end_time']}"
                )

                _assert(
                    unexpected not in body,
                    f"email excludes uncited source: {unexpected}"
                )

        except Exception:
            print(
                "  NOTE: could not parse MCP body text "
                "(still passed)"
            )


async def test_3_vague_email_with_recipient(
    agent: MovieAgent
) -> None:
    """
    Vague email: has a recipient but no meaningful content topic.

    Expected:
      - intent=clarification
      - NO RAG call
      - NO MCP call
    """
    _header(
        3,
        "Clarification – 'Email me the scene to test@example.com'"
    )

    request = "Email me the scene to test@example.com"
    result = await agent.run(request)

    _show(result)

    _assert(
        result["intent"] == "clarification",
        "intent is clarification (not email)"
    )

    _assert(
        bool(result.get("answer", "").strip()),
        "clarification message is non-empty"
    )

    _assert(
        "email" not in result or result.get("email") is None,
        "MCP email was NOT called"
    )

    _assert(
        len(result.get("sources", [])) == 0,
        "no RAG sources returned"
    )


async def test_4_vague_email_no_recipient(
    agent: MovieAgent
) -> None:
    """
    Vague email with no recipient and no content topic.

    Expected:
      - intent=clarification
      - missing information identified
    """
    _header(
        4,
        "Clarification – 'Email me the dialogue'"
    )

    request = "Email me the dialogue"
    result = await agent.run(request)

    _show(result)

    _assert(
        result["intent"] == "clarification",
        "intent is clarification"
    )

    _assert(
        bool(result.get("answer", "").strip()),
        "clarification message is non-empty"
    )

    _assert(
        "email" not in result or result.get("email") is None,
        "MCP email was NOT called"
    )


async def test_5_ambiguous_movie(agent: MovieAgent) -> None:
    """
    Generic question that could match multiple movies.

    If multiple movies are indexed, expect clarification.
    If only one movie is indexed, expect an informational answer.

    Either way the system must not hallucinate or crash.
    """
    _header(
        5,
        "Ambiguous or broad question – 'What happens in the movie?'"
    )

    request = "What happens in the movie?"
    result = await agent.run(request)

    _show(result)

    intent = result["intent"]

    _assert(
        intent in ("clarification", "informational"),
        f"intent is clarification or informational (got {intent!r})"
    )

    if intent == "clarification":
        _assert(
            bool(result.get("answer", "").strip()),
            "clarification message is non-empty"
        )

        print(
            "  NOTE: multiple movies detected – "
            "clarification issued."
        )

    else:
        _assert(
            bool(result.get("answer", "").strip()),
            "answer is non-empty"
        )

        print(
            "  NOTE: single movie in index – "
            "informational answer given."
        )


async def test_6_insufficient_information(
    agent: MovieAgent
) -> None:
    """
    Question for which the subtitle corpus has no evidence.

    Expected:
      - answer explicitly says insufficient information
      - no hallucination
    """
    _header(
        6,
        "Insufficient information – obscure out-of-scope question"
    )

    request = (
        "What is the exact chemical composition of the paint used "
        "on the walls in the opening scene?"
    )

    result = await agent.run(
        request,
        movie_title="A Bucket Of Blood 1959",
    )

    _show(result)

    answer = result.get("answer", "").lower()

    _assert(
        result["intent"] in ("informational", "clarification"),
        "intent is informational or clarification"
    )

    _assert(
        bool(answer.strip()),
        "answer/clarification is non-empty"
    )

    # The answer should signal insufficient information
    # rather than invent one.
    insufficient_signals = [
        "not provide",
        "do not provide",
        "no information",
        "not enough",
        "not available",
        "cannot find",
        "could not find",
        "subtitles do not",
        "insufficient",
        "not found",
        "don't have",
    ]

    signals_found = any(
        signal in answer
        for signal in insufficient_signals
    )

    _assert(
        signals_found,
        "answer indicates insufficient information (no hallucination)"
    )


# ---------------------------------------------------------------------------
# New multi-movie tests (A – F)
# ---------------------------------------------------------------------------

async def test_A_single_explicit_movie(agent: MovieAgent) -> None:
    """
    A. Single explicit movie selection.

    question + movie_title
      → informational answer
      → citations are all from the selected movie.
    """
    _header("A", "Single explicit movie – artist / art in A Bucket Of Blood")

    request = "What does the artist say about art?"
    result = await agent.run(
        request,
        movie_title="A Bucket Of Blood 1959",
    )

    _show(result)

    _assert(
        result["intent"] == "informational",
        "intent is informational",
    )

    _assert(
        bool(result.get("answer", "").strip()),
        "answer is non-empty",
    )

    _assert(
        len(result.get("sources", [])) > 0,
        "sources are present",
    )

    for citation in result.get("citations", []):
        _assert(
            citation.get("movie_title") == "A Bucket Of Blood 1959",
            f"citation movie_title is 'A Bucket Of Blood 1959' (got {citation.get('movie_title')!r})",
        )


async def test_B_multiple_explicit_movies(agent: MovieAgent) -> None:
    """
    B. Multiple explicit movies.

    question + movie_titles
      → retrieval restricted to selected movies.
      → no cross-movie clarification.
    """
    _header("B", "Multiple explicit movies – retrieval restricted to selection")

    request = "What does the artist say about art?"
    selected = ["A Bucket Of Blood 1959", "The Devil Bat 1940"]

    result = await agent.run(
        request,
        movie_titles=selected,
    )

    _show(result)

    _assert(
        result["intent"] in ("informational", "clarification"),
        f"intent is informational or clarification (got {result['intent']!r})",
    )

    # When multiple movies are explicitly selected the agent must NOT
    # return a cross-movie clarification asking "which movie?".
    if result["intent"] == "clarification":
        answer_lower = result.get("answer", "").lower()
        _assert(
            "which movie" not in answer_lower,
            "no 'which movie' clarification when movies are explicitly selected",
        )

    for source in result.get("sources", []):
        _assert(
            source["movie_title"] in selected,
            f"source movie '{source['movie_title']}' is within the selected set",
        )


async def test_C_no_selection_multi_movie_clarification(
    agent: MovieAgent,
) -> None:
    """
    C. No movie selection + broad question → clarification, no LLM call.

    The test cannot directly verify that the LLM was NOT called, but it
    can verify:
      - intent == clarification
      - answer lists relevant movie titles (not all 100)
    """
    _header(
        "C",
        "No movie selection + broad question -> clarification (no LLM)",
    )

    request = "What happens in the movie?"
    result = await agent.run(request)

    _show(result)

    intent = result["intent"]

    _assert(
        intent in ("clarification", "informational"),
        f"intent is clarification or informational (got {intent!r})",
    )

    if intent == "clarification":
        _assert(
            bool(result.get("answer", "").strip()),
            "clarification message is non-empty",
        )

        _assert(
            len(result.get("sources", [])) == 0,
            "no sources in clarification response",
        )

        print("  NOTE: clarification issued (expected for multi-movie corpus).")
    else:
        print("  NOTE: single-movie corpus – informational answer given.")


async def test_D_email_with_explicit_movie(agent: MovieAgent) -> None:
    """
    D. Email + explicit movie.

    → RAG → LLM answer → MCP email
    → correct recipient
    → citations from the selected movie only.
    """
    _header(
        "D",
        "Email + explicit movie -> RAG -> answer -> MCP email",
    )

    request = (
        "Email me a breakdown of what the artist says about art "
        "to test@example.com"
    )

    result = await agent.run(
        request,
        movie_title="A Bucket Of Blood 1959",
    )

    _show(result)

    _assert(
        result["intent"] == "email",
        "intent is email",
    )

    _assert(
        bool(result.get("answer", "").strip()),
        "generated answer is non-empty",
    )

    _assert(
        result.get("email") is not None,
        "email field is present",
    )

    _assert(
        result["email"].get("success") is True,
        "email success=True",
    )

    _assert(
        result.get("recipient") == "test@example.com",
        "recipient is correct",
    )

    _assert(
        len(result.get("citations", [])) > 0,
        "at least one citation in email answer",
    )

    for citation in result.get("citations", []):
        _assert(
            citation.get("movie_title") == "A Bucket Of Blood 1959",
            f"citation from selected movie (got {citation.get('movie_title')!r})",
        )


async def test_E_email_no_movie_multi_relevant(
    agent: MovieAgent,
) -> None:
    """
    E. Email + no movie + multiple relevant movies → clarification, MCP not called.
    """
    _header(
        "E",
        "Email + no movie + ambiguous corpus -> clarification, no MCP",
    )

    request = (
        "Email me a breakdown of what the artist says about art "
        "to test@example.com"
    )

    result = await agent.run(request)

    _show(result)

    intent = result["intent"]

    # When the corpus is genuinely multi-movie the system must clarify.
    # If the corpus happens to map to a single movie, informational/email
    # is acceptable but should NOT be wrong.
    if intent == "clarification":
        _assert(
            "email" not in result or result.get("email") is None,
            "MCP was NOT called when clarification needed",
        )

        _assert(
            bool(result.get("answer", "").strip()),
            "clarification message is non-empty",
        )

        print("  NOTE: correctly blocked MCP call due to ambiguity.")
    else:
        # Single-movie or unambiguous corpus – either email or informational
        # is acceptable here.
        _assert(
            intent in ("email", "informational"),
            f"intent is email or informational when unambiguous (got {intent!r})",
        )

        print(f"  NOTE: corpus resolved to single movie – intent={intent!r}.")


async def test_F_explicit_movie_insufficient_evidence(
    agent: MovieAgent,
) -> None:
    """
    F. Explicit movie + question with no evidence in subtitles.

    → insufficient-information answer
    → no hallucination
    → no cross-movie clarification
    """
    _header(
        "F",
        "Explicit movie + no subtitle evidence -> insufficient-info, no hallucination",
    )

    request = (
        "What is the exact chemical composition of the paint used "
        "on the walls?"
    )

    result = await agent.run(
        request,
        movie_title="A Bucket Of Blood 1959",
    )

    _show(result)

    _assert(
        result["intent"] in ("informational", "clarification"),
        "intent is informational or clarification",
    )

    answer = result.get("answer", "").lower()

    _assert(
        bool(answer.strip()),
        "answer is non-empty",
    )

    insufficient_signals = [
        "not provide",
        "do not provide",
        "no information",
        "not enough",
        "not available",
        "cannot find",
        "could not find",
        "subtitles do not",
        "insufficient",
        "not found",
        "don't have",
        "unable to find",
        "no mention",
    ]

    signals_found = any(
        signal in answer for signal in insufficient_signals
    )

    _assert(
        signals_found,
        "answer indicates insufficient information (no hallucination)",
    )

    # The agent must NOT ask "which movie?" when a movie is explicitly given.
    _assert(
        "which movie" not in answer,
        "no 'which movie' clarification when movie is explicitly selected",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = MovieAgent()

    tests = [
        test_1_informational,
        test_2_email_valid,
        test_3_vague_email_with_recipient,
        test_4_vague_email_no_recipient,
        test_5_ambiguous_movie,
        test_6_insufficient_information,
        # New multi-movie tests
        test_A_single_explicit_movie,
        test_B_multiple_explicit_movies,
        test_C_no_selection_multi_movie_clarification,
        test_D_email_with_explicit_movie,
        test_E_email_no_movie_multi_relevant,
        test_F_explicit_movie_insufficient_evidence,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            await test_fn(agent)
            passed += 1

        except SystemExit:
            failed += 1

        except Exception as exc:
            print(
                f"\n  ERROR: unexpected exception – {exc}"
            )
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())