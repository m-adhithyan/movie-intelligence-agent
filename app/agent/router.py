import re
from dataclasses import dataclass


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

EMAIL_ACTION_PATTERN = re.compile(
    r"\b(email|e-mail|send|mail|dispatch)\b",
    re.IGNORECASE,
)

CONTENT_PATTERN = re.compile(
    r"\b("
    r"scene|scenes|dialogue|dialog|quote|quotes|"
    r"analysis|analyze|breakdown|summary|summarize|"
    r"explain|explaining|conversation|moment|"
    r"says|said|speech|"
    r"about|what|why|how"
    r")\b",
    re.IGNORECASE,
)

# Words that contribute no real topic when left alone
_STOP_WORDS = frozenset(
    "me the a an to from for it this that of and or "
    "please can could would will i my us our is are was "
    "them they he she him her we you your".split()
)

# Vague content words: meaningful-sounding but describe no actual topic
VAGUE_CONTENT_WORDS = frozenset(
    "scene scenes dialogue dialog quote quotes "
    "analysis breakdown summary conversation moment speech".split()
)


@dataclass
class RouteDecision:
    intent: str
    query: str | None = None
    recipient: str | None = None
    missing: list[str] | None = None
    clarification: str | None = None


def _is_content_vague(content_text: str) -> bool:
    """
    Return True when the text (with action verbs and email addresses
    already removed) contains no substantive topic beyond bare vague
    content words and stop words.

    Examples that ARE vague:
        "me the scene to"    → stripped = {"scene"}       → vague
        "the dialogue"       → stripped = {"dialogue"}    → vague

    Examples that are NOT vague:
        "what the artist says about art"  → has "artist", "art"
        "breakdown of Walter's sculpture" → has "breakdown", "Walter"
    """
    tokens = re.findall(r"\b[a-z]+\b", content_text.lower())
    non_stop = [t for t in tokens if t not in _STOP_WORDS]

    if not non_stop:
        return True  # nothing left at all

    # If every remaining token is a vague content word, it's vague.
    return all(t in VAGUE_CONTENT_WORDS for t in non_stop)


class AgentRouter:
    def route(self, user_request: str) -> dict:
        request = user_request.strip()

        if not request:
            return {
                "intent": "clarification",
                "missing": ["request"],
                "clarification": "What would you like me to do?",
            }

        # ---------------------------------------------------------
        # 1. Detect whether this is an email/action request
        # ---------------------------------------------------------
        is_email_action = bool(
            EMAIL_ACTION_PATTERN.search(request)
        )

        if not is_email_action:
            return {
                "intent": "informational",
                "query": request,
            }

        # ---------------------------------------------------------
        # 2. Extract recipient
        # ---------------------------------------------------------
        email_match = EMAIL_PATTERN.search(request)
        recipient = email_match.group(0) if email_match else None

        # ---------------------------------------------------------
        # 3. Determine whether the request contains substantive
        #    movie content (not just vague words like "the scene")
        # ---------------------------------------------------------
        request_without_email = EMAIL_PATTERN.sub(" ", request)
        request_without_email = re.sub(
            r"\s+", " ", request_without_email
        ).strip()

        # Text with action verbs removed — used for vagueness check
        content_text = re.sub(
            r"\b(email|e-mail|send|mail|dispatch)\b",
            " ",
            request_without_email,
            flags=re.IGNORECASE,
        )
        content_text = re.sub(r"\s+", " ", content_text).strip()

        # A request has content only when it passes the surface
        # pattern check AND is not merely vague placeholder words.
        has_surface_content = bool(
            CONTENT_PATTERN.search(request_without_email)
        )
        has_content = has_surface_content and not _is_content_vague(
            content_text
        )

        missing = []

        if not has_content:
            missing.append("movie_content")

        if not recipient:
            missing.append("recipient")

        # ---------------------------------------------------------
        # 4. Missing required information → clarification
        # ---------------------------------------------------------
        if missing:
            if "movie_content" in missing and "recipient" in missing:
                clarification = (
                    "Which movie scene, dialogue, quote, or analysis "
                    "would you like me to email, and what email address "
                    "should I send it to?"
                )

            elif "movie_content" in missing:
                clarification = (
                    "Which movie scene, dialogue, quote, or analysis "
                    "would you like me to email?"
                )

            else:
                clarification = (
                    "What email address should I send the movie "
                    "information to?"
                )

            return {
                "intent": "clarification",
                "missing": missing,
                "clarification": clarification,
            }

        # ---------------------------------------------------------
        # 5. Valid email request — build a clean query for RAG
        # ---------------------------------------------------------
        query = EMAIL_PATTERN.sub(" ", request)
        query = re.sub(
            r"\b(email|e-mail|send|mail|dispatch)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(r"\s+", " ", query).strip()

        return {
            "intent": "email",
            "query": query,
            "recipient": recipient,
        }