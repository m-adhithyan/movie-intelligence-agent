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


@dataclass
class RouteDecision:
    intent: str
    query: str | None = None
    recipient: str | None = None
    missing: list[str] | None = None
    clarification: str | None = None


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
        # 3. Determine whether the request contains movie content
        # ---------------------------------------------------------
        request_without_email = EMAIL_PATTERN.sub(" ", request)
        request_without_email = re.sub(r"\s+", " ", request_without_email).strip()

        has_content = bool(
            CONTENT_PATTERN.search(request_without_email)
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
        # 5. Valid email request
        # ---------------------------------------------------------
        return {
            "intent": "email",
            "query": request,
            "recipient": recipient,
        }