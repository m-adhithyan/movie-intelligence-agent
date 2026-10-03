import asyncio

from mcp import Client
from app.mcp.email_server import mcp
from app.agent.router import AgentRouter
from app.agent.ambiguity import AmbiguityHandler
from app.rag.answerer import RAGAnswerer


class MovieAgent:
    def __init__(self):
        self.router = AgentRouter()
        self.ambiguity_handler = AmbiguityHandler()
        self.answerer = RAGAnswerer()

    async def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
    ) -> dict:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "send_email",
                {
                    "recipient": recipient,
                    "subject": subject,
                    "body": body,
                },
            )

            if result.is_error:
                return {
                    "success": False,
                    "error": "MCP email tool rejected the request.",
                }

            return {
                "success": True,
                "result": result,
            }

    async def run(self, user_request: str, movie_title: str | None = None) -> dict:
        """
        Process a user request end-to-end.

        Supports three intent paths:
          - informational  → RAG retrieval + LLM answer + citations
          - email          → RAG + LLM + MCP email tool
          - clarification  → ask the user for more information
                             (no RAG, no email)

        Args:
            user_request: The raw text from the user.
            movie_title:  Optional movie filter (e.g. set by the UI
                          movie selector). When supplied, ambiguity
                          checking is skipped.
        """

        # ----------------------------------------------------------
        # 1. Route the request
        # ----------------------------------------------------------
        route = self.router.route(user_request)

        # ----------------------------------------------------------
        # 2. Immediate clarification from the router
        # ----------------------------------------------------------
        if route["intent"] == "clarification":
            return {
                "intent": "clarification",
                "answer": route["clarification"],
                "sources": [],
            }

        # ----------------------------------------------------------
        # 3. Retrieve context from the vector store
        # ----------------------------------------------------------
        # Both informational and email paths need RAG.
        rag_result = self.answerer.answer(
            question=route["query"],
            movie_title=movie_title,
        )

        # ----------------------------------------------------------
        # 4. Ambiguity check (skip when user already chose a movie)
        # ----------------------------------------------------------
        ambiguity = self.ambiguity_handler.check(
            sources=rag_result.get("sources", []),
            movie_title=movie_title,
        )

        if ambiguity.is_ambiguous:
            return {
                "intent": "clarification",
                "answer": ambiguity.clarification,
                "sources": [],
            }

        # ----------------------------------------------------------
        # 5. Informational path
        # ----------------------------------------------------------
        if route["intent"] == "informational":
            return {
                "intent": "informational",
                "answer": rag_result["answer"],
                "sources": rag_result.get("sources", []),
                "citations": rag_result.get("citations", []),
            }

        # ----------------------------------------------------------
        # 6. Email path
        # ----------------------------------------------------------
        answer = rag_result["answer"]

        # Guard: never send an empty email.
        if not answer or not answer.strip():
            return {
                "intent": "clarification",
                "answer": (
                    "I was not able to generate a meaningful response "
                    "from the available subtitles. Could you clarify "
                    "what specific information you would like emailed?"
                ),
                "sources": [],
            }

        email_subject = "Movie Information"

        email_body = f"{answer}\n\nSources:\n"

        for source in rag_result.get("sources", []):
            email_body += (
                f"- {source['movie_title']} | "
                f"{source['start_time']} --> "
                f"{source['end_time']}\n"
            )

        email_result = await self.send_email(
            recipient=route["recipient"],
            subject=email_subject,
            body=email_body,
        )

        return {
            "intent": "email",
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "citations": rag_result.get("citations", []),
            "email": email_result,
            "recipient": route["recipient"],
        }