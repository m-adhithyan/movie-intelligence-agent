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

    async def run(
        self,
        user_request: str,
        movie_title: str | None = None,
        movie_titles: list[str] | None = None,
    ) -> dict:
        """
        Process a user request end-to-end.

        Supports three intent paths:
          - informational → RAG retrieval + LLM answer + citations
          - email → RAG + LLM + MCP email tool
          - clarification → ask the user for more information
                             (no RAG, no email)

        Args:
            user_request:
                The raw text from the user.

            movie_title:
                Optional single movie filter for backward compatibility.

            movie_titles:
                Optional list of movie filters selected by the UI.
                When supplied, retrieval is restricted to these movies
                and cross-movie ambiguity is automatically suppressed.
        """

        # ----------------------------------------------------------
        # 0. Normalize movie filters
        # ----------------------------------------------------------
        # Preserve backward compatibility with callers that still
        # provide a single movie_title.
        if movie_titles is None and movie_title is not None:
            movie_titles = [movie_title]

        # Treat an empty movie selection as no explicit filter.
        if movie_titles is not None and len(movie_titles) == 0:
            movie_titles = None

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
        # 3. Retrieve context from the vector store (NO LLM yet)
        # ----------------------------------------------------------
        # Both informational and email paths use RAG retrieval.
        # We deliberately stop here before calling the LLM so that
        # we can inspect the sources and check for ambiguity first.
        sources = self.answerer.retrieve(
            question=route["query"],
            movie_titles=movie_titles,
        )

        # ----------------------------------------------------------
        # 4. Ambiguity check (BEFORE LLM generation)
        # ----------------------------------------------------------
        # If the caller explicitly selected one or more movies,
        # movie_titles is non-None and ambiguity is suppressed.
        ambiguity = self.ambiguity_handler.check(
            sources=sources,
            movie_titles=movie_titles,
        )

        if ambiguity.is_ambiguous:
            return {
                "intent": "clarification",
                "answer": ambiguity.clarification,
                "sources": [],
            }

        # ----------------------------------------------------------
        # 5. LLM generation (only reached when NOT ambiguous)
        # ----------------------------------------------------------
        rag_result = self.answerer.generate(
            question=route["query"],
            sources=sources,
        )

        # ----------------------------------------------------------
        # 6. Informational path
        # ----------------------------------------------------------
        if route["intent"] == "informational":
            return {
                "intent": "informational",
                "answer": rag_result["answer"],
                "sources": rag_result.get("sources", []),
                "citations": rag_result.get("citations", []),
            }

        # ----------------------------------------------------------
        # 7. Email path
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

        # Only include sources that were actually cited by the LLM.
        # This prevents unrelated retrieved chunks from appearing
        # in the email.
        sources_by_chunk_id = {
            source["chunk_id"]: source
            for source in rag_result.get("sources", [])
        }

        for citation in rag_result.get("citations", []):
            source = sources_by_chunk_id.get(citation["chunk_id"])

            if source is None:
                continue

            email_body += (
                f"- {source['movie_title']} | "
                f"{source['start_time']} --> "
                f"{source['end_time']}\n"
            )

        # ----------------------------------------------------------
        # 8. Dispatch email through MCP
        # ----------------------------------------------------------
        email_result = await self.send_email(
            recipient=route["recipient"],
            subject=email_subject,
            body=email_body,
        )

        # ----------------------------------------------------------
        # 9. Return complete result
        # ----------------------------------------------------------
        return {
            "intent": "email",
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "citations": rag_result.get("citations", []),
            "email": email_result,
            "recipient": route["recipient"],
        }