from app.agent.router import AgentRouter
from app.agent.ambiguity import AmbiguityHandler
from app.rag.answerer import RAGAnswerer


class MovieAgent:
    def __init__(self):
        self.router = AgentRouter()
        self.ambiguity_handler = AmbiguityHandler()
        self.answerer = RAGAnswerer()

    def run(self, user_request: str) -> dict:
        """
        Process a user request through the agent router and RAG system.

        Step 10.1 currently handles informational requests only.
        Email actions will be connected after this path is tested.
        """

        # 1. Decide what the user wants
        route = self.router.route(user_request)

        # 2. Handle clarification requests
        if route["intent"] == "clarification":
            return {
                "intent": "clarification",
                "answer": route["clarification"],
                "sources": [],
            }

        # 3. Informational request → RAG
        if route["intent"] == "informational":
            result = self.answerer.answer(
                route["query"]
            )


            # 4. Check whether retrieved information is ambiguous
            ambiguity = self.ambiguity_handler.check(
                sources=result.get("sources", [])
            )

            if ambiguity.is_ambiguous:
                return {
                    "intent": "clarification",
                    "answer": ambiguity.clarification,
                    "sources": [],
                }

            return {
                "intent": "informational",
                "answer": result["answer"],
                "sources": result.get("sources", []),
            }

        # Email handling will be added in Step 10.2
        return {
            "intent": "clarification",
            "answer": "This action is not connected yet.",
            "sources": [],
        }