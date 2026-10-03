from app.agent.llm import LocalLLM
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker


SYSTEM_PROMPT = """
You are a movie subtitle question-answering assistant.

Answer questions using ONLY the provided subtitle sources.

Rules:

1. Do not use outside knowledge.
2. Do not invent events, dialogue, characters, or quotes.
3. If the provided sources do not contain enough information,
   say that the available subtitles do not provide enough information.
4. Keep the answer concise and directly answer the question.
5. Each factual claim must reference one or more source numbers.
6. Do not create or modify timestamps yourself.
7. Never invent citations.

The application will convert source numbers into exact
movie/timestamp citations after you answer.
"""


class RAGAnswerer:

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LocalLLM | None = None,
        reranker: Reranker | None = None,
        use_reranker: bool = True,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LocalLLM()

        self.use_reranker = use_reranker

        if use_reranker:
            self.reranker = reranker or Reranker()
        else:
            self.reranker = None

    def answer(
        self,
        question: str,
        movie_title: str | None = None,
        n_results: int = 5,
    ) -> dict:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Retrieve a larger candidate set when reranking.
        retrieval_count = n_results

        if self.use_reranker:
            retrieval_count = max(n_results * 2, 5)

        sources = self.retriever.retrieve(
            query=question,
            n_results=retrieval_count,
            movie_title=movie_title,
        )

        if not sources:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the available subtitles."
                ),
                "citations": [],
                "sources": [],
            }

        # Optional semantic reranking.
        if self.use_reranker and self.reranker is not None:
            sources = self.reranker.rerank(
                query=question,
                results=sources,
                top_k=n_results,
            )

        context_parts = []

        for index, source in enumerate(sources, start=1):

            context_parts.append(
                f"""
SOURCE {index}

Movie: {source["movie_title"]}
Time: {source["start_time"]} --> {source["end_time"]}

Dialogue:
{source["text"]}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
Answer the following question using ONLY the sources below.

QUESTION:
{question}

SOURCES:
{context}

For every factual statement, include the relevant source number
in square brackets, for example [1] or [1][3].

Do not write movie timestamps yourself.
Do not invent source numbers.

Answer:
"""

        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        citations = []

        for index, source in enumerate(sources, start=1):

            citations.append(
                {
                    "source_number": index,
                    "movie_title": source["movie_title"],
                    "start_time": source["start_time"],
                    "end_time": source["end_time"],
                    "chunk_id": source["chunk_id"],
                }
            )

        return {
            "answer": answer.strip(),
            "citations": citations,
            "sources": sources,
        }