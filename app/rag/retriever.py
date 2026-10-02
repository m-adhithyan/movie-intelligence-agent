from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        movie_title: str | None = None,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be greater than 0.")

        # Build metadata filter when a movie is specified.
        where = None

        if movie_title:
            where = {
                "movie_title": movie_title
            }

        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            where=where,
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for document, metadata, distance, chunk_id in zip(
            documents,
            metadatas,
            distances,
            ids,
        ):
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "movie_title": metadata["movie_title"],
                    "start_time": metadata["start_time"],
                    "end_time": metadata["end_time"],
                    "start_seconds": metadata["start_seconds"],
                    "end_seconds": metadata["end_seconds"],
                    "distance": distance,
                }
            )

        return retrieved