from .vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        min_distance: float | None = None,
    ):
        self.vector_store = (
            vector_store or VectorStore()
        )

        self.min_distance = min_distance

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        movie_title: str | None = None,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if n_results <= 0:
            raise ValueError(
                "n_results must be greater than 0."
            )

        where = None

        if movie_title:
            where = {
                "movie_title": movie_title
            }

        # Retrieve extra candidates when filtering is enabled.
        # This prevents weak matches from reducing the final
        # result count unnecessarily.
        candidate_count = n_results

        if self.min_distance is not None:
            candidate_count = max(
                n_results * 2,
                n_results + 2,
            )

        results = self.vector_store.search(
            query=query,
            n_results=candidate_count,
            where=where,
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        seen_chunk_ids = set()

        for document, metadata, distance, chunk_id in zip(
            documents,
            metadatas,
            distances,
            ids,
        ):

            # Prevent duplicate chunks from reaching the LLM.
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)

            # Lower Chroma distance means a more similar result.
            if (
                self.min_distance is not None
                and distance > self.min_distance
            ):
                continue

            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "movie_title": metadata[
                        "movie_title"
                    ],
                    "movie_id": metadata.get(
                        "movie_id"
                    ),
                    "start_time": metadata[
                        "start_time"
                    ],
                    "end_time": metadata[
                        "end_time"
                    ],
                    "start_seconds": metadata[
                        "start_seconds"
                    ],
                    "end_seconds": metadata[
                        "end_seconds"
                    ],
                    "subtitle_indices": metadata.get(
                        "subtitle_indices",
                        "",
                    ),
                    "distance": distance,
                }
            )

            # Stop once we have enough valid results.
            if len(retrieved) >= n_results:
                break

        return retrieved