from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Rerank retrieved subtitle chunks using a cross-encoder.

    The cross-encoder scores each query/document pair
    directly, allowing more precise relevance ranking
    than embedding similarity alone.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if not results:
            return []

        pairs = [
            (
                query,
                result["text"],
            )
            for result in results
        ]

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores,
        ):
            updated_result = dict(result)

            updated_result[
                "rerank_score"
            ] = float(score)

            reranked.append(
                updated_result
            )

        reranked.sort(
            key=lambda result: result[
                "rerank_score"
            ],
            reverse=True,
        )

        return reranked[:top_k]