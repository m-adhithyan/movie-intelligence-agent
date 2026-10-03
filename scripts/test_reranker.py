from app.rag.retriever import Retriever
from app.rag.reranker import Reranker


# --------------------------------------------------
# Retrieve candidates
# --------------------------------------------------

query = "What does the artist say about art and creation?"

retriever = Retriever()

results = retriever.retrieve(
    query,
    n_results=5,
)

print("=" * 70)
print("INITIAL RETRIEVAL")
print("=" * 70)

for result in results:

    print(
        f"\n{result['chunk_id']} "
        f"| distance={result['distance']:.4f}"
    )

    print(
        result["text"][:250]
    )


# --------------------------------------------------
# Rerank candidates
# --------------------------------------------------

print("\n\nLoading reranker...")

reranker = Reranker()

reranked = reranker.rerank(
    query=query,
    results=results,
    top_k=3,
)


print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)

for result in reranked:

    print(
        f"\n{result['chunk_id']} "
        f"| "
        f"distance={result['distance']:.4f} "
        f"| "
        f"rerank_score="
        f"{result['rerank_score']:.4f}"
    )

    print(
        result["text"][:400]
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert results, (
    "Initial retrieval returned no results."
)

assert reranked, (
    "Reranker returned no results."
)

assert len(reranked) <= 3

# Every result must have a reranking score.
assert all(
    "rerank_score" in result
    for result in reranked
)

# Scores must be descending.
scores = [
    result["rerank_score"]
    for result in reranked
]

assert scores == sorted(
    scores,
    reverse=True,
), "Reranked results are not sorted correctly."


# Existing metadata must survive reranking.
for result in reranked:

    assert result["chunk_id"]
    assert result["movie_title"]
    assert result["movie_id"]
    assert result["start_time"]
    assert result["end_time"]
    assert result["text"]


print("\n" + "=" * 70)
print("ALL RERANKER TESTS PASSED")
print("=" * 70)