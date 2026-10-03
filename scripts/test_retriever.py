from app.rag.retriever import Retriever


retriever = Retriever()


# --------------------------------------------------
# Test 1: General retrieval
# --------------------------------------------------

query = "What does the artist say about art?"

results = retriever.retrieve(
    query,
    n_results=3,
)

print("=" * 70)
print("GENERAL RETRIEVAL")
print("=" * 70)

print(f"Query: {query}")
print(f"Results: {len(results)}")

for result in results:
    print("\n" + "-" * 70)

    print(f"Chunk ID: {result['chunk_id']}")

    print(
        f"Movie: {result['movie_title']}"
    )

    print(
        f"Time: "
        f"{result['start_time']} --> "
        f"{result['end_time']}"
    )

    print(
        f"Distance: "
        f"{result['distance']:.4f}"
    )

    print(
        f"Text:\n{result['text'][:500]}"
    )


# --------------------------------------------------
# Test 2: Movie-filtered retrieval
# --------------------------------------------------

movie_title = "A Bucket Of Blood 1959"

results_filtered = retriever.retrieve(
    query,
    n_results=3,
    movie_title=movie_title,
)

print("\n\n" + "=" * 70)
print("MOVIE-FILTERED RETRIEVAL")
print("=" * 70)

print(f"Query: {query}")
print(f"Movie filter: {movie_title}")
print(f"Results: {len(results_filtered)}")

for result in results_filtered:
    print("\n" + "-" * 70)

    print(
        f"Movie: {result['movie_title']}"
    )

    print(
        f"Time: "
        f"{result['start_time']} --> "
        f"{result['end_time']}"
    )

    print(
        f"Distance: "
        f"{result['distance']:.4f}"
    )

    print(
        f"Text:\n{result['text'][:500]}"
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert results
assert len(results) == 3

assert results_filtered
assert len(results_filtered) == 3

# Every filtered result must belong to requested movie
assert all(
    result["movie_title"] == movie_title
    for result in results_filtered
)

# Every result must contain citation information
for result in results_filtered:
    assert result["movie_title"]
    assert result["start_time"]
    assert result["end_time"]
    assert result["text"]
    assert result["chunk_id"]


print("\n" + "=" * 70)
print("ALL RETRIEVER TESTS PASSED")
print("=" * 70)

# New metadata validation
for result in results_filtered:
    assert result["movie_id"]
    assert result["subtitle_indices"]


# Results must be ordered from most similar
# to least similar.
distances = [
    result["distance"]
    for result in results_filtered
]

assert distances == sorted(distances), (
    "Retriever results are not ordered "
    "by similarity."
)


# Chunk IDs must be unique.
chunk_ids = [
    result["chunk_id"]
    for result in results_filtered
]

assert len(chunk_ids) == len(
    set(chunk_ids)
), "Duplicate chunks returned."


print("\nRetrieval metadata validation: PASSED")