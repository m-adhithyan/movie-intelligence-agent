from app.rag.answerer import RAGAnswerer


answerer = RAGAnswerer()


question = "What does the artist say about art?"

movie_title = "A Bucket Of Blood 1959"


result = answerer.answer(
    question=question,
    movie_title=movie_title,
    n_results=5,
)


print("=" * 70)
print("RAG ANSWER")
print("=" * 70)

print(f"\nQuestion:\n{question}")

print("\nAnswer:")
print(result["answer"])


print("\n" + "=" * 70)
print("CITATIONS")
print("=" * 70)

for citation in result["citations"]:

    print(
        f"\n[{citation['source_number']}] "
        f"{citation['movie_title']} — "
        f"{citation['start_time']} → "
        f"{citation['end_time']}"
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert result["answer"]

assert result["citations"]

assert result["sources"]

for citation in result["citations"]:

    assert citation["movie_title"]
    assert citation["start_time"]
    assert citation["end_time"]
    assert citation["chunk_id"]


print("\n" + "=" * 70)
print("RAG PIPELINE TEST PASSED")
print("=" * 70)