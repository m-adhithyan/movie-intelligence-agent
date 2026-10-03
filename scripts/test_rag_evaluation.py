from app.rag.answerer import RAGAnswerer
from app.rag.citations import extract_source_numbers

def validate_result(result: dict):
    answer = result["answer"]
    citations = result["citations"]
    sources = result["sources"]

    assert answer.strip(), "Answer must not be empty."

    assert len(answer.split()) >= 5, (
    "Answer is too short or appears incomplete."
    )

    assert citations, (
        "A substantive RAG answer must contain at least one citation."
    )

    assert "[" in answer and "]" in answer, (
        "RAG answer must contain source citations."
    )

    source_numbers = {
        citation["source_number"]
        for citation in citations
    }

    referenced_numbers = set(
        extract_source_numbers(answer)
    )

    assert referenced_numbers == source_numbers, (
        "Returned citations must exactly match "
        "the source references used in the answer."
    )
    assert all(
        1 <= number <= len(sources)
        for number in source_numbers
    ), "Citation references must point to existing sources."

    for citation in citations:
        source = sources[citation["source_number"] - 1]

        assert citation["movie_title"] == source["movie_title"]
        assert citation["start_time"] == source["start_time"]
        assert citation["end_time"] == source["end_time"]
        assert citation["chunk_id"] == source["chunk_id"]

    assert result["invalid_source_numbers"] == []


answerer = RAGAnswerer()


TEST_CASES = [
    {
        "name": "art and creation",
        "question": "What does the artist say about art and creation?",
        "movie_title": "A Bucket Of Blood 1959",
    },
    {
        "name": "dialogue about sculpture",
        "question": "What is said about sculpture?",
        "movie_title": "A Bucket Of Blood 1959",
    },
    {
        "name": "artist discussion",
        "question": "What does the dialogue say about being an artist?",
        "movie_title": "A Bucket Of Blood 1959",
    },
]


print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)

for test_case in TEST_CASES:

    print()
    print("-" * 70)
    print(f"TEST: {test_case['name']}")
    print("-" * 70)

    result = answerer.answer(
        question=test_case["question"],
        movie_title=test_case["movie_title"],
        n_results=5,
    )

    print(f"Question: {test_case['question']}")
    print()
    print("Answer:")
    print(result["answer"])

    print()
    print("Citations:")

    for citation in result["citations"]:
        print(
            f"[{citation['source_number']}] "
            f"{citation['movie_title']} — "
            f"{citation['start_time']} → "
            f"{citation['end_time']}"
        )

    validate_result(result)

    print()
    print("Validation: PASSED")


print()
print("=" * 70)
print("ALL RAG EVALUATION TESTS PASSED")
print("=" * 70)