from pathlib import Path

from app.rag.parser import parse_srt
from app.rag.chunker import create_chunks
from app.rag.vector_store import VectorStore


# --------------------------------------------------
# Parse subtitles
# --------------------------------------------------

subtitle_file = next(
    Path("subtitles").glob("*.srt")
)

entries = parse_srt(subtitle_file)

chunks = create_chunks(
    entries,
    window_seconds=30,
    overlap_seconds=5,
)

print(f"Movie: {entries[0].movie_title}")
print(f"Subtitle entries: {len(entries)}")
print(f"Chunks: {len(chunks)}")


# --------------------------------------------------
# Create vector store
# --------------------------------------------------

store = VectorStore()

print("\nAdding chunks to ChromaDB...")

store.add_chunks(chunks)

print(
    f"Chunks currently in ChromaDB: "
    f"{store.count()}"
)


# --------------------------------------------------
# Test semantic search
# --------------------------------------------------

query = "What does the artist say about art?"

print(f"\nQuery: {query}")

results = store.search(
    query,
    n_results=3,
)


print("\nTop results:\n")

for i in range(3):

    document = results["documents"][0][i]
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print("=" * 70)

    print(
        f"Movie: {metadata['movie_title']}"
    )

    print(
        f"Time: "
        f"{metadata['start_time']} --> "
        f"{metadata['end_time']}"
    )

    print(
        f"Distance: {distance:.4f}"
    )

    print(
        f"Text:\n{document[:500]}"
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert store.count() >= len(chunks), (
    "Not all chunks were stored."
)

assert "documents" in results
assert "metadatas" in results
assert "distances" in results

assert len(results["documents"][0]) == 3

for metadata in results["metadatas"][0]:

    assert metadata["movie_title"]
    assert metadata["start_time"]
    assert metadata["end_time"]

print("\n" + "=" * 70)
print("ALL VECTOR STORE TESTS PASSED")
print("=" * 70)