from pathlib import Path

from app.rag.parser import parse_srt
from app.rag.chunker import create_chunks


subtitle_file = next(Path("subtitles").glob("*.srt"))

entries = parse_srt(subtitle_file)

chunks = create_chunks(
    entries,
    window_seconds=30,
    overlap_seconds=5,
)


print(f"Movie: {entries[0].movie_title}")
print(f"Movie ID: {entries[0].movie_id}")
print(f"Subtitle entries: {len(entries)}")
print(f"Chunks created: {len(chunks)}")


print("\nFirst 3 chunks:\n")

for chunk in chunks[:3]:
    print("=" * 70)

    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Movie: {chunk.movie_title}")
    print(f"Movie ID: {chunk.movie_id}")

    print(
        f"Time: "
        f"{chunk.start_time} --> {chunk.end_time}"
    )

    print(
        f"Seconds: "
        f"{chunk.start_seconds:.3f} --> "
        f"{chunk.end_seconds:.3f}"
    )

    print(
        f"Subtitle IDs: "
        f"{chunk.subtitle_indices[0]} --> "
        f"{chunk.subtitle_indices[-1]}"
    )

    print(f"Text:\n{chunk.text[:500]}")


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert chunks, "No chunks were created."

# Every chunk must contain text.
assert all(
    chunk.text.strip()
    for chunk in chunks
), "Empty chunk found."


# Every chunk must have valid timestamps.
assert all(
    chunk.start_seconds <= chunk.end_seconds
    for chunk in chunks
), "Invalid chunk timestamps."


# Movie metadata must be preserved.
assert all(
    chunk.movie_title == entries[0].movie_title
    for chunk in chunks
), "Movie title was lost."


assert all(
    chunk.movie_id == entries[0].movie_id
    for chunk in chunks
), "Movie ID was lost."


# Chunk IDs must be unique.
chunk_ids = [
    chunk.chunk_id
    for chunk in chunks
]

assert len(chunk_ids) == len(set(chunk_ids)), (
    "Duplicate chunk IDs found."
)


# Chunk IDs must contain the movie ID.
assert all(
    chunk.chunk_id.startswith(
        f"{entries[0].movie_id}_"
    )
    for chunk in chunks
), "Chunk ID does not contain movie ID."


# Chunk numbering must be deterministic.
for number, chunk in enumerate(chunks, start=1):

    expected_id = (
        f"{entries[0].movie_id}_{number:04d}"
    )

    assert chunk.chunk_id == expected_id, (
        f"Unexpected chunk ID: "
        f"{chunk.chunk_id}"
    )


# Make sure no subtitle entry disappeared.
original_indices = {
    entry.index
    for entry in entries
}

chunk_indices = {
    index
    for chunk in chunks
    for index in chunk.subtitle_indices
}

missing_indices = (
    original_indices - chunk_indices
)

assert not missing_indices, (
    f"Subtitle entries were lost: "
    f"{missing_indices}"
)


# Every chunk must contain valid subtitle indices.
assert all(
    chunk.subtitle_indices
    for chunk in chunks
), "Chunk contains no subtitle indices."


# Subtitle indices inside each chunk must be ordered.
for chunk in chunks:

    assert chunk.subtitle_indices == sorted(
        chunk.subtitle_indices
    ), (
        f"Subtitle indices are not ordered "
        f"in {chunk.chunk_id}"
    )


print("\n" + "=" * 70)
print("ALL CHUNKING TESTS PASSED")
print("=" * 70)