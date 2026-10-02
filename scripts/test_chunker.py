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
print(f"Subtitle entries: {len(entries)}")
print(f"Chunks created: {len(chunks)}")


print("\nFirst 3 chunks:\n")

for chunk in chunks[:3]:
    print("=" * 70)

    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Movie: {chunk.movie_title}")

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

# Every chunk must contain text
assert all(
    chunk.text.strip()
    for chunk in chunks
), "Empty chunk found."

# Every chunk must have valid timestamps
assert all(
    chunk.start_seconds <= chunk.end_seconds
    for chunk in chunks
), "Invalid chunk timestamps."

# Movie metadata must be preserved
assert all(
    chunk.movie_title == entries[0].movie_title
    for chunk in chunks
), "Movie title was lost."

# Chunk IDs must be unique
chunk_ids = [chunk.chunk_id for chunk in chunks]

assert len(chunk_ids) == len(set(chunk_ids)), (
    "Duplicate chunk IDs found."
)

# Make sure no subtitle entry disappeared
original_indices = {
    entry.index
    for entry in entries
}

chunk_indices = {
    index
    for chunk in chunks
    for index in chunk.subtitle_indices
}

missing_indices = original_indices - chunk_indices

assert not missing_indices, (
    f"Subtitle entries were lost: {missing_indices}"
)

print("\n" + "=" * 70)
print("ALL CHUNKING TESTS PASSED")
print("=" * 70)