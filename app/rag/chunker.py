from dataclasses import dataclass
import re

from .parser import SubtitleEntry


@dataclass
class SubtitleChunk:
    chunk_id: str
    movie_title: str
    movie_id: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    text: str
    subtitle_indices: list[int]


def slugify_movie_title(movie_title: str) -> str:
    """
    Create a stable identifier from a movie title.
    """

    slug = movie_title.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)

    return slug.strip("_")


def create_chunks(
    entries: list[SubtitleEntry],
    window_seconds: float = 30.0,
    overlap_seconds: float = 5.0,
) -> list[SubtitleChunk]:
    """
    Create overlapping, time-based subtitle chunks.

    Each chunk preserves:
    - movie title
    - movie ID
    - exact subtitle timestamps
    - subtitle indices
    - combined dialogue text
    """

    if not entries:
        return []

    if window_seconds <= 0:
        raise ValueError(
            "window_seconds must be greater than 0"
        )

    if overlap_seconds < 0:
        raise ValueError(
            "overlap_seconds cannot be negative"
        )

    if overlap_seconds >= window_seconds:
        raise ValueError(
            "overlap_seconds must be smaller than window_seconds"
        )

    movie_title = entries[0].movie_title
    movie_id = entries[0].movie_id

    chunks: list[SubtitleChunk] = []

    start_index = 0
    chunk_number = 1

    while start_index < len(entries):

        chunk_start = entries[start_index].start_seconds
        window_end = chunk_start + window_seconds

        chunk_entries: list[SubtitleEntry] = []

        index = start_index

        while index < len(entries):

            entry = entries[index]

            if (
                entry.start_seconds < window_end
                or not chunk_entries
            ):
                chunk_entries.append(entry)
                index += 1
            else:
                break

        first_entry = chunk_entries[0]
        last_entry = chunk_entries[-1]

        text = " ".join(
            entry.text
            for entry in chunk_entries
        )

        chunk_id = (
            f"{movie_id}_{chunk_number:04d}"
        )

        chunks.append(
            SubtitleChunk(
                chunk_id=chunk_id,
                movie_title=movie_title,
                movie_id=movie_id,
                start_time=first_entry.start_time,
                end_time=last_entry.end_time,
                start_seconds=first_entry.start_seconds,
                end_seconds=last_entry.end_seconds,
                text=text,
                subtitle_indices=[
                    entry.index
                    for entry in chunk_entries
                ],
            )
        )

        # Advance by window - overlap.
        next_start_time = (
            chunk_start
            + window_seconds
            - overlap_seconds
        )

        next_index = start_index + 1

        while (
            next_index < len(entries)
            and entries[next_index].start_seconds
            < next_start_time
        ):
            next_index += 1

        start_index = next_index
        chunk_number += 1

    return chunks