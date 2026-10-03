from dataclasses import dataclass
from pathlib import Path
import re

import pysrt


# Tiny backward overlaps caused by rounding/precision differences.
OVERLAP_TOLERANCE_SECONDS = 0.01

# Backward jumps up to this size are treated as local ordering quirks
# (e.g. song lyrics listed after overlapping dialogue) and fixed by sorting.
# Larger jumps suggest a corrupt file (bad merge, timing reset, typo'd
# timestamp), where sorting would hide the problem, so the file is rejected.
MAX_BACKWARD_JUMP_SECONDS = 5.0


@dataclass
class SubtitleEntry:
    index: int            # sequential position AFTER chronological sorting (0-based)
    original_index: int   # the index as written in the SRT file
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    text: str
    movie_title: str
    movie_id: str


def timestamp_to_seconds(timestamp) -> float:
    """
    Convert a pysrt SubRipTime object into seconds
    while preserving millisecond precision.
    """
    return (
        timestamp.hours * 3600
        + timestamp.minutes * 60
        + timestamp.seconds
        + timestamp.milliseconds / 1000
    )


def clean_text(text: str) -> str:
    """
    Clean subtitle formatting while preserving dialogue.
    """

    # Remove HTML/XML subtitle tags such as <i>, <b>, etc.
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize line breaks and repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def movie_title_from_filename(filepath: Path) -> str:
    """
    Convert an SRT filename into a readable movie title.

    Example:
        a-bucket-of-blood-1959-en.srt
        -> A Bucket Of Blood 1959
    """

    name = filepath.stem

    # Remove common language suffix.
    name = re.sub(r"-en$", "", name, flags=re.IGNORECASE)

    # Convert separators to spaces.
    name = re.sub(r"[-_]+", " ", name)

    # Normalize whitespace.
    name = re.sub(r"\s+", " ", name)

    return name.strip().title()


def movie_id_from_title(movie_title: str) -> str:
    """
    Create a stable identifier for a movie title.
    """

    movie_id = movie_title.lower()
    movie_id = re.sub(r"[^a-z0-9]+", "_", movie_id)

    return movie_id.strip("_")


def _open_srt(filepath: Path):
    """
    Open an SRT file using UTF-8 first and fall back to
    common legacy encodings when necessary.
    """

    try:
        return pysrt.open(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pysrt.open(filepath, encoding="cp1252")
        except UnicodeDecodeError:
            return pysrt.open(filepath, encoding="latin-1")


def parse_srt(filepath: str | Path) -> list[SubtitleEntry]:
    """
    Parse an SRT file into structured subtitle entries.

    Preserves:
    - original subtitle index (original_index)
    - exact timestamp strings
    - timestamp values in seconds
    - cleaned dialogue
    - movie title
    - stable movie ID

    Small out-of-order entries are repaired by sorting chronologically.
    Large backward jumps raise a ValueError.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Subtitle file not found: {filepath}"
        )

    if filepath.suffix.lower() != ".srt":
        raise ValueError(
            f"Expected an .srt file, got: {filepath.suffix}"
        )

    movie_title = movie_title_from_filename(filepath)
    movie_id = movie_id_from_title(movie_title)

    subtitles = _open_srt(filepath)

    raw_entries: list[SubtitleEntry] = []

    previous_start_seconds = None
    max_backward_jump = 0.0
    max_backward_jump_entry = None

    for subtitle in subtitles:
        text = clean_text(subtitle.text)

        # Ignore empty subtitle blocks.
        if not text:
            continue

        start_seconds = timestamp_to_seconds(subtitle.start)
        end_seconds = timestamp_to_seconds(subtitle.end)

        # Invalid subtitle timing should not silently enter
        # the retrieval database.
        if end_seconds < start_seconds:
            raise ValueError(
                f"Invalid subtitle timing in {filepath.name}: "
                f"entry {subtitle.index} ends before it starts."
            )

        # Track how far backward the file jumps. Tiny overlaps are ignored;
        # the size of real jumps is evaluated after the loop.
        if (
            previous_start_seconds is not None
            and start_seconds < previous_start_seconds - OVERLAP_TOLERANCE_SECONDS
        ):
            jump = previous_start_seconds - start_seconds
            if jump > max_backward_jump:
                max_backward_jump = jump
                max_backward_jump_entry = subtitle.index

        previous_start_seconds = start_seconds

        raw_entries.append(
            SubtitleEntry(
                index=-1,  # assigned after sorting
                original_index=subtitle.index,
                start_time=str(subtitle.start),
                end_time=str(subtitle.end),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
                movie_title=movie_title,
                movie_id=movie_id,
            )
        )

    # Large backward jumps point to a corrupt file rather than a
    # harmless ordering quirk, so reject instead of sorting.
    if max_backward_jump > MAX_BACKWARD_JUMP_SECONDS:
        raise ValueError(
            f"Non-chronological subtitle timing in {filepath.name}: "
            f"entry {max_backward_jump_entry} jumps back "
            f"{max_backward_jump:.1f}s "
            f"(limit {MAX_BACKWARD_JUMP_SECONDS:.1f}s)."
        )

    raw_entries.sort(
        key=lambda e: (e.start_seconds, e.end_seconds, e.original_index)
    )

    # Re-number so `index` always reflects chronological position.
    for position, entry in enumerate(raw_entries):
        entry.index = position

    return raw_entries