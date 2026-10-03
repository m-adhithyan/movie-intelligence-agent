from dataclasses import dataclass
from pathlib import Path
import re

import pysrt


@dataclass
class SubtitleEntry:
    index: int
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
    - subtitle index
    - exact timestamp strings
    - timestamp values in seconds
    - cleaned dialogue
    - movie title
    - stable movie ID
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

    entries: list[SubtitleEntry] = []

    previous_start_seconds = -1.0

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

        # Subtitle files should normally be chronological.
        if start_seconds < previous_start_seconds:
            raise ValueError(
                f"Non-chronological subtitle timing in "
                f"{filepath.name}: entry {subtitle.index}."
            )

        previous_start_seconds = start_seconds

        entries.append(
            SubtitleEntry(
                index=subtitle.index,
                start_time=str(subtitle.start),
                end_time=str(subtitle.end),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
                movie_title=movie_title,
                movie_id=movie_id,
            )
        )

    return entries