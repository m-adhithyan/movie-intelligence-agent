from dataclasses import dataclass
from pathlib import Path
import pysrt
import re


@dataclass
class SubtitleEntry:
    index: int
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    text: str
    movie_title: str


def timestamp_to_seconds(timestamp) -> float:
    """
    Convert a pysrt SubRipTime object into seconds.
    """
    return (
        timestamp.hours * 3600
        + timestamp.minutes * 60
        + timestamp.seconds
        + timestamp.milliseconds / 1000
    )


def clean_text(text: str) -> str:
    """
    Clean subtitle text while preserving the actual dialogue.
    """

    # Remove HTML/XML-style tags such as <i>...</i>
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def movie_title_from_filename(filepath: Path) -> str:
    """
    Convert a subtitle filename into a readable movie title.

    Example:
        a-bucket-of-blood-1959-en.srt
        ->
        A Bucket Of Blood 1959
    """

    name = filepath.stem

    # Remove language suffix
    name = re.sub(r"-en$", "", name, flags=re.IGNORECASE)

    # Replace separators
    name = name.replace("-", " ")

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name)

    return name.strip().title()


def parse_srt(filepath: str | Path) -> list[SubtitleEntry]:
    """
    Parse an SRT file into structured subtitle entries.
    """

    filepath = Path(filepath)

    movie_title = movie_title_from_filename(filepath)

    subtitles = pysrt.open(filepath, encoding="utf-8")

    entries = []

    for subtitle in subtitles:

        text = clean_text(subtitle.text)

        # Ignore completely empty subtitles
        if not text:
            continue

        entry = SubtitleEntry(
            index=subtitle.index,
            start_time=str(subtitle.start),
            end_time=str(subtitle.end),
            start_seconds=timestamp_to_seconds(subtitle.start),
            end_seconds=timestamp_to_seconds(subtitle.end),
            text=text,
            movie_title=movie_title,
        )

        entries.append(entry)

    return entries