from pathlib import Path

from app.rag.parser import parse_srt


def main():
    subtitle_file = Path("subtitles/a-bucket-of-blood-1959-en.srt")

    entries = parse_srt(subtitle_file)

    print(f"Movie: {entries[0].movie_title}")
    print(f"Movie ID: {entries[0].movie_id}")
    print(f"Subtitle entries: {len(entries)}")

    print("\nFirst 5 entries:")

    for entry in entries[:5]:
        print(
            f"[{entry.start_time} --> {entry.end_time}] "
            f"{entry.text}"
        )

    # Validation checks
    assert entries, "Parser returned no subtitle entries."

    assert entries[0].movie_title == "A Bucket Of Blood 1959"

    assert entries[0].movie_id == "a_bucket_of_blood_1959"

    assert entries[0].start_seconds >= 0

    assert entries[0].end_seconds >= entries[0].start_seconds

    # Verify chronological ordering.
    for previous, current in zip(entries, entries[1:]):
        assert (
            current.start_seconds >= previous.start_seconds
        ), (
            f"Subtitle ordering error: "
            f"{previous.index} -> {current.index}"
        )

    # Verify every retained subtitle has dialogue.
    for entry in entries:
        assert entry.text.strip(), (
            f"Empty subtitle retained: {entry.index}"
        )

    print("\nParser validation: PASSED")


if __name__ == "__main__":
    main()