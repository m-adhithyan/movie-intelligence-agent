from pathlib import Path

import pysrt


def find_violations(filename: str):
    filepath = Path("subtitles") / filename
    subtitles = pysrt.open(filepath, encoding="utf-8")

    print("=" * 80)
    print(filename)
    print("=" * 80)

    previous_start = -1.0

    for position, subtitle in enumerate(subtitles, start=1):
        start = (
            subtitle.start.hours * 3600
            + subtitle.start.minutes * 60
            + subtitle.start.seconds
            + subtitle.start.milliseconds / 1000
        )

        if start < previous_start - 0.01:
            print(
                f"VIOLATION at physical position {position}"
            )
            print(
                f"  SRT index: {subtitle.index}"
            )
            print(
                f"  Previous start: {previous_start:.3f}"
            )
            print(
                f"  Current start:  {start:.3f}"
            )
            print(
                f"  Current timing: {subtitle.start} --> {subtitle.end}"
            )
            print(
                f"  Text: {subtitle.text.replace(chr(10), ' ')}"
            )
            print()

        previous_start = start

    print("Finished.")
    print()


find_violations("gulliver's-travels-1939-en.srt")
find_violations("the-devil-bat-1940-en.srt")