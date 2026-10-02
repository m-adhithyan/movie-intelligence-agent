from pathlib import Path

from app.rag.parser import parse_srt


subtitle_file = next(Path("subtitles").glob("*.srt"))

entries = parse_srt(subtitle_file)

print(f"Movie: {entries[0].movie_title}")
print(f"Subtitle entries: {len(entries)}")

print("\nFirst 5 entries:\n")

for entry in entries[:5]:
    print(
        f"[{entry.start_time} --> {entry.end_time}] "
        f"{entry.text}"
    )