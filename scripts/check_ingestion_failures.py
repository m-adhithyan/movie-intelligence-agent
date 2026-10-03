from pathlib import Path

from app.rag.parser import parse_srt


SUBTITLE_DIR = Path("subtitles")


def main():
    files = sorted(SUBTITLE_DIR.glob("*.srt"))[:100]

    failed = []

    for index, filepath in enumerate(files, start=1):
        try:
            parse_srt(filepath)
        except Exception as exc:
            failed.append(
                (
                    index,
                    filepath.name,
                    str(exc),
                )
            )

    print("=" * 70)
    print("INGESTION FAILURE CHECK")
    print("=" * 70)

    print(f"Files checked: {len(files)}")
    print(f"Failed files: {len(failed)}")
    print()

    for index, filename, error in failed:
        print(f"[{index}] {filename}")
        print(f"    Error: {error}")
        print()

    if not failed:
        print("All files passed parser validation.")


if __name__ == "__main__":
    main()