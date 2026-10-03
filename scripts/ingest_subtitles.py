from pathlib import Path
import argparse

from app.rag.parser import parse_srt
from app.rag.chunker import create_chunks
from app.rag.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBTITLE_DIR = PROJECT_ROOT / "subtitles"


def ingest_subtitle_file(
    subtitle_file: Path,
    vector_store: VectorStore,
) -> tuple[str, int, int]:
    """
    Parse, chunk, and store one subtitle file.

    Returns:
        movie_title, subtitle_count, chunk_count
    """

    entries = parse_srt(subtitle_file)

    if not entries:
        raise ValueError(
            f"No valid subtitle entries found in {subtitle_file.name}"
        )

    chunks = create_chunks(entries)

    if not chunks:
        raise ValueError(
            f"No chunks created for {subtitle_file.name}"
        )

    vector_store.add_chunks(chunks)

    movie_title = entries[0].movie_title

    return (
        movie_title,
        len(entries),
        len(chunks),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest movie subtitle files into ChromaDB."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of subtitle files to ingest.",
    )

    args = parser.parse_args()

    if not SUBTITLE_DIR.exists():
        raise FileNotFoundError(
            f"Subtitle directory not found: {SUBTITLE_DIR}"
        )

    subtitle_files = sorted(
        SUBTITLE_DIR.glob("*.srt")
    )

    if not subtitle_files:
        raise FileNotFoundError(
            f"No .srt files found in {SUBTITLE_DIR}"
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        subtitle_files = subtitle_files[:args.limit]

    print("=" * 70)
    print("MOVIE SUBTITLE INGESTION")
    print("=" * 70)

    print(f"Subtitle directory: {SUBTITLE_DIR}")
    print(f"Files to process: {len(subtitle_files)}")
    print()

    vector_store = VectorStore()

    successful = 0
    failed = 0
    total_entries = 0
    total_chunks = 0

    for index, subtitle_file in enumerate(
        subtitle_files,
        start=1,
    ):
        print(
            f"[{index}/{len(subtitle_files)}] "
            f"{subtitle_file.name}"
        )

        try:
            (
                movie_title,
                entry_count,
                chunk_count,
            ) = ingest_subtitle_file(
                subtitle_file,
                vector_store,
            )

            successful += 1
            total_entries += entry_count
            total_chunks += chunk_count

            print(
                f"  Movie: {movie_title}"
            )
            print(
                f"  Subtitles: {entry_count}"
            )
            print(
                f"  Chunks: {chunk_count}"
            )
            print("  Status: SUCCESS")
            print()

        except Exception as exc:
            failed += 1

            print(
                f"  Status: FAILED"
            )
            print(
                f"  Error: {exc}"
            )
            print()

    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(f"Successful movies: {successful}")
    print(f"Failed movies:     {failed}")
    print(f"Subtitle entries:  {total_entries}")
    print(f"Chunks added:      {total_chunks}")
    print(
        f"ChromaDB chunks:   {vector_store.count()}"
    )

    if failed:
        print()
        print(
            "WARNING: Some subtitle files failed to ingest."
        )
    else:
        print()
        print("INGESTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()