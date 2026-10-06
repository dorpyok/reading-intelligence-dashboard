from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.reader_community import (
    build_reader_community_evidence_from_files,
)


READER_BOOK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reader_book_evidence.csv"
)

LEIDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "leiden"
    / "leiden_assignments.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
    / "reading_dna"
)

COMMUNITY_OUTPUT_PATH = (
    OUTPUT_DIR / "reader_community_evidence.csv"
)

UNASSIGNED_OUTPUT_PATH = (
    OUTPUT_DIR / "reader_unassigned_evidence.csv"
)


def main() -> None:
    print("=" * 72)
    print("BUILD READER COMMUNITY EVIDENCE")
    print("=" * 72)

    print(f"Reader/book evidence: {READER_BOOK_PATH}")
    print(f"Leiden assignments:   {LEIDEN_PATH}")
    print()

    community_evidence, unassigned_evidence = (
        build_reader_community_evidence_from_files(
            READER_BOOK_PATH,
            LEIDEN_PATH,
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    community_evidence.to_csv(
        COMMUNITY_OUTPUT_PATH,
        index=False,
    )

    unassigned_evidence.to_csv(
        UNASSIGNED_OUTPUT_PATH,
        index=False,
    )

    print(f"Community rows: {len(community_evidence):,}")
    print(f"Readers: {community_evidence['reader_id'].nunique():,}")
    print(
        "Communities represented: "
        f"{community_evidence['community_id'].nunique():,}"
    )
    print()

    print("Reader/community evidence:")
    print(community_evidence.to_string(index=False))

    print()
    print("Unassigned books:")
    print(unassigned_evidence.to_string(index=False))

    print()
    print(f"Saved: {COMMUNITY_OUTPUT_PATH}")
    print(f"Saved: {UNASSIGNED_OUTPUT_PATH}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()