from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.openlibrary_identity import (  # noqa: E402
    extract_edition_id,
    extract_work_id,
)


DATA_DIR = PROJECT_ROOT / "data" / "raw"

READER_FILES = {
    "you": DATA_DIR / "you_books_enriched.csv",
    "sarah": DATA_DIR / "sarah_books_enriched.csv",
    "shannon": DATA_DIR / "shannon_books_enriched.csv",
}


def parse_json(value):
    """Parse a JSON object stored as a CSV string."""
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_identity_from_row(row: pd.Series) -> tuple[str | None, str | None]:
    """Extract Work and Edition IDs from an enriched row."""

    raw_work = parse_json(row.get("raw_work"))
    raw_edition = parse_json(row.get("raw_edition"))

    work_id = extract_work_id(
        work_key=row.get("work_id"),
        edition_work_key=row.get("edition_work_id"),
        raw_work=raw_work,
    )

    edition_id = extract_edition_id(
        edition_key=row.get("edition_id"),
        raw_edition=raw_edition,
    )

    return work_id, edition_id


def main() -> None:
    print("=" * 70)
    print("OPEN LIBRARY IDENTITY DIAGNOSTIC")
    print("=" * 70)

    all_rows = []

    for reader, path in READER_FILES.items():
        if not path.exists():
            print(f"\nMISSING: {path}")
            continue

        df = pd.read_csv(path)

        print(f"\n{reader.upper()}")
        print("-" * 70)
        print(f"Rows: {len(df):,}")

        identities = []

        for _, row in df.iterrows():
            work_id, edition_id = extract_identity_from_row(row)

            identities.append(
                {
                    "reader": reader,
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "work_id": work_id,
                    "edition_id": edition_id,
                }
            )

        identity_df = pd.DataFrame(identities)

        work_count = identity_df["work_id"].notna().sum()
        edition_count = identity_df["edition_id"].notna().sum()

        print(f"Work IDs found:    {work_count:,}")
        print(f"Edition IDs found: {edition_count:,}")

        if work_count:
            print(
                f"Unique Work IDs:   "
                f"{identity_df['work_id'].nunique():,}"
            )

        all_rows.append(identity_df)

    if not all_rows:
        print("\nNo enriched files were found.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    print("\n" + "=" * 70)
    print("POOLED RESULTS")
    print("=" * 70)

    print(f"Total reader/book rows: {len(combined):,}")

    print(
        "Rows with Work ID:      "
        f"{combined['work_id'].notna().sum():,}"
    )

    print(
        "Unique Work IDs:        "
        f"{combined['work_id'].dropna().nunique():,}"
    )

    duplicate_work_ids = (
        combined.dropna(subset=["work_id"])
        .groupby("work_id")
        .agg(
            readers=("reader", "nunique"),
            books=("title", "count"),
            title=("title", "first"),
        )
        .query("readers > 1")
        .sort_values(
            ["readers", "books"],
            ascending=False,
        )
    )

    print(
        f"\nWorks shared by multiple readers: "
        f"{len(duplicate_work_ids):,}"
    )

    if not duplicate_work_ids.empty:
        print("\nTop shared works:")
        print(duplicate_work_ids.head(25).to_string())

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "openlibrary_identity_diagnostic.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False)

    print(f"\nSaved diagnostic to:")
    print(output_path)


if __name__ == "__main__":
    main()