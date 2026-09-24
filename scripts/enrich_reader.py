from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.enrichment.openlibrary import OpenLibraryClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

READER_FILES = {
    "sarah": RAW_DIR / "sarah_books.csv",
    "shannon": RAW_DIR / "shannon_books.csv",
      "you": RAW_DIR / "goodreads_books.csv",
}

OUTPUT_FILES = {
    "sarah": RAW_DIR / "sarah_books_enriched.csv",
    "shannon": RAW_DIR / "shannon_books_enriched.csv",
    "you": RAW_DIR / "you_books_enriched.csv",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich a reader's Goodreads books with Open Library metadata."
    )

    parser.add_argument(
        "--reader",
        choices=READER_FILES.keys(),
        required=True,
        help="Reader to enrich.",
    )

    return parser.parse_args()


def enrich_reader(reader: str) -> None:
    input_file = READER_FILES[reader]
    output_file = OUTPUT_FILES[reader]

    print("Open Library Reader Enrichment")
    print("=" * 70)
    print(f"Reader: {reader}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    books = pd.read_csv(input_file)

    print(f"Books loaded: {len(books)}")
    print()

    client = OpenLibraryClient()

    results = []

    for index, row in books.iterrows():
        title = str(row.get("title", "") or "").strip()
        author = str(row.get("author", "") or "").strip()
        isbn = str(row.get("isbn", "") or "").strip()

        print(
            f"[{index + 1}/{len(books)}] "
            f"{title} — {author}"
        )

        enrichment = client.match_book(
            title=title,
            author=author,
            isbn=isbn,
        )

        result = asdict(enrichment)

        result["source_book_id"] = row.get("source_book_id")

        results.append(result)

    enriched = pd.DataFrame(results)

    # Preserve the original Goodreads data and append Open Library fields.
    enriched = books.merge(
        enriched,
        on="source_book_id",
        how="left",
        suffixes=("", "_openlibrary"),
    )

    # Convert list/dict fields to JSON strings so the CSV remains readable
    # and can be parsed safely later.
    list_columns = [
        "authors",
        "subjects",
        "subject_people",
        "subject_places",
        "subject_times",
    ]

    dict_columns = [
        "raw_work",
        "raw_edition",
    ]

    for column in list_columns:
        if column in enriched.columns:
            enriched[column] = enriched[column].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
                if isinstance(value, list)
                else value
            )

    for column in dict_columns:
        if column in enriched.columns:
            enriched[column] = enriched[column].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
                if isinstance(value, dict)
                else value
            )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    print(f"Reader: {reader}")
    print(f"Books: {len(enriched)}")
    print(f"Saved: {output_file}")

    if "status" in enriched.columns:
        print()
        print("Open Library status:")
        print(enriched["status"].value_counts(dropna=False))

    if "matched_by" in enriched.columns:
        print()
        print("Matched by:")
        print(enriched["matched_by"].value_counts(dropna=False))


def main() -> None:
    args = parse_arguments()
    enrich_reader(args.reader)


if __name__ == "__main__":
    main()