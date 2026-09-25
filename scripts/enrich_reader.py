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
    "you": RAW_DIR / "goodreads_books.csv",
    "sarah": RAW_DIR / "sarah_books.csv",
    "shannon": RAW_DIR / "shannon_books.csv",
}

OUTPUT_FILES = {
    "you": RAW_DIR / "you_books_enriched.csv",
    "sarah": RAW_DIR / "sarah_books_enriched.csv",
    "shannon": RAW_DIR / "shannon_books_enriched.csv",
}

LIST_COLUMNS = [
    "authors",
    "subjects",
    "subject_people",
    "subject_places",
    "subject_times",
]

DICT_COLUMNS = [
    "raw_work",
    "raw_edition",
]

CHECKPOINT_EVERY = 100


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich a reader's Goodreads books with Open Library metadata."
        )
    )

    parser.add_argument(
        "--reader",
        choices=READER_FILES.keys(),
        required=True,
        help="Reader to enrich.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-enrich every book instead of resuming from "
            "a previous enriched output."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N books.",
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="Delay between uncached Open Library requests.",
    )

    return parser.parse_args()


def _parse_json_value(value):
    if pd.isna(value):
        return value

    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()

    if not text:
        return value

    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def _prepare_enrichment_for_csv(
    enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    output = enrichment_df.copy()

    for column in LIST_COLUMNS:
        if column in output.columns:
            output[column] = output[column].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
                if isinstance(value, list)
                else value
            )

    for column in DICT_COLUMNS:
        if column in output.columns:
            output[column] = output[column].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
                if isinstance(value, dict)
                else value
            )

    return output


def _save_enriched(
    books: pd.DataFrame,
    results_by_id: dict[str, dict],
    output_file: Path,
) -> pd.DataFrame:
    results = list(results_by_id.values())

    if results:
        enrichment_df = pd.DataFrame(results)

        enriched = books.merge(
            enrichment_df,
            on="source_book_id",
            how="left",
            suffixes=("", "_openlibrary"),
        )
    else:
        enriched = books.copy()

    enriched = _prepare_enrichment_for_csv(enriched)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        output_file,
        index=False,
    )

    return enriched


def _load_existing_results(
    output_file: Path,
) -> dict[str, dict]:
    if not output_file.exists():
        return {}

    existing = pd.read_csv(output_file)

    if "source_book_id" not in existing.columns:
        return {}

    if "status" not in existing.columns:
        return {}

    results = {}

    for _, row in existing.iterrows():
        source_book_id = str(row.get("source_book_id", "")).strip()
        status = str(row.get("status", "")).strip()

        if not source_book_id:
            continue

        # Only reuse completed matches. Unmatched records are retried
        # because the current search cache may contain new information.
        if status != "matched":
            continue

        record = {}

        for column in [
            "source",
            "openlibrary_work_id",
            "openlibrary_edition_id",
            "matched_by",
            "match_score",
            "title_openlibrary",
            "title",
            "authors",
            "publication_year",
            "subjects",
            "subject_people",
            "subject_places",
            "subject_times",
            "isbn_10",
            "isbn_13",
            "cover_url",
            "raw_work",
            "raw_edition",
            "status",
            "source_book_id",
        ]:
            if column in existing.columns:
                record[column] = _parse_json_value(
                    row[column]
                )

        # The merge normally creates title_openlibrary when the
        # original Goodreads title is also named "title".
        # The enrichment dataframe itself needs the enrichment title
        # under the original column name.
        if "title_openlibrary" in record:
            record["title"] = record.pop("title_openlibrary")

        record["source_book_id"] = source_book_id

        results[source_book_id] = record

    return results


def enrich_reader(
    reader: str,
    force: bool = False,
    progress_every: int = CHECKPOINT_EVERY,
    request_delay: float = 0.1,
) -> Path:
    input_file = READER_FILES[reader]
    output_file = OUTPUT_FILES[reader]

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found for {reader}: {input_file}"
        )

    books = pd.read_csv(input_file)

    required = {
        "source_book_id",
        "title",
        "author",
        "isbn",
    }

    missing = required - set(books.columns)

    if missing:
        raise ValueError(
            f"{reader} is missing required columns: {sorted(missing)}"
        )

    print()
    print("=" * 70)
    print(f"OPEN LIBRARY ENRICHMENT — {reader.upper()}")
    print("=" * 70)
    print(f"Books: {len(books):,}")
    print(f"Output: {output_file}")

    results_by_id = {}

    if not force:
        results_by_id = _load_existing_results(output_file)

        if results_by_id:
            print(
                f"Resuming: {len(results_by_id):,} completed matches "
                "found in existing output."
            )

    remaining = 0

    client = OpenLibraryClient(
        request_delay=request_delay
    )

    total = len(books)

    for position, (_, row) in enumerate(
        books.iterrows(),
        start=1,
    ):
        source_book_id = str(
            row.get("source_book_id", "")
        ).strip()

        if (
            not force
            and source_book_id in results_by_id
        ):
            continue

        title = str(
            row.get("title", "") or ""
        ).strip()

        author = str(
            row.get("author", "") or ""
        ).strip()

        isbn = str(
            row.get("isbn", "") or ""
        ).strip()

        enrichment = client.match_book(
            title=title,
            author=author,
            isbn=isbn,
        )

        result = asdict(enrichment)
        result["source_book_id"] = source_book_id

        results_by_id[source_book_id] = result
        remaining += 1

        if (
            position % progress_every == 0
            or position == total
        ):
            matched = sum(
                1
                for value in results_by_id.values()
                if value.get("status") == "matched"
            )

            unmatched = sum(
                1
                for value in results_by_id.values()
                if value.get("status") == "unmatched"
            )

            print(
                f"  Progress {position:,}/{total:,} "
                f"| matched={matched:,} "
                f"| unmatched={unmatched:,}"
            )

            _save_enriched(
                books=books,
                results_by_id=results_by_id,
                output_file=output_file,
            )

    enriched = _save_enriched(
        books=books,
        results_by_id=results_by_id,
        output_file=output_file,
    )

    print()
    print("ENRICHMENT COMPLETE")
    print(f"Reader: {reader}")
    print(f"Books: {len(enriched):,}")
    print(f"Newly processed this run: {remaining:,}")
    print(f"Saved: {output_file}")

    if "status" in enriched.columns:
        print()
        print("Status:")
        print(
            enriched["status"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "matched_by" in enriched.columns:
        print()
        print("Matched by:")
        print(
            enriched["matched_by"]
            .value_counts(dropna=False)
            .to_string()
        )

    return output_file


def main() -> None:
    args = parse_arguments()

    enrich_reader(
        reader=args.reader,
        force=args.force,
        progress_every=max(1, args.progress_every),
        request_delay=max(0.0, args.request_delay),
    )


if __name__ == "__main__":
    main()
