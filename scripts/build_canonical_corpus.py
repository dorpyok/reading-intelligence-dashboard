from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROCESSED_DIR / "canonical"


READER_FILES = {
    "you": RAW_DIR / "you_books_enriched.csv",
    "sarah": RAW_DIR / "sarah_books_enriched.csv",
    "shannon": RAW_DIR / "shannon_books_enriched.csv",
}


# ---------------------------------------------------------------------------
# ID HELPERS
# ---------------------------------------------------------------------------


def create_id(*parts: Any) -> str:
    """
    Create a deterministic identifier from one or more values.

    This is used for reader-book relationship records.

    Canonical book IDs are created separately so that the identity
    hierarchy remains explicit.
    """

    normalized_parts = [
        str(part).strip()
        for part in parts
    ]

    value = "|".join(normalized_parts)

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """
    Normalize text for fallback identity matching.

    This is intentionally conservative. We are not attempting to
    determine whether two books are semantically identical here.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Normalize common punctuation to spaces.
    normalized = []

    for character in value:
        if character.isalnum():
            normalized.append(character)
        else:
            normalized.append(" ")

    value = "".join(normalized)

    return " ".join(value.split())


def normalize_isbn(value: Any) -> str:
    """
    Normalize an ISBN to digits, preserving a trailing X for ISBN-10.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    return "".join(
        character
        for character in value
        if character.isdigit() or character == "X"
    )


# ---------------------------------------------------------------------------
# OPEN LIBRARY ID NORMALIZATION
# ---------------------------------------------------------------------------


def normalize_openlibrary_id(value: Any) -> str:
    """
    Normalize Open Library Work/Edition IDs.

    Accepts values such as:

        OL123W
        /works/OL123W
        https://openlibrary.org/works/OL123W
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = value.rstrip("/")

    if "/works/" in value:
        value = value.split("/works/")[-1]

    if "/books/" in value:
        value = value.split("/books/")[-1]

    return value


# ---------------------------------------------------------------------------
# CANONICAL BOOK ID
# ---------------------------------------------------------------------------


def build_canonical_book_id(
    *,
    openlibrary_work_id: str = "",
    openlibrary_edition_id: str = "",
    isbn: str = "",
    title: str = "",
    author: str = "",
    source: str = "",
    source_book_id: str = "",
) -> tuple[str, str, str]:
    """
    Build a deterministic canonical book ID.

    Identity hierarchy:

        1. Open Library Work
        2. Open Library Edition
        3. ISBN
        4. normalized title + author
        5. source + source book ID

    Returns:

        canonical_book_id
        identity_method
        identity_confidence
    """

    work_id = normalize_openlibrary_id(
        openlibrary_work_id
    )

    edition_id = normalize_openlibrary_id(
        openlibrary_edition_id
    )

    isbn_value = normalize_isbn(isbn)

    normalized_title = normalize_text(title)
    normalized_author = normalize_text(author)

    if work_id:
        identity_key = f"openlibrary_work|{work_id}"

        method = "openlibrary_work"
        confidence = "high"

    elif edition_id:
        identity_key = f"openlibrary_edition|{edition_id}"

        method = "openlibrary_edition"
        confidence = "high"

    elif isbn_value:
        identity_key = f"isbn|{isbn_value}"

        method = "isbn"
        confidence = "high"

    elif normalized_title and normalized_author:
        identity_key = (
            f"title_author|"
            f"{normalized_title}|"
            f"{normalized_author}"
        )

        method = "title_author"
        confidence = "medium"

    else:
        identity_key = (
            f"source|"
            f"{source}|"
            f"{source_book_id}"
        )

        method = "source_fallback"
        confidence = "low"

    digest = hashlib.sha256(
        identity_key.encode("utf-8")
    ).hexdigest()[:16]

    canonical_book_id = f"book_{digest}"

    return (
        canonical_book_id,
        method,
        confidence,
    )


# ---------------------------------------------------------------------------
# VALUE HELPERS
# ---------------------------------------------------------------------------


def clean_value(value: Any) -> Any:
    """
    Convert pandas missing values to None while preserving real values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return value


def parse_json_list(value: Any) -> list[str]:
    """
    Parse list-valued fields stored as JSON strings in the enriched CSVs.
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)

    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, list):
        return []

    return [
        str(item)
        for item in parsed
        if str(item).strip()
    ]


# ---------------------------------------------------------------------------
# READING STATUS
# ---------------------------------------------------------------------------


def derive_reading_status(row: pd.Series) -> str:
    """
    Derive the canonical reading status from Goodreads evidence.

    This mirrors the established project precedence:

        did-not-finish
        currently-reading
        to-read
        date_read
        rating
        unknown

    date_added is deliberately NOT treated as evidence of reading.
    """

    shelves = str(
        row.get("shelves", "") or ""
    ).lower()

    if "did-not-finish" in shelves:
        return "did_not_finish"

    if (
        "currently-reading" in shelves
        or "currently_reading" in shelves
    ):
        return "currently_reading"

    if "to-read" in shelves:
        return "to_read"

    date_read = clean_value(
        row.get("date_read")
    )

    if date_read:
        return "read"

    rating = clean_value(
        row.get("user_rating")
    )

    if rating is not None:
        try:
            if float(rating) > 0:
                return "read"
        except (TypeError, ValueError):
            pass

    return "unknown"


# ---------------------------------------------------------------------------
# READER CONFIGURATION
# ---------------------------------------------------------------------------


def get_reader_id(
    reader: str,
    row: pd.Series,
) -> str:
    """
    Build a stable application reader ID.

    The current datasets represent Goodreads readers, so the source
    identifier is retained rather than inventing a new identity.
    """

    source_user_id = clean_value(
        row.get("source_user_id")
    )

    if source_user_id:
        return f"goodreads_{source_user_id}"

    # Current enriched datasets may not carry source_user_id.
    # Fall back to the configured reader name.
    return f"goodreads_{reader}"


# ---------------------------------------------------------------------------
# LOAD READERS
# ---------------------------------------------------------------------------


def load_reader_data() -> pd.DataFrame:
    """
    Load all reader-level enriched datasets into one dataframe.

    Every input row is preserved.
    """

    frames: list[pd.DataFrame] = []

    for reader, path in READER_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing enriched reader file for "
                f"{reader}: {path}"
            )

        frame = pd.read_csv(path)

        frame["reader_name"] = reader

        frames.append(frame)

        print(
            f"Loaded {reader}: "
            f"{len(frame):,} records"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


# ---------------------------------------------------------------------------
# BUILD CANONICAL DATASETS
# ---------------------------------------------------------------------------


def build_canonical_datasets(
    combined: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build:

        canonical_books
        reader_books
    """

    canonical_records: dict[
        str,
        dict[str, Any],
    ] = {}

    reader_records: list[
        dict[str, Any]
    ] = []

    for _, row in combined.iterrows():

        reader = str(
            row.get("reader_name", "")
        )

        source = "goodreads"

        source_book_id = clean_value(
            row.get("source_book_id")
        )

        source_book_id = (
            str(source_book_id)
            if source_book_id is not None
            else ""
        )

        title = clean_value(
            row.get("title")
        )

        author = clean_value(
            row.get("author")
        )

        title = (
            str(title)
            if title is not None
            else ""
        )

        author = (
            str(author)
            if author is not None
            else ""
        )

        openlibrary_work_id = (
            clean_value(
                row.get(
                    "openlibrary_work_id"
                )
            )
        )

        openlibrary_edition_id = (
            clean_value(
                row.get(
                    "openlibrary_edition_id"
                )
            )
        )

        isbn = clean_value(
            row.get("isbn")
        )

        isbn = (
            str(isbn)
            if isbn is not None
            else ""
        )

        (
            canonical_book_id,
            identity_method,
            identity_confidence,
        ) = build_canonical_book_id(
            openlibrary_work_id=(
                str(openlibrary_work_id)
                if openlibrary_work_id is not None
                else ""
            ),
            openlibrary_edition_id=(
                str(openlibrary_edition_id)
                if openlibrary_edition_id is not None
                else ""
            ),
            isbn=isbn,
            title=title,
            author=author,
            source=source,
            source_book_id=source_book_id,
        )

        # ---------------------------------------------------------------
        # Canonical book
        # ---------------------------------------------------------------

        if canonical_book_id not in canonical_records:

            canonical_records[
                canonical_book_id
            ] = {
                "canonical_book_id": canonical_book_id,
                "title": title,
                "author": author,
                "isbn": clean_value(
                    row.get("isbn")
                ),
                "pages": clean_value(
                    row.get("pages")
                ),
                "publication_year": clean_value(
                    row.get("publication_year")
                ),
                "description": clean_value(
                    row.get("description")
                ),
                "cover_url": clean_value(
                    row.get("cover_url")
                ),
                "openlibrary_work_id": (
                    normalize_openlibrary_id(
                        openlibrary_work_id
                    )
                    if openlibrary_work_id
                    else None
                ),
                "openlibrary_edition_id": (
                    normalize_openlibrary_id(
                        openlibrary_edition_id
                    )
                    if openlibrary_edition_id
                    else None
                ),
                "identity_method": identity_method,
                "identity_confidence": identity_confidence,
                "subjects": json.dumps(
                    parse_json_list(
                        row.get("subjects")
                    ),
                    ensure_ascii=False,
                ),
                "subject_people": json.dumps(
                    parse_json_list(
                        row.get("subject_people")
                    ),
                    ensure_ascii=False,
                ),
                "subject_places": json.dumps(
                    parse_json_list(
                        row.get("subject_places")
                    ),
                    ensure_ascii=False,
                ),
                "subject_times": json.dumps(
                    parse_json_list(
                        row.get("subject_times")
                    ),
                    ensure_ascii=False,
                ),
            }

        # ---------------------------------------------------------------
        # Reader/book relationship
        # ---------------------------------------------------------------

        reader_id = get_reader_id(
            reader,
            row,
        )

        reading_status = derive_reading_status(
            row
        )

        reading_record_id = create_id(
            source,
            reader_id,
            source_book_id,
        )

        reader_records.append(
            {
                "reading_record_id": reading_record_id,
                "reader_id": reader_id,
                "reader_name": reader,
                "canonical_book_id": canonical_book_id,
                "source": source,
                "source_book_id": source_book_id,
                "reading_status": reading_status,
                "user_rating": clean_value(
                    row.get("user_rating")
                ),
                "date_read": clean_value(
                    row.get("date_read")
                ),
                "date_added": clean_value(
                    row.get("date_added")
                ),
                "date_created": clean_value(
                    row.get("date_created")
                ),
                "shelves": clean_value(
                    row.get("shelves")
                ),
                "description": clean_value(
                    row.get("description")
                ),
                "identity_method": identity_method,
                "identity_confidence": identity_confidence,
            }
        )

    canonical_books = pd.DataFrame(
        list(
            canonical_records.values()
        )
    )

    reader_books = pd.DataFrame(
        reader_records
    )

    return (
        canonical_books,
        reader_books,
    )


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------


def validate_outputs(
    source: pd.DataFrame,
    canonical_books: pd.DataFrame,
    reader_books: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("CANONICAL IDENTITY VALIDATION")
    print("=" * 70)

    source_count = len(source)
    reader_count = len(reader_books)
    canonical_count = len(canonical_books)

    print(
        f"Source reader/book records: "
        f"{source_count:,}"
    )

    print(
        f"Reader/book records retained: "
        f"{reader_count:,}"
    )

    print(
        f"Canonical books: "
        f"{canonical_count:,}"
    )

    # ---------------------------------------------------------------
    # Record preservation
    # ---------------------------------------------------------------

    if source_count != reader_count:
        raise AssertionError(
            "Record count changed during "
            "canonical identity integration."
        )

    print(
        "Record preservation: PASS"
    )

    # ---------------------------------------------------------------
    # Missing canonical IDs
    # ---------------------------------------------------------------

    missing_ids = reader_books[
        reader_books["canonical_book_id"]
        .isna()
        | (
            reader_books[
                "canonical_book_id"
            ].astype(str).str.strip()
            == ""
        )
    ]

    print(
        f"Missing canonical IDs: "
        f"{len(missing_ids):,}"
    )

    if len(missing_ids) > 0:
        raise AssertionError(
            "Some reader/book records do not "
            "have canonical IDs."
        )

    # ---------------------------------------------------------------
    # Identity method
    # ---------------------------------------------------------------

    print()
    print("Identity method:")
    print(
        reader_books[
            "identity_method"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print("Identity confidence:")
    print(
        reader_books[
            "identity_confidence"
        ].value_counts(
            dropna=False
        )
    )

    # ---------------------------------------------------------------
    # Shared canonical books
    # ---------------------------------------------------------------

    reader_counts = (
        reader_books.groupby(
            "canonical_book_id"
        )["reader_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    shared_books = reader_counts[
        reader_counts > 1
    ]

    print()
    print(
        f"Canonical books shared by "
        f"multiple readers: "
        f"{len(shared_books):,}"
    )

    # ---------------------------------------------------------------
    # Work IDs
    # ---------------------------------------------------------------

    with_work = reader_books[
        reader_books[
            "canonical_book_id"
        ].notna()
    ]

    print()
    print(
        "Open Library Work coverage:"
    )

    if (
        "openlibrary_work_id"
        in canonical_books.columns
    ):
        work_count = (
            canonical_books[
                "openlibrary_work_id"
            ]
            .notna()
            .sum()
        )

        print(
            f"Canonical books with Work ID: "
            f"{work_count:,}"
        )

    # ---------------------------------------------------------------
    # Reader-level counts
    # ---------------------------------------------------------------

    print()
    print("Reader record counts:")

    print(
        reader_books.groupby(
            "reader_name"
        ).size()
    )

    # ---------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------

    print()
    print("Reading status:")

    print(
        reader_books[
            "reading_status"
        ].value_counts(
            dropna=False
        )
    )


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------


def save_outputs(
    canonical_books: pd.DataFrame,
    reader_books: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_path = (
        OUTPUT_DIR
        / "canonical_books.csv"
    )

    reader_books_path = (
        OUTPUT_DIR
        / "reader_books.csv"
    )

    canonical_books.to_csv(
        canonical_path,
        index=False,
    )

    reader_books.to_csv(
        reader_books_path,
        index=False,
    )

    print()
    print(
        f"Saved canonical books: "
        f"{canonical_path}"
    )

    print(
        f"Saved reader books: "
        f"{reader_books_path}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build canonical book identity "
            "datasets from enriched reader data."
        )
    )

    parser.parse_args()

    print()
    print("=" * 70)
    print(
        "READING INTELLIGENCE DASHBOARD"
    )
    print(
        "CANONICAL IDENTITY INTEGRATION"
    )
    print("=" * 70)
    print()

    combined = load_reader_data()

    print()
    print(
        f"Total reader/book records: "
        f"{len(combined):,}"
    )

    canonical_books, reader_books = (
        build_canonical_datasets(
            combined
        )
    )

    validate_outputs(
        source=combined,
        canonical_books=canonical_books,
        reader_books=reader_books,
    )

    save_outputs(
        canonical_books=canonical_books,
        reader_books=reader_books,
    )

    print()
    print("=" * 70)
    print("CANONICAL IDENTITY INTEGRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()