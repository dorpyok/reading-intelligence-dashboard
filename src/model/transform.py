import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from database import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_id(*parts: str) -> str:
    """
    Create a stable ID from one or more values.

    Stable IDs mean that running the same data through the pipeline
    again produces the same canonical IDs.
    """

    value = "|".join(str(part) for part in parts)

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def split_shelves(shelf_value) -> list[str]:
    """Convert a Goodreads shelf value into a list of shelf names."""

    if pd.isna(shelf_value):
        return []

    shelves = str(shelf_value).split(",")

    return [
        shelf.strip()
        for shelf in shelves
        if shelf.strip()
    ]


def normalize_rating(value):
    """
    Convert Goodreads rating values into canonical ratings.

    Goodreads uses 0 to mean the user has not rated the book.
    """

    if pd.isna(value):
        return None

    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None

    if rating == 0:
        return None

    return rating


def derive_reading_status(row, shelves):
    """
    Derive canonical reading status from Goodreads evidence.

    Evidence hierarchy:
    1. did-not-finish shelf
    2. currently-reading shelf
    3. to-read shelf
    4. date_read exists
    5. rating > 0
    6. otherwise unknown

    Important:
    - date_added is NOT a read date.
    - Blank shelves do NOT imply read or unread.
    - Ratings are treated as evidence that the book was read.
    - Custom shelves are preserved but are not interpreted as reading status.
    """

    if "did-not-finish" in shelves:
        return "did_not_finish"

    if "currently-reading" in shelves:
        return "currently_reading"

    if "to-read" in shelves:
        return "to_read"

    date_read = row.get("date_read")
    if pd.notna(date_read) and str(date_read).strip():
        return "read"

    rating = row.get("user_rating")
    if pd.notna(rating):
        try:
            if float(rating) > 0:
                return "read"
        except (TypeError, ValueError):
            pass

    return "unknown"


def load_raw_data(input_file: Path) -> pd.DataFrame:
    """Load the raw Goodreads CSV."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {input_file}"
        )

    return pd.read_csv(input_file)


def insert_reader(
    connection,
    reader_id: str,
    display_name: str,
    source: str,
    source_user_id: str,
    created_at: str,
) -> None:
    """Insert or update a reader."""

    connection.execute(
        """
        INSERT INTO reader (
            reader_id,
            display_name,
            source,
            source_user_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(reader_id) DO UPDATE SET
            display_name = excluded.display_name,
            source = excluded.source,
            source_user_id = excluded.source_user_id
        """,
        (
            reader_id,
            display_name,
            source,
            source_user_id,
            created_at,
        ),
    )


def transform_reader(
    connection,
    dataframe: pd.DataFrame,
    display_name: str,
    source: str,
    source_user_id: str,
) -> None:
    """Transform one reader's raw Goodreads data into canonical tables."""

    ingested_at = datetime.now(timezone.utc).isoformat()

    reader_id = f"{source}_{source_user_id}"

    insert_reader(
        connection=connection,
        reader_id=reader_id,
        display_name=display_name,
        source=source,
        source_user_id=source_user_id,
        created_at=ingested_at,
    )

    for _, row in dataframe.iterrows():

        source_book_id = str(row["source_book_id"])

        book_id = f"{source}_{source_book_id}"

        reading_record_id = create_id(
            source,
            source_user_id,
            source_book_id,
        )

        insert_book(
            connection=connection,
            book_id=book_id,
            row=row,
            created_at=ingested_at,
        )

        shelves = split_shelves(row.get("shelves"))

        reading_status = derive_reading_status(
            row=row,
            shelves=shelves,
        )

        insert_reading_record(
            connection=connection,
            reading_record_id=reading_record_id,
            reader_id=reader_id,
            book_id=book_id,
            row=row,
            reading_status=reading_status,
        )

        insert_shelves(
            connection=connection,
            reading_record_id=reading_record_id,
            shelves=shelves,
        )


def insert_book(
    connection,
    book_id: str,
    row: pd.Series,
    created_at: str,
) -> None:
    """Insert a canonical book."""

    connection.execute(
        """
        INSERT INTO book (
            book_id,
            title,
            author,
            isbn,
            pages,
            publication_year,
            description,
            cover_url,
            goodreads_book_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(book_id) DO UPDATE SET
            title = excluded.title,
            author = excluded.author,
            isbn = excluded.isbn,
            pages = excluded.pages,
            publication_year = excluded.publication_year,
            description = excluded.description,
            cover_url = excluded.cover_url,
            goodreads_book_id = excluded.goodreads_book_id
        """,
        (
            book_id,
            row.get("title"),
            row.get("author"),
            row.get("isbn"),
            clean_integer(row.get("pages")),
            clean_integer(row.get("publication_year")),
            row.get("description"),
            row.get("cover_url"),
            row.get("source_book_id"),
            created_at,
        ),
    )


def insert_reading_record(
    connection,
    reading_record_id: str,
    reader_id: str,
    book_id: str,
    row: pd.Series,
    reading_status: str,
) -> None:
    """Insert a canonical reading record."""

    connection.execute(
        """
        INSERT INTO reading_record (
            reading_record_id,
            reader_id,
            book_id,
            reading_status,
            rating,
            date_added,
            date_read,
            review,
            source,
            source_record_id,
            ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(reading_record_id) DO UPDATE SET
            reading_status = excluded.reading_status,
            rating = excluded.rating,
            date_added = excluded.date_added,
            date_read = excluded.date_read,
            review = excluded.review,
            source = excluded.source,
            source_record_id = excluded.source_record_id,
            ingested_at = excluded.ingested_at
        """,
        (
            reading_record_id,
            reader_id,
            book_id,
            reading_status,
            normalize_rating(row.get("user_rating")),
            clean_date(row.get("date_added")),
            clean_date(row.get("date_read")),
            row.get("review"),
            row.get("source"),
            row.get("source_book_id"),
            row.get("ingested_at"),
        ),
    )


def insert_shelves(
    connection,
    reading_record_id: str,
    shelves: list[str],
) -> None:
    """Insert the reader's original shelf labels."""

    # Remove existing shelves for this reading record so that
    # rerunning the transformation does not create duplicates.
    connection.execute(
        """
        DELETE FROM shelf
        WHERE reading_record_id = ?
        """,
        (reading_record_id,),
    )

    for shelf_name in shelves:

        shelf_id = create_id(
            reading_record_id,
            shelf_name,
        )

        connection.execute(
            """
            INSERT INTO shelf (
                shelf_id,
                reading_record_id,
                shelf_name
            )
            VALUES (?, ?, ?)
            """,
            (
                shelf_id,
                reading_record_id,
                shelf_name,
            ),
        )


def clean_integer(value):
    """Convert a value to an integer or return None."""

    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def clean_date(value):
    """Return a clean date string or None."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Transform Goodreads raw data into the canonical model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw Goodreads CSV.",
    )

    parser.add_argument(
        "--display-name",
        required=True,
        help="Display name for the reader.",
    )

    parser.add_argument(
        "--source-user-id",
        required=True,
        help="Source-system user ID.",
    )

    parser.add_argument(
        "--source",
        default="goodreads",
        help="Source system. Defaults to Goodreads.",
    )

    args = parser.parse_args()

    input_file = Path(args.input)

    dataframe = load_raw_data(input_file)

    print(f"Loaded {len(dataframe):,} raw records.")
    print(f"Reader: {args.display_name}")
    print(f"Source user ID: {args.source_user_id}")

    with get_connection() as connection:

        transform_reader(
            connection=connection,
            dataframe=dataframe,
            display_name=args.display_name,
            source=args.source,
            source_user_id=args.source_user_id,
        )

    print("Canonical transformation complete.")


if __name__ == "__main__":
    main()