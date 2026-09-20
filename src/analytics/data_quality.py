import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "reading_intelligence.db"


def get_connection() -> sqlite3.Connection:
    """Connect to the canonical reading database."""
    return sqlite3.connect(DATABASE_FILE)


def get_reading_status_summary() -> pd.DataFrame:
    """Return the number of books in each reading status."""

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                reading_status,
                COUNT(*) AS book_count
            FROM reading_record
            GROUP BY reading_status
            ORDER BY reading_status;
            """,
            connection,
        )


def get_data_quality_summary() -> dict:
    """Return high-level reading data quality and library metrics."""

    with get_connection() as connection:

        total_books = pd.read_sql_query(
            """
            SELECT COUNT(*) AS count
            FROM reading_record;
            """,
            connection,
        ).iloc[0]["count"]

        read_books = pd.read_sql_query(
            """
            SELECT COUNT(*) AS count
            FROM reading_record
            WHERE reading_status = 'read';
            """,
            connection,
        ).iloc[0]["count"]

        needs_confirmation = pd.read_sql_query(
            """
            SELECT COUNT(*) AS count
            FROM reading_record
            WHERE reading_status = 'unknown';
            """,
            connection,
        ).iloc[0]["count"]

        unrated_reads = pd.read_sql_query(
            """
            SELECT COUNT(*) AS count
            FROM reading_record
            WHERE reading_status = 'read'
              AND (rating IS NULL OR rating = 0);
            """,
            connection,
        ).iloc[0]["count"]

        reads_without_reviews = pd.read_sql_query(
            """
            SELECT COUNT(*) AS count
            FROM reading_record
            WHERE reading_status = 'read'
              AND (review IS NULL OR TRIM(review) = '');
            """,
            connection,
        ).iloc[0]["count"]

    return {
        "total_books": int(total_books),
        "read_books": int(read_books),
        "needs_confirmation": int(needs_confirmation),
        "unrated_reads": int(unrated_reads),
        "reads_without_reviews": int(reads_without_reviews),
    }


if __name__ == "__main__":
    print("Reading Intelligence Data Quality")
    print("=" * 40)

    print("\nReading status:")
    print(
        get_reading_status_summary()
        .to_string(index=False)
    )

    print("\nData quality and library metrics:")

    summary = get_data_quality_summary()

    for metric, value in summary.items():
        print(f"{metric}: {value}")