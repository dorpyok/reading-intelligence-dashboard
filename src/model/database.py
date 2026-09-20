import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_FILE = DATA_DIR / "reading_intelligence.db"


def get_connection() -> sqlite3.Connection:
    """Create a connection to the canonical reading database."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_FILE)

    # Enforce foreign-key relationships.
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection


def create_tables() -> None:
    """Create the canonical data model tables."""

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reader (
                reader_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_user_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS book (
                book_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                isbn TEXT,
                pages INTEGER,
                publication_year INTEGER,
                description TEXT,
                cover_url TEXT,
                goodreads_book_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reading_record (
                reading_record_id TEXT PRIMARY KEY,
                reader_id TEXT NOT NULL,
                book_id TEXT NOT NULL,
                reading_status TEXT NOT NULL,
                rating REAL,
                date_added TEXT,
                date_read TEXT,
                review TEXT,
                source TEXT NOT NULL,
                source_record_id TEXT,
                ingested_at TEXT NOT NULL,

                FOREIGN KEY (reader_id)
                    REFERENCES reader (reader_id),

                FOREIGN KEY (book_id)
                    REFERENCES book (book_id)
            );
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shelf (
                shelf_id TEXT PRIMARY KEY,
                reading_record_id TEXT NOT NULL,
                shelf_name TEXT NOT NULL,

                FOREIGN KEY (reading_record_id)
                    REFERENCES reading_record (reading_record_id)
            );
            """
        )


if __name__ == "__main__":
    create_tables()
    print(f"Canonical database created at: {DATABASE_FILE}")